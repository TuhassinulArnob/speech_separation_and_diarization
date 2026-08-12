"""FastAPI application that wraps the existing audio pipeline modules."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Librosa imports Numba-decorated functions at module import time. Under the
# FastAPI/Uvicorn process on this Windows/Python combination, Numba can fail
# while resolving its cache locator. Disabling JIT for the API process keeps the
# existing preprocessing module usable without changing the backend pipeline.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT_DIR / "uploads"
RESULTS_DIR = ROOT_DIR / "results"
RUNTIME_DIR = ROOT_DIR / ".web_runtime"
LOG_FILE = RUNTIME_DIR / "backend_api.log"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}

RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)


@dataclass
class JobState:
    status: Literal["idle", "uploaded", "running", "complete", "error"] = "idle"
    stage: str = "Waiting for audio"
    progress: int = 0
    logs: list[str] = field(default_factory=list)
    job_id: str | None = None
    upload_path: Path | None = None
    original_filename: str | None = None
    result_dir: Path | None = None
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None


STATE = JobState()
LOCK = threading.Lock()

app = FastAPI(
    title="Speech Separation and Speaker Diarization API",
    version="1.0.0",
    description="Thin FastAPI wrapper around the completed Python audio pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StatusResponse(BaseModel):
    status: str
    stage: str
    progress: int
    uploaded_file: str | None
    job_id: str | None
    processing_time: float | None
    error: str | None


class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    kind: str
    previewable: bool


class ResultsResponse(BaseModel):
    job_id: str | None
    original_audio: FileInfo | None
    separated_audio: list[FileInfo]
    figures: list[FileInfo]
    files: list[FileInfo]
    statistics: dict[str, Any]
    transcript: dict[str, Any] | None = None


def _safe_error(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def _log(message: str, progress: int | None = None) -> None:
    timestamp = time.strftime("%H:%M:%S")
    STATE.logs.append(f"[{timestamp}] {message}")
    STATE.stage = message
    if progress is not None:
        STATE.progress = progress
    LOGGER.info(message)


def _current_processing_time() -> float | None:
    if STATE.started_at is None:
        return None
    end = STATE.finished_at or time.time()
    return round(end - STATE.started_at, 2)


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".wav", ".mp3", ".m4a"}:
        return "audio"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix in {".json", ".csv", ".txt", ".rttm"}:
        return "text"
    if suffix == ".zip":
        return "archive"
    return "file"


def _file_info(path: Path) -> FileInfo:
    relative = path.relative_to(RESULTS_DIR).as_posix()
    kind = _file_kind(path)
    return FileInfo(
        name=path.name,
        path=relative,
        size=path.stat().st_size,
        kind=kind,
        previewable=kind in {"audio", "image", "text"},
    )


def _upload_file_info(path: Path) -> FileInfo:
    kind = _file_kind(path)
    relative = path.relative_to(UPLOAD_DIR).as_posix()
    return FileInfo(
        name=path.name,
        path=relative,
        size=path.stat().st_size,
        kind=kind,
        previewable=kind == "audio",
    )


def _collect_results() -> ResultsResponse:
    result_dir = STATE.result_dir or RESULTS_DIR
    files = [path for path in result_dir.rglob("*") if path.is_file() and path.name != ".gitkeep"]
    infos = [_file_info(path) for path in sorted(files)]

    separated = [item for item in infos if item.kind == "audio" and item.name.startswith("speaker")]
    figures = [item for item in infos if item.kind == "image"]
    original = _upload_file_info(STATE.upload_path) if STATE.upload_path and STATE.upload_path.exists() else None

    transcript_data = None
    transcript_json = result_dir / "transcript.json"
    if transcript_json.exists():
        try:
            transcript_data = json.loads(transcript_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            transcript_data = None

    return ResultsResponse(
        job_id=STATE.job_id,
        original_audio=original,
        separated_audio=separated,
        figures=figures,
        files=infos,
        statistics=_statistics(result_dir, infos),
        transcript=transcript_data,
    )


def _statistics(result_dir: Path, files: list[FileInfo]) -> dict[str, Any]:
    timeline_path = result_dir / "timeline.json"
    evaluation_path = result_dir / "evaluation.json"
    timeline: list[dict[str, Any]] = []
    evaluation: dict[str, Any] = {}

    if timeline_path.exists():
        try:
            timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            timeline = []
    if evaluation_path.exists():
        try:
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            evaluation = {}

    speakers = {
        row.get("speaker_id", row.get("speaker", row.get("label")))
        for row in timeline
        if isinstance(row, dict)
    }
    durations = [
        float(row.get("end", 0)) - float(row.get("start", 0))
        for row in timeline
        if isinstance(row, dict)
    ]
    embedding_dim = _embedding_dimension(result_dir)

    models = ["SpeechBrain SepFormer", "WavLM", "Spectral Clustering"]
    if any(path.name.startswith("transcript") for path in files):
        models.append("Whisper")

    return {
        "numberOfSpeakers": len([speaker for speaker in speakers if speaker is not None]),
        "speechSegments": len(timeline),
        "embeddingDimension": embedding_dim,
        "processingTime": _current_processing_time(),
        "modelsUsed": models,
        "outputFilesGenerated": len(files),
        "totalSpeechDuration": round(sum(max(duration, 0.0) for duration in durations), 2),
        "evaluation": evaluation,
    }


def _embedding_dimension(result_dir: Path) -> int | None:
    embeddings_path = result_dir / "normalized_embeddings.npz"
    if not embeddings_path.exists():
        return None
    try:
        import numpy as np

        with np.load(embeddings_path) as data:
            vectors = data.get("vectors")
            if vectors is not None and len(vectors.shape) == 2:
                return int(vectors.shape[1])
    except Exception:
        return None
    return None


def _zip_results(result_dir: Path) -> Path:
    zip_path = result_dir / "results.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in result_dir.rglob("*"):
            if path.is_file() and path.name not in {".gitkeep", "results.zip"}:
                archive.write(path, path.relative_to(result_dir))
    return zip_path


def _runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT_DIR)
        if not python_path
        else f"{ROOT_DIR}{os.pathsep}{python_path}"
    )
    hf_home = env.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    env["HF_HOME"] = hf_home
    env["TRANSFORMERS_CACHE"] = env.get("TRANSFORMERS_CACHE") or str(Path(hf_home) / "hub")
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("NUMBA_DISABLE_CACHE", "1")
    env.setdefault("NUMBA_CACHE_DIR", str(RUNTIME_DIR / "numba_cache"))
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def _run_subprocess(
    command: list[str],
    *,
    label: str,
    progress: int,
) -> None:
    _log(label, progress)
    LOGGER.info("Starting subprocess: %s", " ".join(command))

    return_code = _run_subprocess_blocking(command)
    if return_code != 0:
        raise RuntimeError(f"{label} failed with exit code {return_code}.")


def _run_subprocess_blocking(command: list[str]) -> int:
    process = subprocess.Popen(
        command,
        cwd=ROOT_DIR,
        env=_runtime_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        text = line.strip()
        if text:
            _log(text)
    return process.wait()


def _run_job(mode: Literal["separation", "diarization", "complete", "transcription"]) -> None:
    with LOCK:
        if STATE.upload_path is None or STATE.job_id is None:
            STATE.status = "error"
            STATE.error = "Upload an audio file before running the pipeline."
            return

        STATE.status = "running"
        STATE.error = None
        STATE.started_at = time.time()
        STATE.finished_at = None
        STATE.result_dir = RESULTS_DIR / STATE.job_id

        if mode in {"separation", "diarization", "complete"}:
            _clean_dir(STATE.result_dir)

        try:
            if mode in {"separation", "complete"}:
                _run_subprocess(
                    [
                        sys.executable,
                        "-m",
                        "Project.ssl_pipeline.speech_separation",
                        str(STATE.upload_path),
                        "--output-dir",
                        str(STATE.result_dir),
                    ],
                    label="Running Speech Separation...",
                    progress=18 if mode == "complete" else 10,
                )
                _log("Speech Separation finished.", 35 if mode == "complete" else 100)

            if mode in {"diarization", "complete"}:
                _run_subprocess(
                    [
                        sys.executable,
                        "-m",
                        "backend_api.pipeline_runner",
                        "diarization",
                        "--input",
                        str(STATE.upload_path),
                        "--output-dir",
                        str(STATE.result_dir),
                    ],
                    label="Running Diarization Pipeline...",
                    progress=42 if mode == "complete" else 10,
                )
                _log("Diarization finished.", 96)

            if mode in {"complete", "transcription"}:
                timeline_path = STATE.result_dir / "timeline.json"
                if not timeline_path.exists():
                    raise RuntimeError("Diarization timeline not found. Run diarization before transcription.")
                _run_subprocess(
                    [
                        sys.executable,
                        "-m",
                        "backend_api.pipeline_runner",
                        "transcription",
                        "--input",
                        str(STATE.result_dir / "processed.wav"),
                        "--timeline",
                        str(timeline_path),
                        "--output-dir",
                        str(STATE.result_dir),
                    ],
                    label="Running Whisper Transcription...",
                    progress=80 if mode == "complete" else 60,
                )
                _log("Transcription finished.", 98)

            _zip_results(STATE.result_dir)
            STATE.finished_at = time.time()
            STATE.status = "complete"
            _log("Finished.", 100)
        except Exception as exc:
            LOGGER.exception("Pipeline job failed")
            STATE.finished_at = time.time()
            STATE.status = "error"
            STATE.error = _safe_error(exc)
            _log(f"Failed: {STATE.error}")


@app.on_event("startup")
def startup() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    LOGGER.warning("HTTP error %s: %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status": "error"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled API exception")
    return JSONResponse(
        status_code=500,
        content={"detail": _safe_error(exc), "status": "error"},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Only wav, mp3, and m4a files are supported.")

        job_id = uuid4().hex
        _clean_dir(UPLOAD_DIR)
        upload_path = UPLOAD_DIR / job_id / f"{uuid4().hex}{suffix}"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        with upload_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)

        STATE.job_id = job_id
        STATE.status = "uploaded"
        STATE.stage = "Audio uploaded"
        STATE.progress = 0
        STATE.logs = [f"[{time.strftime('%H:%M:%S')}] Uploaded {file.filename}"]
        STATE.upload_path = upload_path
        STATE.original_filename = file.filename
        STATE.result_dir = RESULTS_DIR / job_id
        STATE.started_at = None
        STATE.finished_at = None
        STATE.error = None
        LOGGER.info("Uploaded %s to %s", file.filename, upload_path)

        return {
            "filename": file.filename,
            "storedName": upload_path.relative_to(UPLOAD_DIR).as_posix(),
            "size": upload_path.stat().st_size,
            "url": f"/download-upload/{upload_path.relative_to(UPLOAD_DIR).as_posix()}",
            "jobId": job_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {_safe_error(exc)}") from exc


def _start_job(mode: Literal["separation", "diarization", "complete", "transcription"]) -> StatusResponse:
    if STATE.status == "running":
        raise HTTPException(status_code=409, detail="A job is already running.")
    if STATE.upload_path is None or not STATE.upload_path.exists():
        raise HTTPException(status_code=400, detail="Upload an audio file before running the pipeline.")
    if mode == "transcription" and (STATE.result_dir is None or not (STATE.result_dir / "timeline.json").exists()):
        raise HTTPException(status_code=400, detail="Diarization results are required before transcription.")
    STATE.status = "running"
    STATE.error = None
    STATE.started_at = time.time()
    STATE.finished_at = None
    STATE.progress = 1
    STATE.stage = f"Queued {mode} job"
    _log(f"Queued {mode} job", 1)
    threading.Thread(target=_run_job, args=(mode,), daemon=True).start()
    return get_status()


@app.post("/run-separation")
async def run_separation(_background_tasks: BackgroundTasks) -> StatusResponse:
    return _start_job("separation")


@app.post("/run-diarization")
async def run_diarization(_background_tasks: BackgroundTasks) -> StatusResponse:
    return _start_job("diarization")


@app.post("/run-complete")
async def run_complete(_background_tasks: BackgroundTasks) -> StatusResponse:
    return _start_job("complete")


@app.post("/run-transcription")
async def run_transcription(_background_tasks: BackgroundTasks) -> StatusResponse:
    return _start_job("transcription")


@app.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    return StatusResponse(
        status=STATE.status,
        stage=STATE.stage,
        progress=STATE.progress,
        uploaded_file=STATE.original_filename,
        job_id=STATE.job_id,
        processing_time=_current_processing_time(),
        error=STATE.error,
    )


@app.get("/logs")
def get_logs() -> dict[str, list[str]]:
    return {"logs": STATE.logs[-250:]}


@app.get("/results", response_model=ResultsResponse)
def get_results() -> ResultsResponse:
    if STATE.job_id is None or STATE.result_dir is None:
        raise HTTPException(status_code=404, detail="No results are available yet.")
    if STATE.status not in {"complete", "error"} and not any(STATE.result_dir.rglob("*")):
        raise HTTPException(status_code=404, detail="No results are available yet.")
    return _collect_results()


@app.get("/download/{filename:path}")
def download_result(filename: str) -> FileResponse:
    path = (RESULTS_DIR / filename).resolve()
    if RESULTS_DIR.resolve() not in path.parents and path != RESULTS_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid file path.")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, filename=path.name)


@app.get("/download-upload/{filename:path}")
def download_upload(filename: str) -> FileResponse:
    path = (UPLOAD_DIR / filename).resolve()
    if UPLOAD_DIR.resolve() not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid file path.")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, filename=STATE.original_filename or path.name)


@app.delete("/results")
def clear_results() -> dict[str, str]:
    if STATE.status == "running":
        raise HTTPException(status_code=409, detail="Cannot clear results while processing.")
    _clean_dir(RESULTS_DIR)
    STATE.job_id = None
    STATE.result_dir = None
    STATE.upload_path = None
    STATE.original_filename = None
    STATE.logs.append(f"[{time.strftime('%H:%M:%S')}] Cleared results")
    return {"status": "cleared"}
