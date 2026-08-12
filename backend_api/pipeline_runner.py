"""Web subprocess entry points for the existing pipeline modules."""

from __future__ import annotations

import argparse
from pathlib import Path


def _disable_numba_disk_cache() -> None:
    """Keep JIT enabled while avoiding broken cache locators in web subprocesses."""

    try:
        import numba
        import numba.core.decorators
    except Exception:
        return

    real_jit = numba.jit

    def jit_without_cache(*args, **kwargs):
        kwargs["cache"] = False
        return real_jit(*args, **kwargs)

    numba.jit = jit_without_cache
    numba.core.decorators.jit = jit_without_cache

    real_guvectorize = getattr(numba, "guvectorize", None)
    if real_guvectorize is not None:

        def guvectorize_without_cache(*args, **kwargs):
            kwargs["cache"] = False
            return real_guvectorize(*args, **kwargs)

        numba.guvectorize = guvectorize_without_cache

    real_vectorize = getattr(numba, "vectorize", None)
    if real_vectorize is not None:

        def vectorize_without_cache(*args, **kwargs):
            kwargs["cache"] = False
            return real_vectorize(*args, **kwargs)

        numba.vectorize = vectorize_without_cache


def run_diarization(args: argparse.Namespace) -> None:
    _disable_numba_disk_cache()

    from Project.run_diarization import run_pipeline

    run_pipeline(
        args.input,
        output_dir=args.output_dir,
        reference_rttm=args.reference_rttm,
        device=args.device,
        backend=args.backend,
        num_speakers=args.num_speakers,
        oracle_num_speakers=args.oracle_num_speakers,
        save_json=args.save_json,
        save_csv=args.save_csv,
        save_rttm=args.save_rttm,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run web-safe pipeline subprocesses.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diarization = subparsers.add_parser("diarization")
    diarization.add_argument("--input", required=True, type=Path)
    diarization.add_argument("--output-dir", required=True, type=Path)
    diarization.add_argument("--reference-rttm", type=Path, default=None)
    diarization.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    diarization.add_argument("--backend", choices=["auto", "pyannote", "webrtc", "energy"], default=None)
    diarization.add_argument("--num-speakers", type=int, default=None)
    diarization.add_argument("--oracle-num-speakers", action=argparse.BooleanOptionalAction, default=None)
    diarization.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=True)
    diarization.add_argument("--save-csv", action=argparse.BooleanOptionalAction, default=True)
    diarization.add_argument("--save-rttm", action=argparse.BooleanOptionalAction, default=True)

    transcription = subparsers.add_parser("transcription")
    transcription.add_argument("--input", required=True, type=Path)
    transcription.add_argument("--timeline", required=True, type=Path)
    transcription.add_argument("--output-dir", required=True, type=Path)
    transcription.add_argument("--model", default="base")
    transcription.add_argument("--language", default="auto")
    transcription.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    transcription.add_argument("--no-json", action="store_true", default=False)
    transcription.add_argument("--no-txt", action="store_true", default=False)
    transcription.add_argument("--no-srt", action="store_true", default=False)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "diarization":
        run_diarization(args)
    elif args.command == "transcription":
        run_transcription(args)


def run_transcription(args: argparse.Namespace) -> None:
    _disable_numba_disk_cache()

    from Project.transcription.whisper_transcription import transcribe_speaker_timeline

    device = None if args.device == "auto" else args.device
    transcribe_speaker_timeline(
        audio_path=args.input,
        timeline_path=args.timeline,
        output_dir=args.output_dir,
        model_name=args.model,
        language=args.language,
        device=device,
        save_json=not args.no_json,
        save_txt=not args.no_txt,
        save_srt=not args.no_srt,
    )


if __name__ == "__main__":
    main()
