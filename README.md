# Speech Separation & Speaker Diarization

Production web interface for an existing Python speech separation and speaker diarization project. The core research pipeline remains in `Project/`; the new app adds a FastAPI wrapper and a modern React dashboard for uploads, processing logs, waveform playback, visual results, and downloads.

## What The Pipeline Does

```text
Input Audio
-> Speech Separation
-> Audio Preprocessing
-> Voice Activity Detection
-> Speech Segmentation
-> WavLM Embedding Extraction
-> Embedding Normalization
-> Spectral Clustering
-> Speaker Timeline Reconstruction
-> RTTM Output
-> DER/JER Evaluation
```

## Folder Structure

```text
Project/                  Existing backend algorithms
backend_api/              FastAPI wrapper around existing modules
frontend/                 React 19 + Vite application
uploads/                  Runtime uploaded audio
results/                  Runtime generated artifacts and ZIP
tests/                    Existing backend unit tests
```

## Backend API

Install Python dependencies:

```powershell
pip install -r Project/requirements.txt
pip install -r backend_api/requirements.txt
```

Run the API:

```powershell
uvicorn backend_api.main:app --host 127.0.0.1 --port 8001
```

Endpoints:

```text
POST /upload
POST /run-separation
POST /run-diarization
POST /run-complete
POST /run-transcription
GET  /status
GET  /results
GET  /download/{filename}
GET  /logs
GET  /health
DELETE /results
```

The API calls `Project.ssl_pipeline.speech_separation.separate_speech`, `Project.run_diarization.run_pipeline`, and `Project.transcription.whisper_transcription.transcribe_speaker_timeline`. It does not duplicate or rewrite the existing algorithms.

## Whisper Transcription

A speaker-attributed Whisper transcription stage is available after diarization. It consumes `timeline.json` and the processed audio file to produce transcript artifacts using the existing diarization speaker labels.

Transcript outputs include:

```text
transcript.json
transcript.txt
transcript.srt
```

## Frontend

Install dependencies:

```powershell
cd frontend
npm install
```

Run the app:

```powershell
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8001`.

## Web App Pages

- Home: SaaS-style hero, project actions, animated waveform visual, feature cards.
- About: project overview, architecture, technologies, pipeline stages.
- Process: drag-and-drop upload, audio player, WaveSurfer preview, processing buttons, progress, realtime logs.
- Results: original audio, separated speaker tracks, timeline images, statistics, charts, generated file downloads, ZIP download.
- Settings: theme toggle in the header, backend health, model info, clear results.

## Generated Files

Default output artifacts include:

```text
processed.wav
speaker1.wav
speaker2.wav
timeline.json
timeline.csv
timeline.txt
hypothesis.rttm
evaluation.json
figures/timeline.png
figures/speaker_activity.png
figures/duration_histogram.png
results.zip
```

## Testing

Existing backend tests:

```powershell
python -m unittest discover -s tests
```

Frontend production build:

```powershell
cd frontend
npm run build
```

Manual web checks:

1. Start FastAPI on port `8001`.
2. Start Vite on port `5173`.
3. Upload a `.wav`, `.mp3`, or `.m4a` file.
4. Run separation, diarization, or the complete pipeline.
5. Confirm buttons disable while processing.
6. Confirm `/status` and `/logs` update.
7. Confirm Results previews audio, images, charts, and downloads.

## Security

Do not commit Hugging Face access tokens. Pyannote and model downloads read credentials from the `HUGGINGFACE_TOKEN` environment variable when needed.
