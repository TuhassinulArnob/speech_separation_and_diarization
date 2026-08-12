# Speech Separation & Speaker Diarization

A Python speech separation, speaker diarization, and speaker-attributed speech transcription system with a FastAPI backend and React/Vite web interface.

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
-> Whisper Speech-to-Text
-> Speaker-attributed Transcript
-> DER/JER Evaluation
```

Diarization provides speaker labels and timestamps while Whisper provides speech-to-text. Whisper is not used for speaker identification.

## Folder Structure

```text
Project/                  Existing backend algorithms and pipeline code
backend_api/              FastAPI orchestration and API wrapper
frontend/                 React + Vite web interface
inputs/                   Example input audio and test data
uploads/                  Runtime uploaded audio files
results/                  Runtime generated artifacts and ZIP
outputs/                  Output folders for pipeline runs
tests/                    Backend unit tests
README.md                 Project documentation
run_diarization.py        CLI entry point for diarization
```

## Backend API

Create and activate the virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r Project/requirements.txt
python -m pip install -r backend_api/requirements.txt
```

Start the backend:

```powershell
python -m uvicorn backend_api.main:app --host 127.0.0.1 --port 8001
```

Backend URL:

```text
http://127.0.0.1:8001
```

Health check:

```text
http://127.0.0.1:8001/health
```

Implemented API endpoints:

```text
POST /upload
POST /run-separation
POST /run-diarization
POST /run-transcription
POST /run-complete
GET  /status
GET  /results
GET  /download/{filename}
GET  /logs
GET  /health
DELETE /results
```

The FastAPI backend wraps the existing `Project` modules, rather than duplicating the core algorithms.

## Whisper Transcription

Whisper is implemented for audio-to-text transcription. It consumes the diarization timeline and processed audio, then writes speaker-attributed transcript artifacts.

Generated transcript files include:

```text
transcript.json
transcript.txt
transcript.srt
```

Example transcript output:

```text
[00:00 - 00:04] Speaker 1:
Hello everyone.

[00:04 - 00:08] Speaker 2:
Hello.
```

Whisper does not perform diarization; speaker labels come from the diarization timeline.

## Frontend

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Run the frontend:

```powershell
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:5173
```

The Vite frontend communicates with the FastAPI backend on port `8001`.

Pages include:

- Home
- About
- Process
- Results
- Settings

## Generated Files

Typical output artifacts include:

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
transcript.json
transcript.txt
transcript.srt
results.zip
```

## CLI Usage

Speech separation:

```powershell
python -m Project.ssl_pipeline.speech_separation inputs\mixed.wav --output-dir outputs\separated_audio
```

Diarization:

```powershell
python run_diarization.py --input inputs\mixed.wav --output-dir outputs
```

## Testing

Backend tests:

```powershell
python -m unittest discover -s tests
```

Compile Python sources:

```powershell
python -m compileall -q Project tests backend_api
```

Frontend build:

```powershell
cd frontend
npm run build
```

## Manual Web Checks

1. Start backend on port `8001`.
2. Start frontend on port `5173`.
3. Open the frontend in a browser.
4. Upload WAV, MP3, or M4A audio.
5. Run separation, diarization, transcription, or the complete pipeline.
6. Check realtime status and logs.
7. Verify audio playback, timeline visuals, transcript output, and downloads.

## Practical Applications

- Meeting analysis
- Interview transcription
- Lecture analysis
- Call-center analysis
- Conference recordings
- Podcast processing
- Speaker-attributed transcription
- Speech and diarization research

## Technology Stack

- Python
- PyTorch
- SpeechBrain
- SepFormer
- Pyannote
- WavLM
- Whisper
- NumPy
- librosa
- soundfile
- scikit-learn
- FastAPI
- React
- Vite
- Tailwind CSS
- Framer Motion
- WaveSurfer.js
- Recharts
- Axios

## Limitations

The current implementation is validated primarily for two-speaker separation. Performance varies with noise, reverberation, overlapping speech, recording quality, speaker similarity, language/accent, and model configuration. The system is not claimed to support arbitrary unseen speaker counts.

## Security

- Never commit Hugging Face tokens.
- Use `HUGGINGFACE_TOKEN` as an environment variable when required.
- Do not commit API keys, passwords, or private credentials.
- Keep runtime directories such as `uploads/`, `results/`, and `.web_runtime/` out of Git where appropriate.

## Project Status

The project currently includes:

- Two-speaker speech separation
- Speaker diarization
- WavLM speaker embeddings
- Spectral clustering
- Speaker timeline reconstruction
- RTTM generation
- Visualization
- Whisper speech-to-text
- Speaker-attributed transcript
- FastAPI backend
- React/Vite web interface
- Local end-to-end workflow

Remaining academic work includes experimental evaluation, dataset-based testing, performance/error analysis, final documentation, and research paper preparation.

## Local Development URLs

Frontend:

```text
http://127.0.0.1:5173
```

Backend:

```text
http://127.0.0.1:8001
```

Health:

```text
http://127.0.0.1:8001/health
```

## License

This project is intended for academic and research purposes.
