# Speech Separation & Speaker Diarization

A production-grade Python speech separation, speaker diarization, and speaker-attributed transcription system with a FastAPI backend, React + Vite web interface, and full DER/JER evaluation suite.

GitHub Repository: [https://github.com/TuhassinulArnob/speech_separation_and_diarization](https://github.com/TuhassinulArnob/speech_separation_and_diarization)

---

## 1. System Architecture & Pipeline Workflow

```text
Input Audio (.wav / .flac / .mp3 / .m4a / .ogg / .opus / .sph)
   │
   ▼
1. Speech Separation (SpeechBrain SepFormer-WHAMR) ───► [Separated Channels (if multi-speaker)]
   │
   ▼
2. Audio Preprocessing (16 kHz mono conversion & peak normalization)
   │
   ▼
3. Voice Activity Detection (Adaptive Energy / WebRTC / Pyannote fallback)
   │
   ▼
4. Speech Segmentation (Configurable sliding window 2.0s / hop 1.5s)
   │
   ▼
5. WavLM Embedding Extraction (microsoft/wavlm-base-plus, 768-dim)
   │
   ▼
6. Embedding Normalization (L2 norm, Centering, Standard Scaler, PCA)
   │
   ▼
7. Spectral Clustering (Cosine affinity + Eigengap / Oracle speaker count)
   │
   ▼
8. Speaker Timeline Reconstruction (Adjacency merging & overlap resolution)
   │
   ▼
9. RTTM Export (Standard 10-field NIST format)
   │
   ▼
10. Whisper ASR Speech-to-Text (OpenAI Whisper base model)
   │
   ▼
11. Speaker-Attributed Transcription (.json, .txt, .srt subtitle export)
   │
   ▼
12. Dataset DER / JER Evaluation (NIST md-eval compatible Hungarian scoring)
```

> **Note on Separation vs Diarization**: Single-speaker recordings bypass the SepFormer stage to avoid artificial distortion. Diarization provides temporal boundaries and speaker labels, while Whisper provides linguistic transcription; Whisper is not used for speaker identity assignment.

---

## 2. Directory Structure

```text
├── Project/
│   ├── config/default.json              # Central diarization & model configuration
│   └── ssl_pipeline/
│       ├── baseline.py                  # Classical MFCC + K-Means diarization baseline
│       ├── config.py                    # Configuration schema & dataclasses
│       ├── dataset_evaluation.py        # Dataset-level DER/JER evaluation engine
│       ├── embedding_normalization.py   # L2, standard scaling, and PCA normalization
│       ├── evaluation.py                # NIST-compatible DER/JER scorer & Hungarian mapping
│       ├── preprocessing.py             # Resampling, mono conversion & normalization
│       ├── rttm.py                      # RTTM parsing, validation & serialization
│       ├── segmentation.py              # VAD-guided windowed speech segmentation
│       ├── spectral_clustering.py       # Spectral clustering, cosine affinity, eigengap
│       ├── speech_separation.py         # SpeechBrain SepFormer separation wrapper
│       ├── ssl_embedding.py             # WavLM model loader & batch embedding extractor
│       ├── timeline_reconstruction.py   # Chronological interval merging & overlap handling
│       └── vad.py                       # Adaptive energy, WebRTC, and Pyannote VAD backends
├── backend_api/
│   ├── main.py                          # FastAPI application endpoints & CORS setup
│   ├── pipeline_runner.py               # Asynchronous pipeline & evaluation execution
│   └── requirements.txt                 # Backend-specific Python dependencies
├── frontend/                            # React + Vite + Tailwind CSS web interface
│   ├── src/pages/
│   │   ├── Home.jsx                     # Landing page
│   │   ├── Process.jsx                  # Pipeline execution & parameter controls
│   │   ├── Results.jsx                  # Audio player, visual waveforms & transcript view
│   │   ├── Evaluation.jsx               # Dataset DER/JER evaluation dashboard
│   │   ├── About.jsx                    # System architecture & documentation
│   │   └── Settings.jsx                 # Pipeline threshold & device configuration
├── evaluation/                          # Reference evaluation datasets (placed locally)
│   ├── LDC2021S10.flac                  # Reference audio (DIHARD / LDC conversational speech)
│   ├── LDC2021S10.rttm                  # Ground-truth speaker turn annotations
│   ├── LDC2021S10.lab                   # Ground-truth speech activity detection (SAD) labels
│   ├── README.md                        # Dataset drop-in specifications
│   ├── manual_tests/                    # Automated testing & diagnostic suite
│   │   ├── test_A_clean/                # Clean single-speaker audio (LibriSpeech dev-clean-2)
│   │   ├── test_B_noise/                # Known silence & controlled Gaussian noise audio
│   │   ├── create_test_audio.py         # Test B audio generator script (fixed seed)
│   │   ├── run_test_a.py                # Test A runner
│   │   ├── run_test_b.py                # Test B runner
│   │   ├── run_test_c.py                # Test C runner & diagnostic analysis
│   │   ├── generate_summary.py          # Summary aggregator
│   │   └── results/                     # Test artifacts, logs, figures & audit reports
│   │       ├── summary.md               # Master summary report
│   │       ├── summary.json             # Machine-readable master summary
│   │       └── test_C_annotation_and_clustering_audit.md # Technical audit report
├── inputs/                              # Sample audio files for testing and demo runs
├── tests/                               # Comprehensive unit tests (130 automated tests)
├── run_diarization.py                   # Standalone CLI entry point for diarization
├── requirements.txt                     # Unified production Python dependencies
├── .python-version                      # Pinned Python version (3.11.9) for Render
├── render.yaml                          # Render Blueprint deployment configuration
└── README.md                            # Complete project documentation
```

---

## 3. Installation & Quick Start

### 3.1 Backend Setup

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the FastAPI backend:

```powershell
python -m uvicorn backend_api.main:app --host 127.0.0.1 --port 8001
```

- **Backend API URL**: `http://127.0.0.1:8001`
- **Interactive OpenAPI Docs**: `http://127.0.0.1:8001/docs`
- **Health Check**: `http://127.0.0.1:8001/health`

### 3.2 Frontend Setup

Install frontend dependencies:

```powershell
cd frontend
npm install
```

Start the Vite development server:

```powershell
npm run dev
```

- **Frontend URL**: `http://127.0.0.1:5173`

### 3.3 Render Deployment (Cloud Production)

The repository is configured for automated deployment on **Render** as a Python Web Service.

| Setting | Value |
|---|---|
| **Environment** | `Python` |
| **Python Version** | `3.11.9` (automatically set via `.python-version` or `PYTHON_VERSION=3.11.9`) |
| **Root Directory** | `.` (repository root) |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn backend_api.main:app --host 0.0.0.0 --port $PORT` |
| **Health Check Path** | `/health` |

#### Environment Variables (Render Dashboard):
- `PYTHON_VERSION`: `3.11.9` (prevents default selection of unsupported Python 3.14.x)
- `NUMBA_DISABLE_JIT`: `1` (prevents numba cache conflicts in containerized environments)
- `MPLBACKEND`: `Agg` (headless matplotlib rendering)
- `CORS_ORIGINS`: Your frontend URL (e.g. `https://your-frontend.onrender.com`)

Render Blueprints are also supported via the included [`render.yaml`](render.yaml).


---

## 4. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload audio file (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.opus`) |
| `POST` | `/run-separation` | Run SepFormer two-speaker separation |
| `POST` | `/run-diarization` | Run WavLM embedding extraction + Spectral clustering |
| `POST` | `/run-transcription` | Run Whisper speaker-attributed transcription |
| `POST` | `/run-complete` | Run complete end-to-end pipeline |
| `GET` | `/status` | Check pipeline execution status & progress |
| `GET` | `/results` | Fetch generated metadata, timelines, and transcripts |
| `GET` | `/download/{filename}` | Download generated artifacts (`.wav`, `.rttm`, `.json`, `.zip`) |
| `DELETE` | `/results` | Clear runtime output files |
| `POST` | `/run-evaluation` | Run DER/JER evaluation across all datasets in `evaluation/` |
| `GET` | `/evaluation/files` | Discover valid reference audio and RTTM pairs |
| `GET` | `/evaluation/status` | Check dataset evaluation execution status |
| `GET` | `/evaluation/results` | Fetch overall & recording-wise evaluation metrics |
| `DELETE` | `/evaluation/results` | Clear evaluation results while preserving reference files |

---

## 5. Automated Unit & Integration Testing

The project contains a comprehensive automated unit test suite covering every pipeline stage:

```powershell
$env:PYTHONPATH="."
.\.venv\Scripts\python.exe -m unittest discover tests
```

- **Total Tests**: **130 passed, 2 skipped, 0 failures**
- **Test Modules**:
  - `tests/test_vad.py`: Adaptive noise floor thresholding, short speech filtering, silence handling.
  - `tests/test_segmentation.py`: Sliding window, hop durations, boundary compliance.
  - `tests/test_ssl_embedding.py`: WavLM inference, batching, caching, pooling methods.
  - `tests/test_embedding_normalization.py`: L2, Standard Scaler, PCA transforms.
  - `tests/test_spectral_clustering.py`: Cosine affinity matrix, connected components, eigengap speaker estimation.
  - `tests/test_timeline_reconstruction.py`: Gap merging, smoothing, chronological ordering.
  - `tests/test_evaluation.py`: Exact synthetic scorer tests, permutation invariance, collar logic.
  - `tests/test_dataset_evaluation.py`: Dataset discovery, paired stem matching, UEM clipping.
  - `tests/test_speech_separation.py`: SepFormer integration & mock tests.
  - `tests/test_baseline.py`: Classical MFCC + K-Means baseline validation.
  - `tests/test_backend_api.py`: FastAPI endpoints, file uploads, background execution.

---

## 6. Manual Testing Workflow & Technical Audit (Tests A, B, C)

A dedicated reproducible manual testing suite is available in `evaluation/manual_tests/`:

### Test A: Clean Single-Speaker Audio
- **Input**: `evaluation/manual_tests/test_A_clean/clean_speech.flac` (LibriSpeech dev-clean-2, 15.56s, 16 kHz mono).
- **Results**:
  - **VAD**: Detected 8 speech regions (11.55s total speech).
  - **Diarization**: Detected **1 speaker** (`speaker_0`). Eigengap spectral estimation independently confirmed 1 speaker.
  - **Speech Separation (SepFormer)**: Bypassed as intended for single-speaker audio.
  - **Whisper Transcription**: Accurate transcript generated (*"John Mangles therefore hoped that the wretched hole would reach poor..."*).
- **Reproduction**:
  ```powershell
  $env:PYTHONPATH="."
  .\.venv\Scripts\python.exe evaluation/manual_tests/run_test_a.py
  ```

### Test B: Silence and Background Noise Robustness
- **Generator**: `evaluation/manual_tests/create_test_audio.py` (fixed seed=42).
- **Test Audio 1 (`speech_silence.wav`)**: Clean speech padded with 3.0s leading and 3.0s trailing digital silence (21.56s duration).
  - **Known Silence Duration**: 6.000s.
  - **False Speech in Known Silence**: **0.000s** (0.00s leading, 0.00s trailing). Perfect rejection of digital silence.
- **Test Audio 2 (`speech_noise.wav`)**: Clean speech mixed with controlled additive Gaussian white noise at target SNR=15.0 dB (measured 15.00 dB).
  - **Detected Speech Duration**: 11.250s ($\Delta = -0.30\text{s}$ from clean baseline 11.55s).
- **Reproduction**:
  ```powershell
  $env:PYTHONPATH="."
  .\.venv\Scripts\python.exe evaluation/manual_tests/create_test_audio.py
  .\.venv\Scripts\python.exe evaluation/manual_tests/run_test_b.py
  ```

### Test C: Two-Speaker Diarization & Technical Audit
- **Input Audio**: `evaluation/LDC2021S10.flac` (120.000s, 16 kHz mono, DIHARD / LDC conversational speech).
- **Reference RTTM**: `evaluation/LDC2021S10.rttm` (Recording ID: `DH_0052`, 56 speech turns, 2 speakers).
- **Reference LAB**: `evaluation/LDC2021S10.lab` (Ground-truth speech activity detection).
- **Standard Protocol**: Collar = 0.0s, Overlap = Scored (`skip_overlap=False`), UEM = full_file.
- **Diarization Output**: 46 hypothesis turns, 2 detected speakers.
- **Scoring Results**:
  - **DER**: **133.09%**
  - **JER**: **80.86%**
  - **Missed Speech**: 15.374s (27.00%)
  - **False Alarm**: 44.442s (78.05%)
  - **Speaker Confusion**: 15.966s (28.04%)
  - **Scored Reference Duration**: 56.942s
- **Reproduction**:
  ```powershell
  $env:PYTHONPATH="."
  .\.venv\Scripts\python.exe evaluation/manual_tests/run_test_c.py
  .\.venv\Scripts\python.exe evaluation/manual_tests/generate_summary.py
  ```

---

## 7. Technical Audit Findings & Root-Cause Diagnosis

A rigorous technical audit ([`test_C_annotation_and_clustering_audit.md`](file:///c:/Users/USER/Desktop/Projects/SpeechSepatarion_and_Diarization%20-%20Final/evaluation/manual_tests/results/test_C_annotation_and_clustering_audit.md)) resolved the root cause of the reported DER (133.09%):

### 7.1 Why DER Exceeds 100%
In NIST md-eval / DIHARD conventions:
$$\text{DER} = \frac{\text{Missed} + \text{False Alarm} + \text{Confusion}}{\text{Scored Reference Duration}}$$
The denominator is the total reference speech time (**56.942s**), **not** the 120s audio duration. Because the hypothesis detected 86.010s of speech, False Alarm alone (44.442s) is $78.05\%$ of the reference speech duration.

### 7.2 Duration Discrepancy Explained (58.432s vs 56.942s)
The raw reference RTTM lists 58.432s of speech. However, two turns extend past the audio boundary:
- `speaker117`: $120.095\text{s}-120.475\text{s}$ (0.380s)
- `speaker118`: $120.120\text{s}-121.230\text{s}$ (1.110s)
Total speech past 120.0s is **1.490s**. When clipped to the 120.0s audio boundary:
$$58.432\text{s} - 1.490\text{s} = \mathbf{56.942s}$$

### 7.3 False Alarm Root Cause: Incomplete Reference Annotation
- **Stage Origin**: 100.0% of the false alarm occurs directly inside VAD speech regions (VAD = 88.170s $\rightarrow$ Segments = 86.070s $\rightarrow$ Hypothesis = 86.010s).
- **Ground-Truth Corroboration**: `LDC2021S10.lab` specifies **88.696s** of active speech in the recording.
- **Whisper ASR Corroboration**: Whisper confirms continuous, intelligible conversational English throughout the unannotated intervals (e.g. $[28.2\text{s}-34.4\text{s}]$ and $[79.4\text{s}-90.9\text{s}]$).
- **Attribution**:
  - **87.0% (38.675s)** of the False Alarm is **genuine human speech** documented in `LDC2021S10.lab` but omitted from `LDC2021S10.rttm`.
  - Only **13.0% (5.767s)** represents false detection in acoustic non-speech (room reverberation, breathing, mic bumps).
- **Independent VAD Performance vs LAB Ground Truth**:
  - **Precision**: 86.27%
  - **Recall**: 85.75%
  - **F1-Score**: **86.01%**
  - **Accuracy**: 79.38%

### 7.4 Speaker Confusion & Clustering Benchmark
Evaluated across unambiguous reference segments (23 segments of `speaker117` vs 10 segments of `speaker118`):

| Clustering Method | Cluster Sizes | Diagnostic Purity | DER (%) | Speaker Confusion (s) |
|---|:---:|---:|---:|---:|
| **Baseline: Spectral (kmeans)** | [30, 38] | **54.5%** | **133.09%** | 15.966s |
| **Alt 1: Spectral (discretize)** | [38, 30] | **54.5%** | **133.09%** | 15.966s |
| **Alt 2: AHC (cosine, average)** | [26, 42] | **51.5%** | **136.58%** | 17.953s |
| **Alt 3: AHC (cosine, complete)** | [41, 27] | **57.6%** | **135.76%** | 17.477s |
| **Alt 4: KMeans (L2 normalized)** | [31, 37] | **54.5%** | **132.03%** | 15.366s |
| *WavLM Base (unsupervised)* | [30, 38] | **54.5%** | 133.09% | 15.966s |

**Technical Finding**: Changing clustering algorithms has minimal impact (~132% - 136% DER). The bottleneck is the **upstream self-supervised representation**: base WavLM models (`wavlm-base` / `wavlm-base-plus`) lack supervised speaker metric learning (e.g. angular margin / ECAPA-TDNN loss). In single-microphone conversational speech, intra-speaker phonetic variability exceeds inter-speaker vocal tract differences.

### 7.5 Overlap Handling
The pipeline enforces single-speaker hard partitions in `Project.ssl_pipeline.timeline_reconstruction._resolve_overlaps` (`current.start = resolved[-1].end`). The system **does not support overlapping-speaker diarization**, resulting in 1.988s of overlapping reference speech being classified as missed speech.

---

## 8. Technology Stack

- **Deep Learning & Audio**: PyTorch, SpeechBrain (SepFormer), Hugging Face Transformers (WavLM Base+), OpenAI Whisper
- **Scientific Computing**: NumPy, SciPy, scikit-learn, soundfile
- **Backend API**: FastAPI, Uvicorn, Pydantic, Python Standard Library
- **Frontend Web UI**: React 18, Vite, Tailwind CSS, Framer Motion, WaveSurfer.js, Recharts, Axios
- **Evaluation & Formats**: NIST RTTM, NIST md-eval compatible Hungarian algorithm, WebVTT/SRT subtitles

---

## 9. Known Limitations

1. **Reference RTTM Incompleteness**: Whole-file evaluation on `LDC2021S10` cannot produce a calibrated benchmark score without an oracle UEM specifying which intervals were transcribed.
2. **Speaker Representation Metric Head**: Raw base WavLM frame representations capture phonetic and acoustic room properties. High-accuracy speaker clustering on natural single-channel conversational speech requires a downstream speaker-discriminative model (e.g. ECAPA-TDNN trained with AAM-softmax on VoxCeleb).
3. **Single-Speaker Timeline Partitioning**: Overlapping speech from secondary speakers is collapsed into single-speaker turns.
4. **Speech Separation Domain**: SpeechBrain SepFormer is trained on two-speaker synthetic mixtures; separation performance varies on multi-speaker or noisy real-world recordings.

---

## 10. License

This project is intended for academic, research, and educational purposes.
