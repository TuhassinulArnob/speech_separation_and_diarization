# 🎙️ Speech Separation and Speaker Diarization

A complete Python-based speech processing project that performs **speech separation** and **speaker diarization** on multi-speaker audio recordings.

The system separates mixed speech into individual speaker audio streams and identifies **who spoke when** using pretrained **WavLM embeddings** and **Spectral Clustering**.

---

# ✨ Features

## Speech Separation

- Pretrained SpeechBrain SepFormer
- Asteroid Conv-TasNet fallback
- CPU/GPU support
- Saves separated audio
- Waveform comparison
- Spectrogram visualization

---

## Speaker Diarization

- Audio preprocessing
- Voice Activity Detection (VAD)
- Speech Segmentation
- WavLM Embedding Extraction
- Embedding Normalization
- Spectral Clustering
- Speaker Timeline Reconstruction
- RTTM Generation
- DER & JER Evaluation

---

## Baseline System

Classical speaker diarization pipeline:

MFCC → K-Means → Timeline → RTTM → DER/JER

Used for comparison with the proposed method.

---

# 🏗️ Project Pipeline

```text
                Mixed Audio
                     │
                     ▼
          Speech Separation
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    Speaker 1.wav          Speaker 2.wav
                     │
                     ▼
          Audio Preprocessing
                     │
                     ▼
      Voice Activity Detection
                     │
                     ▼
        Speech Segmentation
                     │
                     ▼
     WavLM Embedding Extraction
                     │
                     ▼
     Embedding Normalization
                     │
                     ▼
        Spectral Clustering
                     │
                     ▼
    Speaker Timeline Reconstruction
                     │
                     ▼
            RTTM Generation
                     │
                     ▼
            DER / JER Evaluation
```

---

# 📁 Project Structure

```text
SpeechSeparation_and_Diarization/

│
├── inputs/
│   └── mixed.wav
│
├── outputs/
│   ├── separated_audio/
│   ├── embeddings/
│   ├── timeline/
│   ├── rttm/
│   ├── evaluation/
│   ├── figures/
│   └── logs/
│
├── Project/
│   ├── config/
│   ├── ssl_pipeline/
│   └── ...
│
├── frontend/
├── tests/
├── docs/
│
├── README.md
├── requirements.txt
└── run_project.py
```

---

# ⚙️ Requirements

- Python 3.10+
- PyTorch
- torchaudio
- librosa
- SpeechBrain
- Transformers
- pyannote.audio
- pyannote.metrics
- scikit-learn
- NumPy
- SciPy
- matplotlib
- pandas

Install:

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## 1. Put audio inside

```text
inputs/
```

Example:

```text
inputs/mixed.wav
```

---

## 2. Run Speech Separation

```bash
python -m Project.ssl_pipeline.speech_separation inputs/mixed.wav
```

Output:

```text
outputs/separated_audio/

speaker1.wav

speaker2.wav
```

---

## 3. Run Speaker Diarization

```bash
python run_project.py --input inputs/mixed.wav
```

The pipeline performs:

- Audio preprocessing
- VAD
- Segmentation
- WavLM embeddings
- Spectral clustering
- Timeline reconstruction
- RTTM generation
- Evaluation (optional)

---

# 📊 Outputs

After execution:

```text
outputs/

separated_audio/

timeline/

rttm/

evaluation/

figures/

embeddings/

logs/
```

---

# 📈 Evaluation Metrics

The project supports:

- DER (Diarization Error Rate)
- JER (Jaccard Error Rate)
- Missed Speech
- False Alarm
- Speaker Confusion

---

# 🧪 Baseline

A classical baseline is implemented for comparison.

Pipeline:

```text
MFCC

↓

K-Means

↓

Timeline

↓

RTTM

↓

DER/JER
```

---

# 🖥️ Hardware

Recommended

- NVIDIA GPU (CUDA)
- 8 GB+ RAM

Minimum

- CPU
- 8 GB RAM

---

# 📚 Technologies Used

- Python
- PyTorch
- SpeechBrain
- Hugging Face Transformers
- WavLM
- pyannote.audio
- pyannote.metrics
- scikit-learn
- librosa
- matplotlib
- NumPy
- pandas

---

# 📄 License

This project is intended for educational and research purposes.

---

# 👨‍💻 Author

**Sarkar Tuhassinul Arnob**

Department of Information and Communication Technology (ICT)

Islamic University, Kushtia

Bangladesh

GitHub:
https://github.com/TuhassinulArnob

---

# ⭐ Acknowledgements

- Microsoft (WavLM)
- SpeechBrain
- Hugging Face
- pyannote.audio
- Asteroid
- scikit-learn