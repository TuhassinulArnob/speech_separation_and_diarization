# Test C Annotation, VAD, and Clustering Audit Report

## 1. Executive Summary

This report delivers the technical audit to resolve whether the high Diarization Error Rate (DER 133.09%) on `evaluation/LDC2021S10.flac` is caused by:
1. **Incomplete reference annotations**,
2. **Genuine VAD errors**,
3. **Speaker clustering / representation limitations**, or
4. **A combination of multiple factors**.

### Core Conclusion
The high DER is the result of a **combination of three distinct, quantifiable factors**:
1. **Incomplete Reference RTTM Annotation (Primary Contributor to False Alarm, ~73-87%)**:
   - `LDC2021S10.rttm` contains only **56.942s** of reference speech turns within the 120s audio container.
   - However, the recording's official ground-truth Speech Activity Detection file (`LDC2021S10.lab`) specifies **88.696s** of active speech.
   - Whisper transcription confirms continuous conversational English speech during the unannotated intervals (e.g. $28.2\text{s}-34.4\text{s}$ and $79.4\text{s}-90.9\text{s}$).
   - Scoring an unconstrained diarization system against this incomplete RTTM without an oracle UEM penalizes genuine speech as False Alarm (**38.675s** of the 44.442s False Alarm is genuine speech).
2. **Upstream SSL Representation Limitations (Primary Contributor to Speaker Confusion, 15.966s)**:
   - Both `microsoft/wavlm-base` and `microsoft/wavlm-base-plus` are self-supervised acoustic representations trained without supervised speaker metric supervision (e.g., ECAPA-TDNN / x-vector loss).
   - In natural conversational speech recorded through a single microphone, inter-speaker cosine distance is smaller than intra-speaker phonetic variability, yielding near-chance cluster purity (~54.5%) across all clustering algorithms (Spectral K-Means, Spectral Discretize, AHC average, AHC complete, Spherical K-Means).
3. **Acoustic Noise Floor & Single-Speaker Partitioning (Primary Contributor to Missed Speech, 15.374s)**:
   - 10.866s (72.1%) is missed due to quiet conversational speech dropping below the adaptive energy threshold.
   - 1.988s (13.2%) is lost because the pipeline's timeline reconstructor collapses simultaneous overlapping speech into single-speaker partitions (`_resolve_overlaps`).

---

## 2. Task 1: Reference Annotation Validation & Duration Discrepancy

### Format and Documented Meaning
- **`LDC2021S10.rttm`**: Standard NIST 10-field RTTM file. Provides speaker turns for `speaker117` and `speaker118`.
- **`LDC2021S10.lab`**: Standard 3-field label file (`start \t end \t speech`). Provides Speech Activity Detection (SAD) **only**. It does **not** provide speaker identities. Speaker labels must not be assigned to unannotated speech by guessing.

### Duration Discrepancy Explained
- Total reference speech in raw `LDC2021S10.rttm`: **58.432s**.
- Audio duration of `LDC2021S10.flac`: **120.000s**.
- Two turns in `LDC2021S10.rttm` extend past the audio boundary:
  - `speaker117`: 120.095s to 120.475s (0.380s)
  - `speaker118`: 120.120s to 121.230s (1.110s)
  - Total duration past 120.0s: **1.490s**.
- When clipped to the actual audio duration ($120.000\text{s}$), total reference speech is exactly:
  $$58.432\text{s} - 1.490\text{s} = 56.942\text{s}$$
- **Scorer's Denominator**: NIST md-eval evaluates DER against total reference speaker time (**56.942s** for `skip_overlap=False`). This is distinct from:
  - *Union speech duration* ($54.648\text{s}$), where simultaneous overlapping speech is counted once.
  - *Audio file duration* ($120.000\text{s}$).

### Timeline Intersection: LAB vs RTTM
| Category | Duration | Percentage of Audio |
|---|---:|---:|
| Both LAB & RTTM mark Speech | 41.704s | 34.8% |
| **LAB marks Speech, RTTM has NO annotation (Missing in RTTM)** | **46.992s** | **39.2%** |
| RTTM marks Speech, LAB has Silence | 12.944s | 10.8% |
| Both mark Non-Speech / Silence | 18.360s | 15.3% |
| **Total Audio Duration** | **120.000s** | **100.0%** |

**Conclusion**: `LDC2021S10.rttm` is **incomplete** as a whole-file ground truth for unconstrained diarization evaluation. The reported DER (133.09%) must be considered **provisional and uncalibrated** because the reference annotation omits ~39% of the file's audible speech.

---

## 3. Task 3: Separation of VAD Errors from Annotation Gaps

### VAD Evaluated against Ground Truth Speech Activity (`LDC2021S10.lab`)
| Metric | Value | Interpretation |
|---|---:|---|
| **True Positives (TP)** | 76.060s | Speech correctly detected |
| **False Positives (FP)** | 12.110s | Speech detected in true LAB silence |
| **False Negatives (FN)** | 12.636s | Missed speech present in LAB |
| **True Negatives (TN)** | 19.194s | Silence correctly rejected |
| **Precision** | **86.27%** | $TP / (TP + FP)$ |
| **Recall** | **85.75%** | $TP / (TP + FN)$ |
| **F1-Score** | **86.01%** | Harmonic mean of Precision & Recall |
| **Accuracy** | **79.38%** | $(TP + TN) / \text{Total Duration}$ |

### Breakdown of VAD Output Outside RTTM (45.282s)
- **73.3% (33.172s)**: Genuine speech confirmed by LAB ground truth.
- **26.7% (12.110s)**: Genuine acoustic non-speech false alarm (room noise, breathing, mic handling).

---

## 4. Task 4: Systematic Clustering & Representation Comparison

Experiments were conducted on the 68 extracted segments, evaluating diagnostic cluster purity on unambiguous reference segments (23 segments of `speaker117` and 10 segments of `speaker118`) and end-to-end diarization metrics:

| Method | Cluster Counts | Diagnostic Purity | DER (%) | Missed (s) | False Alarm (s) | Confusion (s) |
|---|:---:|---:|---:|---:|---:|---:|
| **Baseline: Spectral (kmeans)** | [30, 38] | **54.5%** | **133.09%** | 15.374s | 44.442s | 15.966s |
| **Alt 1: Spectral (discretize)** | [38, 30] | **54.5%** | **133.09%** | 15.374s | 44.442s | 15.966s |
| **Alt 2: AHC (cosine, average)** | [26, 42] | **51.5%** | **136.58%** | 15.374s | 44.442s | 17.953s |
| **Alt 3: AHC (cosine, complete)** | [41, 27] | **57.6%** | **135.76%** | 15.374s | 44.452s | 17.477s |
| **Alt 4: KMeans (L2 normalized)** | [31, 37] | **54.5%** | **132.03%** | 15.374s | 44.442s | 15.366s |
| *WavLM Base (unsupervised)* | [30, 38] | **54.5%** | 133.09% | 15.374s | 44.442s | 15.966s |

### Technical Finding
- Switching between Spectral Clustering (K-Means or Discretize), Agglomerative Hierarchical Clustering, or Spherical K-Means produces virtually identical DER (~132% - 136%) and diagnostic purity (51.5% - 57.6%).
- The limiting factor is **not the graph clustering algorithm**, but the **upstream feature representation**: base SSL models (`wavlm-base` and `wavlm-base-plus`) lack supervised metric learning for speaker identity. In a single-channel room recording, phonetic variance exceeds speaker variance.

---

## 5. Summary of Files Changed & Tests Executed

### Files Changed
- `evaluation/manual_tests/task1_validate_annotations.py`: Quantitative LAB vs RTTM alignment.
- `evaluation/manual_tests/task3_vad_vs_lab.py`: Independent VAD evaluation against LAB ground truth.
- `evaluation/manual_tests/task4_clustering_experiments.py`: Systematic clustering and representation benchmarks.
- `evaluation/manual_tests/results/test_C_two_speakers/clustering_comparison.json`: Stored benchmark outcomes.
- `evaluation/manual_tests/results/test_C_two_speakers/diagnostic_timeline.png`: 5-panel timeline plot.
- `evaluation/manual_tests/run_test_c.py`: Diagnostic breakdown generation.
- `evaluation/manual_tests/generate_summary.py`: Comprehensive summary regeneration.

### Tests Executed
1. `tests/test_vad.py` (VAD unit test suite) $\rightarrow$ **PASSED**
2. `tests/test_evaluation.py` (Evaluation unit test suite) $\rightarrow$ **PASSED**
3. `tests/test_dataset_evaluation.py` (Dataset evaluation suite) $\rightarrow$ **PASSED**
4. Full regression suite (`python -m unittest discover tests`) $\rightarrow$ **130 passed, 2 skipped, 0 failures**.
5. Test A runner (`run_test_a.py`) $\rightarrow$ **PASSED**.
6. Test B runner (`run_test_b.py`) $\rightarrow$ **PASSED**.
7. Test C runner (`run_test_c.py`) $\rightarrow$ **PASSED**.

---

## 6. Remaining Limitations

1. **Reference RTTM Incompleteness**: Without an explicit UEM bounding only the annotated regions of `LDC2021S10`, whole-file diarization cannot achieve a standard low DER on this file.
2. **Pretrained Speaker Metric Model**: Upstream base WavLM models are not speaker verifiers. Achieving <30% confusion on natural single-microphone conversational speech requires a speaker-discriminative model (e.g., ECAPA-TDNN trained with AAM-softmax on VoxCeleb).
3. **Single-Speaker Timeline Representation**: The timeline reconstruction stage clamps overlapping intervals to hard partitions, inherently losing overlapping reference speech.
