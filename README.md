# Speaker Diarization Using Pretrained WavLM Embeddings

This repository contains an undergraduate thesis implementation of an offline
speaker diarization pipeline using pretrained WavLM embeddings and spectral
clustering.

## Current Pipeline

```text
Input Audio
-> Audio Preprocessing
-> Voice Activity Detection
-> Speech Segmentation
-> WavLM Embedding Extraction
-> Embedding Normalization
-> Spectral Clustering
-> Speaker Timeline Reconstruction
-> RTTM Output
-> DER and JER Evaluation
```

The repository is being improved one module at a time so that it stays runnable
after every change.

## Repository Layout

```text
Project/
  config/
    default.json
  ssl_pipeline/
    config.py
    preprocessing.py
    vad.py
    segmentation.py
    segmantation.py
    ssl_embedding.py
    embedding_normalization.py
    spectral_clustering.py
    timeline_reconstruction.py
    rttm.py
    evaluation.py
    baseline.py
    speech_separation.py
tests/
  test_baseline.py
  test_config.py
  test_embedding_normalization.py
  test_evaluation.py
  test_rttm.py
  test_spectral_clustering.py
  test_speech_separation.py
  test_timeline_reconstruction.py
  test_ssl_embedding.py
  test_segmentation.py
  test_vad.py
test_diarization.py
```

`segmantation.py` is kept only as a compatibility wrapper for the original
misspelled file name. New code should import `Project.ssl_pipeline.segmentation`.

## Security

Do not commit Hugging Face access tokens. Pyannote scripts read the token from
the `HUGGINGFACE_TOKEN` environment variable.

PowerShell example:

```powershell
$env:HUGGINGFACE_TOKEN = "your-token-here"
python test_diarization.py
```

## Testing

Run the current unit tests with:

```powershell
python -m unittest discover -s tests
```

## Complete Diarization Pipeline

Run the end-to-end diarization runner from one input audio file:

```powershell
python run_diarization.py --input inputs/mixed.wav --output-dir outputs
```

The module form is also supported:

```powershell
python -m Project.run_diarization --input inputs/mixed.wav --output-dir outputs
```

Optional evaluation runs automatically when a reference RTTM is supplied:

```powershell
python run_diarization.py --input inputs/mixed.wav --output-dir outputs --reference-rttm reference.rttm
```

Useful overrides:

- `--backend`: VAD backend, such as `energy`, `webrtc`, `pyannote`, or `auto`.
- `--device`: WavLM embedding device, such as `cpu`, `cuda`, or `auto`.
- `--num-speakers`: configured speaker count for clustering.
- `--oracle-num-speakers` / `--no-oracle-num-speakers`: use or estimate the
  clustering speaker count.

Default outputs:

```text
outputs/
  processed.wav
  vad_regions.json
  segments.json
  embeddings/
    embeddings.npz
  normalized_embeddings.npz
  clustering.json
  timeline.json
  timeline.csv
  timeline.txt
  hypothesis.rttm
  evaluation.json
  figures/
    timeline.png
    speaker_activity.png
    duration_histogram.png
```

## Voice Activity Detection

VAD is the official speech detection stage of the pipeline. It returns speech
timestamps as `(start_time, end_time)` pairs in seconds and can export them to
JSON.

The default backend is `auto`, which tries pyannote first, then WebRTC VAD, then
a lightweight energy backend for local fallback/testing. For thesis experiments,
prefer pyannote when model access and runtime are available, or WebRTC VAD as
the clean non-neural fallback.

CLI example:

```powershell
python -m Project.ssl_pipeline.vad Project/ssl_pipeline/processed.wav --backend auto --output outputs/vad_regions.json
```

Python example:

```python
from Project.ssl_pipeline.vad import detect_speech_regions

regions = detect_speech_regions("Project/ssl_pipeline/processed.wav")
print(regions)
```

## Speech Segmentation

Speech segmentation consumes VAD timestamps and creates embedding-ready speech
windows. It does not segment the whole raw recording. This keeps silence out of
the WavLM stage and makes speaker embeddings cleaner.

The default strategy uses a `2.0` second window and a `1.5` second hop, which
creates `0.5` seconds of overlap. Overlap helps reduce boundary errors because a
speaker change or word boundary is less likely to destroy a single embedding.
Very short speech regions below `segment_min_duration_seconds` are skipped.

CLI example:

```powershell
python -m Project.ssl_pipeline.segmentation Project/ssl_pipeline/processed.wav --vad-backend energy --output outputs/segments.json
```

Python example using existing VAD timestamps:

```python
from Project.ssl_pipeline.segmentation import segment_speech

segments = segment_speech([(0.32, 4.11), (4.92, 5.88)])
print([segment.to_dict() for segment in segments])
```

## WavLM Embedding Extraction

The WavLM stage consumes `SpeechSegment` objects and extracts one embedding per
segment. The default model is `microsoft/wavlm-base-plus`, loaded lazily through
Hugging Face Transformers. Embeddings are not normalized in this stage; that is
the next pipeline module.

Configuration options live in `Project/config/default.json`:

- `embedding_model_name`: Hugging Face model name.
- `embedding_pooling_method`: `mean`, `max`, `cls`, or `statistics`.
- `embedding_hidden_layer`: hidden layer index, with `-1` meaning final layer.
- `embedding_batch_size`: number of speech segments per inference batch.
- `embedding_device`: `auto`, `cpu`, or `cuda`.
- `embedding_cache_enabled`: save/load `.npz` embedding caches.

Token priority is environment variable, `.env` file, then anonymous public model
download. Do not commit tokens.

CLI example:

```powershell
python -m Project.ssl_pipeline.ssl_embedding Project/ssl_pipeline/processed.wav --segments outputs/segments.json --cache outputs/embeddings/sample.npz --plot outputs/embedding_stats.png
```

Python example:

```python
from Project.ssl_pipeline.segmentation import load_segments
from Project.ssl_pipeline.ssl_embedding import extract_segment_embeddings

segments = load_segments("outputs/segments.json")
embeddings = extract_segment_embeddings(
    "Project/ssl_pipeline/processed.wav",
    segments,
    cache_path="outputs/embeddings/sample.npz",
)
print(embeddings[0].dimension)
```

### Pooling Strategies

Mean pooling is the default because it averages information across all WavLM
frames in a speech segment. It is stable, simple, and usually robust for speaker
diarization because every frame contributes to the segment-level speaker
representation.

- `mean`: averages all frame embeddings. It is robust to short frame-level noise
  and keeps the WavLM hidden size unchanged, but it can smooth away brief speaker
  cues.
- `max`: keeps the strongest activation per dimension. It can preserve salient
  cues, but it is more sensitive to outlier frames and noise.
- `cls`: uses the first token representation. It is cheap and common in some
  transformer pipelines, but WavLM is not primarily trained around a CLS token
  for speaker embeddings, so it may be less reliable.
- `statistics`: concatenates per-dimension mean and standard deviation. It can
  capture both central tendency and variation, but it doubles the vector
  dimension and may need more data downstream.

## Embedding Normalization

Embedding normalization is the official stage between WavLM embedding extraction
and spectral clustering. It consumes `SegmentEmbedding` objects and returns
`NormalizedEmbedding` objects while preserving `segment_id`, `start`, `end`,
`model_name`, and source metadata.

Configuration options:

- `embedding_normalization_method`: `l2` by default, or `standard`, or `none`.
- `embedding_pca_enabled`: enable/disable PCA.
- `embedding_pca_components`: number of PCA components when PCA is enabled.

L2 normalization makes every embedding have comparable length, so clustering is
driven more by direction/cosine similarity than by vector magnitude. This helps
spectral clustering because the affinity matrix should reflect speaker
similarity rather than arbitrary embedding scale.

Mathematically, for an embedding vector `x`, L2 normalization is:

```text
x_normalized = x / max(||x||_2, epsilon)
```

where `||x||_2 = sqrt(sum_i x_i^2)` and `epsilon` prevents division by zero.

Standard scaling transforms each dimension to approximately zero mean and unit
variance across the current embedding set. Optional PCA can reduce dimensionality
after normalization/scaling before the future clustering stage.

CLI example from a raw embedding cache:

```powershell
python -m Project.ssl_pipeline.embedding_normalization outputs/embeddings/sample.npz --method l2 --plot-norms outputs/norm_distribution.png --plot-stats outputs/vector_statistics.png
```

Python example:

```python
from Project.ssl_pipeline.embedding_normalization import normalize_embeddings
from Project.ssl_pipeline.ssl_embedding import load_embedding_cache

embeddings = load_embedding_cache("outputs/embeddings/sample.npz")
normalized = normalize_embeddings(embeddings, normalization_method="l2")
print(normalized[0].dimension)
```

Limitations:

- PCA is fitted on the provided embedding batch; persist fitted transforms before
  using this for multi-recording experiments.
- Normalization does not assign speakers, reconstruct timelines, write RTTM, or
  compute DER/JER.

## Spectral Clustering

Spectral clustering is the official speaker label assignment stage. It consumes
`NormalizedEmbedding` objects and returns `ClusteredEmbedding` objects with a
`cluster_id` for each speech segment. It preserves `segment_id`, `start`, `end`,
`vector`, and metadata. This stage does not reconstruct a timeline, write RTTM,
or compute DER/JER.

Configuration options:

- `clustering_algorithm`: currently `spectral`.
- `clustering_affinity_metric`: currently `cosine`.
- `clustering_affinity_threshold`: removes weak graph edges below this value.
- `clustering_num_speakers`: oracle number of speakers.
- `clustering_oracle_num_speakers`: `true` by default for reproducibility.
- `clustering_speaker_count_method`: currently `eigengap`.
- `clustering_max_estimated_speakers`: upper bound for automatic estimation.
- `clustering_random_seed`: deterministic clustering seed.

Cosine similarity compares embedding directions:

```text
cos(x_i, x_j) = (x_i . x_j) / (||x_i||_2 ||x_j||_2)
```

The affinity matrix is the graph used by spectral clustering. In this
implementation, cosine similarities are clipped to the non-negative range,
thresholded, symmetrized, and given a diagonal value of `1.0`:

```text
A_ij = max(cos(x_i, x_j), 0)
A_ij = 0 if A_ij < threshold
A_ii = 1
```

The normalized graph Laplacian is:

```text
L_sym = I - D^(-1/2) A D^(-1/2)
```

where `A` is the affinity matrix and `D` is the degree matrix. Spectral
clustering computes eigenvectors of this Laplacian:

```text
L_sym u = lambda u
```

Rows of the selected eigenvector matrix are clustered to minimize graph cut
costs, so strongly connected speech segments tend to receive the same speaker
label while weakly connected segments separate.

Oracle speaker count mode uses the configured `clustering_num_speakers`. This is
the default because thesis experiments are easier to reproduce when the number
of speakers is controlled. Estimated mode uses the eigengap heuristic: the
speaker count is chosen from the largest gap between consecutive small
eigenvalues of `L_sym`.

Spectral clustering is used instead of plain K-Means because diarization
embeddings often form non-spherical, graph-like structures. K-Means clusters by
distance to centroids and assumes compact Euclidean groups. Spectral clustering
uses pairwise affinity, which better matches cosine-similarity speaker graphs
and can separate clusters that are not well represented by a single centroid.

Python example:

```python
from Project.ssl_pipeline.spectral_clustering import cluster_embeddings

clustered = cluster_embeddings(
    normalized_embeddings,
    num_speakers=2,
    oracle_num_speakers=True,
    random_seed=42,
)
print(clustered[0].segment_id, clustered[0].cluster_id)
```

Plot helpers:

```python
from Project.ssl_pipeline.spectral_clustering import (
    build_affinity_matrix,
    compute_cosine_similarity_matrix,
    plot_affinity_matrix,
    plot_cluster_distribution,
    plot_similarity_matrix,
)

similarity = compute_cosine_similarity_matrix(vectors)
affinity = build_affinity_matrix(similarity, threshold=0.0)
plot_similarity_matrix(similarity, "outputs/similarity_matrix.png")
plot_affinity_matrix(affinity, "outputs/affinity_matrix.png")
plot_cluster_distribution(clustered, "outputs/cluster_distribution.png")
```

Limitations:

- Estimated speaker count is heuristic and can be sensitive to noisy affinity
  graphs.
- Cluster IDs are relative labels, not final speaker identities.
- This stage does not merge adjacent segments, reconstruct speaker turns, write
  RTTM, or evaluate diarization quality.

## Speaker Timeline Reconstruction

Timeline reconstruction is the official post-processing stage after spectral
clustering. Clustering assigns a speaker label to each embedding window, while
timeline reconstruction converts those per-segment labels into clean
chronological speaker intervals. It consumes `ClusteredEmbedding` objects and
returns `SpeakerTimeline` rows with `speaker_id`, `start`, `end`, `duration`, and
`segment_count`.

This stage is required because embedding windows may overlap, repeat the same
speaker across adjacent windows, contain tiny isolated fragments, or include
short silences between segments from the same speaker. Post-processing improves
diarization readability by merging same-speaker runs, bridging very small gaps,
removing tiny fragments, and optionally smoothing isolated label flips.

Configuration options:

- `timeline_merge_adjacent`: merge touching same-speaker segments.
- `timeline_merge_silence_threshold_seconds`: merge same-speaker segments across
  short silence gaps.
- `timeline_min_duration_seconds`: remove very short speaker fragments.
- `timeline_label_smoothing_enabled`: relabel one-segment islands between
  matching speakers.

Python example:

```python
from Project.ssl_pipeline.timeline_reconstruction import (
    export_timeline_json,
    reconstruct_speaker_timeline,
)

timeline = reconstruct_speaker_timeline(
    clustered_embeddings,
    merge_silence_threshold_seconds=0.2,
    min_duration_seconds=0.1,
)
export_timeline_json(timeline, "outputs/speaker_timeline.json")
```

Example timeline:

```text
speaker_0  0.00-3.42  duration=3.42s  segments=4
speaker_1  3.42-5.87  duration=2.45s  segments=2
speaker_0  5.87-8.23  duration=2.36s  segments=3
```

Export helpers:

```python
from Project.ssl_pipeline.timeline_reconstruction import (
    export_timeline_csv,
    export_timeline_json,
    export_timeline_txt,
)

export_timeline_json(timeline, "outputs/speaker_timeline.json")
export_timeline_csv(timeline, "outputs/speaker_timeline.csv")
export_timeline_txt(timeline, "outputs/speaker_timeline.txt")
```

Plot helpers:

```python
from Project.ssl_pipeline.timeline_reconstruction import (
    plot_speaker_activity_chart,
    plot_speaker_duration_histogram,
    plot_timeline,
)

plot_timeline(timeline, "outputs/timeline.png")
plot_speaker_activity_chart(timeline, "outputs/speaker_activity.png")
plot_speaker_duration_histogram(timeline, "outputs/duration_histogram.png")
```

Timeline reconstruction differs from clustering in its purpose: clustering
decides which speaker label each segment receives, while reconstruction turns
those labels into a chronological sequence of speaker turns. The next RTTM
module can consume the timeline rows directly because each row already contains
the speaker ID, start time, and duration needed for RTTM formatting.

Limitations:

- Overlapping different-speaker segments are clamped chronologically rather than
  modeled as simultaneous speech.
- Tiny-fragment removal may drop real but very short turns if the threshold is
  too high.
- This stage does not generate RTTM and does not evaluate DER/JER.

## RTTM Reader / Writer

RTTM, or Rich Transcription Time Marked format, is the standard text format used
to represent diarization speaker turns. Each `SPEAKER` line stores the recording
ID, channel, start time, duration, and speaker name. This repository writes
standard 10-field RTTM rows and can read both full 10-field rows and the common
9-field diarization variant.

RTTM is required because most diarization evaluation tools expect hypothesis and
reference annotations in the same time-marked format. DER uses RTTM by comparing
reference speaker turns against system speaker turns and measuring missed
speech, false alarm speech, and speaker confusion. This module only creates and
validates the RTTM files; it does not compute DER or JER.

Configuration options:

- `rttm_recording_id`: default recording/file ID used by the writer.
- `rttm_channel_id`: default channel ID, usually `1`.
- `rttm_decimal_places`: decimal precision for start and duration values.

Example RTTM:

```text
SPEAKER recording1 1 0.00 2.34 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER recording1 1 2.34 1.87 <NA> <NA> speaker_1 <NA> <NA>
```

Python example:

```python
from Project.ssl_pipeline.rttm import (
    export_rttm_json,
    parse_rttm,
    validate_rttm,
    write_rttm,
)

write_rttm(
    timeline,
    "outputs/hypothesis.rttm",
    recording_id="recording1",
    channel_id="1",
)
validate_rttm("outputs/hypothesis.rttm")
turns = parse_rttm("outputs/hypothesis.rttm")
export_rttm_json(turns, "outputs/hypothesis_rttm.json")
```

CLI examples:

```powershell
python -m Project.ssl_pipeline.rttm validate outputs/hypothesis.rttm
python -m Project.ssl_pipeline.rttm to-json outputs/hypothesis.rttm outputs/hypothesis_rttm.json
```

Validation checks:

- Speaker, recording, and channel names must be non-empty tokens without
  whitespace.
- Start times must be non-negative.
- Durations must be positive.
- Empty RTTM files are rejected unless explicitly allowed.
- Overlapping turns within the same recording/channel are rejected.

Limitations:

- RTTM reading preserves speaker turns, but `SpeakerTimeline.segment_count` is
  set to `1` when converting from RTTM because RTTM does not store the original
  embedding segment count.
- Overlapped speech RTTM is rejected by default for this pipeline's single-label
  timeline output.
- This stage does not compute DER/JER.

## Baseline Speaker Diarization System

The baseline is a standalone classical diarization system that does not modify
the proposed WavLM pipeline. It follows:

```text
Audio
-> MFCC Feature Extraction
-> K-Means Clustering
-> Speaker Timeline Reconstruction
-> RTTM Output
-> DER/JER Evaluation
```

MFCC + K-Means is selected as the baseline because it is a simple, well-known
classical approach: MFCCs summarize short-term spectral envelope information
from speech, and K-Means gives a reproducible unsupervised clustering method for
speaker-like acoustic groups. This makes it a useful comparison point for the
proposed WavLM embedding pipeline.

Configuration options:

- `baseline_mfcc_coefficients`: number of MFCC coefficients.
- `baseline_frame_length_seconds`: MFCC frame length.
- `baseline_frame_hop_seconds`: MFCC frame hop.
- `baseline_kmeans_random_seed`: deterministic K-Means seed.
- `baseline_num_speakers`: oracle speaker count for K-Means.
- `baseline_oracle_num_speakers`: use configured speaker count when `true`.

Python example:

```python
from Project.ssl_pipeline.baseline import run_baseline

result = run_baseline(
    "Project/ssl_pipeline/processed.wav",
    output_rttm_path="outputs/baseline.rttm",
    reference_rttm_path="outputs/reference.rttm",
    recording_id="recording1",
)
print(result.evaluation_report.der, result.evaluation_report.jer)
```

CLI example:

```powershell
python -m Project.ssl_pipeline.baseline Project/ssl_pipeline/processed.wav --rttm outputs/baseline.rttm --reference-rttm outputs/reference.rttm --recording-id recording1
```

Visualization helpers:

```python
from Project.ssl_pipeline.baseline import (
    plot_baseline_cluster_distribution,
    plot_baseline_timeline,
    plot_mfcc_features,
)

plot_mfcc_features(result.frames, "outputs/baseline_mfcc.png")
plot_baseline_cluster_distribution(result.frames, "outputs/baseline_clusters.png")
plot_baseline_timeline(result.timeline, "outputs/baseline_timeline.png")
```

Compared with the WavLM pipeline, the baseline is faster, simpler, and easier to
interpret, but it uses low-level acoustic features and K-Means' centroid
assumptions. The proposed WavLM pipeline uses pretrained contextual speech
representations and spectral clustering, which can better capture speaker
identity but requires heavier models and more dependencies.

To compare systems, generate one hypothesis RTTM from the proposed pipeline and
one from the baseline, then evaluate both against the same reference RTTM using
the existing evaluation module. Compare DER, JER, missed speech, false alarm,
and speaker confusion under the same recording IDs and scoring settings.

## True Speech Separation

Speech separation is an independent preprocessing option that reconstructs
speaker waveforms from mixed speech. It is not diarization and it is not
timeline-based audio cutting. Diarization estimates who spoke when; separation
estimates the audio signal for each source speaker.

The separation module uses pretrained models only. The preferred backend is
SpeechBrain SepFormer because it is a well-supported open-source speech
separation model family with public pretrained checkpoints. If SpeechBrain is
not available and `auto` backend is selected, the module can fall back to an
Asteroid Conv-TasNet style backend when installed. No Hugging Face token is
hardcoded; `HUGGINGFACE_TOKEN` is read from the environment only if a selected
model requires access.

Configuration options:

- `separation_model_backend`: `speechbrain`, `asteroid`, or `auto`.
- `separation_model_name`: default SpeechBrain model name.
- `separation_fallback_model_name`: Asteroid fallback model.
- `separation_device`: `auto`, `cpu`, or `cuda`.
- `separation_output_dir`: default output directory.
- `separation_sample_rate`: target separation sample rate. The default is
  `8000` for `speechbrain/sepformer-whamr`, matching that pretrained model.
- `separation_num_speakers`: number of silent outputs to create for silent input.
- `separation_chunk_duration_seconds`: process longer recordings in chunks to
  reduce peak memory use; set to `0` to disable chunking.
- `separation_chunk_overlap_seconds`: overlap used when stitching separated
  chunks.

CLI example:

```powershell
python -m Project.ssl_pipeline.speech_separation mixed.wav --output-dir outputs/separated_audio --backend speechbrain --device auto
```

Example outputs:

```text
outputs/separated_audio/speaker1.wav
outputs/separated_audio/speaker2.wav
```

Python example:

```python
from Project.ssl_pipeline.speech_separation import separate_speech

result = separate_speech(
    "mixed.wav",
    output_dir="outputs/separated_audio",
    backend="speechbrain",
)
print(result.output_paths)
```

To listen to the separated audio, open `outputs/separated_audio/speaker1.wav`
and `outputs/separated_audio/speaker2.wav` in any audio player, or load them in
a notebook/audio editor for inspection.

Visualization helpers:

```python
from Project.ssl_pipeline.speech_separation import (
    load_audio,
    plot_spectrogram_comparison,
    plot_waveform_comparison,
)

mixture, sample_rate = load_audio("mixed.wav", target_sample_rate=16000)
sources = [load_audio(path, target_sample_rate=16000)[0] for path in result.output_paths]
plot_waveform_comparison(mixture, sources, sample_rate, "outputs/separation_waveforms.png")
plot_spectrogram_comparison(mixture, sources, sample_rate, "outputs/separation_spectrograms.png")
```

Integration with diarization is optional and happens after separation. Each
`speakerN.wav` file can be passed into the existing preprocessing/VAD/
segmentation/WavLM pipeline as ordinary audio input. The separation module does
not modify or replace the diarization modules.

Limitations:

- Pretrained two-speaker models may not cleanly separate three or more speakers.
- Performance depends strongly on the training domain of the selected model.
- Music, noise, reverberation, and non-speech sounds can reduce separation
  quality.
- The first run may download model weights unless they are already cached.
- Chunked inference reduces memory pressure on long recordings, but very hard
  speaker changes exactly at chunk boundaries can still need listening checks.

## Next Planned Module

The next implementation step is experiment orchestration and thesis reporting
around the proposed pipeline and baseline outputs.
