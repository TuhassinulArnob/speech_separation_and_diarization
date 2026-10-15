# Evaluation Datasets

Place reference evaluation data in **this folder**. Nothing here is downloaded
automatically and nothing here is overwritten by the application.

## What to place here

For each recording, two files that share the same stem:

```text
<name>.<audio ext>   Reference audio (.flac, .wav, .mp3, .m4a, .ogg, .opus, .sph)
<name>.rttm          Ground-truth annotation (required)
<name>.lab           Optional speech/non-speech labels
```

Example:

```text
evaluation/
  LDC2021S10.flac
  LDC2021S10.rttm
  LDC2021S10.lab
```

Pairing is by file stem, so no filename is hardcoded. Add as many pairs as you like and
they are all discovered automatically. Audio with no matching `.rttm` is reported as
unpaired and skipped — it is never scored against a guessed reference.

The reference RTTM may contain overlapping speaker turns; the reader accepts them. The
number of speakers is read from the reference RTTM, so recordings with more than two
speakers are handled without configuration.

## Reference RTTM format

Standard 10-field NIST RTTM lines:

```text
SPEAKER <recording-id> 1 <start> <duration> <NA> <NA> <speaker-id> <NA> <NA>
```

The `<recording-id>` does not have to match the audio filename. Scoring relabels the
generated hypothesis to the reference recording ID instead of editing your annotation.

## What the application writes

Only inside `results/`:

```text
evaluation/results/summary.json
evaluation/results/<name>/evaluation.json
evaluation/results/<name>/hypothesis.rttm
evaluation/results/<name>/pipeline/
```

Your reference files stay at the top level of `evaluation/` and are never mixed with
generated output.

## Safety

- The reference `.rttm` is opened read-only. It is never modified.
- `Clear Evaluation Results` (and `DELETE /evaluation/results`) removes only
  `evaluation/results/`. The audio, `.rttm`, and `.lab` files you placed here are never
  deleted.

## Running

From the website: open the **Evaluation** page and press **Run Evaluation**.

From the CLI:

```powershell
python -m backend_api.pipeline_runner evaluation --evaluation-dir evaluation
```

See the Evaluation section of the top-level [README.md](../README.md) for the DER
definition, the scoring protocol, and the output field reference.

## Git

Reference audio and annotations are ignored by Git so large datasets are not committed.
This README is tracked so the drop-in location stays documented. Keep your dataset
wherever it is licensed to live and copy it in locally.
