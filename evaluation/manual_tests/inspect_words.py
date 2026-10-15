"""Inspect Whisper word-level timestamps between 20s and 40s in LDC2021S10."""

import whisper

def main():
    model = whisper.load_model("base")
    result = model.transcribe("evaluation/LDC2021S10.flac", word_timestamps=True)
    for seg in result["segments"]:
        if (seg["start"] < 40 and seg["end"] > 20) or (seg["start"] < 93 and seg["end"] > 78):
            print(f"Segment: {seg['start']:5.2f} - {seg['end']:5.2f} | {seg['text']}")

if __name__ == "__main__":
    main()
