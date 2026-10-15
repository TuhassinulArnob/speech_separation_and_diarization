"""Transcribe LDC2021S10 with Whisper to observe active speech segments."""

import whisper

def main():
    model = whisper.load_model("base")
    result = model.transcribe("evaluation/LDC2021S10.flac")
    print("Detected language:", result.get("language"))
    print("Total Whisper segments:", len(result["segments"]))
    whisper_dur = sum(s["end"] - s["start"] for s in result["segments"])
    print(f"Total Whisper speech duration: {whisper_dur:.3f}s")
    for seg in result["segments"][:15]:
        print(f"  [{seg['start']:6.2f} - {seg['end']:6.2f}] {seg['text']}")

if __name__ == "__main__":
    main()
