import json
import mimetypes
import time
from pathlib import Path
from urllib import parse, request

SERVER = "127.0.0.1:8000"
ROOT = Path(__file__).resolve().parent

def post_multipart(url, fields, files):
    boundary = "----------ThIs_Is_tHe_bouNdaRY_$"
    lines = []
    for name, value in fields.items():
        lines.append(f"--{boundary}")
        lines.append(f"Content-Disposition: form-data; name=\"{name}\"")
        lines.append("")
        lines.append(value)
    for name, filename, value in files:
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        lines.append(f"--{boundary}")
        lines.append(
            f"Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"")
        lines.append(f"Content-Type: {content_type}")
        lines.append("")
        if isinstance(value, str):
            value = value.encode("utf-8")
        lines.append(value)
    lines.append(f"--{boundary}--")
    lines.append("")
    body = b"\r\n".join(
        item if isinstance(item, bytes) else item.encode("utf-8") for item in lines
    )
    req = request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    return request.urlopen(req)


def get_json(path):
    with request.urlopen(f"http://{SERVER}{path}") as r:
        return json.load(r)


def wait_complete():
    while True:
        status = get_json("/status")
        print("status", status["status"], status["stage"], status["progress"])
        if status["status"] in ("complete", "error"):
            return status
        time.sleep(2)


audio_files = ["inputs/mixed.wav", "inputs/mixed_old.wav"]
results = []
for audio in audio_files:
    path = ROOT / audio
    print("Uploading", path)
    with path.open("rb") as f:
        res = post_multipart(f"http://{SERVER}/upload", {}, [("file", path.name, f.read())])
    data = json.load(res)
    print("upload response", data)
    status = wait_complete()
    print("final status", status)
    results_data = get_json("/results")
    print("results job_id", results_data.get("job_id"))
    print("figure paths", [f["path"] for f in results_data["figures"]])
    timeline = None
    for file in results_data["files"]:
        if file["name"] == "timeline.json":
            with request.urlopen(f"http://{SERVER}/download/{parse.quote(file['path'])}") as r:
                timeline = json.load(r)
            break
    print("timeline len", len(timeline) if timeline is not None else None)
    if timeline:
        print("timeline first rows", timeline[:3])
    results.append({
        "upload": audio,
        "job_id": results_data.get("job_id"),
        "figures": [f["path"] for f in results_data["figures"]],
        "timeline_count": len(timeline) if timeline is not None else None,
    })
    print("---")
print("SUMMARY")
print(json.dumps(results, indent=2))
