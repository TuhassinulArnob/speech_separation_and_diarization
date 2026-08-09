import requests
from pathlib import Path
import tempfile

p = Path(tempfile.gettempdir()) / 'verify_upload.wav'
p.write_bytes(b'RIFF')
with p.open('rb') as fh:
    files = {'file': ('verify_upload.wav', fh, 'audio/wav')}
    r = requests.post('http://127.0.0.1:8001/upload', files=files, timeout=30)
    print('status', r.status_code)
    print('body', r.text)
