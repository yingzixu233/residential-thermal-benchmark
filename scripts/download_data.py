"""Restore the exact published CSV and verify its SHA-256."""
from pathlib import Path
import hashlib
import io
import urllib.request
import zipfile

URL = "https://zenodo.org/api/records/10156745/files/HarryVallianos/Automated-MultiZone-Model-Generation-v1.0.zip/content"
EXPECTED = "d7da67d1be8e57be1419379f61ddc82fab737fc888141ac48754a898549f7bc8"
blob = urllib.request.urlopen(URL, timeout=60).read()
if hashlib.md5(blob).hexdigest() != "c40600b8d079cef6b293ac2286b81fcc":
    raise RuntimeError("Zenodo archive checksum mismatch")
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    names = [n for n in archive.namelist() if n.endswith("/data_in/house_data.csv")]
    if len(names) != 1:
        raise RuntimeError("Ambiguous CSV in archive")
    data = archive.read(names[0])
if hashlib.sha256(data).hexdigest() != EXPECTED:
    raise RuntimeError("CSV checksum mismatch")
path = Path(__file__).resolve().parents[1]/"data"/"house_data.csv"
path.parent.mkdir(exist_ok=True)
path.write_bytes(data)
print(f"Verified {path.name}: {EXPECTED}")
