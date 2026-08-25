#!/usr/bin/env python3
"""Download ResOpsBR+CARS with curl resume support.

Source DOI: 10.5281/zenodo.16096623.
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
URL = "https://zenodo.org/api/records/16096623/files/ResOpsBR%2BCARS_v10.zip/content"
OUT = PROJECT_ROOT / "data/raw" / "ResOpsBR+CARS_v10.zip"
EXPECTED_SIZE = 327403645
EXPECTED_MD5 = "a449074fbca3657e7cdd5eb7743ae7ea"
MAX_ATTEMPTS = 200
RETRY_DELAY = 3


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_curl_once() -> bool:
    before = OUT.stat().st_size if OUT.exists() else 0
    proc = subprocess.run(
        ["curl", "-L", "-sS", "-C", "-", "--max-time", "600", "-o", str(OUT), URL],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=620,
    )
    after = OUT.stat().st_size if OUT.exists() else 0
    print(f"[br] rc={proc.returncode} before={before} after={after}", flush=True)
    if after < before:
        print("[br] warning: file size decreased, restart from scratch", flush=True)
        OUT.unlink(missing_ok=True)
        return False
    return proc.returncode == 0 and after > before


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[br] attempt {attempt}", flush=True)
        if OUT.exists() and OUT.stat().st_size == EXPECTED_SIZE:
            print("[br] size OK, verifying md5 ...", flush=True)
            actual = md5_of_file(OUT)
            print(f"  actual={actual}", flush=True)
            print(f"  expected={EXPECTED_MD5}", flush=True)
            if actual == EXPECTED_MD5:
                print("[br] OK", flush=True)
                return 0
            print("[br] checksum mismatch, restart", flush=True)
            OUT.unlink()
        run_curl_once()
        time.sleep(RETRY_DELAY)
    print("[br] ERROR too many attempts", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
