#!/usr/bin/env python3
"""Download ResOpsUS+CARS by supervising resumable curl calls.

The outer loop retries failures and validates the final MD5. Manual Range
headers are avoided because Zenodo may reject them with HTTP 403.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
URL = "https://zenodo.org/api/records/15978041/files/ResOpsUS%2BCARS_v10.zip/content"
OUT = PROJECT_ROOT / "data" / "raw" / "ResOpsUS+CARS_v10.zip"
EXPECTED_SIZE = 1597270154
EXPECTED_MD5 = "47af90a5f5a32e2466ffd5cf721816ab"
MAX_ATTEMPTS = 200
RETRY_DELAY = 3


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_curl_once() -> bool:
    """Run one ``curl -C -`` transfer and return whether it succeeded."""
    before = OUT.stat().st_size if OUT.exists() else 0
    cmd = [
        "curl", "-L", "-sS", "-C", "-", "--max-time", "600",
        "-o", str(OUT), URL,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=620)
    after = OUT.stat().st_size if OUT.exists() else 0
    print(f"[download] curl rc={proc.returncode} before={before} after={after}", flush=True)
    # A smaller file indicates a likely partial overwrite; restart the transfer.
    if after < before:
        print("[download] warning: file size decreased, removing and restarting", flush=True)
        OUT.unlink(missing_ok=True)
        return False
    return proc.returncode == 0 and after > before


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"[download] attempt {attempt}", flush=True)

        if OUT.exists() and OUT.stat().st_size == EXPECTED_SIZE:
            print("[download] size OK, verifying md5 ...", flush=True)
            actual = md5_of_file(OUT)
            print(f"  actual:   {actual}", flush=True)
            print(f"  expected: {EXPECTED_MD5}", flush=True)
            if actual == EXPECTED_MD5:
                print("[download] OK", flush=True)
                return 0
            print("[download] checksum mismatch, restarting from 0", flush=True)
            OUT.unlink()

        run_curl_once()
        time.sleep(RETRY_DELAY)

    print("[download] ERROR: too many attempts", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
