#!/usr/bin/env python3
"""Download ResOpsUS+CARS with resumable requests and MD5 validation.

The downloader retries interrupted transfers and avoids overwriting partial
files when a server returns a full-response status.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import requests

URL = "https://zenodo.org/api/records/15978041/files/ResOpsUS%2BCARS_v10.zip/content"
OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "ResOpsUS+CARS_v10.zip"
EXPECTED_SIZE = 1597270154
EXPECTED_MD5 = "47af90a5f5a32e2466ffd5cf721816ab"
MAX_ATTEMPTS = 100
CHUNK_SIZE = 1024 * 256  # 256 KB
RETRY_DELAY = 5


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_once(path: Path, timeout: int = 600) -> bool:
    offset = path.stat().st_size if path.exists() else 0

    if offset >= EXPECTED_SIZE:
        return True

    headers = {"User-Agent": "Mozilla/5.0", "Range": f"bytes={offset}-"}
    print(f"[download] offset={offset} bytes", flush=True)

    try:
        with requests.get(URL, headers=headers, stream=True, timeout=timeout) as r:
            if r.status_code == 206:
                mode = "ab"
            elif r.status_code == 200 and offset == 0:
                mode = "wb"
            else:
                print(f"[download] server returned {r.status_code}, retry later", flush=True)
                return False

            with path.open(mode) as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            return True
    except Exception as e:
        print(f"[download] exception: {e}", flush=True)
        return False


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

        if download_once(OUT):
            print("[download] download_once returned success", flush=True)
            # Recheck after a short delay because a clean connection close can
            # still leave an incomplete transfer.
            time.sleep(1)
            if OUT.exists() and OUT.stat().st_size == EXPECTED_SIZE:
                continue

        time.sleep(RETRY_DELAY)

    print("[download] ERROR: too many attempts", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
