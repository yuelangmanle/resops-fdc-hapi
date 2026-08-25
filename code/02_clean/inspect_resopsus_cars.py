#!/usr/bin/env python3
"""Inspect the ResOpsUS+CARS archive structure and field summary."""
from __future__ import annotations

import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ZIP = PROJECT_ROOT / "data" / "raw" / "ResOpsUS+CARS_v10.zip"
EXTRACT_DIR = PROJECT_ROOT / "data" / "raw" / "ResOpsUS+CARS_v10"


def main() -> None:
    if not RAW_ZIP.exists():
        print(f"[inspect] zip not found: {RAW_ZIP}")
        print("[inspect] Download the archive first: bash code/01_download/download_resopsus_cars.sh")
        return

    print(f"[inspect] zip size: {RAW_ZIP.stat().st_size / 1e6:.1f} MB")

    with zipfile.ZipFile(RAW_ZIP) as zf:
        names = zf.namelist()
        print(f"[inspect] total entries: {len(names)}")
        print("[inspect] first 30 entries:")
        for name in names[:30]:
            print("  ", name)

    # Summarize file types.
        suffixes: dict[str, int] = {}
        for name in names:
            suffix = Path(name).suffix.lower()
            suffixes[suffix] = suffixes.get(suffix, 0) + 1
        print("[inspect] file type counts:", suffixes)

    # Extract the archive when requested.
    if not EXTRACT_DIR.exists():
        print(f"[inspect] Archive is not extracted to {EXTRACT_DIR}")
        print("[inspect] Extraction can be performed in the next step")
    else:
        print(f"[inspect] Extracted directory already exists: {EXTRACT_DIR}")


if __name__ == "__main__":
    main()
