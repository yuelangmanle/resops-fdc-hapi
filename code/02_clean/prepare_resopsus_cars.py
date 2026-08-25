#!/usr/bin/env python3
"""Extract and inventory the ResOpsUS+CARS archive.

The script extracts the archive, inspects its directory structure, and writes
``data/processed/resopsus_cars_inventory.csv``.
"""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ZIP = PROJECT_ROOT / "data" / "raw" / "ResOpsUS+CARS_v10.zip"
EXTRACT_DIR = PROJECT_ROOT / "data" / "raw" / "ResOpsUS+CARS_v10"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INVENTORY_CSV = PROCESSED_DIR / "resopsus_cars_inventory.csv"


def extract_if_needed() -> None:
    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.iterdir()):
        print(f"[prepare] already extracted: {EXTRACT_DIR}")
        return
    if not RAW_ZIP.exists():
        print("[prepare] zip not found, skip")
        raise SystemExit(1)
    print(f"[prepare] extracting {RAW_ZIP.name} ...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RAW_ZIP) as zf:
        zf.extractall(EXTRACT_DIR)
    print("[prepare] extract done")


def build_inventory() -> None:
    rows: list[dict[str, str]] = []
    for path in sorted(EXTRACT_DIR.rglob("*")):
        if path.is_file():
            rows.append({
                "path": str(path.relative_to(EXTRACT_DIR)),
                "size_bytes": str(path.stat().st_size),
                "suffix": path.suffix.lower(),
            })
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with INVENTORY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "size_bytes", "suffix"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[prepare] inventory written: {INVENTORY_CSV} ({len(rows)} files)")


def main() -> None:
    extract_if_needed()
    build_inventory()


if __name__ == "__main__":
    main()
