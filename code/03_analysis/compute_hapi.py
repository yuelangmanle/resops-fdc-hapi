#!/usr/bin/env python3
"""Compute the Hydrological Alteration Prioritization Index (HAPI).

HAPI = 0.5 * FDC alteration + 0.3 * degree of regulation (DOR)
       + 0.2 * storage capacity, after min-max normalization.

Output: data/processed/hapi_scores.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/analysis_dataset.csv"
OUT = ROOT / "data/processed/hapi_scores.csv"

WEIGHTS = {
    "w1_fdc_shape": 0.5,
    "DOR_PC": 0.3,
    "CAP_MCM": 0.2,
}


def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo = s.min()
    hi = s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def main() -> None:
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["w1_fdc_shape", "DOR_PC", "CAP_MCM"]).copy()

    score = pd.Series(0.0, index=df.index)
    for col, w in WEIGHTS.items():
        score = score + w * minmax(df[col])

    df["HAPI"] = score
    df = df.sort_values("HAPI", ascending=False)
    df["HAPI_rank"] = np.arange(1, len(df) + 1)

    out_cols = [
        "GRAND_ID", "RES_NAME", "RIVER", "STATE", "LAT", "LON",
        "w1_fdc_shape", "DOR_PC", "CAP_MCM", "HAPI", "HAPI_rank",
    ]
    df[out_cols].to_csv(OUT, index=False)
    print(f"saved {len(df)} rows to {OUT}")
    print(df[out_cols].head(20).to_string())

if __name__ == "__main__":
    main()
