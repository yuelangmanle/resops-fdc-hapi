#!/usr/bin/env python3
"""Compare HAPI with simple single-component ranking alternatives.

Alternatives are FDC shape alteration, DOR_PC, CAP_MCM, and absolute Q10
alteration; comparisons use Spearman correlations and top-10/top-20 overlap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/analysis_dataset.csv"
HAPI = ROOT / "data/processed/hapi_scores.csv"
OUT_JSON = ROOT / "outputs/hapi_existing_comparison.json"

ALTERNATIVES = {
    "w1_fdc_shape": "FDC shape alteration",
    "DOR_PC": "Degree of regulation",
    "CAP_MCM": "Storage capacity",
    "abs_alter_q10": "Absolute Q10 alteration",
}


def main() -> None:
    df = pd.read_csv(DATA)
    hapi = pd.read_csv(HAPI)
    df = df.merge(hapi[["GRAND_ID", "HAPI"]], on="GRAND_ID", how="inner")
    df["abs_alter_q10"] = df["alter_q10_pct"].abs()
    df = df.dropna(subset=["HAPI"])

    results = {}
    for col, name in ALTERNATIVES.items():
        sub = df.dropna(subset=[col])
        if len(sub) < 2:
            continue
        rho, p = spearmanr(sub["HAPI"], sub[col])
        # top20 overlap by HAPI and by alternative
        top20_hapi = set(sub.nlargest(20, "HAPI")["GRAND_ID"])
        top20_alt = set(sub.nlargest(20, col)["GRAND_ID"])
        overlap20 = len(top20_hapi & top20_alt) / 20
        top10_hapi = set(sub.nlargest(10, "HAPI")["GRAND_ID"])
        top10_alt = set(sub.nlargest(10, col)["GRAND_ID"])
        overlap10 = len(top10_hapi & top10_alt) / 10
        results[name] = {
            "n": int(len(sub)),
            "spearman": float(rho),
            "p_value": float(p),
            "top10_overlap": float(overlap10),
            "top20_overlap": float(overlap20),
        }
        print(name, results[name], flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
