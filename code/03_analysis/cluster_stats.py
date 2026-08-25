#!/usr/bin/env python3
"""Test hydrological and attribute differences among FDC partitions."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import kruskal

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data/processed/analysis_dataset.csv"
CLUSTER = ROOT / "data/processed/fdc_functional_clusters.csv"
OUT_JSON = ROOT / "outputs/cluster_stats.json"

METRICS = ["w1_fdc_shape", "DOR_PC", "CAP_MCM", "mean_inflow", "mean_outflow"]


def main() -> None:
    ana = pd.read_csv(ANALYSIS)
    clu = pd.read_csv(CLUSTER)[["GRAND_ID", "cluster"]]
    df = ana.merge(clu, on="GRAND_ID", how="inner")
    results = {}
    for col in METRICS:
        sub = df.dropna(subset=[col])
        groups = [g[col].to_numpy(dtype=float) for _, g in sub.groupby("cluster")]
        if len(groups) < 2:
            continue
        stat, p = kruskal(*groups)
        results[col] = {
            "n_clusters": len(groups),
            "h_stat": float(stat),
            "p_value": float(p),
            "cluster_means": {int(k): float(g[col].mean()) for k, g in sub.groupby("cluster")},
        }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
