#!/usr/bin/env python3
"""Supplementary statistical tests and bootstrap intervals.

The output also records that the unrecoverable regional Q10 subgroup
calculation is deliberately not reported in the manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data/processed/analysis_dataset.csv"
CLUSTER = ROOT / "data/processed/fdc_functional_clusters.csv"
SENSITIVITY = ROOT / "outputs/hapi_sensitivity_results.json"
OUT_JSON = ROOT / "outputs/additional_stats.json"

METRICS = ["w1_fdc_shape", "DOR_PC", "CAP_MCM"]


def main() -> None:
    ana = pd.read_csv(ANALYSIS)
    clu = pd.read_csv(CLUSTER)[["GRAND_ID", "cluster"]]
    df = ana.merge(clu, on="GRAND_ID", how="inner")

    pairwise = {}
    for metric in METRICS:
        sub = df.dropna(subset=[metric])
        clusters = sorted(sub["cluster"].unique())
        pairs = {}
        for i, c1 in enumerate(clusters):
            for c2 in clusters[i+1:]:
                g1 = sub.loc[sub["cluster"] == c1, metric].to_numpy()
                g2 = sub.loc[sub["cluster"] == c2, metric].to_numpy()
                if len(g1) < 2 or len(g2) < 2:
                    continue
                stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
                pairs[f"{c1}_vs_{c2}"] = {"u": float(stat), "p": float(p)}
        pairwise[metric] = pairs

    sens = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    rng = np.random.default_rng(42)
    spearman_array = np.array(sens.get("spearman_list", []))
    overlap_array = np.array(sens.get("top10_overlap_list", []))
    def bootstrap_ci(arr, n=2000):
        if len(arr) == 0:
            return None
        boot = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)])
        return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    hapi_ci = {
        "spearman_ci": bootstrap_ci(spearman_array),
        "top10_overlap_ci": bootstrap_ci(overlap_array),
    }

    results = {
        "pairwise_cluster_mannwhitney": pairwise,
        "hapi_sensitivity_bootstrap_ci": hapi_ci,
        "regional_q10_stratification": {
            "status": "not_reported",
            "reason": (
                "No auditable regional assignment rule or aggregation record was "
                "preserved; regional subgroup values were excluded from the manuscript "
                "and claim verification."
            ),
            "assignment_rule": None,
            "reported_values": None,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
