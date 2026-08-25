#!/usr/bin/env python3
"""Run the robust US-Brazil comparison.

The analysis compares the US observed-inflow subset with all Brazilian
reservoirs and reports Cliff's delta plus a bootstrap median-difference CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
US = ROOT / "data/processed/flow_regime_alteration.csv"
BR = ROOT / "data/processed/brazil_flow_regime_alteration.csv"
OUT_JSON = ROOT / "outputs/us_brazil_robust.json"


def cliffs_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    m = len(x); n = len(y)
    if m == 0 or n == 0:
        return np.nan
    gt = 0.0; lt = 0.0
    for xi in x:
        gt += np.sum(y < xi)
        lt += np.sum(y > xi)
    return (gt - lt) / (m * n)


def bootstrap_median_diff(x, y, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    diffs = []
    for _ in range(n):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        diffs.append(np.median(bx) - np.median(by))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def main() -> None:
    us = pd.read_csv(US)
    br = pd.read_csv(BR)
    us_obs = us[(us["ref_type"] == "observed")].dropna(subset=["w1_fdc_shape"])
    brn = br.dropna(subset=["w1_fdc_shape"])
    x = us_obs["w1_fdc_shape"].to_numpy()
    y = brn["w1_fdc_shape"].to_numpy()
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    res = {
        "n_us_observed": int(len(x)),
        "n_brazil": int(len(y)),
        "median_us": float(np.median(x)),
        "median_brazil": float(np.median(y)),
        "median_diff": float(np.median(x) - np.median(y)),
        "median_diff_ci95": bootstrap_median_diff(x, y),
        "mannwhitney_u": float(u),
        "p_value": float(p),
        "cliffs_delta": float(cliffs_delta(x, y)),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
