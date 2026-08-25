#!/usr/bin/env python3
"""Observational diagnostics for the causal-forest analysis.

The script estimates propensity scores, checks overlap, and reports
standardized mean differences before and after weighting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/analysis_dataset.csv"
OUT_JSON = ROOT / "outputs/causal_diagnostics.json"

FEATURES = [
    "CAP_MCM", "CATCH_SKM", "AREA_SKM", "DAM_HGT_M", "DEPTH_M",
    "DIS_AVG_LS", "YEAR", "USE_ELEC", "USE_FISH", "USE_IRRI",
    "USE_NAVI", "USE_RECR", "USE_SUPP", "SINGLE_USE",
]


def compute_smd(x: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    if w is None:
        w = np.ones(len(x))
    t = t.astype(bool)
    mask = w > 0
    x = x[mask]; t = t[mask]; w = w[mask]
    w1 = w[t]; w0 = w[~t]
    mean1 = np.average(x[t], weights=w1, axis=0)
    mean0 = np.average(x[~t], weights=w0, axis=0)
    var1 = np.average((x[t] - mean1) ** 2, weights=w1, axis=0)
    var0 = np.average((x[~t] - mean0) ** 2, weights=w0, axis=0)
    pool = np.sqrt((var1 + var0) / 2)
    return ((mean1 - mean0) / (pool + 1e-12)).astype(float)


def main() -> None:
    df = pd.read_csv(DATA).dropna(subset=["w1_fdc_shape", "DOR_PC"]).copy()
    T = (df["DOR_PC"] > df["DOR_PC"].median()).astype(int).to_numpy()
    X_raw = df[FEATURES].fillna(df[FEATURES].median()).to_numpy(dtype=float)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X, T)
    ps = lr.predict_proba(X)[:, 1]

    smd_before = compute_smd(X, T)
    # IPW
    w = np.where(T == 1, 1.0 / np.clip(ps, 0.01, 0.99), 1.0 / np.clip(1 - ps, 0.01, 0.99))
    # Trim extreme weights.
    w = np.clip(w, 0.1, 10)
    smd_after = compute_smd(X, T, w)

    results = {
        "n": int(len(df)),
        "propensity_by_group": {
            "treated": {
                "min": float(ps[T == 1].min()),
                "p5": float(np.percentile(ps[T == 1], 5)),
                "median": float(np.median(ps[T == 1])),
                "max": float(ps[T == 1].max()),
            },
            "control": {
                "min": float(ps[T == 0].min()),
                "p5": float(np.percentile(ps[T == 0], 5)),
                "median": float(np.median(ps[T == 0])),
                "max": float(ps[T == 0].max()),
            },
        },
        "overlap_p5_p95": {
            "treated_p5": float(np.percentile(ps[T == 1], 5)),
            "control_p95": float(np.percentile(ps[T == 0], 95)),
        },
        "smd_before": {f: float(v) for f, v in zip(FEATURES, smd_before)},
        "smd_after": {f: float(v) for f, v in zip(FEATURES, smd_after)},
        "smd_abs_above_0_1_before": int((np.abs(smd_before) > 0.1).sum()),
        "smd_abs_above_0_1_after": int((np.abs(smd_after) > 0.1).sum()),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k not in ["smd_before", "smd_after"]}, indent=2, ensure_ascii=False), flush=True)
    print("SMD before:", results["smd_before"], flush=True)
    print("SMD after:", results["smd_after"], flush=True)


if __name__ == "__main__":
    main()
