#!/usr/bin/env python3
"""Assess HAPI ranking sensitivity to Dirichlet weight perturbations.

For each random weight vector, the script computes the HAPI ranking,
Spearman correlation with the base ranking, and top-10/top-20 overlap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/analysis_dataset.csv"
HAPI_CSV = ROOT / "data/processed/hapi_scores.csv"
OUT_JSON = ROOT / "outputs/hapi_sensitivity_results.json"
N_SIM = 1000
BASE_WEIGHTS = np.array([0.5, 0.3, 0.2])
FEATURES = ["w1_fdc_shape", "DOR_PC", "CAP_MCM"]


def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def main() -> None:
    df = pd.read_csv(DATA).dropna(subset=FEATURES).copy()
    base = pd.read_csv(HAPI_CSV)
    base_ids = set(base["GRAND_ID"].astype(str))
    df = df[df["GRAND_ID"].astype(str).isin(base_ids)].copy()

    for col in FEATURES:
        df[col] = df[col].fillna(df[col].median())
    X = pd.DataFrame({col: minmax(df[col]) for col in FEATURES})

    base_score = X @ BASE_WEIGHTS
    base_rank = base_score.rank(ascending=False).to_numpy()
    base_top10 = set(base_score.nlargest(10).index)
    base_top20 = set(base_score.nlargest(20).index)

    rng = np.random.default_rng(42)
    rho_list, top10_overlap, top20_overlap = [], [], []
    weights_list = []
    for _ in range(N_SIM):
        w = rng.dirichlet(np.ones(3))
        score = X @ w
        rank = score.rank(ascending=False).to_numpy()
        rho, _ = spearmanr(base_rank, rank)
        rho_list.append(rho)
        top10_overlap.append(len(set(score.nlargest(10).index) & base_top10) / 10)
        top20_overlap.append(len(set(score.nlargest(20).index) & base_top20) / 20)
        weights_list.append(w)

    results = {
        "n_sim": N_SIM,
        "base_weights": BASE_WEIGHTS.tolist(),
        "spearman_mean": float(np.mean(rho_list)),
        "spearman_min": float(np.min(rho_list)),
        "spearman_p5": float(np.percentile(rho_list, 5)),
        "top10_overlap_mean": float(np.mean(top10_overlap)),
        "top10_overlap_min": float(np.min(top10_overlap)),
        "top20_overlap_mean": float(np.mean(top20_overlap)),
        "top20_overlap_min": float(np.min(top20_overlap)),
        "spearman_list": [float(x) for x in rho_list],
        "top10_overlap_list": [float(x) for x in top10_overlap],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
