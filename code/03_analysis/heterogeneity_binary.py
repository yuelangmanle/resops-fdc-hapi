#!/usr/bin/env python3
"""Estimate exploratory heterogeneity with a binary regulation treatment.

Treatment is above- versus below-median DOR_PC and the outcome is
w1_fdc_shape. Version 2 adds a bootstrap interval by refitting the forest
200 times and taking the 2.5th and 97.5th percentiles of mean CATE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/analysis_dataset.csv"
OUT_JSON = ROOT / "outputs/heterogeneity_binary_results.json"

FEATURES = [
    "CAP_MCM", "CATCH_SKM", "AREA_SKM", "DAM_HGT_M", "DEPTH_M",
    "DIS_AVG_LS", "YEAR", "USE_ELEC", "USE_FISH", "USE_IRRI",
    "USE_NAVI", "USE_RECR", "USE_SUPP", "SINGLE_USE",
]

N_BOOTSTRAP = 200
SEED = 42


def fit_and_ate(Xs, T, Y, seed):
    from econml.dml import CausalForestDML

    model = CausalForestDML(
        n_estimators=100,
        max_depth=6,
        random_state=seed,
        discrete_treatment=True,
        n_jobs=-1,
    )
    model.fit(Y, T, X=Xs, W=None)
    cate = model.effect(Xs)
    return float(np.mean(cate))


def main() -> None:
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["w1_fdc_shape", "DOR_PC"]).copy()
    X = df[FEATURES].fillna(df[FEATURES].median()).to_numpy(dtype=float)
    T = (df["DOR_PC"] > df["DOR_PC"].median()).astype(int).to_numpy()
    Y = df["w1_fdc_shape"].to_numpy(dtype=float)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-9
    Xs = (X - X_mean) / X_std

    # Trim to the propensity-score common-support region before estimation.
    _lr = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xs, T)
    ps_vals = np.clip(_lr.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
    lo = np.percentile(ps_vals[T == 1], 2.5)
    hi = np.percentile(ps_vals[T == 0], 97.5)
    keep = (ps_vals > max(lo, 0.05)) & (ps_vals < min(hi, 0.95))
    Xt, Tt, Yt = Xs[keep], T[keep], Y[keep]
    n_trim = int((~keep).sum())

    # Fit the primary model on the trimmed common-support sample.
    from econml.dml import CausalForestDML

    model = CausalForestDML(
        n_estimators=300,
        max_depth=6,
        random_state=SEED,
        discrete_treatment=True,
        n_jobs=-1,
    )
    model.fit(Yt, Tt, X=Xt, W=None)
    cate = model.effect(Xt)
    ate = float(np.mean(cate))

    # Refit each bootstrap replicate to quantify estimation uncertainty.
    rng = np.random.default_rng(SEED)
    n = len(Yt)
    boot_ates = []
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        try:
            boot_ates.append(fit_and_ate(Xt[idx], Tt[idx], Yt[idx], seed=SEED + b + 1))
        except Exception:
            continue
    boot_ates = np.array(boot_ates)
    ci_lower = float(np.percentile(boot_ates, 2.5))
    ci_upper = float(np.percentile(boot_ates, 97.5))

    # Estimate feature importance.
    imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances()}).sort_values("importance", ascending=False)

    results = {
        "n_original": int(len(df)),
        "n_after_overlap_trim": int(len(Yt)),
        "n_trimmed": int(n_trim),
        "ps_trim_lo": float(max(lo, 0.05)),
        "ps_trim_hi": float(min(hi, 0.95)),
        "treatment": "DOR_PC > median (high regulation vs low regulation)",
        "ate_mean": ate,
        "ate_std": float(np.std(cate)),
        "ate_min": float(np.min(cate)),
        "ate_max": float(np.max(cate)),
        "pct_positive_cate": float((cate > 0).mean()),
        "ate_bootstrap_ci": {
            "n_bootstrap": len(boot_ates),
            "ci_lower_2.5": ci_lower,
            "ci_upper_97.5": ci_upper,
            "boot_mean": float(np.mean(boot_ates)),
        },
        "feature_importance": imp.to_dict(orient="records"),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
