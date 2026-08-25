#!/usr/bin/env python3
"""Robustness analysis with extended covariates and CATE intervals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/analysis_dataset.csv"
OUT_JSON = ROOT / "outputs/causal_robustness.json"

BASE_FEATURES = [
    "CAP_MCM", "CATCH_SKM", "AREA_SKM", "DAM_HGT_M", "DEPTH_M",
    "DIS_AVG_LS", "YEAR", "USE_ELEC", "USE_FISH", "USE_IRRI",
    "USE_NAVI", "USE_RECR", "USE_SUPP", "SINGLE_USE",
]
BASINATLAS = ROOT / "data/processed/basinatlas_matched.csv"
BASINATLAS_FEATURES = [
    "dis_m3_pyr", "run_mm_syr", "ari_ix_uav", "pre_mm_uyr",
    "tmp_dc_uyr", "sgr_dk_sav", "cly_pc_sav", "for_pc_sse",
    "pop_ct_ssu", "riv_tc_ssu", "urb_pc_sse", "ire_pc_sse",
]
EXTRA_FEATURES = [
    "mean_inflow", "mean_outflow", "cv_inflow", "cv_outflow", "CAP_RESOPS",
]
FEATURES = BASE_FEATURES + EXTRA_FEATURES + BASINATLAS_FEATURES


def main() -> None:
    from econml.dml import CausalForestDML

    df = pd.read_csv(DATA)
    if BASINATLAS.exists():
        basin = pd.read_csv(BASINATLAS)
        keep = ["GRAND_ID", *BASINATLAS_FEATURES]
        df = df.merge(basin[keep], on="GRAND_ID", how="left", validate="one_to_one")
    else:
        raise FileNotFoundError(f"Required BasinATLAS match table not found: {BASINATLAS}")
    df = df.dropna(subset=["w1_fdc_shape", "DOR_PC"]).copy()
    # Retain rows with the extended covariates required by the robustness model.
    df = df.dropna(subset=EXTRA_FEATURES).copy()
    X = df[FEATURES].fillna(df[FEATURES].median()).to_numpy(dtype=float)
    T = (df["DOR_PC"] > df["DOR_PC"].median()).astype(int).to_numpy()
    Y = df["w1_fdc_shape"].to_numpy(dtype=float)

    X_mean = X.mean(axis=0); X_std = X.std(axis=0) + 1e-9
    Xs = (X - X_mean) / X_std

    model = CausalForestDML(n_estimators=500, max_depth=6, random_state=42,
                            discrete_treatment=True, n_jobs=-1)
    model.fit(Y, T, X=Xs, W=None)
    cate = model.effect(Xs)

    # Use interval estimates when the installed econml version provides them.
    ci_lower = ci_upper = None
    try:
        ci = model.effect_interval(Xs, alpha=0.05)
        ci_lower = np.asarray(ci[0])
        ci_upper = np.asarray(ci[1])
    except Exception:
        pass

    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances(),
    }).sort_values("importance", ascending=False)

    results = {
        "n": int(len(df)),
        "treatment": "DOR_PC > median",
        "features": FEATURES,
        "ate_mean": float(np.mean(cate)),
        "ate_std": float(np.std(cate)),
        "pct_positive_cate": float((cate > 0).mean()),
        "ci_available": ci_lower is not None,
        "ci_mean_lower": float(np.mean(ci_lower)) if ci_lower is not None else None,
        "ci_mean_upper": float(np.mean(ci_upper)) if ci_upper is not None else None,
        "feat_importance": imp.to_dict(orient="records"),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "feat_importance"}, indent=2, ensure_ascii=False), flush=True)
    print("top features:", [(f["feature"], round(f["importance"], 3)) for f in imp.head(8).to_dict(orient="records")], flush=True)


if __name__ == "__main__":
    main()
