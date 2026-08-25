#!/usr/bin/env python3
"""Generate FDC functional features for the 1st-99th percentiles.

The output contains inflow FDCs, outflow FDCs, and relative alteration
profiles for downstream PCA and functional clustering.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/raw/ResOpsUS+CARS_v10/v1.0"
SUMMARY = ROOT / "data/processed/reservoir_data_summary.csv"
OUT = ROOT / "data/processed/fdc_functional_features.csv"
MIN_ACTIVE_YEARS = 10
QS = np.round(np.linspace(0.01, 0.99, 99), 3)


def compute_features(path: Path, gid: str) -> dict:
    try:
        cols = pd.read_csv(path, nrows=0).columns
        usecols = ["date", "outflow"]
        if "inflow" in cols:
            usecols.append("inflow")
        if "inflow_sim" in cols:
            usecols.append("inflow_sim")
        df = pd.read_csv(path, usecols=usecols, parse_dates=["date"])
    except Exception:
        return {}
    if "outflow" not in df.columns:
        return {}
    # Prefer observed inflow when available.
    ref_col = None
    if "inflow" in df.columns and df["inflow"].notna().sum() >= 365 * MIN_ACTIVE_YEARS:
        ref_col = "inflow"
    elif "inflow_sim" in df.columns and df["inflow_sim"].notna().sum() >= 365 * MIN_ACTIVE_YEARS:
        ref_col = "inflow_sim"
    if ref_col is None:
        return {}
    sub = df.dropna(subset=["outflow", ref_col])
    if len(sub) < 365 * MIN_ACTIVE_YEARS:
        return {}
    out = sub["outflow"].to_numpy(dtype=float)
    inp = sub[ref_col].to_numpy(dtype=float)
    out = out[np.isfinite(out) & (out >= 0)]
    inp = inp[np.isfinite(inp) & (inp >= 0)]
    if len(inp) < 365 * MIN_ACTIVE_YEARS or len(out) < 365 * MIN_ACTIVE_YEARS:
        return {}

    row = {"GRAND_ID": gid, "ref_type": ref_col}
    in_q = np.quantile(inp, QS)
    out_q = np.quantile(out, QS)
    for i, q in enumerate(QS):
        tag = f"p{int(q*100):02d}"
        row[f"in_{tag}"] = in_q[i]
        row[f"out_{tag}"] = out_q[i]
        row[f"alter_{tag}"] = (out_q[i] - in_q[i]) / in_q[i] if in_q[i] > 0 else np.nan
    return row


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    if "eligible" in summary.columns:
        summary = summary[summary["eligible"].fillna(False)]
    else:
        summary = summary[summary["n_active_years"] >= MIN_ACTIVE_YEARS]
    ids = summary["GRAND_ID"].astype(str).tolist()
    print(f"processing {len(ids)}", flush=True)
    rows = []
    for gid in ids:
        path = BASE / "time_series" / "csv" / f"{gid}.csv"
        if not path.exists():
            continue
        r = compute_features(path, gid)
        if r:
            rows.append(r)
    df = pd.DataFrame(rows)
    # p29 and p58 are excluded from the clustering feature set (quality screening:
    # irregular zero-flow structure in a subset of stations at these two quantiles).
    # This rule is applied here so that re-runs reproduce the 97-quantile matrix.
    DROP_QUANTILES = ["p29", "p58"]
    for tag in DROP_QUANTILES:
        for prefix in ("in_", "out_", "alter_"):
            col = prefix + tag
            if col in df.columns:
                df = df.drop(columns=[col])
    expected = 99 - len(DROP_QUANTILES)
    alter_cols = [c for c in df.columns if c.startswith("alter_")]
    assert len(alter_cols) == expected,         f"expected {expected} alter_ columns after dropping {DROP_QUANTILES}, got {len(alter_cols)}"
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} rows x {len(df.columns)} cols to {OUT}", flush=True)


if __name__ == "__main__":
    main()
