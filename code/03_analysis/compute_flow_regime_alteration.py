#!/usr/bin/env python3
"""Compute reservoir FDC alteration and ecological-flow indicator changes.

Input: ResOpsUS+CARS time-series CSV files.
Output: data/processed/flow_regime_alteration.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "data/raw/ResOpsUS+CARS_v10/v1.0"
SUMMARY = PROJECT_ROOT / "data/processed/reservoir_data_summary.csv"
OUT = PROJECT_ROOT / "data/processed/flow_regime_alteration.csv"
MIN_ACTIVE_YEARS = 10

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def safe_pct_change(out: float, inp: float) -> float:
    if not np.isfinite(inp) or inp <= 0 or not np.isfinite(out):
        return np.nan
    return (out - inp) / inp * 100.0


def fdc_wasserstein(inflow: np.ndarray, outflow: np.ndarray) -> float:
    """Approximate one-dimensional Wasserstein distance from 200 quantiles."""
    if len(inflow) == 0 or len(outflow) == 0:
        return np.nan
    qs = np.linspace(0.01, 0.99, 200)
    xq = np.quantile(inflow, qs)
    yq = np.quantile(outflow, qs)
    return float(np.mean(np.abs(xq - yq)))


def compute_reservoir(path: Path, gid: str) -> dict:
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
    if df.empty or "outflow" not in df.columns:
        return {}
    df = df.dropna(subset=["date"]).set_index("date")

    # Prefer observed inflow and use simulated inflow only when needed.
    ref_type = None
    if "inflow" in df.columns:
        n_in = df["inflow"].notna().sum()
        if n_in >= 365 * MIN_ACTIVE_YEARS:
            ref_type = "observed"
    if ref_type is None and "inflow_sim" in df.columns:
        n_sim = df["inflow_sim"].notna().sum()
        if n_sim >= 365 * MIN_ACTIVE_YEARS:
            ref_type = "simulated"
    if ref_type is None:
        return {}

    ref_col = "inflow" if ref_type == "observed" else "inflow_sim"
    sub = df.dropna(subset=["outflow", ref_col])
    if len(sub) < 365 * MIN_ACTIVE_YEARS:
        return {}

    out = sub["outflow"].to_numpy(dtype=float)
    inp = sub[ref_col].to_numpy(dtype=float)
    out = out[np.isfinite(out) & (out >= 0)]
    inp = inp[np.isfinite(inp) & (inp >= 0)]
    if len(inp) < 365 * MIN_ACTIVE_YEARS or len(out) < 365 * MIN_ACTIVE_YEARS:
        return {}

    mean_in = float(np.mean(inp))
    mean_out = float(np.mean(out))
    w1 = fdc_wasserstein(inp, out)
    row = {
        "GRAND_ID": gid,
        "ref_type": ref_type,
        "n_days": len(inp),
        "start": str(sub.index.min().date()),
        "end": str(sub.index.max().date()),
        "mean_inflow": mean_in,
        "mean_outflow": mean_out,
        "w1_fdc": w1,
        "w1_fdc_norm": (w1 / mean_in) if mean_in > 0 else np.nan,
        "w1_fdc_shape": (
            fdc_wasserstein(inp / mean_in, out / mean_out)
            if mean_in > 0 and mean_out > 0
            else np.nan
        ),
    }

    for q in QUANTILES:
        in_q = float(np.quantile(inp, q))
        out_q = float(np.quantile(out, q))
        row[f"in_q{int(q*100)}"] = in_q
        row[f"out_q{int(q*100)}"] = out_q
        row[f"alter_q{int(q*100)}_pct"] = safe_pct_change(out_q, in_q)

    row["cv_inflow"] = float(np.std(inp) / np.mean(inp)) if np.mean(inp) > 0 else np.nan
    row["cv_outflow"] = float(np.std(out) / np.mean(out)) if np.mean(out) > 0 else np.nan
    return row


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    if "eligible" in summary.columns:
        summary = summary[summary["eligible"].fillna(False)]
    else:
        summary = summary[summary["n_active_years"] >= MIN_ACTIVE_YEARS]
    ids = summary["GRAND_ID"].astype(str).tolist()
    print(f"selected reservoirs: {len(ids)}", flush=True)

    rows = []
    for gid in ids:
        path = BASE / "time_series" / "csv" / f"{gid}.csv"
        if not path.exists():
            continue
        r = compute_reservoir(path, gid)
        if r:
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} rows to {OUT}", flush=True)
    print(df.describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
