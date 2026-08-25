#!/usr/bin/env python3
"""Compute Brazil FDC alteration metrics using the US-aligned pipeline.

The reference is the observed inflow series, compared with reservoir outflow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/raw/ResOpsBR+CARS_v10/v1.0"
OUT = ROOT / "data/processed/brazil_flow_regime_alteration.csv"
MIN_ACTIVE_YEARS = 10
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def safe_pct_change(out: float, inp: float) -> float:
    if not np.isfinite(inp) or inp <= 0 or not np.isfinite(out):
        return np.nan
    return (out - inp) / inp * 100.0


def fdc_wasserstein(inflow: np.ndarray, outflow: np.ndarray) -> float:
    if len(inflow) == 0 or len(outflow) == 0:
        return np.nan
    qs = np.linspace(0.01, 0.99, 200)
    return float(np.mean(np.abs(np.quantile(inflow, qs) - np.quantile(outflow, qs))))


def main() -> None:
    files = sorted((BASE / "time_series/csv").glob("*.csv"))
    rows = []
    for path in files:
        gid = path.stem
        try:
            df = pd.read_csv(path, usecols=["date", "inflow", "outflow"], parse_dates=["date"])
        except Exception:
            continue
        sub = df.dropna(subset=["inflow", "outflow"])
        if len(sub) < 365 * MIN_ACTIVE_YEARS:
            continue
        inp = sub["inflow"].to_numpy(dtype=float)
        out = sub["outflow"].to_numpy(dtype=float)
        inp = inp[np.isfinite(inp) & (inp >= 0)]
        out = out[np.isfinite(out) & (out >= 0)]
        if len(inp) < 365 * MIN_ACTIVE_YEARS or len(out) < 365 * MIN_ACTIVE_YEARS:
            continue
        mean_in = float(np.mean(inp))
        mean_out = float(np.mean(out))
        w1 = fdc_wasserstein(inp, out)
        row = {
            "GDW_ID": gid,
            "n_days": len(inp),
            "start": str(pd.to_datetime(sub["date"]).min().date()),
            "end": str(pd.to_datetime(sub["date"]).max().date()),
            "mean_inflow": mean_in,
            "mean_outflow": mean_out,
            "w1_fdc": w1,
            "w1_fdc_norm": (w1 / mean_in) if mean_in > 0 else np.nan,
            "w1_fdc_shape": (fdc_wasserstein(inp / mean_in, out / mean_out) if mean_in > 0 and mean_out > 0 else np.nan),
        }
        for q in QUANTILES:
            in_q = float(np.quantile(inp, q))
            out_q = float(np.quantile(out, q))
            row[f"in_q{int(q*100)}"] = in_q
            row[f"out_q{int(q*100)}"] = out_q
            row[f"alter_q{int(q*100)}_pct"] = safe_pct_change(out_q, in_q)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} rows to {OUT}", flush=True)
    print(df.describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
