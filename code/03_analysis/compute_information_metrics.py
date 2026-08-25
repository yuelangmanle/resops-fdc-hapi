#!/usr/bin/env python3
"""Compute information-theoretic metrics for inflow and outflow signals.

The outputs include permutation entropy, spectral entropy, mutual information,
and approximate sample entropy. Sample entropy uses subsampling for speed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/raw/ResOpsUS+CARS_v10/v1.0"
SUMMARY = ROOT / "data/processed/reservoir_data_summary.csv"
OUT = ROOT / "data/processed/information_metrics.csv"
MIN_ACTIVE_YEARS = 10


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Compute normalized permutation entropy."""
    n = len(x)
    if n < order * delay + 1:
        return np.nan
    # Subsample long series to keep computation bounded.
    stride = max(1, n // 5000)
    x = x[::stride]
    patterns = []
    for i in range(len(x) - (order - 1) * delay):
        window = x[i:i + (order - 1) * delay + 1:delay]
        patterns.append(tuple(np.argsort(window)))
    counts = pd.Series(patterns).value_counts(normalize=True).values
    return float(-np.sum(counts * np.log2(counts)) / np.log2(math.factorial(order)))


def spectral_entropy(x: np.ndarray) -> float:
    """Compute normalized power-spectral entropy."""
    x = x - np.mean(x)
    if np.std(x) == 0:
        return np.nan
    spec = np.abs(np.fft.rfft(x)) ** 2
    if spec.sum() == 0:
        return np.nan
    p = spec / spec.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(len(p)))


def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Estimate histogram mutual information normalized to [0, 1]."""
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n = min(len(x), len(y))
    if n < 100:
        return np.nan
    x = x[:n]
    y = y[:n]
    cxy, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = cxy / cxy.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
    # Normalize by min(HX, HY).
    hx = -np.sum(px[px > 0] * np.log2(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log2(py[py > 0]))
    denom = min(hx, hy) if min(hx, hy) > 0 else 1
    return float(mi / denom)


def sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2, max_samples: int = 300) -> float:
    """Compute approximate sample entropy after upstream subsampling."""
    x = x[np.isfinite(x)]
    if len(x) < max_samples:
        x = x
    else:
        idx = np.linspace(0, len(x) - 1, max_samples).astype(int)
        x = x[idx]
    n = len(x)
    r = r_factor * np.std(x)
    if r == 0 or n < m + 2:
        return np.nan

    def count_matches(m_len: int) -> int:
        count = 0
        for i in range(n - m_len):
            for j in range(i + 1, n - m_len):
                if np.max(np.abs(x[i:i + m_len] - x[j:j + m_len])) <= r:
                    count += 1
        return count

    b = count_matches(m)
    a = count_matches(m + 1)
    if b == 0 or a == 0:
        return np.nan
    return float(-np.log(a / b))


def compute_for_reservoir(path: Path, gid: str) -> dict:
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
    inp = sub[ref_col].to_numpy(dtype=float)
    out = sub["outflow"].to_numpy(dtype=float)
    inp = inp[np.isfinite(inp) & (inp >= 0)]
    out = out[np.isfinite(out) & (out >= 0)]
    if len(inp) < 365 * MIN_ACTIVE_YEARS or len(out) < 365 * MIN_ACTIVE_YEARS:
        return {}

    # Align mutual-information inputs.
    n = min(len(inp), len(out))
    return {
        "GRAND_ID": gid,
        "ref_type": ref_col,
        "pe_inflow": permutation_entropy(inp),
        "pe_outflow": permutation_entropy(out),
        "spec_entropy_inflow": spectral_entropy(inp),
        "spec_entropy_outflow": spectral_entropy(out),
        "mi_norm": mutual_information(inp[:n], out[:n]),
        "sampen_inflow": sample_entropy(inp),
        "sampen_outflow": sample_entropy(out),
    }


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    if "eligible" in summary.columns:
        summary = summary[summary["eligible"].fillna(False)]
    else:
        summary = summary[summary["n_active_years"] >= MIN_ACTIVE_YEARS]
    ids = summary["GRAND_ID"].astype(str).tolist()
    print(f"processing {len(ids)} reservoirs", flush=True)
    rows = []
    for gid in ids:
        path = BASE / "time_series" / "csv" / f"{gid}.csv"
        if not path.exists():
            continue
        r = compute_for_reservoir(path, gid)
        if r:
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"saved {len(df)} rows to {OUT}", flush=True)
    print(df.describe().to_string(), flush=True)


if __name__ == "__main__":
    main()
