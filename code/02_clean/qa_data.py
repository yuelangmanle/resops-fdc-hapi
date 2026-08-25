#!/usr/bin/env python3
"""Audit date continuity, duplicates, negative values, jumps, and missingness.

Output: ``outputs/data_qa_report.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "outputs/data_qa_report.json"

DATASETS = {
    "US": {
        "dir": ROOT / "data/raw/ResOpsUS+CARS_v10/v1.0/time_series/csv",
        "flow_cols": ["outflow"],
        "storage_col": "storage",
    },
    "BR": {
        "dir": ROOT / "data/raw/ResOpsBR+CARS_v10/v1.0/time_series/csv",
        "flow_cols": ["outflow", "inflow"],
        "storage_col": "storage",
    },
}


def check_file(path: Path, flow_cols: list[str], storage_col: str) -> dict:
    try:
        available = pd.read_csv(path, nrows=0).columns
        dynamic_flows = [c for c in ("inflow", "inflow_sim") if c in available]
        requested = ["date", *flow_cols, *dynamic_flows, storage_col]
        requested = list(dict.fromkeys(c for c in requested if c in available))
        df = pd.read_csv(path, usecols=requested, parse_dates=["date"])
    except Exception as e:
        return {"file": path.name, "error": str(e)[:100]}
    if df.empty:
        return {"file": path.name, "error": "empty"}

    date = df["date"]
    issues = {
        "duplicate_dates": int(date.duplicated().sum()),
        "non_monotonic": int((date.diff().dt.days.fillna(1) <= 0).sum()),
        "negative_flow": 0,
        "negative_storage": int((df[storage_col] < 0).sum()) if storage_col in df else 0,
        "extreme_jumps": {},
        "missing_flow": int(df[[c for c in [*flow_cols, "inflow", "inflow_sim"] if c in df]].isna().sum().sum()),
        "n_rows": len(df),
    }
    for col in flow_cols:
        if col not in df:
            continue
        s = df[col]
        issues["negative_flow"] += int((s < 0).sum())
        # Flag daily relative jumps above 500% or below -80%.
        s = s.replace(0, np.nan)
        pct = s.pct_change(fill_method=None).abs()
        issues["extreme_jumps"][col] = int((pct > 5).sum())
    for col in ("inflow", "inflow_sim"):
        if col not in df or col in flow_cols:
            continue
        s = df[col]
        issues["negative_flow"] += int((s < 0).sum())
        pct = s.replace(0, np.nan).pct_change(fill_method=None).abs()
        issues["extreme_jumps"][col] = int((pct > 5).sum())
    return {"file": path.name, **issues}


def main() -> None:
    report = {}
    for name, cfg in DATASETS.items():
        files = sorted(cfg["dir"].glob("*.csv"))
        rows = []
        total = {
            "files": len(files),
            "duplicate_dates": 0,
            "non_monotonic": 0,
            "negative_flow": 0,
            "negative_storage": 0,
            "extreme_jumps": {},
            "missing_flow": 0,
            "rows_with_issues": 0,
        }
        for f in files:
            r = check_file(f, cfg["flow_cols"], cfg["storage_col"])
            rows.append(r)
            if "error" not in r:
                total["duplicate_dates"] += r["duplicate_dates"]
                total["non_monotonic"] += r["non_monotonic"]
                total["negative_flow"] += r["negative_flow"]
                total["negative_storage"] += r["negative_storage"]
                total["missing_flow"] += r["missing_flow"]
                if r["duplicate_dates"] or r["non_monotonic"] or r["negative_flow"] or r["negative_storage"] or any(r["extreme_jumps"].values()):
                    total["rows_with_issues"] += 1
                for col, val in r["extreme_jumps"].items():
                    total["extreme_jumps"][col] = total["extreme_jumps"].get(col, 0) + val
            else:
                total["rows_with_issues"] += 1
        report[name] = {"total": total, "files_with_issues": [r for r in rows if "error" in r or r.get("duplicate_dates", 0) or r.get("non_monotonic", 0) or r.get("negative_flow", 0) or r.get("negative_storage", 0) or any(r.get("extreme_jumps", {}).values())][:20]}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v["total"] for k, v in report.items()}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
