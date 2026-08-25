#!/usr/bin/env python3
"""Build the reproducible US time-series eligibility inventory.

The analysis first applies the dataset-level availability screen used by the
ResOpsUS+CARS release: date, storage, outflow and simulated inflow must all
be present, with at least ten complete calendar years of outflow/simulated-
inflow overlap.  Within that screened set, observed inflow is preferred when
at least 3,650 values are available; otherwise simulated inflow is used.  A
final 3,650-day paired-record requirement is applied to the selected
reference series.  This preserves the documented 433-reservoir analysis
sample while making the selection rule reproducible and explicit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/raw/ResOpsUS+CARS_v10/v1.0/time_series/csv"
OUT = ROOT / "data/processed/reservoir_data_summary.csv"
MIN_DAYS = 365 * 10


def inspect_file(path: Path) -> dict:
    gid_text = path.stem
    try:
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        required = {"date", "storage", "outflow", "inflow_sim"}
        if not required.issubset(columns):
            return {
                "GRAND_ID": gid_text,
                "eligible": False,
                "screened": False,
                "error": "missing required screening column(s): " + ", ".join(sorted(required - set(columns))),
            }
        candidates = [c for c in ("inflow", "inflow_sim") if c in columns]
        usecols = ["date", "storage", "outflow", *candidates]
        df = pd.read_csv(path, usecols=usecols, parse_dates=["date"])
    except Exception as exc:  # keep one malformed file from hiding the inventory
        return {"GRAND_ID": gid_text, "eligible": False, "error": str(exc)[:160]}

    result = {
        "GRAND_ID": gid_text,
        "n_rows": int(len(df)),
        "start": str(df["date"].min().date()) if df["date"].notna().any() else None,
        "end": str(df["date"].max().date()) if df["date"].notna().any() else None,
        "n_outflow": int(df["outflow"].notna().sum()) if "outflow" in df else 0,
        "n_inflow": int(df["inflow"].notna().sum()) if "inflow" in df else 0,
        "n_inflow_sim": int(df["inflow_sim"].notna().sum()) if "inflow_sim" in df else 0,
        "screened": False,
        "sim_active_years": 0,
        "ref_type": None,
        "n_overlap_days": 0,
        "n_active_years": 0,
        "eligible": False,
        "error": None,
    }
    if "outflow" not in df:
        result["error"] = "missing outflow"
        return result

    # Dataset-level screen: this is the historical 434-record list before the
    # final paired-record check removes GRAND_ID 903.
    sim_pair = df[["date", "outflow", "inflow_sim"]].dropna()
    sim_pair = sim_pair[np.isfinite(sim_pair["outflow"]) & np.isfinite(sim_pair["inflow_sim"])]
    sim_pair = sim_pair[(sim_pair["outflow"] >= 0) & (sim_pair["inflow_sim"] >= 0)]
    if len(sim_pair):
        sim_pair = sim_pair.assign(year=sim_pair["date"].dt.year)
        result["sim_active_years"] = int((sim_pair.groupby("year").size() >= 365).sum())
    result["screened"] = result["sim_active_years"] >= 10
    if not result["screened"]:
        return result

    # Match the analysis scripts exactly: observed is preferred once it has
    # enough values; only otherwise is simulated inflow considered.
    if "inflow" in df and result["n_inflow"] >= MIN_DAYS:
        ref_col = "inflow"
    elif "inflow_sim" in df and result["n_inflow_sim"] >= MIN_DAYS:
        ref_col = "inflow_sim"
    else:
        return result

    result["ref_type"] = "observed" if ref_col == "inflow" else "simulated"
    pair = df[["date", "outflow", ref_col]].dropna()
    pair = pair[np.isfinite(pair["outflow"]) & np.isfinite(pair[ref_col])]
    pair = pair[(pair["outflow"] >= 0) & (pair[ref_col] >= 0)]
    result["n_overlap_days"] = int(len(pair))
    if len(pair):
        pair = pair.assign(year=pair["date"].dt.year)
        result["n_active_years"] = int((pair.groupby("year").size() >= 365).sum())
    result["eligible"] = result["n_overlap_days"] >= MIN_DAYS
    return result


def main() -> None:
    rows = [inspect_file(path) for path in sorted(BASE.glob("*.csv"))]
    summary = pd.DataFrame(rows)
    # Preserve lexicographic filename order.  The fixed random seeds in the
    # clustering stage operate on this deterministic row order; changing it
    # can change KMeans' seeded initialization and hence label assignments.
    summary["GRAND_ID"] = summary["GRAND_ID"].astype(str)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)
    report = {
        "files": int(len(summary)),
        "eligible": int(summary["eligible"].sum()),
        "observed_reference": int((summary["eligible"] & (summary["ref_type"] == "observed")).sum()),
        "simulated_reference": int((summary["eligible"] & (summary["ref_type"] == "simulated")).sum()),
        "minimum_paired_days": MIN_DAYS,
        "selection_rule": "required date/storage/outflow/inflow_sim columns and >=10 full years of outflow-inflow_sim overlap; then observed inflow preferred when >=3650 values, otherwise simulated inflow; >=3650 paired non-negative days required",
    }
    (ROOT / "outputs" / "reservoir_summary_build.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
