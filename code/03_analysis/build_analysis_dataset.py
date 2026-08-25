#!/usr/bin/env python3
"""Merge FDC alteration results with reservoir and basin attributes."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/raw/ResOpsUS+CARS_v10/v1.0"
OUT = ROOT / "data/processed/analysis_dataset.csv"

alter = pd.read_csv(ROOT / "data/processed/flow_regime_alteration.csv")
grand = pd.read_csv(BASE / "attributes/grand.csv")
resops = pd.read_csv(BASE / "attributes/resops.csv")
glofas = pd.read_csv(BASE / "attributes/glofas.csv")

df = alter.merge(grand, on="GRAND_ID", how="left")
df = df.merge(resops[["GRAND_ID", "STATE", "CAP_RESOPS", "INFLOW", "OUTFLOW", "STORAGE"]], on="GRAND_ID", how="left")
df = df.merge(glofas[["GRAND_ID", "CAP", "Qn", "Qf", "Qmin", "Vn"]], on="GRAND_ID", how="left")

feat_cols = [
    "GRAND_ID", "RES_NAME", "RIVER", "STATE", "LAT", "LON", "YEAR",
    "CAP_MCM", "CATCH_SKM", "AREA_SKM", "DAM_HGT_M", "DOR_PC", "DEPTH_M", "DIS_AVG_LS",
    "USE_ELEC", "USE_FISH", "USE_IRRI", "USE_NAVI", "USE_RECR", "USE_SUPP", "SINGLE_USE",
    "CAP_RESOPS", "CAP", "Qn", "Qf", "Qmin", "Vn",
    "mean_inflow", "mean_outflow", "n_days",
    "w1_fdc", "w1_fdc_norm", "w1_fdc_shape",
    "alter_q5_pct", "alter_q10_pct", "alter_q25_pct", "alter_q50_pct",
    "alter_q75_pct", "alter_q90_pct", "alter_q95_pct",
    "cv_inflow", "cv_outflow"
]
missing = [c for c in feat_cols if c not in df.columns]
if missing:
    print("missing:", missing, flush=True)
df = df[feat_cols].copy()
df.to_csv(OUT, index=False)
print("shape:", df.shape, flush=True)
print(df.head(3).to_string(), flush=True)
print("\nmissing rate (selected):", flush=True)
print((df.isna().mean().sort_values(ascending=False)).head(20).to_string(), flush=True)
