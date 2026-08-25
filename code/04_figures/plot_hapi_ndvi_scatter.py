#!/usr/bin/env python3
"""Plot the full HAPI versus MODIS NDVI supplementary scatter plot."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from nf_style import apply_publication_style

apply_publication_style(font_size=8, axes_linewidth=1.0)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/remote_sensing_ndvi_ndwi_all.csv"
FIGDIR = ROOT / "manuscript/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df["ndvi_frac"] = df["ndvi"] / 10000.0

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(df["HAPI"], df["ndvi_frac"], s=8, alpha=0.5, color="steelblue")
ax.set_xlabel("HAPI")
ax.set_ylabel("MODIS NDVI (2015–2020 median)")
ax.set_title("HAPI vs NDVI across 430 reservoirs")
fig.tight_layout()
for ext in ["png", "svg", "pdf", "tiff"]:
    fig.savefig(FIGDIR / f"figS4_hapi_ndvi_scatter.{ext}", bbox_inches="tight")
print("saved figS4_hapi_ndvi_scatter")
