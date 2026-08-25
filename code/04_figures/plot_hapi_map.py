#!/usr/bin/env python3
"""
Plot the spatial distribution of the Hydrological Alteration Prioritization Index (HAPI).
"""
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
DATA = ROOT / "data/processed/hapi_scores.csv"
FIGDIR = ROOT / "manuscript/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
df = df.dropna(subset=["LAT", "LON", "HAPI"])

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(df["LON"], df["LAT"], c=df["HAPI"], cmap="plasma", s=14, alpha=0.85)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.colorbar(sc, label="Hydrological Alteration Prioritization Index (HAPI)")
fig.tight_layout()
for ext in ["png", "svg", "pdf", "tiff"]:
    fig.savefig(FIGDIR / f"fig5_hapi_map.{ext}", bbox_inches="tight")
print("saved fig5_hapi_map")
