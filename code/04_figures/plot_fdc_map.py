#!/usr/bin/env python3
"""Generate the FDC-shape map with the full observed value range."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from nf_style import apply_publication_style, save_pub

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "analysis_dataset.csv"
FIGDIR = ROOT / "manuscript" / "figures"

apply_publication_style(font_size=8, axes_linewidth=1.0)

df = pd.read_csv(DATA).dropna(subset=["w1_fdc_shape", "LAT", "LON"]).copy()
values = df["w1_fdc_shape"].astype(float)
norm = Normalize(vmin=float(values.min()), vmax=float(values.max()))
capacity = df["CAP_MCM"].clip(lower=1).astype(float)
log_capacity = np.log10(capacity)
sizes = 8 + 22 * (log_capacity - log_capacity.min()) / (log_capacity.max() - log_capacity.min() + 1e-9)

fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(
    df["LON"],
    df["LAT"],
    c=values,
    cmap="viridis",
    norm=norm,
    s=sizes,
    alpha=0.8,
    edgecolors="none",
)
ax.set_xlim(-127, -64)
ax.set_ylim(23, 51)
ax.set_xlabel("Longitude (deg)")
ax.set_ylabel("Latitude (deg)")
ax.set_title(f"Spatial distribution of FDC shape alteration (n = {len(df)})")
ax.grid(True, alpha=0.25)
ax.set_axisbelow(True)
ax.text(
    0.985,
    0.965,
    "n = {0}\nMedian = {1:.3f}\nIQR = [{2:.3f}, {3:.3f}]".format(
        len(df), values.median(), values.quantile(0.25), values.quantile(0.75)
    ),
    transform=ax.transAxes,
    fontsize=7,
    va="top",
    ha="right",
    bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.35", "alpha": 0.95},
)
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("FDC shape alteration")
fig.tight_layout()
save_pub(fig, "fig2_fdc_shape_map", FIGDIR)
plt.close(fig)
print("saved fig2_fdc_shape_map", "range", round(norm.vmin, 3), round(norm.vmax, 3))
