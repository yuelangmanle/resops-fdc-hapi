#!/usr/bin/env python3
"""Plot median and IQR relative FDC alteration by functional partition."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nf_style import apply_publication_style

apply_publication_style(font_size=8, axes_linewidth=1.0)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/fdc_functional_clusters.csv"
FIGDIR = ROOT / "manuscript/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
alter_cols = sorted(
    [c for c in df.columns if c.startswith("alter_p")],
    key=lambda c: int(re.search(r"p(\d+)$", c).group(1)),
)
quantiles = np.array([int(re.search(r"p(\d+)$", c).group(1)) for c in alter_cols])
labels = {
    0: "Near-neutral / high-flow increase (n = 254)",
    1: "Extreme amplification (n = 27)",
    2: "Broad reduction (n = 92)",
    3: "Low-flow amplification (n = 60)",
}
colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

fig, ax = plt.subplots(figsize=(6.8, 4.3))
for cid in sorted(df["cluster"].dropna().astype(int).unique()):
    grp = df[df["cluster"] == cid][alter_cols].clip(lower=-1.0, upper=3.0)
    median = grp.median(axis=0).to_numpy(dtype=float) * 100.0
    q25 = grp.quantile(0.25, axis=0).to_numpy(dtype=float) * 100.0
    q75 = grp.quantile(0.75, axis=0).to_numpy(dtype=float) * 100.0
    color = colors[cid % len(colors)]
    ax.fill_between(quantiles, q25, q75, color=color, alpha=0.14, linewidth=0)
    ax.plot(quantiles, median, color=color, linewidth=1.8, label=labels.get(cid, f"Partition {cid}"))

ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
ax.set_xlabel("Flow duration percentile (%)")
ax.set_ylabel("Relative FDC alteration (%)")
ax.set_xlim(1, 99)
ax.set_ylim(-100, 300)
ax.grid(True, alpha=0.25)
ax.set_axisbelow(True)
handles, legend_labels = ax.get_legend_handles_labels()
fig.legend(handles, legend_labels, fontsize=7.0, loc="upper center",
           bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False,
           handlelength=2.2, columnspacing=1.2)
fig.subplots_adjust(top=0.78, left=0.11, right=0.98, bottom=0.14)
for ext in ["png", "svg", "pdf", "tiff"]:
    fig.savefig(FIGDIR / f"fig3_fdc_clusters.{ext}", bbox_inches="tight", dpi=300)
print("saved fig3_fdc_clusters")
