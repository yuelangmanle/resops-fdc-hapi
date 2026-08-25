#!/usr/bin/env python3
"""Plot the US-Brazil FDC-shape comparison from the source data."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from nf_style import apply_publication_style

apply_publication_style(font_size=8, axes_linewidth=1.0)

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "manuscript/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

us = pd.read_csv(ROOT / "data/processed/flow_regime_alteration.csv")
br = pd.read_csv(ROOT / "data/processed/brazil_flow_regime_alteration.csv")
us_values = us["w1_fdc_shape"].dropna().to_numpy(dtype=float)
br_values = br["w1_fdc_shape"].dropna().to_numpy(dtype=float)
u_stat, p_value = stats.mannwhitneyu(us_values, br_values, alternative="two-sided")

fig, ax = plt.subplots(figsize=(6.2, 4.8))
parts = ax.violinplot(
    [us_values, br_values],
    positions=[1, 2],
    widths=0.68,
    showmeans=False,
    showmedians=False,
    showextrema=True,
)
colors = ["#2E5090", "#D62728"]
for body, color in zip(parts["bodies"], colors):
    body.set_facecolor(color)
    body.set_edgecolor("black")
    body.set_alpha(0.62)
    body.set_linewidth(1.0)
for key in ("cbars", "cmins", "cmaxes"):
    if key in parts:
        parts[key].set_color("black")
        parts[key].set_linewidth(1.2)
ax.boxplot(
    [us_values, br_values],
    positions=[1, 2],
    widths=0.22,
    patch_artist=True,
    showfliers=False,
    boxprops={"facecolor": "white", "alpha": 0.65, "linewidth": 1.0},
    medianprops={"color": "#B22222", "linewidth": 1.6},
    whiskerprops={"linewidth": 1.0},
    capprops={"linewidth": 1.0},
)

us_median = float(np.median(us_values))
br_median = float(np.median(br_values))
for x, med, color in [(1, us_median, "#0B3A6B"), (2, br_median, "#8B0000")]:
    ax.text(
        x + 0.20,
        med,
        f"{med:.3f}",
        ha="left",
        va="center",
        fontsize=8,
        color=color,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "linewidth": 0.8},
    )

ymax = max(float(us_values.max()), float(br_values.max()))
y_bracket = ymax * 1.12
ax.plot([1, 1, 2, 2], [y_bracket, y_bracket * 1.04, y_bracket * 1.04, y_bracket], color="black", linewidth=1.0)
ax.text(1.5, y_bracket * 1.055, "p < 0.001", ha="center", va="bottom", fontsize=8)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"United States\n(n = {len(us_values)})", f"Brazil\n(n = {len(br_values)})"])
ax.set_ylabel("FDC shape alteration")
ax.set_title("FDC shape alteration: US versus Brazil")
ax.set_xlim(0.45, 2.55)
ax.set_ylim(0, ymax * 1.24)
ax.grid(True, axis="y", alpha=0.25)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ["png", "svg", "pdf", "tiff"]:
    fig.savefig(FIGDIR / f"fig7_us_brazil.{ext}", bbox_inches="tight", dpi=300)
print("saved fig7_us_brazil", {"u": float(u_stat), "p": float(p_value)})
