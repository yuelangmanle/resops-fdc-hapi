#!/usr/bin/env python3
"""Plot HAPI ranking sensitivity distributions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from nf_style import apply_publication_style

apply_publication_style(font_size=8, axes_linewidth=1.0)

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "manuscript/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
data = json.loads((ROOT / "outputs/hapi_sensitivity_results.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
series = [
    (np.asarray(data["spearman_list"], dtype=float), "#2F6FAE",
     "Spearman correlation", data["spearman_mean"], data["spearman_p5"],
     "mean = {:.2f}; P5 = {:.2f}".format(data["spearman_mean"], data["spearman_p5"])),
    (np.asarray(data["top10_overlap_list"], dtype=float), "#D77A2B",
     "Top-10 overlap", data["top10_overlap_mean"], None,
     "mean = {:.1%}".format(data["top10_overlap_mean"])),
]
for i, (values, color, xlabel, mean, p5, stat_label) in enumerate(series):
    values = np.sort(values)
    y = np.arange(1, values.size + 1, dtype=float) / values.size
    axes[i].step(values, y, where="post", color=color, linewidth=2.0)
    axes[i].fill_between(values, y, step="post", color=color, alpha=0.12)
    axes[i].axvline(mean, color=color, linestyle="--", linewidth=1.0)
    if p5 is not None:
        axes[i].axvline(p5, color="0.35", linestyle=":", linewidth=1.0)
    axes[i].text(0.04, 0.92, stat_label, transform=axes[i].transAxes,
                 fontsize=7, va="top", color="0.15",
                 bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.8", alpha=0.9))
    axes[i].set_xlabel(xlabel)
    axes[i].set_xlim(max(0, values.min() - 0.03), min(1, values.max() + 0.03))
    axes[i].set_ylim(0, 1.02)
    axes[i].grid(True, axis="y", alpha=0.25)
    axes[i].set_axisbelow(True)
axes[0].set_ylabel("Cumulative proportion of perturbations")
fig.suptitle("Stability of HAPI rankings under weight perturbations", y=1.02, fontsize=10)
fig.tight_layout()
for ext in ["png", "svg", "pdf", "tiff"]:
    fig.savefig(FIGDIR / f"fig6_hapi_sensitivity.{ext}", bbox_inches="tight")
print("saved fig6_hapi_sensitivity")
