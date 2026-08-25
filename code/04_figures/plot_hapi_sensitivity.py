#!/usr/bin/env python3
"""Plot HAPI ranking sensitivity distributions."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from nf_style import apply_publication_style

apply_publication_style(font_size=8, axes_linewidth=1.0)

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "manuscript/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
data = json.loads((ROOT / "outputs/hapi_sensitivity_results.json").read_text())

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
axes[0].hist(data["spearman_list"], bins=30, color="steelblue", edgecolor="white")
axes[0].set_xlabel("Spearman correlation with base ranking")
axes[0].set_ylabel("Frequency")
axes[1].hist(data["top10_overlap_list"], bins=30, color="darkorange", edgecolor="white")
axes[1].set_xlabel("Top-10 overlap with base ranking")
axes[1].set_ylabel("Frequency")
fig.tight_layout()
for ext in ["png", "svg", "pdf", "tiff"]:
    fig.savefig(FIGDIR / f"fig6_hapi_sensitivity.{ext}", bbox_inches="tight")
print("saved fig6_hapi_sensitivity")
