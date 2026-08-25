#!/usr/bin/env python3
"""Build the graphical abstract from four data-derived result panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "manuscript/figures"
OUTDIR = ROOT / "outputs/submission"
OUTDIR.mkdir(parents=True, exist_ok=True)

panels = [
    (FIGDIR / "fig1_fdc_shape_distribution.png", "FDC alteration"),
    (FIGDIR / "fig3_fdc_clusters.png", "Functional partitions"),
    (FIGDIR / "fig5_hapi_map.png", "HAPI priority"),
    (FIGDIR / "fig6_hapi_sensitivity.png", "Ranking sensitivity"),
]

fig, axes = plt.subplots(1, 4, figsize=(13.28, 5.31))
for ax, (path, title) in zip(axes.ravel(), panels):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
fig.suptitle("Large-sample FDC alteration screening with HAPI", fontsize=14)
fig.text(0.5, 0.015,
         "HAPI screens reservoirs with strong flow alteration and regulation intensity for follow-up assessment.",
         ha="center", va="bottom", fontsize=10)
fig.tight_layout(rect=[0, 0.07, 1, 0.92])
for ext in ["png", "pdf"]:
    fig.savefig(OUTDIR / f"graphical_abstract.{ext}", dpi=300)
print("saved graphical abstract")
