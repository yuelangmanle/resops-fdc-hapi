"""Figure S2: data processing workflow diagram."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from nf_style import apply_publication_style

apply_publication_style(font_size=8, axes_linewidth=1.0)

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "manuscript" / "figures" / "supplementary"
FIGDIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.4)


def box(x, y, w, h, text, fc="#EAF2FB", ec="#0F4D92", fs=7.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                       facecolor=fc, edgecolor=ec, lw=1.0)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, wrap=True)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#4D4D4D", lw=1.2))


# Row 1: raw data
box(0.2, 3.1, 2.6, 0.9, "ResOpsUS+CARS\n(Zenodo, 676 US reservoirs,\ndaily 1975-2020)", fc="#FDF2E9", ec="#B9770E")
box(3.4, 3.1, 2.4, 0.9, "ResOpsBR+CARS\n(Zenodo, 142 BR reservoirs)", fc="#FDF2E9", ec="#B9770E")
box(6.4, 3.1, 2.4, 0.9, "MODIS NDVI/NDWI\n(GEE, 2015-2020)", fc="#FDF2E9", ec="#B9770E")
box(0.2, 2.0, 2.6, 0.8, "QA + screening\n(>=10 yr overlap)", fc="#EAF2FB")
box(3.4, 2.0, 2.4, 0.8, "QA + screening\n(>=10 yr overlap)", fc="#EAF2FB")

arrow(1.5, 3.1, 1.5, 2.8)
arrow(4.6, 3.1, 4.6, 2.8)

# Row 2: core analysis
box(0.2, 0.9, 4.6, 0.8, "FDC alteration metrics + 99-quantile functional features\n+ entropy metrics (433 US / 134 BR)", fc="#E8F8F5", ec="#117864")
box(5.4, 0.9, 4.4, 0.8, "PCA + KMeans clustering (k=4)\n+ robustness (silhouette, ARI, bootstrap)", fc="#E8F8F5", ec="#117864")

arrow(1.5, 2.0, 1.5, 1.7)
arrow(4.6, 2.0, 4.6, 1.7)

# Row 3: downstream
box(0.2, -0.1, 3.0, 0.8, "Heterogeneity analysis\n(causal forest + diagnostics)", fc="#F9EBEA", ec="#B64342")
box(3.6, -0.1, 3.0, 0.8, "HAPI + sensitivity\n(1000 weight vectors)", fc="#F9EBEA", ec="#B64342")
box(7.0, -0.1, 2.8, 0.8, "Validations\n(GloFAS bias, USGS pilot,\nremote sensing, Brazil)", fc="#F9EBEA", ec="#B64342")

arrow(2.5, 0.9, 1.7, 0.7)
arrow(4.6, 0.9, 5.1, 0.7)
arrow(7.6, 0.9, 8.4, 0.7)

fig.tight_layout()
for ext in ["png", "svg", "pdf", "tiff"]:
    fig.savefig(FIGDIR / ("figS2_data_workflow." + ext), bbox_inches="tight", dpi=300)
print("saved figS2_data_workflow")
