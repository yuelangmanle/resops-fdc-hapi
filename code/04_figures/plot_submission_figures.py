"""Generate the submission-ready composite figures from processed data.

Fig 4 (DOR vs FDC): hexbin density + log-x + Spearman stats box + cluster-colored overlay
Fig 5 (HAPI map):   top-10 reservoirs starred + labeled; point size ~ log capacity
Supplementary Fig S4 (HAPI vs NDVI): cluster-colored scatter + stats box + fit line
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from nf_style import apply_publication_style

apply_publication_style(font_size=8, axes_linewidth=1.0)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
FIGDIR = ROOT / "manuscript" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
NL = chr(10)

CLUSTER_COLORS = ["#0F4D92", "#D95F02", "#1B9E77", "#7570B3"]

# ---------------- Fig 4 ----------------
adf = pd.read_csv(PROC / "analysis_dataset.csv")
_cl = pd.read_csv(PROC / "fdc_functional_clusters.csv")[["GRAND_ID", "cluster"]]
adf = adf.merge(_cl, on="GRAND_ID", how="left")
d = adf.dropna(subset=["w1_fdc_shape", "DOR_PC"]).copy()
rho, pval = stats.spearmanr(d["DOR_PC"], d["w1_fdc_shape"])

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]
# Bin in log10(DOR) space before plotting; binning raw, right-skewed DOR
# values and applying a log axis afterwards creates artificial horizontal bands.
d = d[d["DOR_PC"] > 0].copy()
d["log10_DOR"] = np.log10(d["DOR_PC"])
hb = ax.hexbin(d["log10_DOR"], d["w1_fdc_shape"], gridsize=28, bins="log", cmap="viridis", mincnt=1)
ax.set_xlabel("Degree of regulation (capacity / mean annual inflow)")
ax.set_ylabel("FDC shape alteration (normalized Wasserstein)")
ax.set_xticks([np.log10(0.5), 0, 1, 2, 3, 4])
ax.set_xticklabels(["0.5", "1", "10", "100", "1000", "10000"])
cb = fig.colorbar(hb, ax=ax, pad=0.02)
cb.set_label("Reservoirs (log count)")
lab = "Spearman rho = {0:.3f}{1}p < 0.001".format(rho, NL)
ax.text(0.04, 0.95, lab, transform=ax.transAxes, va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
ax.text(0.01, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

ax2 = axes[1]
for cid, grp in d.groupby("cluster"):
    ax2.scatter(grp["DOR_PC"], grp["w1_fdc_shape"], s=7, alpha=0.55,
                color=CLUSTER_COLORS[int(cid)], label="Cluster " + str(cid))
ax2.set_xscale("log")
ax2.set_xlabel("Degree of regulation")
ax2.set_ylabel("FDC shape alteration")
ax2.legend(fontsize=6, loc="lower right", frameon=False)
ax2.text(0.01, 1.02, "(b)", transform=ax2.transAxes, fontweight="bold")

fig.tight_layout()
for ext in ["png", "svg", "pdf", "tiff"]:
    fig.savefig(FIGDIR / ("fig4_dor_vs_fdc." + ext), bbox_inches="tight", dpi=300)
print("saved fig4_dor_vs_fdc")
plt.close(fig)

# ---------------- Fig 5 ----------------
hapi = pd.read_csv(PROC / "hapi_scores.csv")
hapi = hapi.dropna(subset=["LAT", "LON"]).copy()
top10 = hapi.sort_values("HAPI", ascending=False).head(10).copy()
cap = hapi["CAP_MCM"].clip(lower=1).astype(float)
lo, hi = np.log10(cap).min(), np.log10(cap).max()
sizes = 8 + 22 * (np.log10(cap) - lo) / (hi - lo + 1e-9)

fig, ax = plt.subplots(figsize=(8, 4.6))
sc = ax.scatter(hapi["LON"], hapi["LAT"], c=hapi["HAPI"], cmap="YlOrRd",
                s=sizes, alpha=0.75)
label_offsets = {
    931: (12, 18), 929: (12, -18), 1788: (10, 7), 468: (-46, 10),
    572: (10, 7), 956: (12, 4), 1073: (10, 8), 111: (10, 7),
    508: (10, -12), 519: (-46, -12),
}
focus_ids = {931, 956, 929}
for _, r in top10.iterrows():
    raw_name = r.get("RES_NAME")
    if pd.isna(raw_name) or not str(raw_name).strip():
        name = "GRanD #" + str(int(r["GRAND_ID"]))
    else:
        name = str(raw_name).strip()
    ax.scatter(r["LON"], r["LAT"], marker="*", s=95, color="black", zorder=5,
               edgecolors="white", linewidths=0.4)
    if int(r["GRAND_ID"]) not in focus_ids:
        ax.annotate(name, (r["LON"], r["LAT"]),
                    xytext=label_offsets.get(int(r["GRAND_ID"]), (8, 5)),
                    textcoords="offset points", fontsize=6.0, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                              edgecolor="none", alpha=0.82),
                    arrowprops=dict(arrowstyle="-", color="0.35", lw=0.55,
                                    shrinkA=2.5, shrinkB=3.5,
                                    connectionstyle="arc3,rad=0.0"))

# The three top-ranked sites in the central Plains are geographically close.
# Draw a linked zoom panel so the two nearly coincident GRanD sites are not
# visually merged into one star at the national-map scale.
focus_bounds = (-100.25, 39.25, 1.55, 1.65)
ax.add_patch(Rectangle((focus_bounds[0], focus_bounds[1]), focus_bounds[2], focus_bounds[3],
                       fill=False, linestyle=(0, (3, 2)), linewidth=0.8,
                       edgecolor="0.25", zorder=7))
axins = ax.inset_axes([0.58, 0.57, 0.31, 0.33], zorder=10)
axins.set_facecolor("white")
axins.patch.set_alpha(0.97)
local = hapi[(hapi["LON"] >= -100.25) & (hapi["LON"] <= -98.70) &
             (hapi["LAT"] >= 39.20) & (hapi["LAT"] <= 40.95)]
axins.scatter(local["LON"], local["LAT"], c=local["HAPI"], cmap="YlOrRd",
              norm=sc.norm, s=18, alpha=0.55, linewidths=0)
focus_colors = {931: "#2F6FAE", 956: "#238B45", 929: "#C23B3B"}
focus_text_offsets = {931: (-38, 22), 956: (13, 5), 929: (12, -22)}
focus_markers = {931: "s", 956: "*", 929: "D"}
for _, r in top10[top10["GRAND_ID"].isin(focus_ids)].iterrows():
    gid = int(r["GRAND_ID"])
    raw_name = r.get("RES_NAME")
    name = (str(raw_name).strip() if pd.notna(raw_name) and str(raw_name).strip()
            else "GRanD #" + str(gid))
    color = focus_colors[gid]
    axins.scatter(r["LON"], r["LAT"], marker="o", s=170, facecolors="none",
                  edgecolors=color, linewidths=1.5, zorder=8)
    axins.scatter(r["LON"], r["LAT"], marker=focus_markers[gid], s=105, color=color,
                  edgecolors="black", linewidths=0.55, zorder=9)
    axins.annotate(name, (r["LON"], r["LAT"]),
                   xytext=focus_text_offsets[gid], textcoords="offset points",
                   fontsize=5.6, zorder=10,
                   bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                             edgecolor=color, linewidth=0.6, alpha=0.95),
                   arrowprops=dict(arrowstyle="-", color=color, lw=0.7,
                                   shrinkA=2.0, shrinkB=3.0))
axins.set_xlim(-100.25, -98.70)
axins.set_ylim(39.20, 40.95)
axins.set_title("Nearby top-10 sites", fontsize=5.6, pad=1)
axins.text(0.02, 0.04, "GRanD #931 / #929: 4.9 km apart",
           transform=axins.transAxes, fontsize=4.8, color="0.2",
           bbox=dict(boxstyle="round,pad=0.16", facecolor="white",
                     edgecolor="0.65", alpha=0.9))
axins.tick_params(labelsize=4.8, length=2)
axins.grid(True, alpha=0.2, linewidth=0.45)
mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.35", lw=0.55,
           linestyle=(0, (3, 2)))
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("HAPI (Hydrological Alteration Prioritization Index)")
ax.set_title("HAPI spatial pattern; stars indicate the top-10 reservoirs", fontsize=9)
fig.tight_layout()
for ext in ["png", "svg", "pdf", "tiff"]:
    fig.savefig(FIGDIR / ("fig5_hapi_map." + ext), bbox_inches="tight", dpi=300)
print("saved fig5_hapi_map")
plt.close(fig)

# ---------------- Fig 8 ----------------
rs = pd.read_csv(PROC / "remote_sensing_ndvi_ndwi_all.csv")
cl = pd.read_csv(PROC / "fdc_functional_clusters.csv")[["GRAND_ID", "cluster"]]
m = rs.merge(cl, on="GRAND_ID", how="left")
m["ndvi_frac"] = m["ndvi"] / 10000.0

pr, pp = stats.pearsonr(m["HAPI"], m["ndvi_frac"])
sr, sp = stats.spearmanr(m["HAPI"], m["ndvi_frac"])
slope, intercept, _, _, _ = stats.linregress(m["HAPI"], m["ndvi_frac"])
x_fit = np.linspace(m["HAPI"].min(), m["HAPI"].max(), 100)

fig, ax = plt.subplots(figsize=(5.2, 4.0))
for cid, grp in m.groupby("cluster"):
    ax.scatter(grp["HAPI"], grp["ndvi_frac"], s=9, alpha=0.55,
               color=CLUSTER_COLORS[int(cid)], label="Cluster " + str(cid))
ax.plot(x_fit, slope * x_fit + intercept, "k--", lw=0.9, alpha=0.75)
ax.set_xlabel("Hydrological Alteration Prioritization Index (HAPI)")
ax.set_ylabel("MODIS NDVI (2015-2020 median)")
lab = "Pearson r = {0:.3f} (p < 0.001){1}Spearman rho = {2:.3f} (p < 0.001)".format(pr, NL, sr)
ax.text(0.03, 0.97, lab, transform=ax.transAxes, va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
ax.legend(fontsize=6, loc="lower left", frameon=False)
ax.set_title("HAPI vs NDVI (n = 430)", fontsize=9)
fig.tight_layout()
for ext in ["png", "svg", "pdf", "tiff"]:
    fig.savefig(FIGDIR / ("figS4_hapi_ndvi_scatter." + ext), bbox_inches="tight", dpi=300)
print("saved figS4_hapi_ndvi_scatter")
plt.close(fig)

print("all submission figures saved")
