"""GAGES-II monitoring-context proxy analysis.

Matches the 430 HAPI-scored reservoirs to USGS GAGES-II reference-quality
gaging stations and quantifies the association between HAPI and the local
fraction of reference-quality stations. The station class is a monitoring
context proxy, not a biological response measurement.

Outputs:
  data/processed/gages2_validation.csv
  outputs/gages2_validation_summary.json
  manuscript/figures/figS5_hapi_vs_reference_gage_fraction.png (+svg/pdf/tiff)
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dbfread import DBF
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "code/04_figures"))
from nf_style import apply_publication_style

apply_publication_style(font_size=8, axes_linewidth=1.0)

PROBE = ROOT / "data/enhancement_probe"
HAPI_CSV = ROOT / "data/processed/hapi_scores.csv"
OUT_CSV = ROOT / "data/processed/gages2_validation.csv"
OUT_JSON = ROOT / "outputs/gages2_validation_summary.json"
FIGDIR = ROOT / "manuscript/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
NL = chr(10)

RADIUS_KM = 25.0


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def main():
    # load GAGES-II stations
    dbf = DBF(str(PROBE / "gagesII_9322_sept30_2011.dbf"), encoding="latin-1")
    gages = []
    for rec in dbf:
        try:
            gages.append({
                "sta_id": rec["STAID"],
                "lat": float(rec["LAT_GAGE"]),
                "lon": float(rec["LNG_GAGE"]),
                "class": rec["CLASS"],
                "aggeco": rec["AGGECOREGI"],
                "drain": rec["DRAIN_SQKM"],
            })
        except Exception:
            pass
    print("GAGES stations loaded:", len(gages))

    hapi = pd.read_csv(HAPI_CSV)
    print("HAPI reservoirs:", len(hapi))

    rows = []
    for _, r in hapi.iterrows():
        try:
            lat, lon = float(r["LAT"]), float(r["LON"])
        except Exception:
            continue
        near = []
        for g in gages:
            if haversine(lat, lon, g["lat"], g["lon"]) <= RADIUS_KM:
                near.append(g)
        n_ref = sum(1 for g in near if g["class"] == "Ref")
        n_non = sum(1 for g in near if g["class"] != "Ref")
        rows.append({
            "GRAND_ID": r.get("GRAND_ID") or r.get("gsim_id"),
            "HAPI": r["HAPI"],
            "w1_fdc_shape": r.get("w1_fdc_shape"),
            "DOR_PC": r.get("DOR_PC"),
            "n_gages_25km": len(near),
            "n_ref_gages_25km": n_ref,
            "pct_ref_gages": (n_ref / len(near)) if near else np.nan,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("validation rows:", len(df))

    # ---- stats ----
    sub = df.dropna(subset=["pct_ref_gages"])
    print("reservoirs with >=1 gage in 25km:", len(sub))

    # high vs low HAPI (median split)
    med = sub["HAPI"].median()
    hi = sub[sub["HAPI"] >= med]
    lo = sub[sub["HAPI"] < med]
    u, p = stats.mannwhitneyu(hi["pct_ref_gages"], lo["pct_ref_gages"], alternative="two-sided")
    rho, rp = stats.spearmanr(sub["HAPI"], sub["pct_ref_gages"])
    pear_r, pear_p = stats.pearsonr(sub["HAPI"], sub["pct_ref_gages"])

    hi_mean = hi["pct_ref_gages"].mean()
    lo_mean = lo["pct_ref_gages"].mean()

    # also fraction of reservoirs with any ref gage
    any_ref_hi = (hi["n_ref_gages_25km"] > 0).mean()
    any_ref_lo = (lo["n_ref_gages_25km"] > 0).mean()

    summary = {
        "reservoirs_matched": int(len(sub)),
        "reservoirs_with_ref_gage": int((df["n_ref_gages_25km"] > 0).sum()),
        "radius_km": RADIUS_KM,
        "med_split_threshold": float(med),
        "hi_hapi_n": int(len(hi)),
        "lo_hapi_n": int(len(lo)),
        "hi_hapi_mean_pct_ref": round(float(hi_mean), 4),
        "lo_hapi_mean_pct_ref": round(float(lo_mean), 4),
        "mannwhitney_u": float(u),
        "mannwhitney_p": float(p),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(rp),
        "pearson_r": round(float(pear_r), 4),
        "pearson_p": float(pear_p),
        "hi_hapi_any_ref_fraction": round(float(any_ref_hi), 4),
        "lo_hapi_any_ref_fraction": round(float(any_ref_lo), 4),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax = axes[0]
    ax.scatter(sub["HAPI"], sub["pct_ref_gages"], s=12, alpha=0.6, color="#0F4D92")
    z = np.polyfit(sub["HAPI"], sub["pct_ref_gages"], 1)
    xs = np.linspace(sub["HAPI"].min(), sub["HAPI"].max(), 100)
    ax.plot(xs, np.polyval(z, xs), "k--", lw=0.9, alpha=0.8)
    lab = "Spearman rho = {0:.3f}{1}p = {2:.4f}".format(rho, NL, rp)
    ax.text(0.03, 0.97, lab, transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    ax.set_xlabel("HAPI")
    ax.set_ylabel("Fraction of reference-quality gages within 25 km")
    ax.set_title("(a) HAPI vs reference-gage fraction")

    ax2 = axes[1]
    labels = ["High HAPI", "Low HAPI"]
    vals = [hi["pct_ref_gages"].dropna(), lo["pct_ref_gages"].dropna()]
    parts = ax2.violinplot(vals, showmeans=True, showmedians=True)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(labels)
    lab2 = "Mann-Whitney U{0}p = {1:.4f}".format(NL, p)
    ax2.text(0.05, 0.97, lab2, transform=ax2.transAxes, va="top", fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    ax2.set_ylabel("Fraction of reference-quality gages within 25 km")
    ax2.set_title("(b) High vs low HAPI groups")

    fig.tight_layout()
    for ext in ["png", "svg", "pdf", "tiff"]:
        fig.savefig(FIGDIR / ("figS5_hapi_vs_reference_gage_fraction." + ext), bbox_inches="tight", dpi=300)
    print("saved figS5_hapi_vs_reference_gage_fraction")


if __name__ == "__main__":
    main()
