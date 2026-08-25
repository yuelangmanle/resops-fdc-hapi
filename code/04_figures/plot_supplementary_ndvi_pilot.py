#!/usr/bin/env python3
"""Reproduce Supplementary Figure S1 from the 20-reservoir NDVI pilot table."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from nf_style import apply_publication_style, save_pub

apply_publication_style(font_size=8, axes_linewidth=1.0)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/remote_sensing_ndvi.csv"
OUTDIR = ROOT / "manuscript/figures/supplementary"


def main() -> None:
    df = pd.read_csv(DATA)
    required = {"group", "ndvi_median"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.loc[df["group"].isin(["high", "low"])].copy()
    if set(df["group"]) != {"high", "low"}:
        raise ValueError("The pilot table must contain both high and low HAPI groups")
    df["ndvi"] = df["ndvi_median"] / 10000.0
    order = ["high", "low"]
    groups = [df.loc[df["group"] == label, "ndvi"].to_numpy() for label in order]
    if any(len(values) != 10 for values in groups):
        raise ValueError(f"Expected 10 reservoirs per group, got {[len(v) for v in groups]}")

    u_stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
    medians = [float(np.median(values)) for values in groups]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bp = ax.boxplot(
        groups,
        positions=[1, 2],
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#272727", "linewidth": 1.2},
        whiskerprops={"color": "#4D4D4D", "linewidth": 0.9},
        capprops={"color": "#4D4D4D", "linewidth": 0.9},
        boxprops={"linewidth": 0.9},
    )
    for patch, color in zip(bp["boxes"], ["#D95F02", "#3775BA"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    rng = np.random.default_rng(42)
    for xpos, values, color in zip([1, 2], groups, ["#D95F02", "#3775BA"]):
        jitter = rng.uniform(-0.11, 0.11, size=len(values))
        ax.scatter(
            xpos + jitter,
            values,
            s=22,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.9,
            zorder=3,
        )

    ax.set_xticks([1, 2], ["High HAPI\n(n = 10)", "Low HAPI\n(n = 10)"])
    ax.set_ylabel("MODIS NDVI (median, 2015-2020)")
    ax.set_title("Pilot comparison of NDVI by HAPI group", fontsize=9)
    ax.text(
        0.03,
        0.97,
        "Median: {:.3f} vs {:.3f}\nMann-Whitney p = {:.3g}".format(
            medians[0], medians[1], p_value
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_pub(fig, "figS1_remote_ndvi_20reservoir", OUTDIR, dpi=600)
    plt.close(fig)
    print(
        "saved",
        OUTDIR / "figS1_remote_ndvi_20reservoir.png",
        "n=20",
        "U={:.1f}".format(u_stat),
        "p={:.3g}".format(p_value),
    )


if __name__ == "__main__":
    main()
