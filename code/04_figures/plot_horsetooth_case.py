#!/usr/bin/env python3
"""Reproduce the supplementary Horsetooth hydrological scenario figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/ResOpsUS+CARS_v10/v1.0/time_series/csv/468.csv"
FIGDIR = ROOT / "manuscript/figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
BENCHMARK = 0.3


def main() -> None:
    df = pd.read_csv(RAW, parse_dates=["date"])
    df = df.dropna(subset=["inflow", "outflow"])
    df = df[(df["inflow"] >= 0) & (df["outflow"] >= 0)].copy()
    df["month"] = df["date"].dt.month
    winter = df[df["month"].isin([11, 12, 1, 2, 3])]
    summer = df[df["month"].isin([6, 7, 8, 9])]
    deficit_mcm = ((BENCHMARK - winter["outflow"]).clip(lower=0).sum() * 86400 / 1e6)
    print({"paired_days": len(df), "winter_days": len(winter),
           "summer_days": len(summer), "winter_deficit_mcm": round(deficit_mcm, 3)})

    monthly = df.groupby("month")[["inflow", "outflow"]].median()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    ax = axes[0]
    ax.plot(monthly.index, monthly["inflow"], marker="o", label="Inflow")
    ax.plot(monthly.index, monthly["outflow"], marker="o", label="Outflow")
    ax.axvspan(1, 3.99, color="#bdbdbd", alpha=0.18)
    ax.axvspan(11, 12.99, color="#bdbdbd", alpha=0.18)
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlabel("Month")
    ax.set_ylabel("Daily flow (m3 s-1)")
    ax.set_title("(a) Monthly median flow")
    ax.legend(frameon=False, fontsize=7)

    ax2 = axes[1]
    ax2.boxplot([winter["outflow"], summer["outflow"]], tick_labels=["Winter\nNov-Mar", "Summer\nJun-Sep"],
                showfliers=False)
    ax2.axhline(BENCHMARK, color="#b22222", linestyle="--", linewidth=1,
                label="0.3 m3 s-1 scenario")
    ax2.set_ylabel("Outflow (m3 s-1)")
    ax2.set_title("(b) Seasonal outflow")
    ax2.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf", "tiff"]:
        fig.savefig(FIGDIR / f"figS3_horsetooth_case.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
