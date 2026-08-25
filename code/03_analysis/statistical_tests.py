#!/usr/bin/env python3
"""Run primary non-parametric and binomial descriptive tests.

The tests compare US and Brazil w1_fdc_shape values and evaluate the share
of US reservoirs with negative Q10 alteration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, binomtest

ROOT = Path(__file__).resolve().parents[2]
US = ROOT / "data/processed/flow_regime_alteration.csv"
BR = ROOT / "data/processed/brazil_flow_regime_alteration.csv"
OUT = ROOT / "outputs/statistical_tests.json"

def main() -> None:
    us = pd.read_csv(US).dropna(subset=["w1_fdc_shape"])
    br = pd.read_csv(BR).dropna(subset=["w1_fdc_shape"])
    stat, p = mannwhitneyu(us["w1_fdc_shape"], br["w1_fdc_shape"], alternative="two-sided")

    q10 = pd.read_csv(US)["alter_q10_pct"].dropna()
    neg_count = int((q10 < 0).sum())
    neg_prop = neg_count / len(q10)
    # The manuscript states that all tests are two-sided.  This is a
    # descriptive departure from 50%, not a pre-registered directional test.
    bt = binomtest(neg_count, len(q10), p=0.5, alternative="two-sided")

    results = {
        "us_n": int(len(us)),
        "brazil_n": int(len(br)),
        "us_median": float(us["w1_fdc_shape"].median()),
        "brazil_median": float(br["w1_fdc_shape"].median()),
        "mannwhitney_u": float(stat),
        "p_value": float(p),
        "q10_negative_n": neg_count,
        "q10_negative_n_total": int(len(q10)),
        "q10_negative_prop": float(neg_prop),
        "q10_binomial_p": float(bt.pvalue),
    }
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
