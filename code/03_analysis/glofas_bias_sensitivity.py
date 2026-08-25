#!/usr/bin/env python3
"""Run bounded sensitivity scenarios for GloFAS low-flow bias.

The analysis perturbs only simulated-inflow Q10 values by -30% to +30% and
recomputes the full-sample Q10 alteration. Observed-inflow values remain at
their measured reference values. The scenarios are not a calibrated error
model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/flow_regime_alteration.csv"
OUT_JSON = ROOT / "outputs/glofas_bias_sensitivity.json"
BIASES = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]


def main() -> None:
    df = pd.read_csv(DATA).dropna(subset=["in_q10", "out_q10"])
    df = df[(df["in_q10"] > 0) & (df["out_q10"] >= 0)].copy()
    out = df["out_q10"].to_numpy(dtype=float)
    inflow = df["in_q10"].to_numpy(dtype=float)
    simulated = df["ref_type"].eq("simulated").to_numpy()
    observed = ~simulated
    baseline_alter = df["alter_q10_pct"].to_numpy(dtype=float)

    results = {}
    for bias in BIASES:
        # Positive bias means simulated inflow is above the assumed reference.
        # Keep observed-inflow records unchanged and adjust only the simulated
        # subset before recomputing the full-sample summary.
        true_in = inflow / (1.0 + bias)
        alter = baseline_alter.copy()
        valid_sim = simulated & (true_in > 0)
        alter[valid_sim] = (out[valid_sim] - true_in[valid_sim]) / true_in[valid_sim] * 100.0
        valid = observed | valid_sim
        results[bias] = {
            "n": int(valid.sum()),
            "n_observed": int(observed.sum()),
            "n_simulated": int(valid_sim.sum()),
            "median_alter_q10": float(np.median(alter)) if len(alter) else None,
            "mean_alter_q10": float(np.mean(alter)) if len(alter) else None,
            "negative_ratio": float((alter < 0).mean()) if len(alter) else None,
        }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
