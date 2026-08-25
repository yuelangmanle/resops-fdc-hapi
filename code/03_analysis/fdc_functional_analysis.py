#!/usr/bin/env python3
"""Run PCA and functional clustering on FDC alteration profiles.

Input: data/processed/fdc_functional_features.csv
Outputs: fdc_functional_clusters.csv and fdc_pca_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/processed/fdc_functional_features.csv"
OUT_CSV = ROOT / "data/processed/fdc_functional_clusters.csv"
OUT_JSON = ROOT / "outputs/fdc_pca_summary.json"

N_COMPONENTS = 5
N_CLUSTERS = 4


def main() -> None:
    df = pd.read_csv(FEAT)
    print("shape:", df.shape, flush=True)
    alter_cols = [c for c in df.columns if c.startswith("alter_")]
    X = df[alter_cols].to_numpy(dtype=float)

    # Median-impute missing values and clip extreme relative changes.
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)
    X = np.clip(X, -1.0, 3.0)

    # Standardize before PCA.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    scores = pca.fit_transform(Xs)

    # Cluster the reduced feature matrix.
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    labels = km.fit_predict(scores[:, :N_COMPONENTS])
    df["cluster"] = labels

    df[["GRAND_ID", "cluster"] + alter_cols].to_csv(OUT_CSV, index=False)

    summary = {
        "n_reservoirs": int(len(df)),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "n_clusters": N_CLUSTERS,
        "cluster_counts": {int(k): int(v) for k, v in df["cluster"].value_counts().sort_index().items()},
        "cluster_means": df.groupby("cluster")[alter_cols].mean().to_dict(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", OUT_CSV, flush=True)
    print("saved", OUT_JSON, flush=True)
    print("explained:", summary["explained_variance_ratio"], flush=True)
    print("cluster counts:", summary["cluster_counts"], flush=True)


if __name__ == "__main__":
    main()
