#!/usr/bin/env python3
"""Assess FDC clustering robustness.

The script evaluates silhouette scores for k=2-8, compares KMeans with
agglomerative clustering at k=4, and computes ARI across 80% subsamples.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/processed/fdc_functional_features.csv"
OUT_JSON = ROOT / "outputs/fdc_cluster_robustness.json"
N_BOOT = 100
K_DEFAULT = 4


def preprocess() -> np.ndarray:
    df = pd.read_csv(FEAT)
    alter_cols = [c for c in df.columns if c.startswith("alter_")]
    X = df[alter_cols].to_numpy(dtype=float)
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = np.clip(X, -1.0, 3.0)
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=5, random_state=42).fit_transform(X)
    return X


def main() -> None:
    X = preprocess()
    res = {}

    # Compare silhouette scores across candidate k values.
    sil = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
        sil[k] = float(silhouette_score(X, km.labels_))
    res["silhouette_by_k"] = sil
    print("silhouette:", sil, flush=True)

    # KMeans vs Agglomerative at k=4
    km4 = KMeans(n_clusters=K_DEFAULT, random_state=42, n_init=20).fit(X)
    agg4 = AgglomerativeClustering(n_clusters=K_DEFAULT).fit(X)
    ari_km_agg = float(adjusted_rand_score(km4.labels_, agg4.labels_))
    res["ari_kmeans_agglomerative_k4"] = ari_km_agg
    print("ARI KMeans vs Agglomerative:", ari_km_agg, flush=True)

    # Compare each resampled partition with the full-sample reference labels.
    km_ref = KMeans(n_clusters=K_DEFAULT, random_state=42, n_init=20).fit(X)
    rng = np.random.default_rng(42)
    aris = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
        km_boot = KMeans(n_clusters=K_DEFAULT, random_state=42, n_init=10).fit(X[idx])
        aris.append(adjusted_rand_score(km_ref.labels_[idx], km_boot.labels_))
    res["bootstrap_ari_mean"] = float(np.mean(aris))
    res["bootstrap_ari_std"] = float(np.std(aris))
    res["bootstrap_ari_min"] = float(np.min(aris))
    print("bootstrap ARI (vs full-sample reference):", res["bootstrap_ari_mean"], res["bootstrap_ari_std"], flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
