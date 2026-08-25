#!/usr/bin/env python3
"""Extract MODIS NDVI and NDWI for all HAPI-scored reservoirs."""
from __future__ import annotations

import json
from pathlib import Path

import ee
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
HAPI_CSV = ROOT / "data/processed/hapi_scores.csv"
OUT_CSV = ROOT / "data/processed/remote_sensing_ndvi_ndwi_all.csv"
OUT_JSON = ROOT / "outputs/remote_sensing_all_summary.json"
PROJECT = "yueliang-475414"


def ndwi_func(img):
    green = img.select("sur_refl_b04").multiply(0.0001)
    nir = img.select("sur_refl_b02").multiply(0.0001)
    return (green.subtract(nir)).divide(green.add(nir).add(1e-6)).rename("NDWI")


def main() -> None:
    ee.Initialize(credentials="persistent", project=PROJECT)
    # Set a request timeout.
    try:
        ee.data.setDeadline(30000)
    except Exception:
        pass
    df = pd.read_csv(HAPI_CSV)
    print(f"processing {len(df)} reservoirs", flush=True)

    rows = []
    for i, r in df.iterrows():
        point = ee.Geometry.Point(float(r["LON"]), float(r["LAT"]))
        try:
            ndvi_col = (
                ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterDate("2015-01-01", "2020-12-31")
                .filterBounds(point)
                .select("NDVI")
            )
            ndvi = ndvi_col.median().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point.buffer(1000),
                scale=250,
                maxPixels=1e9,
            ).get("NDVI").getInfo()

            ndwi_col = (
                ee.ImageCollection("MODIS/061/MOD09GA")
                .filterDate("2015-01-01", "2020-12-31")
                .filterBounds(point)
                .select(["sur_refl_b04", "sur_refl_b02"])
            )
            ndwi = ndwi_col.map(ndwi_func).median().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point.buffer(1000),
                scale=500,
                maxPixels=1e9,
            ).get("NDWI").getInfo()

            rows.append({
                "GRAND_ID": r["GRAND_ID"],
                "HAPI": r["HAPI"],
                "ndvi": ndvi,
                "ndwi": ndwi,
            })
        except Exception as e:
            print(f"{r['GRAND_ID']} error {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"processed {i+1}/{len(df)}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    print(f"saved {len(result)} rows", flush=True)

    valid = result.dropna(subset=["ndvi", "ndwi"])
    summary = {"n": len(valid)}
    if len(valid) > 10:
    # Convert scaled NDVI to the conventional range.
        ndvi_norm = valid["ndvi"] / 10000.0
        summary["ndvi_mean"] = float(ndvi_norm.mean())
        summary["ndwi_mean"] = float(valid["ndwi"].mean())
        pr, pp = pearsonr(valid["HAPI"], ndvi_norm)
        sr, sp = spearmanr(valid["HAPI"], ndvi_norm)
        summary["pearson_hapi_ndvi"] = {"r": float(pr), "p": float(pp)}
        summary["spearman_hapi_ndvi"] = {"rho": float(sr), "p": float(sp)}
        pr2, pp2 = pearsonr(valid["HAPI"], valid["ndwi"])
        sr2, sp2 = spearmanr(valid["HAPI"], valid["ndwi"])
        summary["pearson_hapi_ndwi"] = {"r": float(pr2), "p": float(pp2)}
        summary["spearman_hapi_ndwi"] = {"rho": float(sr2), "p": float(sp2)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
