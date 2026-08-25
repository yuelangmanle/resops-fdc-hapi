#!/usr/bin/env python3
"""Extract 2015-2020 MODIS NDVI medians for ten high- and ten low-HAPI sites."""
from __future__ import annotations

import json
from pathlib import Path

import ee
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HAPI_CSV = ROOT / "data/processed/hapi_scores.csv"
OUT_CSV = ROOT / "data/processed/remote_sensing_ndvi.csv"
OUT_JSON = ROOT / "outputs/remote_sensing_validation.json"
PROJECT = "yueliang-475414"


def main() -> None:
    ee.Initialize(credentials="persistent", project=PROJECT)

    df = pd.read_csv(HAPI_CSV)
    high = df.nlargest(10, "HAPI")
    low = df.nsmallest(10, "HAPI")
    sel = pd.concat([high, low]).copy()
    sel["group"] = ["high"] * len(high) + ["low"] * len(low)

    rows = []
    for _, r in sel.iterrows():
        gid = r["GRAND_ID"]
        lat = float(r["LAT"])
        lon = float(r["LON"])
        point = ee.Geometry.Point(lon, lat)
        try:
            ndvi_col = (
                ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterDate("2015-01-01", "2020-12-31")
                .filterBounds(point)
                .select("NDVI")
            )
            # Extract the median within the reservoir and 1-km buffer.
            val = ndvi_col.median().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point.buffer(1000),
                scale=250,
                maxPixels=1e9,
            ).get("NDVI").getInfo()
            rows.append({"GRAND_ID": gid, "group": r["group"], "HAPI": r["HAPI"], "ndvi_median": val})
            print(f"{gid} {r['group']} ndvi={val}", flush=True)
        except Exception as e:
            print(f"{gid} error {e}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    if len(result) >= 2:
        high_ndvi = result.loc[result["group"] == "high", "ndvi_median"].dropna()
        low_ndvi = result.loc[result["group"] == "low", "ndvi_median"].dropna()
        summary = {
            "n_high": int(len(high_ndvi)),
            "n_low": int(len(low_ndvi)),
            "high_ndvi_mean": float(high_ndvi.mean()) if len(high_ndvi) else None,
            "low_ndvi_mean": float(low_ndvi.mean()) if len(low_ndvi) else None,
            "project": PROJECT,
        }
    else:
        summary = {"note": "not enough data", "project": PROJECT}
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
