#!/usr/bin/env python3
"""Run the MODIS NDWI remote-sensing subset analysis in Earth Engine.

NDWI is computed from MOD09GA green and near-infrared bands for the high- and
low-HAPI reservoir groups.
"""
from __future__ import annotations

import json
from pathlib import Path

import ee
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HAPI_CSV = ROOT / "data/processed/hapi_scores.csv"
OUT_CSV = ROOT / "data/processed/remote_sensing_ndwi.csv"
OUT_JSON = ROOT / "outputs/remote_sensing_ndwi.json"
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
        point = ee.Geometry.Point(float(r["LON"]), float(r["LAT"]))
        try:
            col = (
                ee.ImageCollection("MODIS/061/MOD09GA")
                .filterDate("2015-01-01", "2020-12-31")
                .filterBounds(point)
                .select(["sur_refl_b04", "sur_refl_b02"])
            )
            def ndwi_img(img):
                green = img.select("sur_refl_b04").multiply(0.0001)
                nir = img.select("sur_refl_b02").multiply(0.0001)
                return (green.subtract(nir)).divide(green.add(nir).add(1e-6)).rename("NDWI")

            val = col.map(ndwi_img).median().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point.buffer(1000),
                scale=500,
                maxPixels=1e9,
            ).get("NDWI").getInfo()
            rows.append({"GRAND_ID": r["GRAND_ID"], "group": r["group"], "HAPI": r["HAPI"], "ndwi": val})
            print(f"{r['GRAND_ID']} {r['group']} ndwi={val}", flush=True)
        except Exception as e:
            print(f"{r['GRAND_ID']} error {e}", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(OUT_CSV, index=False)
    if len(result) >= 2:
        high_ndwi = result.loc[result["group"] == "high", "ndwi"].dropna()
        low_ndwi = result.loc[result["group"] == "low", "ndwi"].dropna()
        summary = {
            "high_ndwi_mean": float(high_ndwi.mean()) if len(high_ndwi) else None,
            "low_ndwi_mean": float(low_ndwi.mean()) if len(low_ndwi) else None,
            "n_high": int(len(high_ndwi)),
            "n_low": int(len(low_ndwi)),
            "project": PROJECT,
        }
    else:
        summary = {"note": "not enough data", "project": PROJECT}
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
