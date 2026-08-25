"""Extract Landsat 8 30m NDVI for all HAPI-scored reservoirs (GEE).

Median NDVI 2015-2020 within 1 km buffer and 500 m buffer at 30 m scale.
Downloads per-reservoir CSV chunks to data/processed/landsat_ndvi/ then merges.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import ee

ee.Initialize(project="yueliang-475414")

ROOT = Path(__file__).resolve().parents[2]
HAPI = ROOT / "data/processed/hapi_scores.csv"
OUT_DIR = ROOT / "data/processed/landsat_ndvi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(HAPI)))
print("reservoirs:", len(rows))

ls = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterDate("2015-01-01", "2020-12-31")
    .map(lambda img: img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI"))
)
median = ls.median()

done = 0
chunks = []
for i, r in enumerate(rows):
    gid = r.get("GRAND_ID") or str(i)
    try:
        lat, lon = float(r["LAT"]), float(r["LON"])
    except Exception:
        continue
    pt = ee.Geometry.Point([lon, lat])
    out_path = OUT_DIR / ("ndvi_" + str(gid) + ".csv")
    try:
        for buf_m, tag in [(1000, "b1000"), (500, "b500")]:
            val = median.reduceRegion(ee.Reducer.median(), pt.buffer(buf_m), 30).get("NDVI").getInfo()
            if tag == "b1000":
                v1000 = val
            else:
                v500 = val
        with open(out_path, "w") as f:
            f.write("GRAND_ID,ndvi_30m_1km,ndvi_30m_500m" + chr(10))
            f.write(f"{gid},{v1000},{v500}" + chr(10))
        done += 1
    except Exception as e:
        chunks.append((gid, str(e)[:60]))
    if done % 50 == 0 and done > 0:
        print("progress:", done, "/", len(rows))

print("completed:", done)
if chunks:
    print("failures:", chunks[:10])
