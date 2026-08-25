#!/usr/bin/env python3
"""Run the Colorado GloFAS-USGS pilot validation.

The script matches Colorado reservoirs to nearby USGS stations with similar
basin area, downloads 2015-2020 daily flows, and compares specific discharge
with reservoir inflow_sim.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data/processed/analysis_dataset.csv"
OUT_JSON = ROOT / "outputs/glofas_usgs_pilot.json"
STATE = "co"
SITE_URL = (
    "https://waterservices.usgs.gov/nwis/site/"
    f"?format=rdb&siteOutput=expanded&stateCd={STATE}&siteType=ST&parameterCd=00060"
)
DV_URL = (
    "https://waterservices.usgs.gov/nwis/dv/"
    "?format=json&sites={site}&parameterCd=00060&startDT=2015-01-01&endDT=2020-12-31"
)
SQKM_PER_SQMI = 2.58998811
# NWIS parameter 00060 returns ft3/s by default; ResOpsUS+CARS inflow_sim is m3/s.
CFS_TO_CMS = 0.028316846592
MAX_DIST_KM = 100.0


def curl_text(url: str) -> str:
    proc = subprocess.run(["curl", "--http1.1", "-m", "30", "-sS", url], capture_output=True, text=True, timeout=35)
    return proc.stdout


REGULATED_KEYWORDS = [
    "below", "reservoir", "lake", "dam", "diversion", "canal",
    "irrigation", "powerplant", "power plant", "tunnel", "afterbay",
]


def is_reference_like(name: str) -> bool:
    low = (name or "").lower()
    return not any(k in low for k in REGULATED_KEYWORDS)


def fetch_site_list() -> pd.DataFrame:
    text = curl_text(SITE_URL)
    lines = [l for l in text.splitlines() if not l.startswith("#") and l.strip()]
    if len(lines) < 2:
        return pd.DataFrame()
    header = lines[0].split("\t")
    data = [l.split("\t") for l in lines[2:]]
    df = pd.DataFrame(data, columns=header)
    for col in ["dec_lat_va", "dec_long_va", "drain_area_va"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "station_nm" in df.columns:
        df = df[df["station_nm"].apply(is_reference_like)]
    return df


def fetch_daily(site: str) -> pd.DataFrame:
    text = curl_text(DV_URL.format(site=site))
    try:
        data = json.loads(text)
    except Exception:
        return pd.DataFrame()
    try:
        ts = data["value"]["timeSeries"][0]
        values = ts["values"][0]["value"]
        rows = [{"date": v["dateTime"][:10], "flow": float(v["value"]) * CFS_TO_CMS} for v in values]
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    analysis = pd.read_csv(ANALYSIS)
    res = analysis[analysis["STATE"] == "Colorado"].copy()
    sites = fetch_site_list()
    print(f"USGS sites in CO: {len(sites)}", flush=True)

    pairs = []
    catch_map = dict(zip(res["GRAND_ID"], res["CATCH_SKM"]))
    for _, r in res.iterrows():
        lat, lon = r["LAT"], r["LON"]
        catch = r["CATCH_SKM"]
        best = None
        best_dist = MAX_DIST_KM
        for _, s in sites.iterrows():
            if pd.isna(s.get("dec_lat_va")) or pd.isna(s.get("dec_long_va")) or pd.isna(s.get("drain_area_va")):
                continue
            d = math.hypot((s["dec_lat_va"] - lat) * 111.0, (s["dec_long_va"] - lon) * 97.0)
            area_sqkm = s["drain_area_va"] * SQKM_PER_SQMI
            if d < best_dist and 0.5 * catch <= area_sqkm <= 2.0 * catch:
                best_dist = d
                best = s
        if best is not None:
            pairs.append((r["GRAND_ID"], best["site_no"], best_dist))

    print(f"matched pairs: {len(pairs)}", flush=True)
    results = []
    for gid, site, dist in pairs[:8]:
        daily = fetch_daily(site)
        if daily.empty:
            continue
        daily["date"] = pd.to_datetime(daily["date"])
        # reservoir inflow_sim
        csv_path = ROOT / "data/raw/ResOpsUS+CARS_v10/v1.0/time_series/csv" / f"{gid}.csv"
        try:
            rdf = pd.read_csv(csv_path, usecols=["date", "inflow_sim"], parse_dates=["date"])
        except Exception:
            continue
        rdf = rdf[(rdf["date"] >= "2015-01-01") & (rdf["date"] <= "2020-12-31")]
        rdf = rdf.dropna()
        if rdf.empty or daily.empty:
            continue
        # gage drainage area
        site_info = sites[sites["site_no"] == site].iloc[0]
        gage_area = site_info["drain_area_va"] * SQKM_PER_SQMI
        res_area = catch_map[gid]
        merged = daily.merge(rdf[["date", "inflow_sim"]], on="date", how="inner")
        if len(merged) < 100:
            continue
        observed_specific = merged["flow"] / gage_area
        glofas_specific = merged["inflow_sim"] / res_area
        bias = (glofas_specific - observed_specific) / observed_specific
        results.append({
            "GRAND_ID": gid,
            "USGS_site": site,
            "distance_km": dist,
            "n_days": int(len(merged)),
            "median_bias_ratio": float(bias.median()),
            "median_specific_discharge_ratio": float((1.0 + bias).median()),
            "mean_bias_ratio": float(bias.mean()),
            "glofas_specific_median": float(glofas_specific.median()),
            "observed_specific_median": float(observed_specific.median()),
        })
        print(f"Pair {gid} {site}: median bias {bias.median():.2f}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", len(results), "pairs", flush=True)


if __name__ == "__main__":
    main()
