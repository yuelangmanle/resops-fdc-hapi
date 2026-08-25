#!/usr/bin/env python3
"""Run the expanded multi-state GloFAS-USGS validation.

At most one matched pair is sampled per state, with a maximum of 20 pairs.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data/processed/analysis_dataset.csv"
OUT_JSON = ROOT / "outputs/glofas_usgs_allstates.json"

STATE_TO_CODE = {
    "Alabama": "al", "Arizona": "az", "Arkansas": "ar", "California": "ca",
    "Colorado": "co", "Georgia": "ga", "Kansas": "ks", "Kentucky": "ky",
    "Minnesota": "mn", "Missouri": "mo", "Montana": "mt", "Nebraska": "ne",
    "New Mexico": "nm", "North Carolina": "nc", "Ohio": "oh", "Oklahoma": "ok",
    "Oregon": "or", "South Dakota": "sd", "Tennessee": "tn", "Texas": "tx",
    "Utah": "ut", "Washington": "wa", "Wyoming": "wy",
}
MAX_TOTAL_PAIRS = 30
SQKM_PER_SQMI = 2.58998811
# NWIS parameter 00060 returns ft3/s by default; ResOpsUS+CARS inflow_sim is m3/s.
CFS_TO_CMS = 0.028316846592
MAX_DIST_KM = 150.0
AREA_MIN_RATIO = 0.25
AREA_MAX_RATIO = 4.0
REGULATED_KEYWORDS = [
    "below", "reservoir", "lake", "dam", "diversion", "canal",
    "irrigation", "powerplant", "power plant", "tunnel", "afterbay",
]


def is_reference_like(name: str) -> bool:
    low = (name or "").lower()
    return not any(k in low for k in REGULATED_KEYWORDS)


def curl_text(url: str) -> str:
    proc = subprocess.run(["curl", "--http1.1", "-m", "30", "-sS", url], capture_output=True, text=True, timeout=35)
    return proc.stdout


def fetch_site_list(state: str) -> pd.DataFrame:
    url = (
        "https://waterservices.usgs.gov/nwis/site/"
        f"?format=rdb&siteOutput=expanded&stateCd={state}&siteType=ST&parameterCd=00060"
    )
    text = curl_text(url)
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
    url = (
        "https://waterservices.usgs.gov/nwis/dv/"
        f"?format=json&sites={site}&parameterCd=00060&startDT=2015-01-01&endDT=2020-12-31"
    )
    text = curl_text(url)
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
    states = sorted(set(analysis["STATE"].dropna()) & set(STATE_TO_CODE.keys()))
    results = []
    total = 0
    for state_name in states:
        if total >= MAX_TOTAL_PAIRS:
            break
        code = STATE_TO_CODE[state_name]
        sites = fetch_site_list(code)
        res = analysis[analysis["STATE"] == state_name]
        catch_map = dict(zip(res["GRAND_ID"], res["CATCH_SKM"]))
        best_pair = None
        best_dist = MAX_DIST_KM
        for _, r in res.iterrows():
            for _, s in sites.iterrows():
                if pd.isna(s.get("dec_lat_va")) or pd.isna(s.get("dec_long_va")) or pd.isna(s.get("drain_area_va")):
                    continue
                d = math.hypot((s["dec_lat_va"] - r["LAT"]) * 111.0, (s["dec_long_va"] - r["LON"]) * 97.0)
                area_sqkm = s["drain_area_va"] * SQKM_PER_SQMI
                if d < best_dist and AREA_MIN_RATIO * r["CATCH_SKM"] <= area_sqkm <= AREA_MAX_RATIO * r["CATCH_SKM"]:
                    best_dist = d
                    best_pair = (r["GRAND_ID"], s["site_no"], catch_map[r["GRAND_ID"]], s)
        if best_pair is None:
            continue
        gid, site, res_area, site_info = best_pair
        daily = fetch_daily(site)
        if daily.empty:
            continue
        daily["date"] = pd.to_datetime(daily["date"])
        csv_path = ROOT / "data/raw/ResOpsUS+CARS_v10/v1.0/time_series/csv" / f"{gid}.csv"
        try:
            rdf = pd.read_csv(csv_path, usecols=["date", "inflow_sim"], parse_dates=["date"])
        except Exception:
            continue
        rdf = rdf[(rdf["date"] >= "2015-01-01") & (rdf["date"] <= "2020-12-31")].dropna()
        if rdf.empty:
            continue
        gage_area = site_info["drain_area_va"] * SQKM_PER_SQMI
        merged = daily.merge(rdf[["date", "inflow_sim"]], on="date", how="inner")
        if len(merged) < 100:
            continue
        obs_spec = merged["flow"] / gage_area
        glo_spec = merged["inflow_sim"] / res_area
        bias = (glo_spec - obs_spec) / obs_spec
        results.append({
            "state": code,
            "GRAND_ID": gid,
            "USGS_site": site,
            "distance_km": best_dist,
            "n_days": int(len(merged)),
            "median_bias_ratio": float(bias.median()),
            "median_specific_discharge_ratio": float((1.0 + bias).median()),
            "glofas_specific_median": float(glo_spec.median()),
            "observed_specific_median": float(obs_spec.median()),
        })
        total += 1
        print(f"{code} {gid} {site}: bias {bias.median():.2f}", flush=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved", len(results), "pairs", flush=True)


if __name__ == "__main__":
    main()
