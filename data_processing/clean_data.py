#!/usr/bin/env python3
"""Clean the merged modeling table: dtypes, dup dates, obvious bad values. By default drop incomplete rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# usual Open-Meteo daily numeric fields
_DAILY_WEATHER_NUMERIC = (
    "grid_latitude",
    "grid_longitude",
    "weathercode",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "daylight_duration",
    "sunshine_duration",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "winddirection_10m_dominant",
    "shortwave_radiation_sum",
)
# hourly export columns (if you merged hourly weather)
_HOURLY_WEATHER_NUMERIC = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "pressure_msl",
    "shortwave_radiation",
    "cloud_cover",
    "TEMP",
    "PRECIP",
    "RHUM",
    "WS",
    "WD",
    "BARPR",
    "SRAD",
    "CLOUD",
)
_AQI_NUMERIC = ("AQI", "Number of Sites Reporting")
_STRING_COLS = (
    "location_label",
    "State Name",
    "county Name",
    "Category",
    "Defining Parameter",
    "Defining Site",
)
_US_AQI_MAX = 500


def clean_modeling_table(
    df: pd.DataFrame,
    *,
    drop_if_no_aqi: bool = True,
    drop_if_no_weather: bool = True,
) -> pd.DataFrame:
    """
    What this does (kinda boring but useful):
    - Parse date, sort, dedupe by day (keep last row if duplicates)
    - Strip strings; coerce numerics
    - Parse sunrise/sunset if those cols exist
    - If max temp < min temp, swap them (API glitch / weird row)
    - Clamp AQI to [0, 500] US-style; outside -> NaN
    - By default drop rows missing AQI or missing core weather
      (daily: mean or weathercode; hourly: temperature_2m)
    - If both drop flags are on, we remove the helper cols aqi_available / core_weather_available
      since they'd just be True everywhere
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    for col in _STRING_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
            out.loc[out[col].isin(("", "nan", "None")), col] = pd.NA

    numeric_cols = [
        c
        for c in (
            *_DAILY_WEATHER_NUMERIC,
            *_HOURLY_WEATHER_NUMERIC,
            *_AQI_NUMERIC,
        )
        if c in out.columns
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ("sunrise", "sunset"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    if {"temperature_2m_max", "temperature_2m_min"}.issubset(out.columns):
        tmax = out["temperature_2m_max"]
        tmin = out["temperature_2m_min"]
        swap = tmax.notna() & tmin.notna() & (tmax < tmin)
        if swap.any():
            out.loc[swap, ["temperature_2m_max", "temperature_2m_min"]] = out.loc[
                swap, ["temperature_2m_min", "temperature_2m_max"]
            ].to_numpy()

    if "AQI" in out.columns:
        invalid = out["AQI"].notna() & ((out["AQI"] < 0) | (out["AQI"] > _US_AQI_MAX))
        out.loc[invalid, "AQI"] = pd.NA
        out["aqi_available"] = out["AQI"].notna()
    else:
        out["aqi_available"] = False

    wx_markers = [c for c in ("temperature_2m_mean", "weathercode", "temperature_2m") if c in out.columns]
    if wx_markers:
        out["core_weather_available"] = out[wx_markers].notna().any(axis=1)
    else:
        out["core_weather_available"] = True

    if "date" in out.columns:
        out = out.sort_values("date", kind="mergesort")
        out = out.drop_duplicates(subset=["date"], keep="last")

    if drop_if_no_aqi and "aqi_available" in out.columns:
        out = out.loc[out["aqi_available"]].copy()
    if drop_if_no_weather and "core_weather_available" in out.columns:
        out = out.loc[out["core_weather_available"]].copy()

    if drop_if_no_aqi and drop_if_no_weather:
        out = out.drop(columns=["aqi_available", "core_weather_available"], errors="ignore")

    return out.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Load modeling_table.csv, clean it, write back.")
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=data_dir / "processed" / "modeling_table.csv",
        help="Input CSV (default: data/processed/modeling_table.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: overwrite input)",
    )
    parser.add_argument(
        "--keep-incomplete",
        action="store_true",
        help="Keep rows missing AQI or core weather (default: drop them)",
    )
    args = parser.parse_args()

    if not args.in_path.is_file():
        print(f"ERROR: input file not found: {args.in_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.in_path)
    keep_inc = args.keep_incomplete
    cleaned = clean_modeling_table(
        df,
        drop_if_no_aqi=not keep_inc,
        drop_if_no_weather=not keep_inc,
    )
    out_path = args.out or args.in_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  rows={len(cleaned)}  (was {len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
