#!/usr/bin/env python3
"""
fetch_weather_data.py — pull hourly or daily weather from Open-Meteo Historical Archive for one lat/lon.

Hourly mode: kinda matches the cpportal job/openmeteo style (hour_et in your TZ, short cols TEMP, PRECIP, …).
Daily mode: Open-Meteo daily= vars, one row per local day, date_et is the local midnight anchor.

Note: the reanalysis grid might snap your point a bit — check grid_latitude / grid_longitude in the CSV.

API: https://archive-api.open-meteo.com/v1/archive
Docs: https://open-meteo.com/en/docs/historical-weather-api

You can also set env vars (CLI wins): OPEN_METEO_FREQ, OPEN_METEO_LAT, OPEN_METEO_LON,
OPEN_METEO_START, OPEN_METEO_END, OPEN_METEO_TZ, OPEN_METEO_OUT, OPEN_METEO_SLEEP_SEC

Deps: pip install requests pandas
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "raw"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "pressure_msl",
    "shortwave_radiation",
    "cloud_cover",
]
DAILY_VARIABLES = [
    "weathercode",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "sunrise",
    "sunset",
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
]
TOMPKINS_NY_REF_LAT = 42.443961
TOMPKINS_NY_REF_LON = -76.501881
CHUNK_DAYS_HOURLY = 14
CHUNK_DAYS_DAILY = 366


def _date_window_chunks(start: date, end: date, max_days: int) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        win_end = min(cur + timedelta(days=max_days - 1), end)
        out.append((cur, win_end))
        cur = win_end + timedelta(days=1)
    return out


def _parse_ymd(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def _coerce_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float("nan")
    if f != f:
        return float("nan")
    return f


def fetch_archive(
    session: requests.Session,
    lat: float,
    lon: float,
    start: date,
    end: date,
    tz: str,
    *,
    mode: str,
) -> dict[str, Any]:
    q: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": tz,
        "timeformat": "iso8601",
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
        "cell_selection": "nearest",
    }
    if mode == "hourly":
        q["hourly"] = ",".join(HOURLY_VARIABLES)
    else:
        q["daily"] = ",".join(DAILY_VARIABLES)
    r = session.get(ARCHIVE_URL, params=q, timeout=120)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        if not data:
            return {}
        first = data[0]
        return first if isinstance(first, dict) else {}
    if isinstance(data, dict) and ("hourly" in data or "daily" in data):
        return data
    return {}


def build_hourly_dataframe(response: dict[str, Any], location_label: str, tz: str) -> pd.DataFrame:
    h = response.get("hourly") or {}
    if not h or not h.get("time"):
        return pd.DataFrame()

    times = h["time"]
    n = len(times)

    def series(key: str) -> list[float]:
        v = h.get(key)
        if v is None or len(v) != n:
            return [float("nan")] * n
        return [_coerce_float(x) for x in v]

    df = pd.DataFrame(
        {
            "time_local": times,
            "temperature_2m": series("temperature_2m"),
            "relative_humidity_2m": series("relative_humidity_2m"),
            "wind_speed_10m": series("wind_speed_10m"),
            "wind_direction_10m": series("wind_direction_10m"),
            "precipitation": series("precipitation"),
            "pressure_msl": series("pressure_msl"),
            "shortwave_radiation": series("shortwave_radiation"),
            "cloud_cover": series("cloud_cover"),
        }
    )
    zone = ZoneInfo(tz)
    ts = pd.to_datetime(df["time_local"], utc=False)
    if ts.dt.tz is None:
        try:
            ts = ts.dt.tz_localize(zone, ambiguous="infer", nonexistent="shift_forward")
        except (TypeError, ValueError):
            ts = ts.dt.tz_localize("UTC").dt.tz_convert(zone)
    else:
        ts = ts.dt.tz_convert(zone)
    df["hour_et"] = ts.dt.floor("h")
    df["location_label"] = location_label
    df["grid_latitude"] = response.get("latitude")
    df["grid_longitude"] = response.get("longitude")
    df["TEMP"] = df["temperature_2m"].round(2)
    df["PRECIP"] = df["precipitation"].round(2)
    df["RHUM"] = df["relative_humidity_2m"].round(2)
    df["WS"] = df["wind_speed_10m"].round(2)
    df["WD"] = df["wind_direction_10m"].round(2)
    df["BARPR"] = df["pressure_msl"].round(2)
    df["SRAD"] = df["shortwave_radiation"].round(2)
    df["CLOUD"] = df["cloud_cover"].round(2)
    return df


def build_daily_dataframe(response: dict[str, Any], location_label: str, tz: str) -> pd.DataFrame:
    d = response.get("daily") or {}
    if not d or not d.get("time"):
        return pd.DataFrame()
    n = len(d["time"])
    glat, glon = response.get("latitude"), response.get("longitude")
    rows: dict[str, Any] = {
        "date_local": d["time"],
        "location_label": [location_label] * n,
        "grid_latitude": [glat] * n,
        "grid_longitude": [glon] * n,
    }
    for key, val in d.items():
        if key == "time" or not isinstance(val, list) or len(val) != n:
            continue
        rows[key] = list(val)
    df = pd.DataFrame(rows)
    zone = ZoneInfo(tz)
    dtp = pd.to_datetime(df["date_local"], errors="coerce")
    dtp = dtp.dt.tz_localize(zone) if dtp.dt.tz is None else dtp.dt.tz_convert(zone)
    df["date_et"] = dtp.dt.normalize()
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Open-Meteo historical archive weather for one AOI (point) to CSV."
    )
    parser.add_argument("--lat", type=float, default=None, help="WGS84 latitude")
    parser.add_argument("--lon", type=float, default=None, help="WGS84 longitude")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD, inclusive)")
    parser.add_argument(
        "--tz",
        type=str,
        default=None,
        help="IANA timezone for API local times (default: America/New_York)",
    )
    parser.add_argument(
        "--freq",
        type=str,
        choices=("hourly", "daily"),
        default=None,
        help="hourly (default) or daily aggregation",
    )
    parser.add_argument(
        "--tompkins",
        action="store_true",
        help="Use Tompkins County, NY reference point; default output names for this AOI",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="openmeteo_point",
        help="String stored in location_label",
    )
    parser.add_argument("--out", type=str, default=None, help="Output CSV path")
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=None,
        help="Delay between date-window requests (default 4; use 0 for local runs)",
    )
    args = parser.parse_args()

    use_tompkins = bool(args.tompkins)
    env_freq = (os.environ.get("OPEN_METEO_FREQ") or "").strip().lower()
    if args.freq is not None:
        frequency = args.freq
    elif env_freq in ("hourly", "daily"):
        frequency = env_freq
    elif use_tompkins:
        frequency = "daily"
    else:
        frequency = "hourly"
    if frequency not in ("hourly", "daily"):
        print("ERROR: --freq / OPEN_METEO_FREQ must be hourly or daily", file=sys.stderr)
        return 1

    default_out = DEFAULT_OUTPUT_DIR / "weather_hourly.csv"
    if use_tompkins:
        default_out = (
            DEFAULT_OUTPUT_DIR / "daily_weather.csv"
            if frequency == "daily"
            else DEFAULT_OUTPUT_DIR / "weather_hourly.csv"
        )
    if use_tompkins and args.label == "openmeteo_point":
        location_label = "tompkins_county_ny"
    else:
        location_label = args.label

    lat = args.lat
    lon = args.lon
    if use_tompkins and lat is None and os.environ.get("OPEN_METEO_LAT") is None:
        lat = TOMPKINS_NY_REF_LAT
    if use_tompkins and lon is None and os.environ.get("OPEN_METEO_LON") is None:
        lon = TOMPKINS_NY_REF_LON
    if lat is None:
        lat = float(os.environ.get("OPEN_METEO_LAT", "40.7128"))
    else:
        lat = float(lat)
    if lon is None:
        lon = float(os.environ.get("OPEN_METEO_LON", "-74.0060"))
    else:
        lon = float(lon)

    start_s = args.start or os.environ.get("OPEN_METEO_START", "2024-01-01")
    end_s = args.end or os.environ.get("OPEN_METEO_END", "2024-01-07")
    tz = args.tz or os.environ.get("OPEN_METEO_TZ", "America/New_York")
    out = Path(args.out or os.environ.get("OPEN_METEO_OUT", str(default_out)))
    if args.sleep_sec is None:
        sleep_sec = float(os.environ.get("OPEN_METEO_SLEEP_SEC", "4"))
    else:
        sleep_sec = float(args.sleep_sec)

    d0, d1 = _parse_ymd(start_s), _parse_ymd(end_s)
    if d1 < d0:
        print("ERROR: end date before start date", file=sys.stderr)
        return 1

    max_days = CHUNK_DAYS_DAILY if frequency == "daily" else CHUNK_DAYS_HOURLY
    windows = _date_window_chunks(d0, d1, max_days=max_days)
    session = requests.Session()
    frames: list[pd.DataFrame] = []
    for i, (w0, w1) in enumerate(windows):
        if i and sleep_sec > 0:
            time.sleep(sleep_sec)
        payload = fetch_archive(session, lat, lon, w0, w1, tz, mode=frequency)
        if frequency == "hourly":
            part = build_hourly_dataframe(payload, location_label, tz)
        else:
            part = build_daily_dataframe(payload, location_label, tz)
        if part.empty:
            print(f"  warning: no {frequency} data for {w0}..{w1}", file=sys.stderr)
        else:
            frames.append(part)

    if not frames:
        print("ERROR: all date windows empty", file=sys.stderr)
        return 2

    merged = pd.concat(frames, ignore_index=True)
    if frequency == "hourly":
        merged = merged.sort_values("hour_et").drop_duplicates(subset=["hour_et"], keep="last")
    else:
        merged = merged.sort_values("date_et").drop_duplicates(subset=["date_et"], keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Wrote {out}  rows={len(merged)}  freq={frequency}  lat={lat}  lon={lon}  tz={tz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
