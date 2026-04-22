#!/usr/bin/env python3
"""
Merge county-level daily AQI (US-wide CSVs) with weather from fetch_weather_data.py.

- AQI: by default we load daily_aqi_by_county_2024.csv and 2025 from data/raw if they exist,
  pull Tompkins NY rows, and concat. Pass --aqi one or more times for other files.
- Weather: daily (date_et) or hourly (hour_et); merge key is calendar date in America/New_York.

We drop redundant cols (date_et, date_local, EPA Date, State/County codes) before the merge
result is final.

After merging we run clean_data.clean_modeling_table. To only re-clean a table: python clean_data.py

Needs: pandas
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from clean_data import clean_modeling_table

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
NY_TZ = "America/New_York"

_REDUNDANT_MERGE_COLUMNS = ("date_et", "date_local", "Date", "State Code", "County Code")

_DEFAULT_AQI_CANDIDATES = (
    DATA_DIR / "raw" / "daily_aqi_by_county_2024.csv",
    DATA_DIR / "raw" / "daily_aqi_by_county_2025.csv",
)


def load_tompkins_aqi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    state_col = "State Name"
    county_col = "county Name"
    if state_col not in df.columns or county_col not in df.columns:
        raise ValueError(f"AQI CSV missing {state_col!r} or {county_col!r}; got: {list(df.columns)}")
    mask = (df[state_col].astype(str).str.strip() == "New York") & (
        df[county_col].astype(str).str.strip().str.lower() == "tompkins"
    )
    sub = df.loc[mask].copy()
    if sub.empty:
        raise ValueError("No New York / Tompkins rows in this AQI file — check the file or column names.")
    sub["date"] = pd.to_datetime(sub["Date"], errors="coerce").dt.normalize()
    if sub["date"].isna().any():
        bad = sub.loc[sub["date"].isna(), "Date"].head(5).tolist()
        raise ValueError(f"Some Date values could not be parsed, examples: {bad}")
    sub = sub.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return sub


def load_tompkins_aqi_concat(paths: Sequence[Path]) -> pd.DataFrame:
    """Stack Tompkins daily rows from several national AQI files; dedupe by date (last row wins)."""
    if not paths:
        raise ValueError("Need at least one AQI file path")
    parts = [load_tompkins_aqi(p) for p in paths]
    combined = pd.concat(parts, ignore_index=True)
    return combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")


def _timestamps_to_merge_dates(ts: pd.Series) -> pd.Series:
    out = pd.to_datetime(ts, utc=True, errors="coerce")
    out = out.dt.tz_convert(NY_TZ)
    return out.dt.normalize().dt.tz_localize(None)


def _openmeteo_kind_and_dates(df: pd.DataFrame) -> tuple[str, pd.Series]:
    if "hour_et" in df.columns:
        return "hourly", _timestamps_to_merge_dates(df["hour_et"])
    if "date_et" in df.columns:
        return "daily", _timestamps_to_merge_dates(df["date_et"])
    raise ValueError(
        "Weather CSV should have date_et (daily) or hour_et (hourly); "
        f"columns you have: {list(df.columns)}"
    )


def merge_frames(aqi: pd.DataFrame, meteo: pd.DataFrame, *, how: str) -> pd.DataFrame:
    kind, meteo_dates = _openmeteo_kind_and_dates(meteo)
    wx = meteo.copy()
    wx["date"] = meteo_dates

    key_aqi = aqi[["date"] + [c for c in aqi.columns if c != "date"]]
    merged = wx.merge(key_aqi, on="date", how=how)

    sort_col = "hour_et" if kind == "hourly" else "date_et"
    if sort_col in merged.columns:
        merged = merged.sort_values(sort_col, kind="mergesort")
    else:
        merged = merged.sort_values("date", kind="mergesort")

    drop_cols = [c for c in _REDUNDANT_MERGE_COLUMNS if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    return merged


def write_derived_outputs(merged: pd.DataFrame, aqi_paths: Path | Sequence[Path]) -> None:
    proc = DATA_DIR / "processed"
    sample_dir = DATA_DIR / "sample"
    proc.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    paths = [aqi_paths] if isinstance(aqi_paths, Path) else list(aqi_paths)
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    state_col, county_col = "State Name", "county Name"
    mask = (df[state_col].astype(str).str.strip() == "New York") & (
        df[county_col].astype(str).str.strip().str.lower() == "tompkins"
    )
    sub = df.loc[mask].sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    sub.to_csv(proc / "daily_air_quality.csv", index=False)
    n = min(50, len(merged))
    merged.head(n).to_csv(sample_dir / "sample_modeling_table.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge Tompkins AQI + weather CSVs and write the modeling table."
    )
    parser.add_argument(
        "--aqi",
        type=Path,
        nargs="*",
        default=None,
        help="County daily AQI CSV(s); default: any of 2024/2025 files that exist under data/raw",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=DATA_DIR / "raw" / "daily_weather.csv",
        help="CSV from fetch_weather_data.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR / "processed" / "modeling_table.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--no-extra-outputs",
        action="store_true",
        help="Skip processed/daily_air_quality.csv and sample preview",
    )
    parser.add_argument(
        "--how",
        choices=("inner", "left", "right", "outer"),
        default="outer",
        help="pandas merge how= (default outer)",
    )
    args = parser.parse_args()

    aqi_paths: list[Path] = list(args.aqi) if args.aqi else [p for p in _DEFAULT_AQI_CANDIDATES if p.is_file()]
    if not aqi_paths:
        print(
            f"ERROR: no AQI files found. Put CSVs in {DATA_DIR / 'raw'} or pass --aqi explicitly.",
            file=sys.stderr,
        )
        return 1
    for p in aqi_paths:
        if not p.is_file():
            print(f"ERROR: AQI file not found: {p}", file=sys.stderr)
            return 1
    if not args.weather.is_file():
        print(f"ERROR: weather file not found: {args.weather}", file=sys.stderr)
        return 1

    aqi = load_tompkins_aqi_concat(aqi_paths) if len(aqi_paths) > 1 else load_tompkins_aqi(aqi_paths[0])
    meteo = pd.read_csv(args.weather)
    merged = clean_modeling_table(merge_frames(aqi, meteo, how=args.how))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    if not args.no_extra_outputs:
        write_derived_outputs(merged, aqi_paths)
    print(
        f"Wrote {args.out}  rows={len(merged)}  "
        f"AQI_days={len(aqi)}  weather_rows={len(meteo)}  merge={args.how}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
