# Data processing (Tompkins County, NY)

Small pipeline that pulls **daily weather** from Open-Meteo, merges it with **EPA county-level daily AQI** for **Tompkins / New York**, cleans the result, and writes a **modeling table** for downstream analysis or ML.

## Folder layout

```
data_processing/
├── README.md                 # this file
├── fetch_weather_data.py     # download weather → CSV
├── merge_datasets.py         # merge AQI + weather → modeling table (+ extras)
├── clean_data.py             # optional: re-clean an existing modeling CSV
└── data/
    ├── raw/                  # inputs you keep or regenerate
    │   ├── daily_weather.csv
    │   ├── daily_aqi_by_county_2024.csv
    │   └── daily_aqi_by_county_2025.csv
    ├── processed/            # outputs ready for modeling
    │   ├── modeling_table.csv
    │   └── daily_air_quality.csv   # Tompkins-only slice from raw AQI (for quick checks)
    └── sample/
        └── sample_modeling_table.csv   # first ~50 rows of modeling_table
```

## Requirements

```bash
pip install pandas requests
```

Python 3.10+ recommended (uses `zoneinfo`).

## Quick start

From this directory (`data_processing/`):

1. **Weather** — overwrites `data/raw/daily_weather.csv` by default when you use `--tompkins` + daily mode:

   ```bash
   python fetch_weather_data.py --tompkins --start 2024-01-01 --end 2025-12-31 --sleep-sec 0
   ```

2. **Merge + clean** — reads default AQI files under `data/raw/` (2024 and 2025 if both exist), merges with `daily_weather.csv`, runs `clean_modeling_table` (drops rows without both AQI and core weather unless you change that), writes `data/processed/modeling_table.csv`:

   ```bash
   python merge_datasets.py
   ```

3. **Re-clean only** (e.g. after hand-editing the CSV):

   ```bash
   python clean_data.py
   ```

   Keep incomplete rows:

   ```bash
   python clean_data.py --keep-incomplete
   ```

## Scripts

| Script | Role |
|--------|------|
| `fetch_weather_data.py` | Calls Open-Meteo Historical Archive; supports `--freq hourly\|daily`, `--tompkins` (reference point in Tompkins NY), date range, `--out`. |
| `merge_datasets.py` | Loads Tompkins rows from one or more `--aqi` CSVs (default: both `daily_aqi_by_county_2024.csv` and `2025` if present), merges on calendar date (America/New_York), drops redundant columns, then cleans. |
| `clean_data.py` | Standalone cleaner: dtypes, dedupe by `date`, AQI range check, optional row filters. |

## Data sources

What each input file is **supposed** to come from (cite these in papers / homework):

| File(s) in `data/raw/` | Source | Notes |
|------------------------|--------|--------|
| `daily_weather.csv` | **[Open-Meteo Historical API](https://open-meteo.com/en/docs/historical-weather-api)** | This repo calls the archive endpoint `https://archive-api.open-meteo.com/v1/archive`. Gridded reanalysis (commonly **ERA5**-based; see Open-Meteo’s site for the exact attribution they require). Point query uses the nearest grid cell (`grid_latitude` / `grid_longitude` in the CSV). |
| `daily_aqi_by_county_2024.csv`, `daily_aqi_by_county_2025.csv` | **[EPA AQS Air Data — Daily AQI by County](https://aqs.epa.gov/aqsweb/airdata/download_files.html#AQI)** | Download the **“Daily AQI by County”** files for the years you need from that page (this project’s raw AQI CSVs come from there). Column names match the EPA export (state/county, FIPS, `Defining Parameter`, etc.). Follow EPA’s **terms of use / citation** on the Air Data site for reports or publications. |

Processed files (`modeling_table.csv`, `daily_air_quality.csv`, `sample_…`) are **derived** from the above; cite the original weather + AQI sources, not “this repo” as the primary data origin.

## AQI data

Download **Daily AQI by County** from **[EPA AQS Air Data Downloads](https://aqs.epa.gov/aqsweb/airdata/download_files.html#AQI)** and save under `data/raw/` (e.g. `daily_aqi_by_county_2024.csv`). Expected columns include `State Name`, `county Name`, `Date`, `AQI`, etc. The code filters **New York** + **Tompkins**.

If you only have one year, either keep one file in `raw/` and pass it explicitly:

```bash
python merge_datasets.py --aqi data/raw/daily_aqi_by_county_2025.csv
```

## Merge behavior notes

- Default merge is **`outer`**; after cleaning, only rows with **both** valid AQI and core weather remain (unless you use `clean_data.py --keep-incomplete`).
- Weather timestamps may mix EST/EDT offsets; merge logic normalizes to a single calendar `date` for joining.

## Data dictionary

Units and meanings below match **this repo’s** `fetch_weather_data.py` defaults (°C, mm, m/s for wind). See [Open-Meteo historical API docs](https://open-meteo.com/en/docs/historical-weather-api) if you change units in the fetch script.

### `data/processed/modeling_table.csv`

One row per **calendar day** (after `clean_data`): Tompkins NY, weather + AQI aligned on `date`. Redundant merge keys (`date_et`, raw `Date`, FIPS codes) are dropped in `merge_datasets.py`.

| Column | Type | Description |
|--------|------|-------------|
| `location_label` | string | AOI tag from fetch (e.g. `tompkins_county_ny`). |
| `grid_latitude` | float | Nearest reanalysis grid latitude (°N). |
| `grid_longitude` | float | Nearest reanalysis grid longitude (°E, negative = W). |
| `weathercode` | int | Open-Meteo daily [WMO weather code](https://open-meteo.com/en/docs). |
| `temperature_2m_mean` | float | Daily mean air temperature at 2 m (°C). |
| `temperature_2m_max` | float | Daily max temperature at 2 m (°C). |
| `temperature_2m_min` | float | Daily min temperature at 2 m (°C). |
| `sunrise` | datetime | Local sunrise time (parsed in cleaning). |
| `sunset` | datetime | Local sunset time. |
| `daylight_duration` | float | Day length (seconds). |
| `sunshine_duration` | float | Bright sunshine duration (seconds). |
| `precipitation_sum` | float | Total precipitation including rain/snow melt (mm). |
| `rain_sum` | float | Rain component (mm). |
| `snowfall_sum` | float | Snowfall (cm, Open-Meteo convention). |
| `precipitation_hours` | float | Hours with measurable precip. |
| `windspeed_10m_max` | float | Max 10 m wind speed (m/s). |
| `windgusts_10m_max` | float | Max wind gust (m/s). |
| `winddirection_10m_dominant` | int | Prevailing wind direction (°, meteorological). |
| `shortwave_radiation_sum` | float | Sum of shortwave radiation for the day (MJ/m²). |
| `date` | date | **Merge key**: calendar date (America/New_York), normalized midnight. |
| `State Name` | string | EPA row: state (e.g. `New York`). |
| `county Name` | string | EPA row: county (`Tompkins`). |
| `AQI` | int | US AQI for that county day (0–500 scale; invalid/out-of-range set to missing before row drop). |
| `Category` | string | EPA AQI category (e.g. Good, Moderate). |
| `Defining Parameter` | string | Pollutant that drove the reported AQI (e.g. Ozone, PM2.5). |
| `Defining Site` | string | Monitor / site id used for the defining parameter. |
| `Number of Sites Reporting` | int | Count of sites in the county rollup. |

#### Ozone — AQI index bands (for interpretation)

Daily county AQI is already computed by EPA. If **`Defining Parameter`** is **Ozone**, the underlying concentration breakpoints (ppm, 8-hr, etc.) live in the official **[AQI Breakpoints](https://aqs.epa.gov/aqsweb/documents/codetables/aqi_breakpoints.html)** table (AQS reference; look for **Ozone**, parameter **44201**).

For most analysis you only need the **`AQI`** value (and `Category`). The **index** uses the standard EPA category cutoffs (same AQI scale whether ozone, PM2.5, or another pollutant drives the number):

| AQI range | Category |
|-----------|----------|
| 0–50 | Good |
| 51–100 | Moderate |
| 101–150 | Unhealthy for Sensitive Groups |
| 151–200 | Unhealthy |
| 201–300 | Very Unhealthy |
| 301–500 | Hazardous |

The breakpoints page also lists higher AQI bins (e.g. 501+) for some rows; this pipeline treats AQI > 500 as invalid in `clean_data.py`.

### `data/raw/daily_weather.csv`

Daily Open-Meteo export (same weather columns as above) **plus**:

| Column | Type | Description |
|--------|------|-------------|
| `date_local` | string | API “local” calendar day (YYYY-MM-DD). |
| `date_et` | string | Local midnight with NY offset (e.g. `…-05:00` / `…-04:00`); used only before merge, not in `modeling_table`. |

### `data/raw/daily_aqi_by_county_*.csv`

County-level daily AQI from **[EPA AQS Air Data — Daily AQI by County](https://aqs.epa.gov/aqsweb/airdata/download_files.html#AQI)** (national files; code keeps NY + Tompkins).

| Column | Type | Description |
|--------|------|-------------|
| `State Name` | string | Full state name. |
| `county Name` | string | County name (note leading space in header as in source). |
| `State Code` | string | FIPS state code (e.g. `36`). |
| `County Code` | string | FIPS county code (e.g. `109`). |
| `Date` | string | Calendar date `YYYY-MM-DD`. |
| `AQI` | int | Daily AQI. |
| `Category` | string | Air quality category. |
| `Defining Parameter` | string | Pollutant driving the index. |
| `Defining Site` | string | Site identifier. |
| `Number of Sites Reporting` | int | Sites contributing. |

### `data/processed/daily_air_quality.csv`

Tompkins-only rows cut from the raw national AQI file(s); **same columns** as the EPA table above (no weather).

### `data/sample/sample_modeling_table.csv`

First ~50 rows of `modeling_table.csv` for quick previews; same schema as the modeling table.

## License / attribution

- **Weather:** follow [Open-Meteo](https://open-meteo.com/) terms and the attribution for the underlying reanalysis (e.g. ECMWF) as described on their documentation / “License” page.
- **AQI:** follow **[EPA AQS Air Data](https://aqs.epa.gov/aqsweb/airdata/download_files.html#AQI)** terms of use and any citation guidance for **Daily AQI by County** downloads.
- **This pipeline:** your own code; the merged CSV is a derivative product — still cite Open-Meteo + EPA (or your actual AQI source) for the underlying measurements.
