"""
features.py
-----------
Feature engineering module for ozone AQI prediction.
Input:  modeling_table.csv  (daily-level, one row per day)
Output: DataFrame with all engineered features ready for modeling

Columns used from raw data:
    - date                        : date string (YYYY-MM-DD)
    - AQI                         : daily ozone AQI value (target variable)
    - temperature_2m_mean/max/min
    - precipitation_sum
    - windspeed_10m_max
    - winddirection_10m_dominant
    - shortwave_radiation_sum
    - weathercode
    - daylight_duration
    - sunshine_duration
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Lag features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame, col: str = "AQI", lags=None) -> pd.DataFrame:
    """
    Add lagged AQI features (previous n days).

    Args:
        df   : DataFrame sorted by date ascending.
        col  : Column to lag (default: 'AQI').
        lags : List of lag periods in days (default: [1, 2, 3, 7]).

    Returns:
        DataFrame with new columns aqi_lag_1, aqi_lag_2, aqi_lag_3, aqi_lag_7.
    """
    if lags is None:
        lags = [1, 2, 3, 7]
    df = df.copy()
    for lag in lags:
        df[f"aqi_lag_{lag}"] = df[col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# 2. Rolling features
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame, col: str = "AQI", windows=None) -> pd.DataFrame:
    """
    Add rolling mean and (for window=7) rolling std of AQI.
    Uses shift(1) so that today's value is never included in the window.

    Args:
        df      : DataFrame sorted by date ascending.
        col     : Column to roll over (default: 'AQI').
        windows : List of window sizes in days (default: [3, 7]).

    Returns:
        DataFrame with new columns aqi_roll_mean_3, aqi_roll_mean_7, aqi_roll_std_7.
    """
    if windows is None:
        windows = [3, 7]
    df = df.copy()
    shifted = df[col].shift(1)
    for w in windows:
        df[f"aqi_roll_mean_{w}"] = shifted.rolling(window=w).mean()
        if w == 7:
            df["aqi_roll_std_7"] = shifted.rolling(window=w).std()
    return df


# ---------------------------------------------------------------------------
# 3. Trend features
# ---------------------------------------------------------------------------

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add AQI trend (difference) features.
        aqi_diff_1 : short-term trend  (lag_1 - lag_2)
        aqi_diff_7 : week-over-week    (lag_1 - lag_7)

    Requires add_lag_features to have been called first.

    Args:
        df : DataFrame that already contains aqi_lag_1, aqi_lag_2, aqi_lag_7.

    Returns:
        DataFrame with aqi_diff_1 and aqi_diff_7 added.
    """
    df = df.copy()
    if "aqi_lag_1" in df.columns and "aqi_lag_2" in df.columns:
        df["aqi_diff_1"] = df["aqi_lag_1"] - df["aqi_lag_2"]
    if "aqi_lag_1" in df.columns and "aqi_lag_7" in df.columns:
        df["aqi_diff_7"] = df["aqi_lag_1"] - df["aqi_lag_7"]
    return df


# ---------------------------------------------------------------------------
# 4. Time features
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Extract calendar features and apply cyclic (sin/cos) encoding.

    Features added:
        month, weekday, season           (raw integer)
        month_sin, month_cos             (cyclic encoding, period=12)
        weekday_sin, weekday_cos         (cyclic encoding, period=7)

    Season mapping:
        1 = Spring (Mar-May)
        2 = Summer (Jun-Aug)
        3 = Fall   (Sep-Nov)
        4 = Winter (Dec-Feb)

    Note: is_weekend is intentionally excluded because weekday already
    encodes that information (weekday >= 5 => weekend).

    Args:
        df       : DataFrame with a parseable date column.
        date_col : Name of the date column (default: 'date').

    Returns:
        DataFrame with time feature columns added.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])

    df["month"] = dates.dt.month
    df["weekday"] = dates.dt.weekday  # 0 = Monday, 6 = Sunday

    season_map = {
        1: 4, 2: 4,
        3: 1, 4: 1, 5: 1,
        6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3,
        12: 4,
    }
    df["season"] = df["month"].map(season_map)

    # Cyclic encoding prevents the model from seeing a large gap between
    # December (12) and January (1), or Sunday (6) and Monday (0).
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    return df


# ---------------------------------------------------------------------------
# 5. Weather-derived features
# ---------------------------------------------------------------------------

def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived weather features useful for ozone prediction.

    Features added:
        temp_range   : daily temperature range (max - min).
                       Large temp range correlates with ozone formation.
        wind_dir_sin : cyclic sine encoding of wind direction (degrees).
        wind_dir_cos : cyclic cosine encoding of wind direction (degrees).
                       0 deg and 360 deg are physically identical,
                       so raw degrees would mislead the model.

    Args:
        df : DataFrame containing temperature_2m_max, temperature_2m_min,
             and winddirection_10m_dominant columns.

    Returns:
        DataFrame with three new feature columns added.
    """
    df = df.copy()

    # Temperature range
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

    # Wind direction cyclic encoding
    theta = np.deg2rad(df["winddirection_10m_dominant"])
    df["wind_dir_sin"] = np.sin(theta)
    df["wind_dir_cos"] = np.cos(theta)

    return df


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, date_col: str = "date", target_col: str = "AQI") -> pd.DataFrame:
    """
    Run the full feature engineering pipeline and return a clean feature matrix.

    Steps:
        1. Sort by date.
        2. Add lag features        (lag 1, 2, 3, 7).
        3. Add rolling features    (mean 3, mean 7, std 7).
        4. Add trend features      (diff_1, diff_7).
        5. Add time features       (month, weekday, season + cyclic encodings).
        6. Add weather features    (temp_range, wind_dir_sin/cos).
        7. Drop rows with NaN only in core lag/rolling columns
           (avoids over-dropping due to unrelated missing values).

    Args:
        df         : Raw modeling_table DataFrame.
        date_col   : Name of the date column.
        target_col : Name of the AQI column.

    Returns:
        DataFrame ready for model training (NaN rows removed, index reset).
    """
    df = df.sort_values(date_col).reset_index(drop=True)

    df = add_lag_features(df, col=target_col)
    df = add_rolling_features(df, col=target_col)
    df = add_trend_features(df)
    df = add_time_features(df, date_col=date_col)
    df = add_weather_features(df)

    required_cols = [
        "aqi_lag_1", "aqi_lag_2", "aqi_lag_3", "aqi_lag_7",
        "aqi_roll_mean_3", "aqi_roll_mean_7", "aqi_roll_std_7",
    ]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 7. X / y split
# ---------------------------------------------------------------------------

def get_X_y(df: pd.DataFrame, target_col: str = "AQI"):
    """
    Split feature matrix into X (features) and y (target).

    Columns dropped (non-predictive):
        - Target / leakage : target_col (default AQI), Category
        - Date string      : date  (already encoded in time features)
        - Metadata/ID      : Defining Parameter, Defining Site, State Name,
                             county Name, location_label, forecasting
        - Unparseable str  : sunrise, sunset
        - Raw wind degrees : winddirection_10m_dominant (replaced by sin/cos)

    Columns intentionally KEPT in X:
        - weathercode                 : numeric weather category (0=clear, 61=rain...)
        - Number of Sites Reporting   : proxy for AQI data reliability
        - grid_latitude / longitude   : location coordinates
        - All weather numerics and engineered aqi_* / time / weather features

    Args:
        df         : Output of build_features().
        target_col : Target variable column name (default: 'AQI').

    Returns:
        X : pd.DataFrame of input features.
        y : pd.Series of target values.
    """
    drop_cols = [
        target_col, "Category",
        "date",
        "Defining Parameter", "Defining Site",
        "State Name", "county Name",
        "location_label", "forecasting",
        "sunrise", "sunset",
        "winddirection_10m_dominant",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return X, y


# ---------------------------------------------------------------------------
# 8. Save features table (35 features + AQI, 36 cols total)
# ---------------------------------------------------------------------------

def save_features_table(
    input_path: str = "data_processing/data/processed/modeling_table.csv",
    output_path: str = "feature_engineering/features_table.csv",
    date_col: str = "date",
    target_col: str = "AQI",
) -> pd.DataFrame:
    """
    Load raw data, run full pipeline, and save a features table
    containing X (35 feature columns) + AQI in one CSV.
    Non-predictive columns (metadata, date, etc.) are already removed.

    Args:
        input_path  : Path to modeling_table.csv.
        output_path : Path to save the features table CSV.
        date_col    : Name of the date column.
        target_col  : Name of the AQI column.

    Returns:
        features_df : DataFrame with 36 columns (35 features + AQI).

    Usage:
        from feature_engineering.features import save_features_table
        df = save_features_table()
        X = df.drop(columns=["AQI"])
        y = df["AQI"]
    """
    raw = pd.read_csv(input_path)
    feature_df = build_features(raw, date_col=date_col, target_col=target_col)
    X, y = get_X_y(feature_df, target_col=target_col)

    features_df = X.copy()
    features_df[target_col] = y.values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Features table saved to: {output_path}")
    print(f"Shape: {features_df.shape}  (35 features + 1 target)")
    return features_df


# ---------------------------------------------------------------------------
# Quick smoke test (run: python features.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    _input_csv = os.path.join(
        _here, "data_processing", "data", "processed", "modeling_table.csv"
    )
    _features_csv = os.path.join(_here, "feature_engineering", "features_table.csv")

    raw = pd.read_csv(_input_csv)

    feature_df = build_features(raw)
    print(f"Shape after feature engineering: {feature_df.shape}")

    engineered_cols = [c for c in feature_df.columns if any(
        c.startswith(p) for p in ["aqi_", "month", "weekday", "season", "temp_range", "wind_dir"]
    )]
    print("\nEngineered feature columns:")
    for c in engineered_cols:
        print(f"  {c}")
    print(feature_df[engineered_cols].head(3))

    X, y = get_X_y(feature_df)
    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X columns sample: {X.columns[:10].tolist()}")

    save_features_table(input_path=_input_csv, output_path=_features_csv)
