import pandas as pd
from feature_engineering.features import (
    add_lag_features,
    add_rolling_features,
    build_features,
    get_X_y
)

def sample_df():
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "AQI": range(10, 20),
        "temperature_2m_max": [20]*10,
        "temperature_2m_min": [10]*10,
        "winddirection_10m_dominant": [90]*10,
    })

def test_add_lag_features():
    df = sample_df()
    df2 = add_lag_features(df)

    assert "aqi_lag_1" in df2.columns
    assert df2["aqi_lag_1"].iloc[1] == df["AQI"].iloc[0]

def test_add_rolling_features():
    df = sample_df()
    df2 = add_rolling_features(df)

    assert "aqi_roll_mean_3" in df2.columns

def test_build_features_runs():
    df = sample_df()

    # 加一些必须列
    df["precipitation_sum"] = 0
    df["windspeed_10m_max"] = 1
    df["shortwave_radiation_sum"] = 1
    df["weathercode"] = 1
    df["daylight_duration"] = 1
    df["sunshine_duration"] = 1

    result = build_features(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

def test_get_X_y():
    df = sample_df()

    df["precipitation_sum"] = 0
    df["windspeed_10m_max"] = 1
    df["shortwave_radiation_sum"] = 1
    df["weathercode"] = 1
    df["daylight_duration"] = 1
    df["sunshine_duration"] = 1

    df = build_features(df)

    X, y = get_X_y(df)

    assert len(X) == len(y)
    assert "AQI" not in X.columns

def test_add_time_features():
    import pandas as pd
    from feature_engineering.features import add_time_features

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5)
    })

    df2 = add_time_features(df)

    assert "month" in df2.columns
    assert "weekday" in df2.columns

def test_add_weather_features():
    import pandas as pd
    from feature_engineering.features import add_weather_features

    df = pd.DataFrame({
        "temperature_2m_max": [20, 22],
        "temperature_2m_min": [10, 12],
        "winddirection_10m_dominant": [0, 180]
    })

    df2 = add_weather_features(df)

    assert "temp_range" in df2.columns
    assert "wind_dir_sin" in df2.columns

def test_save_features_table(tmp_path):
    import pandas as pd
    from feature_engineering.features import save_features_table

    input_path = tmp_path / "modeling_table.csv"
    output_path = tmp_path / "features_table.csv"

    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        "AQI": range(10, 20),
        "temperature_2m_max": [20] * 10,
        "temperature_2m_min": [10] * 10,
        "winddirection_10m_dominant": [90] * 10,
        "precipitation_sum": [0] * 10,
        "windspeed_10m_max": [1] * 10,
        "shortwave_radiation_sum": [1] * 10,
        "weathercode": [1] * 10,
        "daylight_duration": [1] * 10,
        "sunshine_duration": [1] * 10,
    })

    df.to_csv(input_path, index=False)

    result = save_features_table(
        input_path=str(input_path),
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert "AQI" in result.columns
    assert len(result) > 0