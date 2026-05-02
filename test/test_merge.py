import pandas as pd
from data_processing.merge_datasets import merge_frames
import pandas as pd
import pytest
from pathlib import Path

from data_processing.merge_datasets import (
    load_tompkins_aqi,
    load_tompkins_aqi_concat,
    _openmeteo_kind_and_dates,
    merge_frames,
)


def test_load_tompkins_aqi(tmp_path):
    csv_path = tmp_path / "aqi.csv"

    df = pd.DataFrame({
        "State Name": ["New York", "New York", "California"],
        "county Name": ["Tompkins", "Albany", "Tompkins"],
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "AQI": [40, 50, 60],
    })

    df.to_csv(csv_path, index=False)

    result = load_tompkins_aqi(csv_path)

    assert len(result) == 1
    assert result["AQI"].iloc[0] == 40
    assert "date" in result.columns


def test_load_tompkins_aqi_missing_columns(tmp_path):
    csv_path = tmp_path / "bad_aqi.csv"

    df = pd.DataFrame({
        "State": ["New York"],
        "County": ["Tompkins"],
        "Date": ["2024-01-01"],
        "AQI": [40],
    })

    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_tompkins_aqi(csv_path)


def test_load_tompkins_aqi_no_tompkins_rows(tmp_path):
    csv_path = tmp_path / "no_tompkins.csv"

    df = pd.DataFrame({
        "State Name": ["New York"],
        "county Name": ["Albany"],
        "Date": ["2024-01-01"],
        "AQI": [40],
    })

    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_tompkins_aqi(csv_path)


def test_load_tompkins_aqi_concat(tmp_path):
    path1 = tmp_path / "aqi1.csv"
    path2 = tmp_path / "aqi2.csv"

    df1 = pd.DataFrame({
        "State Name": ["New York"],
        "county Name": ["Tompkins"],
        "Date": ["2024-01-01"],
        "AQI": [40],
    })

    df2 = pd.DataFrame({
        "State Name": ["New York"],
        "county Name": ["Tompkins"],
        "Date": ["2024-01-02"],
        "AQI": [50],
    })

    df1.to_csv(path1, index=False)
    df2.to_csv(path2, index=False)

    result = load_tompkins_aqi_concat([path1, path2])

    assert len(result) == 2
    assert result["AQI"].tolist() == [40, 50]


def test_openmeteo_kind_daily():
    df = pd.DataFrame({
        "date_et": ["2024-01-01T00:00:00-05:00"]
    })

    kind, dates = _openmeteo_kind_and_dates(df)

    assert kind == "daily"
    assert dates.iloc[0] == pd.Timestamp("2024-01-01")


def test_openmeteo_kind_hourly():
    df = pd.DataFrame({
        "hour_et": ["2024-01-01T12:00:00-05:00"]
    })

    kind, dates = _openmeteo_kind_and_dates(df)

    assert kind == "hourly"
    assert dates.iloc[0] == pd.Timestamp("2024-01-01")


def test_openmeteo_kind_missing_column():
    df = pd.DataFrame({
        "wrong_col": ["2024-01-01"]
    })

    with pytest.raises(ValueError):
        _openmeteo_kind_and_dates(df)
def test_merge_frames():
    aqi = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "AQI": [10, 20, 30]
    })

    meteo = pd.DataFrame({
        "date_et": pd.date_range("2024-01-01", periods=3),
        "temperature": [1,2,3]
    })

    result = merge_frames(aqi, meteo, how="inner")

    assert isinstance(result, pd.DataFrame)

def test_merge_left_join():
    import pandas as pd
    from data_processing.merge_datasets import merge_frames

    aqi = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "AQI": [10, 20, 30]
    })

    meteo = pd.DataFrame({
        "date_et": pd.date_range("2024-01-01", periods=3),
        "temperature": [1, 2, 3]
    })

    result = merge_frames(aqi, meteo, how="left")

    assert len(result) >= 3


def test_merge_outer_join():
    import pandas as pd
    from data_processing.merge_datasets import merge_frames

    aqi = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=2),
        "AQI": [10, 20]
    })

    meteo = pd.DataFrame({
        "date_et": pd.date_range("2024-01-02", periods=2),
        "temperature": [2, 3]
    })

    result = merge_frames(aqi, meteo, how="outer")

    assert isinstance(result, pd.DataFrame)

def test_write_derived_outputs(tmp_path, monkeypatch):
    import pandas as pd
    from data_processing import merge_datasets

    # 临时把 DATA_DIR 改到 tmp_path，避免写进真实项目 data 文件夹
    monkeypatch.setattr(merge_datasets, "DATA_DIR", tmp_path)

    aqi_path = tmp_path / "aqi.csv"

    aqi_df = pd.DataFrame({
        "State Name": ["New York", "New York", "California"],
        "county Name": ["Tompkins", "Albany", "Tompkins"],
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "AQI": [40, 50, 60],
    })

    aqi_df.to_csv(aqi_path, index=False)

    merged = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "AQI": [40, 45, 50],
        "temperature": [1, 2, 3],
    })

    merge_datasets.write_derived_outputs(merged, aqi_path)

    daily_air_quality = tmp_path / "processed" / "daily_air_quality.csv"
    sample_modeling_table = tmp_path / "sample" / "sample_modeling_table.csv"

    assert daily_air_quality.exists()
    assert sample_modeling_table.exists()

    daily_df = pd.read_csv(daily_air_quality)
    sample_df = pd.read_csv(sample_modeling_table)

    assert len(daily_df) == 1
    assert daily_df["county Name"].iloc[0] == "Tompkins"
    assert len(sample_df) == 3

def test_load_tompkins_aqi_bad_date(tmp_path):
    import pandas as pd
    import pytest
    from data_processing.merge_datasets import load_tompkins_aqi

    csv_path = tmp_path / "bad_date.csv"

    df = pd.DataFrame({
        "State Name": ["New York"],
        "county Name": ["Tompkins"],
        "Date": ["not-a-date"],
        "AQI": [40],
    })

    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_tompkins_aqi(csv_path)

def test_main_success(tmp_path, monkeypatch):
    import pandas as pd
    from data_processing import merge_datasets

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    aqi_path = raw_dir / "aqi.csv"
    weather_path = raw_dir / "weather.csv"
    out_path = tmp_path / "processed" / "modeling_table.csv"

    pd.DataFrame({
        "State Name": ["New York"],
        "county Name": ["Tompkins"],
        "Date": ["2024-01-01"],
        "AQI": [40],
    }).to_csv(aqi_path, index=False)

    pd.DataFrame({
        "date_et": ["2024-01-01T00:00:00-05:00"],
        "temperature": [10],
    }).to_csv(weather_path, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "merge_datasets.py",
            "--aqi", str(aqi_path),
            "--weather", str(weather_path),
            "--out", str(out_path),
            "--no-extra-outputs",
        ],
    )

    monkeypatch.setattr(
        merge_datasets,
        "clean_modeling_table",
        lambda df: df,
    )

    result = merge_datasets.main()

    assert result == 0
    assert out_path.exists()