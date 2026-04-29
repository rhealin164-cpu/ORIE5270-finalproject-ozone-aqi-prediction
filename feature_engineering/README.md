# Feature Engineering (Tompkins County, NY)

Feature engineering module that takes the cleaned modeling table and
produces a ready-to-use feature matrix for AQI prediction.

## Folder layout

```
feature_engineering/
├── README.md             # this file
├── features.py           # feature engineering pipeline
└── features_table.csv    # ready-to-use table (35 features + AQI)
```

## Requirements

- Python 3.10+
- pandas
- numpy

```
pip install pandas numpy
```

## Quick start

From the project root:

```
python feature_engineering/features.py
```

This will generate `features_table.csv` under `feature_engineering/`.

## Usage

```python
from feature_engineering.features import build_features, get_X_y, save_features_table

# Option 1: load ready-to-use table directly
import pandas as pd
df = pd.read_csv("feature_engineering/features_table.csv")
X = df.drop(columns=["AQI"])
y = df["AQI"]

# Option 2: run pipeline from scratch
raw = pd.read_csv("data_processing/data/processed/modeling_table.csv")
feature_df = build_features(raw)
X, y = get_X_y(feature_df)
```

## Engineered / derived features (19 columns)

| Category | Features |
|----------|----------|
| Lag | aqi_lag_1, aqi_lag_2, aqi_lag_3, aqi_lag_7 |
| Rolling | aqi_roll_mean_3, aqi_roll_mean_7, aqi_roll_std_7 |
| Trend | aqi_diff_1, aqi_diff_7 |
| Time | month, weekday, season, month_sin, month_cos, weekday_sin, weekday_cos |
| Weather-derived | temp_range, wind_dir_sin, wind_dir_cos |

## Output: features_table.csv

- Shape: 581 rows × 36 columns (35 features + AQI)
- Non-predictive columns removed (metadata, date, sunrise/sunset, raw wind direction)
- No missing values (NaN)
- Target column AQI is placed as the last column
