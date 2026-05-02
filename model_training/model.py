import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def load_data(path="feature_engineering/features_table.csv"):
    return pd.read_csv(path)


def prepare_features(df, target="AQI"):
    numeric_df = df.select_dtypes(include=["number"])
    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def plot_actual_vs_predicted(y_test, y_pred, title):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], "r-")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title(title)
    plt.show()


def plot_feature_importance(model, feature_names, title):
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(by="Importance", ascending=False).head(10)

    plt.figure(figsize=(8, 5))
    plt.barh(importance["Feature"], importance["Importance"])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

    return importance


def run_model_pipeline(data_path="feature_engineering/features_table.csv", show_plots=True):
    df = load_data(data_path)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    rf = train_random_forest(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_results = evaluate_model(y_test, y_pred_rf)

    xgb = train_xgboost(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    xgb_results = evaluate_model(y_test, y_pred_xgb)

    results = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "MSE": [rf_results["MSE"], xgb_results["MSE"]],
        "MAE": [rf_results["MAE"], xgb_results["MAE"]],
        "R2": [rf_results["R2"], xgb_results["R2"]]
    })

    rf_importance = plot_feature_importance(
        rf, X.columns, "Top 10 Feature Importance - Random Forest"
    ) if show_plots else None

    xgb_importance = plot_feature_importance(
        xgb, X.columns, "Top 10 Feature Importance - XGBoost"
    ) if show_plots else None

    if show_plots:
        plot_actual_vs_predicted(y_test, y_pred_rf, "Random Forest: Actual vs Predicted AQI")
        plot_actual_vs_predicted(y_test, y_pred_xgb, "XGBoost: Actual vs Predicted AQI")

    return results, rf_importance, xgb_importance


if __name__ == "__main__":
    results, rf_importance, xgb_importance = run_model_pipeline()
    print(results)
