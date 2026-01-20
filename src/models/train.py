from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.settings import RESULTS_DIR

# Try XGBoost, else fallback to sklearn GradientBoosting
USE_XGBOOST = True
try:
    from xgboost import XGBRegressor
except Exception:
    USE_XGBOOST = False

from sklearn.ensemble import GradientBoostingRegressor


FEATURE_COLS = [
    "sst_proxy",
    "precip_mm",
    "flood_proxy",
    "chlorophyll_proxy",
    "drought_index",
    "pop_density",
    "improved_water_access",
    "sanitation_access",
    "mobility_disruption",
]

TARGET_COL = "risk_score"


def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if USE_XGBOOST:
        model = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
        )
        model_name = "XGBRegressor"
    else:
        model = GradientBoostingRegressor(random_state=42)
        model_name = "GradientBoostingRegressor"

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    report = {
        "model": model_name,
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "metrics": {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)},
    }

    return model, report


def main() -> None:
    data_path = RESULTS_DIR / "synthetic_training_data.csv"
    df = pd.read_csv(data_path)

    model, report = train_model(df)

    report_path = RESULTS_DIR / "model_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Trained:", report["model"])
    print("Saved report:", report_path)
    print("Metrics:", report["metrics"])


if __name__ == "__main__":
    main()
