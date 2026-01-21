from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.settings import RESULTS_DIR

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


def _make_model():
    try:
        from xgboost import XGBRegressor
        return (
            XGBRegressor(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
            ),
            "XGBRegressor",
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=42), "GradientBoostingRegressor"


def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model, model_name = _make_model()
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


def save_model(model, model_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "XGBRegressor":
        model_path = out_dir / "model.xgb.json"
        model.save_model(str(model_path))
        return model_path

    import joblib
    model_path = out_dir / "model.sklearn.joblib"
    joblib.dump(model, model_path)
    return model_path


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "synthetic_training_data.csv")
    model, report = train_model(df)

    model_path = save_model(model, report["model"], RESULTS_DIR)
    report["model_artifact"] = model_path.name

    with open(RESULTS_DIR / "model_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Trained:", report["model"])
    print("Saved model:", model_path)
    print("Saved report:", RESULTS_DIR / "model_report.json")
    print("Metrics:", report["metrics"])


if __name__ == "__main__":
    main()
