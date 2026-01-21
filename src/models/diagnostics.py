from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import RESULTS_DIR
from src.models.train import FEATURE_COLS, TARGET_COL, _make_model


def save_fit_plot(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.xlabel("True risk_score")
    plt.ylabel("Predicted risk_score")
    plt.title("Model Fit: True vs Predicted")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def save_residual_plot(residuals: np.ndarray, out_path: Path) -> None:
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (true - predicted)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def get_feature_importance(model, model_name: str) -> dict:
    if model_name == "XGBRegressor":
        # XGBoost: feature_importances_ aligns with input column order
        importances = model.feature_importances_.tolist()
        return {c: float(i) for c, i in zip(FEATURE_COLS, importances)}

    # sklearn GradientBoosting has feature_importances_
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_.tolist()
        return {c: float(i) for c, i in zip(FEATURE_COLS, importances)}

    return {c: 0.0 for c in FEATURE_COLS}


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "synthetic_training_data.csv")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model, model_name = _make_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    residuals = (y_test.to_numpy() - preds)

    # Save plots
    fit_path = RESULTS_DIR / "model_diagnostics_fit.png"
    resid_path = RESULTS_DIR / "model_diagnostics_residuals.png"
    save_fit_plot(y_test.to_numpy(), preds, fit_path)
    save_residual_plot(residuals, resid_path)

    # Update model_report.json with feature importance + diagnostics filenames
    report_path = RESULTS_DIR / "model_report.json"
    with open(report_path, "r") as f:
        report = json.load(f)

    report["diagnostics"] = {
        "fit_plot": fit_path.name,
        "residuals_plot": resid_path.name,
    }
    report["feature_importance"] = get_feature_importance(model, model_name)

    # Add a sorted top-5 list for readability
    top5 = sorted(report["feature_importance"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    report["top_features"] = [{"feature": k, "importance": float(v)} for k, v in top5]

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Wrote diagnostics:")
    print(" -", fit_path)
    print(" -", resid_path)
    print("Updated report with feature importance:", report_path)


if __name__ == "__main__":
    main()
