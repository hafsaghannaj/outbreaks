from __future__ import annotations

import json
import numpy as np
import pandas as pd

from src.config.settings import RESULTS_DIR, RISK_MIN, RISK_MAX

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


def load_saved_model():
    report_path = RESULTS_DIR / "model_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing {report_path}. Run: python -m src.models.train"
        )

    with open(report_path, "r") as f:
        report = json.load(f)

    model_name = report.get("model")
    artifact = report.get("model_artifact")
    if not artifact:
        raise FileNotFoundError(
            "model_artifact not found in model_report.json. Re-run training."
        )

    artifact_path = RESULTS_DIR / artifact
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {artifact_path}")

    if model_name == "XGBRegressor":
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.load_model(str(artifact_path))
        return model

    import joblib
    return joblib.load(artifact_path)


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "synthetic_training_data.csv")

    model = load_saved_model()
    preds = model.predict(df[FEATURE_COLS].to_numpy())

    preds = np.clip(preds, RISK_MIN, RISK_MAX)

    scored = df[["lat", "lon", "date"]].copy()
    scored["predicted_risk_score"] = preds

    out_path = RESULTS_DIR / "risk_scored_points.csv"
    scored.to_csv(out_path, index=False)

    print("Loaded saved model and scored points.")
    print("Wrote:", out_path)
    print("Rows:", len(scored))
    print(
        "Pred min/max:",
        float(scored["predicted_risk_score"].min()),
        float(scored["predicted_risk_score"].max()),
    )


if __name__ == "__main__":
    main()
