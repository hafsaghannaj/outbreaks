from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import RESULTS_DIR, RISK_MIN, RISK_MAX

# Same feature set used in training
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


def load_model():
    # Re-train quickly for MVP simplicity (we'll persist model artifacts later)
    # This keeps it deterministic given the same dataset.
    from src.models.train import train_model

    df = pd.read_csv(RESULTS_DIR / "synthetic_training_data.csv")
    model, report = train_model(df)
    return model


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "synthetic_training_data.csv")

    model = load_model()
    preds = model.predict(df[FEATURE_COLS])

    preds = np.clip(preds, RISK_MIN, RISK_MAX)

    scored = df[["lat", "lon", "date"]].copy()
    scored["predicted_risk_score"] = preds

    out_path = RESULTS_DIR / "risk_scored_points.csv"
    scored.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print("Rows:", len(scored))
    print("Pred min/max:", float(scored["predicted_risk_score"].min()), float(scored["predicted_risk_score"].max()))


if __name__ == "__main__":
    main()
