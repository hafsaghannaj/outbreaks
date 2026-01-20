from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n_points: int = 500
    random_seed: int = 42
    # Default bbox loosely around a coastal region (can change later)
    min_lat: float = 28.0
    max_lat: float = 38.0
    min_lon: float = -5.0
    max_lon: float = 5.0
    start_date: date = date(2023, 1, 1)
    end_date: date = date(2024, 12, 31)


def _random_dates(rng: np.random.Generator, n: int, start: date, end: date) -> np.ndarray:
    days = (end - start).days
    offsets = rng.integers(0, days + 1, size=n)
    return np.array([start + timedelta(days=int(o)) for o in offsets], dtype="datetime64[D]")


def generate_synthetic_dataset(cfg: SyntheticConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)

    lat = rng.uniform(cfg.min_lat, cfg.max_lat, size=cfg.n_points)
    lon = rng.uniform(cfg.min_lon, cfg.max_lon, size=cfg.n_points)
    dt = _random_dates(rng, cfg.n_points, cfg.start_date, cfg.end_date)

    # --- Environmental precursors (synthetic but plausible patterns) ---
    # Sea surface temperature proxy (warmer near lower latitudes + seasonal component)
    doy = pd.to_datetime(dt).dayofyear.to_numpy()
    sst = 22 + 6 * np.cos((doy - 200) * 2 * np.pi / 365.25) + (38 - lat) * 0.15 + rng.normal(0, 0.8, cfg.n_points)

    # Precipitation (skewed) + flood proxy (heavy rain increases flood risk)
    precip = rng.gamma(shape=2.0, scale=5.0, size=cfg.n_points)  # mm/day-ish
    flood = np.clip((precip - 8) / 12 + rng.normal(0, 0.15, cfg.n_points), 0, 1)

    # Chlorophyll-a proxy (algae bloom), loosely linked to warmth + runoff/flooding
    chl = np.clip(1.2 + 0.08 * (sst - 24) + 1.0 * flood + rng.normal(0, 0.25, cfg.n_points), 0, None)

    # Drought index proxy (higher = drier); inversely related to recent precip
    drought = np.clip(2.0 - 0.06 * precip + rng.normal(0, 0.25, cfg.n_points), 0, 4)

    # --- Socio-economic / infrastructure (synthetic) ---
    pop_density = np.clip(rng.lognormal(mean=4.0, sigma=0.6, size=cfg.n_points), 10, 5000)  # people/km^2
    improved_water_access = np.clip(rng.beta(a=3, b=2, size=cfg.n_points), 0, 1)  # 0..1
    sanitation_access = np.clip(rng.beta(a=2.5, b=2.5, size=cfg.n_points), 0, 1)  # 0..1

    # Mobility/sentinel proxy: spikes after flooding (displacement)
    mobility_disruption = np.clip(0.2 + 0.9 * flood + rng.normal(0, 0.1, cfg.n_points), 0, 1)

    # --- Construct a synthetic "true" risk (0..100) ---
    # Higher risk with: warm water + flooding + chlorophyll + higher pop density
    # Lower risk with: improved water access + sanitation
    raw = (
        0.9 * (sst - 20)
        + 18.0 * flood
        + 6.0 * np.log1p(chl)
        + 3.0 * np.log1p(pop_density)
        - 22.0 * improved_water_access
        - 18.0 * sanitation_access
        + 6.0 * mobility_disruption
        + 2.0 * drought
        + rng.normal(0, 2.5, cfg.n_points)
    )

    # Squash + scale to 0..100
    risk = 100 * (1 / (1 + np.exp(-(raw - np.mean(raw)) / (np.std(raw) + 1e-6))))

    df = pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "date": pd.to_datetime(dt).astype("datetime64[ns]"),
            "sst_proxy": sst,
            "precip_mm": precip,
            "flood_proxy": flood,
            "chlorophyll_proxy": chl,
            "drought_index": drought,
            "pop_density": pop_density,
            "improved_water_access": improved_water_access,
            "sanitation_access": sanitation_access,
            "mobility_disruption": mobility_disruption,
            "risk_score": risk,
        }
    )

    return df


def main() -> None:
    cfg = SyntheticConfig()
    df = generate_synthetic_dataset(cfg)
    print("rows:", len(df), "cols:", df.shape[1])
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
