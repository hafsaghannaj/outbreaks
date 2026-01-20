from __future__ import annotations

import argparse
from pathlib import Path

from src.config.settings import RESULTS_DIR, DEFAULT_N_POINTS, DEFAULT_RANDOM_SEED
from src.data.synthetic import SyntheticConfig, generate_synthetic_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic multimodal dataset for Outbreaks MVP.")
    p.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS)
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--out", type=str, default=str(RESULTS_DIR / "synthetic_training_data.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SyntheticConfig(n_points=args.n_points, random_seed=args.seed)
    df = generate_synthetic_dataset(cfg)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print("Shape:", df.shape)
    print("Risk min/max:", float(df["risk_score"].min()), float(df["risk_score"].max()))


if __name__ == "__main__":
    main()
