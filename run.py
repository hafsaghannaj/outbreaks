from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Outbreaks MVP pipeline runner")
    p.add_argument("--n-points", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    # Publishing filter: only show points >= this on the published map
    p.add_argument("--min-risk", type=float, default=0.0)

    # High-risk layer threshold (layer toggle cutoff)
    p.add_argument("--high-threshold", type=float, default=65.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    run([sys.executable, "-m", "src.data.make_dataset", "--n-points", str(args.n_points), "--seed", str(args.seed)])
    run([sys.executable, "-m", "src.models.train"])
    run([sys.executable, "-m", "src.models.score"])
    run([sys.executable, "-m", "src.models.diagnostics"])

    # Map now supports layer threshold
    run([sys.executable, "-m", "src.viz.make_map", "--high-threshold", str(args.high_threshold)])

    run([
        "bash", "-lc",
        "mkdir -p docs/data docs/assets "
        "&& cp results/risk_map.html docs/index.html "
        "&& cp results/risk_scored_points.csv docs/data/risk_scored_points.csv "
        "&& cp results/model_report.json docs/data/model_report.json "
        "&& cp results/model_diagnostics_fit.png docs/assets/model_diagnostics_fit.png "
        "&& cp results/model_diagnostics_residuals.png docs/assets/model_diagnostics_residuals.png"
    ])

    print("\n✅ Pipeline complete. Open:")
    print(" - docs/index.html")
    print(f" - High-risk layer threshold: {args.high_threshold}")


if __name__ == "__main__":
    main()
