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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Generate dataset
    run([sys.executable, "-m", "src.data.make_dataset", "--n-points", str(args.n_points), "--seed", str(args.seed)])

    # 2) Train + persist model artifact
    run([sys.executable, "-m", "src.models.train"])

    # 3) Score using saved model
    run([sys.executable, "-m", "src.models.score"])

    # 4) Diagnostics + feature importance (writes pngs + updates model_report.json)
    run([sys.executable, "-m", "src.models.diagnostics"])

    # 5) Render map to results/risk_map.html
    run([sys.executable, "-m", "src.viz.make_map"])

    # 6) Copy outputs into docs/ for GitHub Pages
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
    print(" - results/risk_map.html")
    print(" - docs/index.html")
    print(" - docs/assets/model_diagnostics_fit.png")
    print(" - docs/assets/model_diagnostics_residuals.png")


if __name__ == "__main__":
    main()
