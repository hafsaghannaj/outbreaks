from pathlib import Path

# Project root = .../Outbreaks
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Output folders
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DATA_DIR = DOCS_DIR / "data"

# Ensure dirs exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Synthetic dataset defaults (MVP)
DEFAULT_N_POINTS = 500
DEFAULT_RANDOM_SEED = 42

# Risk score bounds
RISK_MIN = 0.0
RISK_MAX = 100.0
