from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
OUTPUT_DIR = ROOT / "outputs"
EXPERIMENTS_DIR = ROOT / "experiments"
TESTS_DIR = ROOT / "tests"

CM500_DIR = DATA_DIR / "CM500"

LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

LOG_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)