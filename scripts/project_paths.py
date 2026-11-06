"""Shared project layout for scripts and notebooks.

All defaults live inside the checkout. TEAM_DIR optionally relocates data;
PROJECT_DIR and STEP5_DIR can override the two stage output directories.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
TEAM_DIR = Path(os.environ.get("TEAM_DIR", REPO_ROOT)).expanduser().resolve()
DATA_DIR = TEAM_DIR / "files csv"
NOV2_DATA = DATA_DIR  # compatibility alias for the parts 1-2 notebooks
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", TEAM_DIR / "3_4_surprise_returns")).expanduser().resolve()
STEP5_DIR = Path(os.environ.get("STEP5_DIR", TEAM_DIR / "5_data cleaning")).expanduser().resolve()
PART34_CODE = REPO_ROOT / "3_4_surprise_returns" / "code"
STEP5_CODE = REPO_ROOT / "5_data cleaning" / "code"
RESULTS_DIR = TEAM_DIR / "results"
MODEL_DIR = TEAM_DIR / "6_Models, GPU setup and inference"


def data_path(name):
    return str(DATA_DIR / name)
