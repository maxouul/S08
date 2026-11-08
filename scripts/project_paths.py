"""Shared project layout for scripts and notebooks.

All defaults live inside the checkout. TEAM_DIR optionally relocates data;
PROJECT_DIR, STEP5_DIR and STEP78_DIR can override the stage output directories.

Every part follows the same layout:
  <part>/code     Python modules (always read from the checkout)
  <part>/data     inputs and built tables (private data, ignored by Git)
  <part>/cache    downloads and intermediate files (ignored by Git)
  <part>/results  tables and figures for the report
  <part>/logs     run logs
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
TEAM_DIR = Path(os.environ.get("TEAM_DIR", REPO_ROOT)).expanduser().resolve()
DATA_DIR = TEAM_DIR / "files csv"
NOV2_DATA = DATA_DIR  # compatibility alias for the parts 1-2 notebooks

PART1_DIR = TEAM_DIR / "1_2nov"
PART1_CODE = REPO_ROOT / "1_2nov" / "code"
PART2_DIR = TEAM_DIR / "2_EDGAR text"
PART2_CODE = REPO_ROOT / "2_EDGAR text" / "code"
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", TEAM_DIR / "3_4_surprise_returns")).expanduser().resolve()
STEP5_DIR = Path(os.environ.get("STEP5_DIR", TEAM_DIR / "5_data cleaning")).expanduser().resolve()
PART34_CODE = REPO_ROOT / "3_4_surprise_returns" / "code"
STEP5_CODE = REPO_ROOT / "5_data cleaning" / "code"
STEP78_DIR = Path(os.environ.get("STEP78_DIR", TEAM_DIR / "7_8_baselines_evaluation")).expanduser().resolve()
STEP78_CODE = REPO_ROOT / "7_8_baselines_evaluation" / "code"
MODEL_DIR = TEAM_DIR / "6_Models, GPU setup and inference"


def data_path(name):
    return str(DATA_DIR / name)
