"""
Shared settings for part 5: where it reads and where it writes.

Every script in code/ takes its default paths from here, so the folders are
defined once. Override the two roots with environment variables (on Colab, WRDS
or a laptop) instead of editing paths:
  TEAM_DIR   the shared "S08 NLP" folder (default: the shortcut in My Drive)
  STEP5_DIR  this folder (default: TEAM_DIR / "5_data cleaning")

Part 5 writes only inside STEP5_DIR. It reads, without changing them:
  1_2nov/nov2_data      event_table.csv, extraction_report*.csv, three_source_timing.csv
  1_2nov/files .py      fetch_releases.py, whose download and choice rules are reused
  3_4_surprise_returns  the release clock, the trading calendar, the CRSP names and
                        I/B/E/S link already pulled from WRDS, and SPLITS in common.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

TEAM_DIR = Path(os.environ.get("TEAM_DIR", "/content/drive/MyDrive/S08 NLP"))
STEP5_DIR = Path(os.environ.get("STEP5_DIR", TEAM_DIR / "5_data cleaning"))

CODE = STEP5_DIR / "code"
CACHE = STEP5_DIR / "cache"        # raw EDGAR bytes and model outputs
DATA = STEP5_DIR / "data"          # built tables keyed by accession
RESULTS = STEP5_DIR / "results"    # tables for the report
LOGS = STEP5_DIR / "logs"

# Inputs from the other parts
NOV2_DATA = TEAM_DIR / "1_2nov" / "nov2_data"
NOV2_CODE = TEAM_DIR / "1_2nov" / "files .py"
PART34 = TEAM_DIR / "3_4_surprise_returns"
CLOCK = PART34 / "data" / "clock_provisional.csv"
CALENDAR = PART34 / "data" / "trading_calendar.csv"
CRSP_NAMES = PART34 / "cache" / "crsp_names.parquet"      # WRDS data: private
IBES_LINK = PART34 / "cache" / "link_ibes_crsp.parquet"   # WRDS data: private
EVENT_TABLE = NOV2_DATA / "event_table.csv"
TIMING = NOV2_DATA / "three_source_timing.csv"
EXTRACTION_REPORT = NOV2_DATA / "extraction_report.csv"   # shared copy: no accession

# The full report of fetch_releases.py (accession, cik, filename) is not in the
# shared folder yet; put it in either place and cache_raw.py uses it.
EXTRACTION_FULL_CANDIDATES = [DATA / "extraction_report_full.csv",
                              NOV2_DATA / "extraction_report_full.csv"]

# Outputs
RAW_DIR = CACHE / "raw"
RAW_MANIFEST = CACHE / "raw_manifest.parquet"
MODEL_CACHE = CACHE / "model_outputs.sqlite"
CLEAN_DIR = DATA / "clean"
ENTITIES = DATA / "entities.parquet"
CHUNK_DIR = DATA / "chunks"      # pieces per model: the tokenized inputs
TOKEN_STATS = DATA / "token_stats.parquet"
TOKEN_QUANTILES = RESULTS / "token_quantiles.csv"
DEDUP_FLAGS = DATA / "dedup_flags.parquet"
SPLITS_DRAFT = DATA / "splits_draft.parquet"
HOLDOUT_IDS = DATA / "holdout_ids.txt"
MANIFEST = STEP5_DIR / "manifest.json"

# fetch_releases.py is imported from part 1-2's folder, not copied.
if str(NOV2_CODE) not in sys.path:
    sys.path.append(str(NOV2_CODE))


def extraction_full() -> Path | None:
    return next((p for p in EXTRACTION_FULL_CANDIDATES if p.exists()), None)


def common_3_4():
    """Part 3-4's common.py, loaded from its file under another module name so it
    never shadows anything here. SPLITS and split_of come from it, so the texts and
    the returns are split by the same rule."""
    path = PART34 / "code" / "common.py"
    spec = importlib.util.spec_from_file_location("common_3_4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
