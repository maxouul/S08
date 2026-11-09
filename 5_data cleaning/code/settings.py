"""
Shared settings for part 5 (data cleaning and preparation): where it reads and where
it writes.

Every script in code/ takes its default paths from here, so the folders are
defined once. Override the two roots with environment variables (on a laptop, a WRDS
server or any other machine) instead of editing paths:
  TEAM_DIR   the private data root (default: the repository root)
  STEP5_DIR  this folder (default: TEAM_DIR / "5_data cleaning")

Part 5 writes only inside STEP5_DIR. It reads, without changing them:
  files csv/            parts 1-2's shared tables (1_2nov/nov2_data instead, when present):
                        extraction_report.csv, the document fetch_releases.py chose for
                        each filing (accession, cik, filing_date, source, filename), and
                        event_table.csv for the company names
  1_2nov/code/          fetch_releases.py, whose download and choice rules are reused
  3_4_surprise_returns  data/events_final.parquet (event keys, reaction dates, splits and
                        sample flags), data/clock_provisional.csv (permno and dates of
                        every filing), data/trading_calendar.csv, the WRDS pulls
                        cache/crsp_names.parquet and cache/link_ibes_crsp.parquet, and
                        SPLITS in code/common.py

Parts 6 to 10 read what part 5 writes (see the README): the cleaned texts, the pieces
cut for each model, the model-output cache and the frozen holdout.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEAM_DIR = Path(os.environ.get("TEAM_DIR", REPO_ROOT)).expanduser().resolve()
STEP5_DIR = Path(os.environ.get("STEP5_DIR", TEAM_DIR / "5_data cleaning")).expanduser().resolve()

CODE = REPO_ROOT / "5_data cleaning" / "code"
CACHE = STEP5_DIR / "cache"        # raw EDGAR bytes and model outputs
DATA = STEP5_DIR / "data"          # built tables keyed by accession
RESULTS = STEP5_DIR / "results"    # tables for the report (no row-level data)
LOGS = STEP5_DIR / "logs"

# Inputs from parts 1-2, found by the same rule as scripts/project_paths.py
NOV2_DATA = TEAM_DIR / "1_2nov" / "nov2_data"
if not NOV2_DATA.is_dir():
    NOV2_DATA = TEAM_DIR / "files csv"
NOV2_CODE = REPO_ROOT / "1_2nov" / "code"   # part 1 pipeline scripts
EVENT_TABLE = NOV2_DATA / "event_table.csv"
EXTRACTION_REPORT = NOV2_DATA / "extraction_report.csv"
# The report of fetch_releases.py names the document chosen for each filing. A rerun of
# fetch_releases.py writes extraction_report_full.csv; the shared extraction_report.csv
# has the same columns. The first one found is used.
FETCH_REPORT_CANDIDATES = [NOV2_DATA / "extraction_report_full.csv", EXTRACTION_REPORT]
FETCH_REPORT_COLUMNS = {"accession", "cik", "filename", "source"}

# Inputs from parts 3-4
PART34 = TEAM_DIR / "3_4_surprise_returns"
EVENTS_FINAL = PART34 / "data" / "events_final.parquet"   # WRDS-derived: private
CLOCK = PART34 / "data" / "clock_provisional.csv"
CALENDAR = PART34 / "data" / "trading_calendar.csv"
CRSP_NAMES = PART34 / "cache" / "crsp_names.parquet"      # WRDS data: private
IBES_LINK = PART34 / "cache" / "link_ibes_crsp.parquet"   # WRDS data: private

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
HOLDOUT_FREEZE = DATA / "holdout_freeze.json"
MANIFEST = STEP5_DIR / "manifest.json"

# fetch_releases.py is imported from 1_2nov/code/, not copied.
if str(NOV2_CODE) not in sys.path:
    sys.path.append(str(NOV2_CODE))


def fetch_report() -> Path | None:
    """The report of fetch_releases.py naming each filing's document, or None, in which
    case cache_raw.py reads every index page again to find it."""
    for p in FETCH_REPORT_CANDIDATES:
        if p.exists():
            with open(p, encoding="utf-8-sig") as f:
                header = {c.strip().strip('"') for c in f.readline().split(",")}
            if FETCH_REPORT_COLUMNS <= header:
                return p
    return None


def common_3_4():
    """Part 3-4's common.py, loaded from its file under another module name so it
    never shadows anything here. SPLITS and split_of come from it, so the texts and
    the returns are split by the same rule."""
    path = REPO_ROOT / "3_4_surprise_returns" / "code" / "common.py"
    spec = importlib.util.spec_from_file_location("common_3_4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
