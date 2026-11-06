"""
Shared settings for parts 7 (baselines and model comparisons) and 8 (financial
evaluation and statistical analysis).

Why a settings module
---------------------
Every other part of the project keeps its paths and constants in one place
(``3_4_surprise_returns/code/common.py``, ``5_data cleaning/code/settings.py``).
We follow the same convention so that:

* nobody edits paths inside the analysis code (they change between Colab,
  WRDS cloud and a laptop);
* the sample splits, the holdout lock and the outcome definitions are taken
  **from the parts that own them**, never restated here. If Beatriz changes the
  split rule in ``common.py``, parts 7-8 follow automatically.

Override the roots with environment variables instead of editing this file::

    TEAM_DIR   the shared "S08 NLP" folder
               (default: the folder that contains this repository, which is
               right both on a laptop and on Google Drive, since the layout is
               the same)
    STEP78_DIR this folder (default: TEAM_DIR / "7_8_baselines_evaluation")

Parts 7-8 write only inside STEP78_DIR. They read, without changing them:

    3_4_surprise_returns/data/events_final.parquet   the event-level table
    3_4_surprise_returns/code/common.py              SPLITS, split_of, guard_holdout
    3_4_surprise_returns/code/returns.py             newey_west_se, rank_ic_by_season
    3_4_surprise_returns/cache/crsp_names.parquet    SIC codes for sectors
    5_data cleaning/data/clean/clean_*.parquet       cleaned release texts
    5_data cleaning/cache/model_outputs.sqlite       model scores (part 6)
    5_data cleaning/data/holdout_ids.txt             frozen test set (once it exists)
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent            # .../7_8_baselines_evaluation/code
STEP78_DIR = Path(os.environ.get("STEP78_DIR", _HERE.parent))
TEAM_DIR = Path(os.environ.get("TEAM_DIR", STEP78_DIR.parent))

CODE = STEP78_DIR / "code"
DATA = STEP78_DIR / "data"            # row-level, WRDS-derived: never committed
RESULTS = STEP78_DIR / "results"      # tables and figures: summary statistics only
TABLES = RESULTS / "tables"
REGRESSIONS = RESULTS / "regressions"
FIGURES = RESULTS / "figures"
LOGS = STEP78_DIR / "logs"
RESOURCES = STEP78_DIR / "resources"

# Inputs owned by other parts ------------------------------------------------
PART34 = TEAM_DIR / "3_4_surprise_returns"
STEP5 = TEAM_DIR / "5_data cleaning"

EVENTS = PART34 / "data" / "events_final.parquet"
CRSP_NAMES = PART34 / "cache" / "crsp_names.parquet"          # WRDS data: private
CLEAN_DIR = STEP5 / "data" / "clean"                          # clean_{year}.parquet
SPLITS_DRAFT = STEP5 / "data" / "splits_draft.parquet"
HOLDOUT_IDS = STEP5 / "data" / "holdout_ids.txt"
MODEL_CACHE = STEP5 / "cache" / "model_outputs.sqlite"

# Older text export from part 2 (accession, text). Used only when part 5 has not
# cleaned a filing yet, and always marked as such (text_source column).
RELEASE_TEXTS_FALLBACK = [TEAM_DIR / "results" / "2_results" / "release_texts.csv",
                          TEAM_DIR / "files csv" / "release_texts.csv"]

# The Loughran-McDonald master dictionary is licensed for academic use and is not
# redistributed here: download it once into resources/lm/ (see the README there).
LM_DIR = RESOURCES / "lm"

# Outputs of parts 7-8 -------------------------------------------------------
PANEL = DATA / "analysis_panel.parquet"          # events + texts metadata + controls
SIGNALS = DATA / "signals.parquet"               # one column per text/surprise signal
TEXT_FEATURES = DATA / "text_features"           # cached tokenised counts per year
BOOK_TO_MARKET = DATA / "book_to_market.parquet" # event_id -> Compustat ceqq / mktcap (book_to_market.py)
RUN_LOG = LOGS / "runs.jsonl"


# ---------------------------------------------------------------------------
# Research design constants (keep in sync with docs/METHODOLOGY.md)
# ---------------------------------------------------------------------------
# Outcomes. All are built by part 4 (returns.py); we never recompute returns.
#   car_0_1_ff4      reaction on days 0 and +1: "is the text priced in at release?"
#   post_car_1_ff4   from the first price after R to the close of +1
#   post_car_60_ff4  from the first price after R to the close of +60 ("drift")
#   car_1_60_ff4     close of day 0 to +60: drift that does not depend on the open
#   ar_m1            day -1 abnormal return: a PLACEBO. A text signal that "predicts"
#                    it is leaking timing errors, not information.
PRIMARY_OUTCOME = "car_0_1_ff4"
OUTCOMES = ["car_0_1_ff4", "post_car_1_ff4", "post_car_60_ff4", "car_1_60_ff4"]
PLACEBO_OUTCOMES = ["ar_m1"]
MKT_ADJ_OUTCOMES = ["car_0_1_mkt", "post_car_1_mkt", "post_car_60_mkt"]

# The earnings-surprise benchmark. sue_price is part 3's primary definition
# ((actual - median consensus) / price at t-2); sue_disp is its alternative.
SUE = "sue_price"
SUE_ALT = "sue_disp"

# Default sample: part 4's pre-registered primary sample. in_text_sample is the
# robustness sample (all events with text, whatever the timing status).
SAMPLE = os.environ.get("STEP78_SAMPLE", "in_primary")

# Splits (labels come from common.py; only the roles are set here).
TRAIN_SPLIT = "dev"            # fit models
TUNE_SPLIT = "validation"      # choose hyper-parameters, compare models
TEST_SPLIT = "holdout"         # touched once, after the freeze (UNLOCK_HOLDOUT=1)

# Winsorisation of continuous variables before OLS (not before rank statistics):
# 1st/99th percentile within each announcement season, so a crash quarter does
# not set the cut-offs for a calm one.
WINSOR = (0.01, 0.99)

# Minimum events in a season for a per-season statistic (same as part 4).
MIN_SEASON_N = 30
# Newey-West lags across seasons: 60-day windows overlap the next season, so two
# lags (part 4 uses the same choice).
NW_LAGS = 2

# The 2022 rate shock. The FOMC raised rates on 16 March 2022 (first hike since
# 2018) and made its last hike of the cycle on 26 July 2023. These dates define
# the "shock" regime; both sit inside the holdout, so the analysis needs the
# holdout unlocked (see rate_shock.py).
RATE_SHOCK = {
    "pre_shock": ("2021-01-01", "2022-03-15"),
    "hiking": ("2022-03-16", "2023-07-26"),
    "post_hiking": ("2023-07-27", "2025-12-31"),
}

# Llama 3.1's documented knowledge cutoff (December 2023), used by part 4's
# after_llama_cutoff flag. Used here only to group results for part 9.
LLAMA_CUTOFF = pd.Timestamp("2023-12-31")

RANDOM_SEED = 2026


# ---------------------------------------------------------------------------
# Reuse of other parts' code (loaded by file path, so nothing is copied and
# nothing shadows our own module names)
# ---------------------------------------------------------------------------
def _load(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: is TEAM_DIR right? ({TEAM_DIR})")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def common_3_4():
    """Part 3-4's common.py: SPLITS, split_of and guard_holdout."""
    return _load(PART34 / "code" / "common.py", "common_3_4")


def returns_3_4():
    """Part 4's returns.py: newey_west_se and rank_ic_by_season (pure numpy/pandas)."""
    return _load(PART34 / "code" / "returns.py", "returns_3_4")


def ensure_dirs() -> None:
    for d in (DATA, TABLES, REGRESSIONS, FIGURES, LOGS, TEXT_FEATURES):
        d.mkdir(parents=True, exist_ok=True)


def redirect_outputs(tag: str) -> None:
    """
    Send every output of this run to data/_<tag>/ and results/_<tag>/ (used by the
    synthetic demo and the tests, so they never overwrite real results).
    Modules read these attributes at call time, so the redirect applies everywhere.
    """
    global DATA, RESULTS, TABLES, REGRESSIONS, FIGURES, PANEL, SIGNALS, TEXT_FEATURES, RUN_LOG, BOOK_TO_MARKET
    DATA = STEP78_DIR / "data" / f"_{tag}"
    RESULTS = STEP78_DIR / "results" / f"_{tag}"
    TABLES, REGRESSIONS, FIGURES = RESULTS / "tables", RESULTS / "regressions", RESULTS / "figures"
    PANEL, SIGNALS = DATA / "analysis_panel.parquet", DATA / "signals.parquet"
    TEXT_FEATURES = DATA / "text_features"
    BOOK_TO_MARKET = DATA / "book_to_market.parquet"
    RUN_LOG = LOGS / f"runs_{tag}.jsonl"
    ensure_dirs()
