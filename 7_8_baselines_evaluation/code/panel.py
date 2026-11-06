"""
The analysis panel: one row per event, with every variable parts 7-8 need.

Why a single panel
------------------
Task 7 asks for "consistent metrics, event windows and chronological splits"
and task 8 for regressions, 2x2 tables and robustness on the *same* events. The
simplest way to guarantee that is to build one table, once, and have every
baseline and every test read from it. A model that cannot score an event gets a
NaN, and comparisons are always made on the events *all* compared signals cover
(``common_support`` in compare.py), never on each model's own subset.

What it adds to part 4's event table
------------------------------------
* ``season``: the calendar quarter of the reaction date. Earnings
  announcements cluster in four seasons a year, and events in one season share
  market-wide shocks. Every cross-sectional statistic (ranks, deciles, ICs,
  2x2 cells) is therefore computed *within* season, and standard errors are
  clustered or Newey-West'ed *across* seasons.
* within-season ranks of SUE (``sue_rank``, in [-0.5, 0.5]) and SUE deciles: the
  return response to SUE is strongly non-linear (flat in the middle, steep in the
  tails), so the "surprise-only" benchmark must be allowed to be non-linear,
  otherwise a text model could look useful merely by capturing that curvature.
* controls: log market cap, log number of analysts, dispersion scaled by price,
  prior one-month return, log book-to-market (+ missing indicator), FF12 sector,
  size terciles.
* text metadata: whether a cleaned text exists, its length and version.

The holdout guard
-----------------
``build_panel`` passes the table through part 3-4's ``guard_holdout``: unless the
environment variable ``UNLOCK_HOLDOUT=1`` is set, 2021-2025 rows are dropped
before *any* statistic is computed. This is deliberate friction. The test set is
looked at once, after the models, prompts and tests in this repository are
frozen.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import settings as S


# ---------------------------------------------------------------------------
# Small, reusable cross-sectional transforms
# ---------------------------------------------------------------------------
def add_season(df: pd.DataFrame, date_col: str = "reaction_date") -> pd.DataFrame:
    """Calendar quarter of the reaction date, as a string ("2014Q3")."""
    df = df.copy()
    df["season"] = pd.to_datetime(df[date_col]).dt.to_period("Q").astype(str)
    df["year"] = pd.to_datetime(df[date_col]).dt.year
    return df


def cs_rank(x: pd.Series, by: pd.Series) -> pd.Series:
    """
    Within-group percentile rank, centred to [-0.5, 0.5].

    Ranks make signals with very different scales (an LLM's 1-5 score, LM tone,
    a TF-IDF prediction, SUE) directly comparable, are immune to outliers, and a
    regression coefficient on a centred rank reads as "the return difference
    between the top and the bottom of the cross-section". Ties get the average
    rank. NaNs stay NaN.
    """
    r = x.groupby(by).rank(method="average")          # 1..n within group
    n = x.groupby(by).transform("count")
    # (rank - 1) / (n - 1) runs from 0 to 1; a group of one gets the middle (0).
    out = ((r - 1) / (n - 1).where(n > 1) - 0.5).fillna(0.0)
    return out.where(x.notna())


def cs_zscore(x: pd.Series, by: pd.Series) -> pd.Series:
    """Within-group z-score (used when the signal's magnitude matters)."""
    g = x.groupby(by)
    return (x - g.transform("mean")) / g.transform("std")


def winsorize_by(x: pd.Series, by: pd.Series, q=S.WINSOR) -> pd.Series:
    """Clip at the q[0] and q[1] quantiles within each group."""
    lo = x.groupby(by).transform(lambda s: s.quantile(q[0]))
    hi = x.groupby(by).transform(lambda s: s.quantile(q[1]))
    return x.clip(lo, hi)


def quantile_bucket(x: pd.Series, by: pd.Series, k: int) -> pd.Series:
    """1..k bucket of x within each group (rank-based, so ties do not break qcut)."""
    def f(s):
        if s.notna().sum() < k:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s.rank(method="first"), k, labels=False) + 1
    return x.groupby(by, group_keys=False).apply(f).astype("float64")


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
PANEL_ID_COLS = ["event_id", "accession", "permno", "ticker", "reaction_date", "season",
                 "year", "split"]


def build_panel(events: pd.DataFrame, sectors: pd.DataFrame | None = None,
                texts_meta: pd.DataFrame | None = None, sample: str | None = S.SAMPLE,
                include_holdout: bool = False, frozen_holdout: set[str] | None = None,
                book_to_market: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    events          part 4's events_final (io_inputs.load_events)
    sectors         io_inputs.load_sectors output (event_id, siccd, ff12)
    texts_meta      io_inputs.load_texts output without the text itself is enough
                    (accession, n_words_body, text_source, clean_version, sha256_text)
    sample          name of a boolean column of events_final ("in_primary",
                    "in_text_sample"), or None for every event
    include_holdout keep 2021-2025. Requires UNLOCK_HOLDOUT=1 as well: two keys,
                    so that the holdout cannot be opened by a default argument.
    frozen_holdout  part 5's frozen holdout accessions; when given, the holdout
                    events must be exactly those (checked, warning otherwise)
    book_to_market  book_to_market.compute_book_to_market output (event_id, bm);
                    None leaves log_bm at 0 with bm_missing = 1 for every event
                    (the regressions then drop both as constants)
    """
    common = S.common_3_4()
    p = events.copy()

    # 1. sample ------------------------------------------------------------
    n0 = len(p)
    if sample is not None:
        p = p[p[sample].astype(bool)]
    # 2020 is set aside by design (COVID: announcement timing and returns are
    # abnormal for reasons unrelated to the text). It is never used for fitting
    # nor for testing.
    p = p[p["split"] != "excluded_2020"]
    print(f"panel: {n0:,} events -> {len(p):,} in sample '{sample}' without 2020")

    # 2. holdout guard -------------------------------------------------------
    if include_holdout:
        if os.environ.get("UNLOCK_HOLDOUT") != "1":
            raise PermissionError("include_holdout=True needs UNLOCK_HOLDOUT=1 too: the holdout "
                                  "is opened once, after the freeze (docs/METHODOLOGY.md, sec. 2)")
        p = common.guard_holdout(p)       # prints the warning and keeps every row
        if frozen_holdout is not None:
            got = set(p.loc[p["split"] == "holdout", "accession"].dropna())
            if got != frozen_holdout:
                print(f"WARNING: holdout events differ from the frozen list "
                      f"({len(got - frozen_holdout)} extra, {len(frozen_holdout - got)} missing)")
    else:
        p = common.guard_holdout(p)

    # 2b. no-look-ahead sanity check: the reaction session can never be before
    # the release. If this fails, part 1/3's clock has a bug: stop.
    rd = pd.to_datetime(p["reaction_date"])
    r_date = pd.to_datetime(p["R_et"].astype(str).str[:10], errors="coerce")
    bad = (rd < r_date).sum()
    assert bad == 0, f"{bad} events react before their release time: check the clock"

    # 3. seasons and SUE transforms -------------------------------------------
    p = add_season(p)
    p["sue_rank"] = cs_rank(p[S.SUE], p["season"])
    p["sue_decile"] = quantile_bucket(p[S.SUE], p["season"], 10)
    p["sue_w"] = winsorize_by(p[S.SUE], p["season"])
    p["sue_sign"] = np.sign(p["surprise"]).astype("float64")   # +1 beat, -1 miss, 0 meet

    # 4. controls -----------------------------------------------------------
    p["log_mktcap"] = np.log(p["mktcap_t2"].where(p["mktcap_t2"] > 0))
    p["log_analysts"] = np.log(p["n_analysts"].where(p["n_analysts"] > 0))
    p["disp_price"] = (p["dispersion"] / p["price_t2"]).replace([np.inf, -np.inf], np.nan)
    # Single-analyst events have no dispersion; 0 + an indicator keeps them in.
    p["disp_missing"] = p["disp_price"].isna().astype(float)
    p["disp_price"] = winsorize_by(p["disp_price"].fillna(0.0), p["season"])
    p["ret_prior_1m_w"] = winsorize_by(p["ret_prior_1m"], p["season"])
    p["size_tercile"] = quantile_bucket(p["mktcap_t2"], p["season"], 3)
    # Book-to-market (book_to_market.py): log of Compustat ceqq of the last quarter
    # public before the release over mktcap_t2. Same treatment as dispersion: a
    # missing value (no Compustat match, stale or negative book equity) is filled,
    # here with the season's median log(B/M), and flagged by bm_missing, so those
    # events stay in every specification.
    if book_to_market is not None and len(book_to_market):
        p = p.merge(book_to_market[["event_id", "bm"]], on="event_id", how="left",
                    validate="many_to_one")
    elif "bm" not in p.columns:
        p["bm"] = np.nan
    lbm = np.log(p["bm"].where(p["bm"] > 0))
    p["bm_missing"] = lbm.isna().astype(float)
    lbm = winsorize_by(lbm, p["season"])
    p["log_bm"] = lbm.fillna(lbm.groupby(p["season"]).transform("median")).fillna(0.0)

    # Winsorised outcomes, for OLS only (rank statistics use the raw outcome).
    for c in S.OUTCOMES + S.PLACEBO_OUTCOMES + S.MKT_ADJ_OUTCOMES:
        if c in p.columns:
            p[c + "_w"] = winsorize_by(p[c], p["season"])

    # Sign of the reaction, for classification baselines.
    p["up_0_1"] = (p[S.PRIMARY_OUTCOME] > 0).astype(float).where(p[S.PRIMARY_OUTCOME].notna())

    # 5. sectors and text metadata --------------------------------------------
    if sectors is not None:
        p = p.merge(sectors[["event_id", "siccd", "ff12"]], on="event_id", how="left")
        p["ff12"] = p["ff12"].fillna("Unknown")
    if texts_meta is not None and len(texts_meta):
        keep = [c for c in ["accession", "n_words_body", "text_source", "clean_version",
                            "sha256_text"] if c in texts_meta.columns]
        p = p.merge(texts_meta[keep], on="accession", how="left", validate="many_to_one")
    p["has_text"] = p.get("text_source", pd.Series(index=p.index, dtype=object)).notna()
    if "n_words_body" in p.columns:
        p["log_words"] = np.log1p(p["n_words_body"])

    # 6. period labels used by robustness and part 9 ----------------------------
    p["period"] = pd.cut(p["year"], bins=[2005, 2010, 2016, 2019, 2023, 2025],
                         labels=["2006-10", "2011-16", "2017-19", "2021-23", "2024-25"]).astype(str)

    p = p.sort_values(["reaction_date", "event_id"]).reset_index(drop=True)
    return p


def describe_panel(p: pd.DataFrame) -> pd.DataFrame:
    """Counts by split: events, events with text, seasons. Summary statistics only."""
    g = p.groupby("split")
    t = pd.DataFrame({
        "events": g.size(),
        "with_text": g["has_text"].sum(),
        "seasons": g["season"].nunique(),
        "first": g["reaction_date"].min().dt.date,
        "last": g["reaction_date"].max().dt.date,
        "median_sue": g[S.SUE].median(),
        "mean_" + S.PRIMARY_OUTCOME: g[S.PRIMARY_OUTCOME].mean(),
    })
    return t
