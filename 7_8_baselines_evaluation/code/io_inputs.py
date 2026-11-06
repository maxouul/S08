"""
Readers for everything parts 7-8 take from the other parts.

Design rules
------------
1. **Read, never rebuild.** Surprise, returns, timing and sample flags come from
   part 3-4's ``events_final.parquet``; texts from part 5's cleaned parquet files;
   model scores from part 5's model-output cache (filled by part 6). If one of
   those changes, we rerun, we do not patch.
2. **Check the contract on the way in.** Each reader asserts the columns it
   needs (``REQUIRED_*`` below). When a teammate renames a column the pipeline
   stops with a clear message instead of silently producing NaNs.
3. **Everything is keyed on ``event_id`` and ``accession``.** ``event_id`` is the
   I/B/E/S event (ticker|fiscal-period-end), ``accession`` the 8-K the text was
   read from. Part 4 links them one-to-one (checked here).
4. **Degrade loudly.** When an input does not exist yet (the LLM scores, the LM
   dictionary, the frozen holdout), the reader returns an empty frame and prints
   why, so the notebooks still run end-to-end on what exists.

See docs/INTERFACES.md for the column-by-column contract.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S

# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------
REQUIRED_EVENT_COLS = [
    # keys
    "event_id", "accession", "permno", "ticker",
    # timing (part 1 / part 3 clock)
    "R_et", "reaction_date", "session_type", "clock_status", "R_source",
    # surprise (part 3)
    "actual", "consensus_median", "surprise", "sue_price", "sue_disp",
    "n_analysts", "dispersion", "price_t2",
    # returns (part 4)
    "car_0_1_ff4", "post_car_1_ff4", "post_car_60_ff4", "car_1_60_ff4", "ar_m1",
    "ret_prior_1m",
    # firm characteristics and flags (part 4)
    "mktcap_t2", "f_size", "basis_suspect", "timing_suspect", "two_source",
    "after_llama_cutoff", "in_primary", "in_text_sample", "split",
    # versions, carried into every output so results can be traced
    "surprise_version", "returns_version", "sample_version",
]

REQUIRED_TEXT_COLS = ["accession", "clean_version", "text_body", "sha256_text"]


def _check(df: pd.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{what} is missing columns {missing}. The upstream part changed its "
                       f"output: update docs/INTERFACES.md and the reader together.")


# ---------------------------------------------------------------------------
# Events (part 3-4)
# ---------------------------------------------------------------------------
def load_events(path: Path | str | None = None) -> pd.DataFrame:
    """
    One row per I/B/E/S earnings announcement, with surprise, returns and flags.

    Nullable pandas types (Float64, Int64, boolean) are converted to plain numpy
    types: statsmodels and scikit-learn do not accept pd.NA.
    """
    path = Path(path) if path else S.EVENTS
    ev = pd.read_parquet(path)
    _check(ev, REQUIRED_EVENT_COLS, str(path))

    assert ev["event_id"].is_unique, "events_final has duplicate event_id"
    acc = ev["accession"].dropna()
    assert acc.is_unique, "one accession linked to two events: the event-filing link is not 1:1"

    for c in ev.columns:
        dt = str(ev[c].dtype)
        if dt in ("Float64", "Int64", "Float32", "Int32"):
            ev[c] = ev[c].astype("float64")
        elif dt == "boolean":
            ev[c] = ev[c].fillna(False).astype(bool)
        elif dt == "string":
            ev[c] = ev[c].astype(object)
    ev["reaction_date"] = pd.to_datetime(ev["reaction_date"])
    return ev


# ---------------------------------------------------------------------------
# Texts (part 5, with part 2's export as a labelled fallback)
# ---------------------------------------------------------------------------
def load_texts(clean_dir: Path | str | None = None, accessions=None,
               allow_fallback: bool = True, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Cleaned release texts, one row per accession.

    We read ``text_body``: part 5's text *without* the boiler-plate sections (safe
    harbour, "About X", contacts, conference-call details). Those sections are
    nearly identical across a firm's releases and would let a bag-of-words model
    learn the firm's identity rather than the news.

    Columns returned: accession, text_body, sha256_text, clean_version,
    n_words_body, text_source ("clean_<version>" or "fallback_release_texts").
    """
    clean_dir = Path(clean_dir) if clean_dir else S.CLEAN_DIR
    want = set(accessions) if accessions is not None else None
    final = ["accession", "clean_version", "text_body", "sha256_text", "n_words_body",
             "text_source"]
    extra = [c for c in (columns or []) if c not in final]   # e.g. "sentences"
    frames = []
    for f in sorted(clean_dir.glob("clean_*.parquet")):
        d = pd.read_parquet(f, columns=final[:-1] + extra + ["error"])
        _check(d, REQUIRED_TEXT_COLS, str(f))
        if want is not None:
            d = d[d["accession"].isin(want)]
        frames.append(d)
    out = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=final[:-1] + extra + ["error"]))
    # A document part 5 could not clean carries an error string: drop it here and
    # report it, rather than scoring an empty text.
    bad = out["error"].fillna("").astype(str).str.len() > 0
    if bad.any():
        print(f"texts: {int(bad.sum())} documents with a cleaning error dropped")
    out = out.loc[~bad].drop(columns="error")
    out["text_source"] = "clean_" + out["clean_version"].astype(str)
    print(f"texts: {len(out):,} cleaned documents from {len(frames)} year files in {clean_dir}")

    if allow_fallback:
        fb = next((p for p in S.RELEASE_TEXTS_FALLBACK if p.exists()), None)
        if fb is not None:
            r = pd.read_csv(fb, usecols=["accession", "text"])
            r = r[~r["accession"].isin(out.get("accession", pd.Series(dtype=str)))]
            if want is not None:
                r = r[r["accession"].isin(want)]
            if len(r):
                import hashlib
                r = r.rename(columns={"text": "text_body"})
                r["text_body"] = r["text_body"].fillna("").astype(str)
                r["sha256_text"] = r["text_body"].map(lambda t: hashlib.sha256(t.encode()).hexdigest())
                r["clean_version"] = "fallback"
                r["n_words_body"] = r["text_body"].str.split().str.len()
                r["text_source"] = "fallback_release_texts"
                print(f"texts: {len(r):,} more from the fallback export {fb.name} "
                      f"(not cleaned by part 5: use for development only)")
                out = pd.concat([out, r.reindex(columns=out.columns)], ignore_index=True)
    out = out.drop_duplicates("accession", keep="first").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Model scores (part 6, stored by part 5's ModelCache)
# ---------------------------------------------------------------------------
# How to read a number out of score_json. Part 6 decides the prompts; the
# agreed contract (docs/INTERFACES.md) is that score_json is either a bare number
# or an object with a numeric "score" field. Prompt-specific keys can be added
# here without touching any other code, e.g. {"themes_v1": "net_tone"}.
SCORE_KEYS: dict[str, str] = {}
DEFAULT_SCORE_KEY = "score"


def _extract_score(score_json: str | None, prompt_id: str) -> float:
    if score_json is None:
        return np.nan
    try:
        v = json.loads(score_json)
    except (TypeError, ValueError):
        return np.nan
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, dict):
        key = SCORE_KEYS.get(prompt_id, DEFAULT_SCORE_KEY)
        x = v.get(key)
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return float(x)
    return np.nan


def signal_name(model_id: str, prompt_id: str, variant: str) -> str:
    """Column name of a model signal: short, filesystem-safe and unique."""
    short = model_id.split("/")[-1].lower().replace(".", "").replace("-", "_")
    return f"llm__{short}__{prompt_id}__{variant}"


def load_model_scores(db_path: Path | str | None = None, clean_version: str | None = None,
                      variants: tuple[str, ...] = ("raw", "mask")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wide table of model scores: one row per accession, one column per
    (model, prompt, variant), plus a long "provenance" table that records which
    model revision and cleaning version produced each column.

    If the same accession was scored twice under one (model, prompt, variant),
    e.g. after a cleaning-version bump, the most recent record wins and the
    duplicate count is reported. Pass ``clean_version`` to pin one version.
    """
    db_path = Path(db_path) if db_path else S.MODEL_CACHE
    empty = (pd.DataFrame(columns=["accession"]), pd.DataFrame())
    if not db_path.exists():
        print(f"model scores: {db_path} does not exist yet (part 6 has not run)")
        return empty
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        q = ("select accession, clean_version, variant, model_id, model_revision, prompt_id, "
             "score_json, created_at_utc from outputs")
        long = pd.read_sql_query(q, con)
    finally:
        con.close()
    if long.empty:
        print("model scores: the cache is empty (part 6 has not run)")
        return empty
    if clean_version is not None:
        long = long[long["clean_version"] == clean_version]
    long = long[long["variant"].isin(variants)]
    long["score"] = [_extract_score(j, p) for j, p in zip(long["score_json"], long["prompt_id"])]
    long["signal"] = [signal_name(m, p, v) for m, p, v in
                      zip(long["model_id"], long["prompt_id"], long["variant"])]

    long = long.sort_values("created_at_utc")
    dups = long.duplicated(["accession", "signal"], keep="last")
    if dups.any():
        print(f"model scores: {int(dups.sum())} older duplicate records ignored")
    long = long.loc[~dups]
    n_bad = int(long["score"].isna().sum())
    if n_bad:
        print(f"model scores: {n_bad} records whose score could not be parsed (kept as NaN)")

    wide = long.pivot(index="accession", columns="signal", values="score").reset_index()
    wide.columns.name = None
    prov = (long.groupby("signal")
                .agg(model_id=("model_id", "first"), prompt_id=("prompt_id", "first"),
                     variant=("variant", "first"),
                     model_revisions=("model_revision", lambda s: ",".join(sorted(set(s)))),
                     clean_versions=("clean_version", lambda s: ",".join(sorted(set(s)))),
                     n_scored=("score", "size"), n_parsed=("score", "count"))
                .reset_index())
    print(f"model scores: {len(prov)} signals over {len(wide):,} documents")
    return wide, prov


# ---------------------------------------------------------------------------
# Sectors (Fama-French 12, from the CRSP SIC code valid on the event date)
# ---------------------------------------------------------------------------
# Ken French's 12-industry definitions (SIC ranges). Anything not listed is "Other".
FF12 = {
    "NoDur": [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    "Durbl": [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714),
              (3716, 3716), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    "Manuf": [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569),
              (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749),
              (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)],
    "Enrgy": [(1200, 1399), (2900, 2999)],
    "Chems": [(2800, 2829), (2840, 2899)],
    "BusEq": [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    "Telcm": [(4800, 4899)],
    "Utils": [(4900, 4949)],
    "Shops": [(5000, 5999), (7200, 7299), (7600, 7699)],
    "Hlth": [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    "Money": [(6000, 6999)],
}


def sic_to_ff12(sic: pd.Series) -> pd.Series:
    s = pd.to_numeric(sic, errors="coerce")
    out = pd.Series("Other", index=s.index, dtype=object)
    for name, ranges in FF12.items():
        for lo, hi in ranges:
            out[s.between(lo, hi)] = name
    out[s.isna()] = "Unknown"
    return out


def load_sectors(events: pd.DataFrame, names_path: Path | str | None = None) -> pd.DataFrame:
    """
    SIC code and FF12 sector of each event, from the CRSP name record valid on
    the reaction date (point in time: a firm that changed industry is classified
    as it was then, not as it is today).
    """
    names_path = Path(names_path) if names_path else S.CRSP_NAMES
    out = events[["event_id"]].copy()
    if not names_path.exists():
        print(f"sectors: {names_path} missing, every event is 'Unknown'")
        out["siccd"] = np.nan
        out["ff12"] = "Unknown"
        return out
    nm = pd.read_parquet(names_path, columns=["permno", "namedt", "nameenddt", "siccd"])
    nm["namedt"] = pd.to_datetime(nm["namedt"])
    nm["nameenddt"] = pd.to_datetime(nm["nameenddt"]).fillna(pd.Timestamp("2099-12-31"))
    k = events[["event_id", "permno", "reaction_date"]].copy()
    k["permno"] = k["permno"].astype("int64")
    nm["permno"] = nm["permno"].astype("int64")
    m = k.merge(nm, on="permno", how="left")
    m = m[(m["reaction_date"] >= m["namedt"]) & (m["reaction_date"] <= m["nameenddt"])]
    m = m.sort_values("namedt").drop_duplicates("event_id", keep="last")
    out = out.merge(m[["event_id", "siccd"]], on="event_id", how="left")
    out["ff12"] = sic_to_ff12(out["siccd"])
    print(f"sectors: {out['siccd'].notna().mean():.1%} of events have a point-in-time SIC")
    return out


# ---------------------------------------------------------------------------
# Frozen test set (part 5's splits.py --freeze)
# ---------------------------------------------------------------------------
def load_frozen_holdout() -> set[str] | None:
    """The frozen holdout accessions, or None while the freeze has not happened."""
    if not S.HOLDOUT_IDS.exists():
        return None
    ids = {line.strip() for line in S.HOLDOUT_IDS.read_text().splitlines() if line.strip()}
    return ids
