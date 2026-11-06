"""
The release clock: one row per event, saying when the release became public
(R) and which trading session first reacts to it.

Parts 3 and 4 read ONLY this table for timing:
  - part 3 keeps an estimate only if its ANNDATS and ACTDATS are before R_date;
  - part 4 opens every return window on reaction_date and enters the
    post-release trade at entry_field ('open' or 'close') on that date.

Rule (instructors' feedback and correction email):
  R is the wire time; EDGAR (header / index page, Eastern) only if there is
  no wire; I/B/E/S ANNDATS/ANNTIMS is a check, never the source. Never the
  earliest of noisy dates. Keep an event in the primary sample only if two
  sources put it in the same session.

This file builds a PROVISIONAL clock while the EDGAR times are re-pulled:
  R = Dow Jones wire; I/B/E/S is the second source. Events with no wire use
  I/B/E/S as a stand-in (status single_source_ibes, development only). The
  EDGAR times from the submissions JSON are NOT used: about 30% are off by
  exactly the Eastern offset. When the corrected clock arrives with the same
  columns, swap the file and rerun; nothing downstream changes.

Usage:
  python clock.py --timing three_source_timing.csv --events event_table.csv \
                  --calendar trading_calendar.csv --out clock_provisional.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import ET, PROJECT_DIR, data_path, load_trading_calendar

CLOCK_VERSION = "provisional_v0_wire_then_ibes"
OPEN_MIN = 9 * 60 + 30

CLOCK_COLUMNS = [
    "accession",            # EDGAR accession number, the event key
    "permno",               # CRSP security
    "cik",                  # EDGAR filer
    "filing_date",          # EDGAR filingDate
    "R_et",                 # release time, ISO 8601 with Eastern offset
    "R_date",               # Eastern calendar date of R (estimate cut-off)
    "R_source",             # wire | edgar_header | edgar_6am_rule | ibes_standin
    "session_type",         # pre_open | intraday | after_close | non_trading_day
    "reaction_date",        # first trading session that can react to R
    "entry_field",          # first CRSP price after R on reaction_date: open | close
    "second_source",        # source used for the agreement check
    "second_reaction_date", # reaction date implied by the second source
    "status",               # see STATUSES
    "primary_eligible",     # True only when two sources agree on the session
    "ibes_anndats",         # carried for part 3's matching, not for timing
    "ibes_anntims",
    "clock_version",
]

STATUSES = {
    "two_source_agree",     # primary sample
    "two_source_conflict",  # excluded from primary, kept for robustness
    "single_source",        # only one source: excluded from primary
    "single_source_ibes",   # I/B/E/S stand-in, development only
    "no_clock",             # nothing usable yet
}


def assign_session(r_et: pd.Series, cal: pd.DataFrame) -> pd.DataFrame:
    """
    Map release times (tz-aware, Eastern) to the session that first reacts.

    On a trading day d:  R < 09:30 -> d, entry at the open
                         09:30 <= R < close(d) -> d, entry at the close
                         R >= close(d) -> next trading day, entry at the open
    On a non-trading day: next trading day, entry at the open.
    Early closes come from the calendar's close_et column.
    """
    out = pd.DataFrame(index=r_et.index, columns=["session_type", "reaction_date", "entry_field"])
    ok = r_et.notna()
    if not ok.any():
        return out

    r = r_et[ok]
    wall = r.dt.tz_convert(ET).dt.tz_localize(None)
    d = wall.dt.normalize()
    minutes = (wall.dt.hour * 60 + wall.dt.minute + wall.dt.second / 60).to_numpy()

    days = cal["date"].to_numpy(dtype="datetime64[ns]")
    close_min = (
        cal["close_et"].str.slice(0, 2).astype(int) * 60
        + cal["close_et"].str.slice(3, 5).astype(int)
    ).to_numpy()

    pos = np.searchsorted(days, d.to_numpy(dtype="datetime64[ns]"), side="left")
    if (pos >= len(days) - 1).any():
        raise ValueError("trading calendar ends too early for some release times")
    is_trading = days[pos] == d.to_numpy(dtype="datetime64[ns]")
    cmin = close_min[pos]

    pre = is_trading & (minutes < OPEN_MIN)
    intra = is_trading & (minutes >= OPEN_MIN) & (minutes < cmin)
    after = is_trading & (minutes >= cmin)

    react_pos = np.where(after, pos + 1, pos)  # non-trading days: pos is already the next trading day
    out.loc[ok, "reaction_date"] = pd.to_datetime(days[react_pos])
    out.loc[ok, "session_type"] = np.select(
        [pre, intra, after], ["pre_open", "intraday", "after_close"], "non_trading_day"
    )
    out.loc[ok, "entry_field"] = np.where(intra, "close", "open")
    out["reaction_date"] = pd.to_datetime(out["reaction_date"])
    return out


def _ibes_time(anndats: pd.Series, anntims: pd.Series) -> pd.Series:
    """I/B/E/S announcement time in Eastern. 00:00:00 means missing, not midnight."""
    tims = anntims.astype("string")
    missing = tims.isna() | tims.eq("00:00:00") | anndats.isna()
    naive = pd.to_datetime(anndats.astype("string") + " " + tims, errors="coerce")
    naive = naive.mask(missing)
    return naive.dt.tz_localize(ET, ambiguous="NaT", nonexistent="NaT")


def build_provisional_clock(timing_csv: str, events_csv: str, cal: pd.DataFrame) -> pd.DataFrame:
    t = pd.read_csv(
        timing_csv,
        usecols=["accession", "permno", "anndats", "anntims", "wire_dt"],
        dtype={"accession": "string", "anndats": "string", "anntims": "string"},
    )
    n_in = len(t)

    # 1,949 rows in the Nov 2 timing file lost their accession number; without
    # it an event cannot be joined to its text, so it cannot be used.
    no_acc = t["accession"].isna()
    t = t.loc[~no_acc]
    n_dup = int(t.duplicated("accession").sum())
    t = t.drop_duplicates("accession", keep="first").reset_index(drop=True)
    print(f"timing rows {n_in:,}: dropped {int(no_acc.sum()):,} with no accession, "
          f"{n_dup:,} duplicate accession(s)")

    wire = pd.to_datetime(t["wire_dt"], utc=True, errors="coerce").dt.tz_convert(ET)
    ibes = _ibes_time(t["anndats"], t["anntims"])

    s_wire = assign_session(wire, cal)
    s_ibes = assign_session(ibes, cal)

    has_w, has_i = wire.notna(), ibes.notna()

    # R: wire first; I/B/E/S only as a development stand-in where no wire exists.
    r = wire.where(has_w, ibes)
    sess = pd.DataFrame({c: s_wire[c].where(has_w, s_ibes[c]) for c in s_wire.columns})

    out = pd.DataFrame({
        "accession": t["accession"],
        "permno": t["permno"].astype("Int64"),
        "R_et": r,
        "R_source": np.select([has_w, has_i], ["wire", "ibes_standin"], ""),
        "session_type": sess["session_type"],
        "reaction_date": pd.to_datetime(sess["reaction_date"]),
        "entry_field": sess["entry_field"],
        "ibes_anndats": t["anndats"],
        "ibes_anntims": t["anntims"],
    })
    out["R_source"] = out["R_source"].replace("", pd.NA)

    # Second source for the agreement rule. Until EDGAR header times exist,
    # only wire events can have one (I/B/E/S).
    both = has_w & has_i
    out["second_source"] = pd.Series(np.where(both, "ibes", pd.NA), dtype="string")
    out["second_reaction_date"] = pd.to_datetime(s_ibes["reaction_date"]).where(both)
    agree = both & (out["reaction_date"] == out["second_reaction_date"])

    out["status"] = np.select(
        [agree, both & ~agree, has_w & ~has_i, ~has_w & has_i],
        ["two_source_agree", "two_source_conflict", "single_source", "single_source_ibes"],
        "no_clock",
    )
    out["primary_eligible"] = out["status"].eq("two_source_agree")

    ev = pd.read_csv(events_csv, usecols=["accession", "cik", "filing_date"],
                     dtype={"accession": "string", "cik": "string", "filing_date": "string"})
    ev = ev.drop_duplicates("accession")
    out = out.merge(ev, on="accession", how="left", validate="one_to_one")

    out["R_date"] = out["R_et"].dt.strftime("%Y-%m-%d")
    out["R_et"] = out["R_et"].map(lambda x: x.isoformat() if pd.notna(x) else pd.NA)
    out["reaction_date"] = out["reaction_date"].dt.strftime("%Y-%m-%d")
    out["second_reaction_date"] = out["second_reaction_date"].dt.strftime("%Y-%m-%d")
    out["clock_version"] = CLOCK_VERSION
    return out[CLOCK_COLUMNS]


def validate_clock(clk: pd.DataFrame, cal: pd.DataFrame) -> None:
    """Checks every clock must pass, provisional or final, before parts 3-4 use it."""
    missing_cols = set(CLOCK_COLUMNS) - set(clk.columns)
    assert not missing_cols, f"clock is missing columns: {missing_cols}"
    assert clk["accession"].notna().all(), "accession must never be missing"
    assert clk["accession"].is_unique, "one row per accession"
    assert set(clk["status"].unique()) <= STATUSES, "unknown status values"

    timed = clk["R_et"].notna()
    r = pd.to_datetime(clk.loc[timed, "R_et"], utc=True)
    assert r.notna().all(), "R_et must parse as a timestamp with an offset"

    react = pd.to_datetime(clk.loc[timed, "reaction_date"])
    assert react.isin(set(cal["date"])).all(), "reaction_date must be a trading day"
    assert (react >= pd.to_datetime(clk.loc[timed, "R_date"])).all(), \
        "the market cannot react before the release"
    assert clk.loc[timed, "entry_field"].isin(["open", "close"]).all()
    assert (clk.loc[clk["primary_eligible"], "status"] == "two_source_agree").all()
    print("clock validation passed")


def summarise(clk: pd.DataFrame) -> pd.DataFrame:
    s = clk["status"].value_counts().rename("events").to_frame()
    s["share_pct"] = (100 * s["events"] / len(clk)).round(1)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", default=data_path("three_source_timing.csv"))
    ap.add_argument("--events", default=data_path("event_table.csv"))
    ap.add_argument("--calendar", default=None)
    ap.add_argument("--out", default=str(PROJECT_DIR / "data" / "clock_provisional.csv"))
    args = ap.parse_args()

    cal = load_trading_calendar(args.calendar)
    clk = build_provisional_clock(args.timing, args.events, cal)
    validate_clock(clk, cal)
    print(summarise(clk).to_string())
    print(clk["session_type"].value_counts().to_string())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    clk.to_csv(args.out, index=False)
    print(f"-> {args.out} ({len(clk):,} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
