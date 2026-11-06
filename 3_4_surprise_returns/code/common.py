"""
Shared settings for parts 3 and 4.

Everything downstream (SUE, returns, funnel) imports from here, so the
sample splits, the holdout lock and the trading calendar are defined once.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ET = "America/New_York"

# Folder holding the inputs and outputs. Override with the PROJECT_DIR
# environment variable (on WRDS, Colab or a laptop) instead of editing paths.
TEAM_DIR = Path(os.environ.get("TEAM_DIR", Path(__file__).resolve().parents[2])).expanduser().resolve()
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", TEAM_DIR / "3_4_surprise_returns")).expanduser().resolve()

# ---------------------------------------------------------------------------
# Splits, as registered in the proposal. 2020 is set aside.
# ---------------------------------------------------------------------------
SPLITS = {
    "dev": (2006, 2016),
    "validation": (2017, 2019),
    "excluded_2020": (2020, 2020),
    "holdout": (2021, 2025),
}


def split_of(dates: pd.Series) -> pd.Series:
    """Label each date with its split, by calendar year of the reaction date."""
    y = pd.to_datetime(dates).dt.year
    out = pd.Series(pd.NA, index=dates.index, dtype="string")
    for name, (lo, hi) in SPLITS.items():
        out = out.mask(y.between(lo, hi), name)
    return out


def guard_holdout(df: pd.DataFrame, date_col: str = "reaction_date") -> pd.DataFrame:
    """
    We build datasets for every year, but nothing that relates a signal to
    returns may see the holdout before the freeze. Call this on any table
    before analysis: it drops holdout rows unless UNLOCK_HOLDOUT=1 is set.
    """
    if os.environ.get("UNLOCK_HOLDOUT") == "1":
        print("WARNING: holdout unlocked. This should happen once, after the freeze.")
        return df
    is_hold = split_of(df[date_col]).eq("holdout").fillna(False)
    if is_hold.any():
        print(f"holdout guard: dropped {int(is_hold.sum()):,} rows from 2021-2025")
    return df.loc[~is_hold].copy()


# ---------------------------------------------------------------------------
# Trading calendar
# ---------------------------------------------------------------------------
# The authoritative list of trading days is CRSP's own. Early closes (13:00)
# are not in CRSP, so we take them from pandas_market_calendars when it is
# installed and otherwise assume a 16:00 close (about three days a year).

CALENDAR_FILE = "trading_calendar.csv"


# Three large stocks that traded on every session of the sample period.
# A distinct-date scan of all of dsf_v2 is slow; the union of these three
# misses a trading day only if all three were halted at once.
CALENDAR_PERMNOS = (14593, 10107, 22111)  # Apple, Microsoft, Johnson & Johnson


def build_trading_calendar(db, start: str = "2005-01-01", end: str = "2026-12-31") -> pd.DataFrame:
    """Trading days from crsp.dsf_v2, with the close time for each day."""
    lst = ",".join(str(p) for p in CALENDAR_PERMNOS)
    days = db.raw_sql(
        f"""
        select distinct dlycaldt as date
        from crsp.dsf_v2
        where permno in ({lst})
          and dlycaldt between date '{start}' and date '{end}'
        order by 1
        """,
        date_cols=["date"],
    )
    days["close_et"] = "16:00"
    per_year = days["date"].dt.year.value_counts().sort_index()
    print("trading days per year:", per_year.to_dict())

    try:
        import pandas_market_calendars as mcal

        last = days["date"].max().strftime("%Y-%m-%d")
        sched = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=last)
        nyse = set(pd.to_datetime(sched.index))
        crsp = set(days["date"])
        only_nyse = sorted(nyse - crsp)
        only_crsp = sorted(crsp - nyse)
        print(f"days in NYSE calendar but not CRSP: {len(only_nyse)} {only_nyse[:5]}")
        print(f"days in CRSP but not NYSE calendar: {len(only_crsp)} {only_crsp[:5]}")

        close = sched["market_close"].dt.tz_convert(ET).dt.strftime("%H:%M")
        close.index = pd.to_datetime(close.index)
        days["close_et"] = days["date"].map(close).fillna("16:00")
        days["source"] = "crsp"
        print(f"early closes: {int((days['close_et'] != '16:00').sum())}")

        # Releases after the close on CRSP's last day react on a later
        # session that CRSP does not have yet. We extend the calendar with
        # the NYSE schedule so those events still get a reaction date; part 4
        # then reports their returns as missing rather than wrong.
        future = mcal.get_calendar("NYSE").schedule(start_date=last, end_date=end)
        future = future.loc[pd.to_datetime(future.index) > days["date"].max()]
        ext = pd.DataFrame({
            "date": pd.to_datetime(future.index),
            "close_et": future["market_close"].dt.tz_convert(ET).dt.strftime("%H:%M").to_numpy(),
            "source": "nyse_calendar_after_crsp",
        })
        days = pd.concat([days, ext], ignore_index=True)
        print(f"extended past CRSP's last day with {len(ext)} NYSE sessions")
    except ImportError:
        raise ImportError("pip install pandas_market_calendars: needed for early closes "
                          "and for sessions after CRSP's last date")
    return days


def load_trading_calendar(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else PROJECT_DIR / "data" / CALENDAR_FILE
    cal = pd.read_csv(path, dtype={"close_et": str}, parse_dates=["date"])
    cal = cal.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    assert cal["date"].is_monotonic_increasing
    return cal
