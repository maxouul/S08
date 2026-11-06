"""
Book-to-market control (the open item of docs/METHODOLOGY.md, section 8).

Definition
----------
For each event, book equity is Compustat's quarterly common equity ``ceqq``
($ millions) of the **most recent fiscal quarter that was public strictly before
the earnings release** of the event:

    available_date = rdq                      (Compustat's report date of quarterly earnings)
                   = datadate + 90 days       when rdq is missing (a conservative lag:
                                              the 10-Q deadline is 40-45 days)
    use the latest quarter with available_date < release date (date of R_et),
    and datadate no older than MAX_AGE_DAYS before the release

The quarter being announced has rdq equal to the release date, so it is never
used: its balance sheet is not public before the release (no look-ahead).

    bm = ceqq * 1000 / mktcap_t2              (ceqq in $ millions, CRSP mktcap_t2 in $ thousands)

Book equity <= 0 gives no B/M (its log is undefined); like a missing Compustat
match, the event is kept with ``bm_missing = 1`` (see panel.py).

Data
----
Part 3-4's ``cache/comp_fundq.parquet`` does not contain ``ceqq``, and parts 7-8
do not edit part 3-4's files, so this module pulls the few columns it needs from
WRDS itself and stores them under ``data/`` (row-level, WRDS-derived: private,
git-ignored):

    data/comp_fundq_ceqq.parquet     gvkey, datadate, rdq, ceqq (raw pull)
    data/book_to_market.parquet      event_id, gvkey, bm_datadate, bm_available, ceqq, bm

Run once (WRDS asks for a Duo approval on the first connection, so run it from a
terminal where you can approve it)::

    TEAM_DIR=... python code/book_to_market.py

The pipeline only reads ``data/book_to_market.parquet``; if it does not exist the
regressions run without the control and say so.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S

MAX_AGE_DAYS = 365          # book equity older than a year is treated as missing
RDQ_FALLBACK_DAYS = 90      # reporting lag when rdq is missing

FUNDQ_SQL = """
select gvkey, datadate, rdq, ceqq
from comp.fundq
where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C'
  and datadate >= %(start)s and gvkey = any(%(gvkeys)s)
"""


# ---------------------------------------------------------------------------
# Pure computation (tested on synthetic data)
# ---------------------------------------------------------------------------
def compute_book_to_market(events: pd.DataFrame, fundq: pd.DataFrame,
                           max_age_days: int = MAX_AGE_DAYS,
                           rdq_fallback_days: int = RDQ_FALLBACK_DAYS) -> pd.DataFrame:
    """
    events: event_id, gvkey, R_et (release time, string or timestamp), mktcap_t2 ($000s)
    fundq:  gvkey, datadate, rdq, ceqq ($ millions)
    Returns one row per event with a gvkey: event_id, gvkey, bm_datadate,
    bm_available, ceqq, bm (NaN when no usable quarter or book equity <= 0).
    """
    ev = events[["event_id", "gvkey", "R_et", "mktcap_t2"]].dropna(subset=["gvkey"]).copy()
    ev["gvkey"] = ev["gvkey"].astype(str).str.zfill(6)
    ev["release_date"] = pd.to_datetime(ev["R_et"].astype(str).str[:10], errors="coerce")
    ev = ev.dropna(subset=["release_date"])

    f = fundq[["gvkey", "datadate", "rdq", "ceqq"]].copy()
    f["gvkey"] = f["gvkey"].astype(str).str.zfill(6)
    f["datadate"] = pd.to_datetime(f["datadate"])
    f["rdq"] = pd.to_datetime(f["rdq"])
    f["ceqq"] = pd.to_numeric(f["ceqq"], errors="coerce").astype(float)  # psycopg gives Decimal
    f = f.dropna(subset=["ceqq", "datadate"])
    f["bm_available"] = f["rdq"].fillna(f["datadate"] + pd.Timedelta(days=rdq_fallback_days))
    f["bm_source"] = np.where(f["rdq"].notna(), "rdq", f"datadate+{rdq_fallback_days}")
    # A quarter restated or duplicated (fiscal-year change): keep the latest datadate
    # among quarters that became public on the same day.
    f = (f.sort_values(["gvkey", "bm_available", "datadate"])
          .drop_duplicates(["gvkey", "bm_available"], keep="last"))

    # merge_asof needs identical datetime resolutions (pandas >= 2 can mix s / us / ns).
    ev["release_date"] = ev["release_date"].astype("datetime64[ns]")
    for c in ("datadate", "bm_available"):
        f[c] = f[c].astype("datetime64[ns]")
    ev = ev.sort_values("release_date")
    f = f.sort_values("bm_available").rename(columns={"datadate": "bm_datadate"})
    m = pd.merge_asof(ev, f[["gvkey", "bm_available", "bm_datadate", "ceqq", "bm_source"]],
                      left_on="release_date", right_on="bm_available", by="gvkey",
                      direction="backward", allow_exact_matches=False)   # strictly before
    stale = (m["release_date"] - m["bm_datadate"]).dt.days > max_age_days
    m.loc[stale, ["bm_available", "bm_datadate", "ceqq", "bm_source"]] = np.nan
    mcap = m["mktcap_t2"].where(m["mktcap_t2"] > 0)
    m["bm"] = (m["ceqq"] * 1000.0 / mcap).where(m["ceqq"] > 0)
    return m[["event_id", "gvkey", "bm_datadate", "bm_available", "bm_source", "ceqq", "bm"]] \
        .reset_index(drop=True)


def load_book_to_market(path: Path | str | None = None) -> pd.DataFrame | None:
    """data/book_to_market.parquet, or None (with a message) if not built yet."""
    path = Path(path or S.BOOK_TO_MARKET)
    if not path.exists():
        print(f"book-to-market: {path} not found, the log(B/M) control is left out "
              f"(build it with: python code/book_to_market.py)")
        return None
    return pd.read_parquet(path)


def coverage(bm: pd.DataFrame) -> dict:
    return {"events": len(bm), "with_bm": int(bm["bm"].notna().sum()),
            "share": float(bm["bm"].notna().mean()) if len(bm) else np.nan,
            "neg_book": int((bm["ceqq"] <= 0).sum()),
            "median_bm": float(bm["bm"].median())}


# ---------------------------------------------------------------------------
# WRDS pull
# ---------------------------------------------------------------------------
def pull_fundq(gvkeys, start: str = "2004-01-01", user: str | None = None,
               timeout: int = 120) -> pd.DataFrame:
    """
    gvkey, datadate, rdq, ceqq from comp.fundq. Credentials come from ~/.pgpass
    (host wrds-pgdata.wharton.upenn.edu, port 9737, db wrds). WRDS requires a Duo
    approval for new connections: if none arrives within ``timeout`` seconds the
    connection fails with a clear message.
    """
    import psycopg

    user = user or os.environ.get("WRDS_USER") or _pgpass_user()
    try:
        con = psycopg.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds",
                              user=user, sslmode="require", connect_timeout=timeout)
    except psycopg.OperationalError as e:
        raise RuntimeError(
            f"WRDS connection failed ({e}). WRDS sends a Duo push on each new connection: run "
            f"`python code/book_to_market.py` from a terminal and approve it on your phone.") from e
    with con:
        cur = con.cursor()
        cur.execute(FUNDQ_SQL, {"start": start, "gvkeys": sorted(set(map(str, gvkeys)))})
        cols = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def _pgpass_user() -> str | None:
    p = Path.home() / ".pgpass"
    if p.exists():
        for line in p.read_text().splitlines():
            parts = line.split(":")
            if len(parts) >= 5 and parts[0].startswith("wrds-pgdata"):
                return parts[3]
    return None


def main() -> int:
    import io_inputs as IO

    S.ensure_dirs()
    ev = IO.load_events()
    raw = S.DATA / "comp_fundq_ceqq.parquet"
    if raw.exists() and "--refresh" not in sys.argv:
        fq = pd.read_parquet(raw)
        print(f"using cached {raw} ({len(fq):,} rows; --refresh to pull again)")
    else:
        fq = pull_fundq(ev["gvkey"].dropna().unique())
        fq.to_parquet(raw, index=False)
        print(f"-> {raw} ({len(fq):,} rows)")
    bm = compute_book_to_market(ev, fq)
    bm.to_parquet(S.BOOK_TO_MARKET, index=False)
    print(f"-> {S.BOOK_TO_MARKET}: {coverage(bm)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
