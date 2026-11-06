#we complete the attrition funnel by applying price, size, and analyst-coverage filters.
#we start with 383,153 filings, of which 328,317 are matched to a PERMNO.
#we process CRSP data year by year using SQL to avoid memory issues.
#WRDS authentication must be set up separately before running the script.
#python wrds_match_and_funnel.py

from __future__ import annotations
from project_paths import DATA_DIR, data_path


import argparse
import sys

import pandas as pd

try:
    import wrds
except ImportError:
    sys.exit("pip install wrds")


#local step: events joined to PERMNO, no WRDS needed


def matched_events(events_csv: str, link_csv: str) -> tuple[pd.DataFrame, int]:
    ev = pd.read_csv(
        events_csv, dtype={"cik": str},
        usecols=["cik", "accession", "filing_date", "session_date", "when",
                 "timing_flag"],
    )
    ev["session_date"] = pd.to_datetime(ev["session_date"])
    ev["cik"] = ev["cik"].str.lstrip("0")
    n_raw = len(ev)

    lk = pd.read_csv(link_csv, dtype=str)
    lk = lk[lk["LINKPRIM"].isin(["P", "C"])].dropna(subset=["cik"])
    lk["cik"] = lk["cik"].str.lstrip("0")
    lk["LINKDT"] = pd.to_datetime(lk["LINKDT"], errors="coerce")
    # 'E' marks a link that is still current.
    lk["LINKENDDT"] = pd.to_datetime(
        lk["LINKENDDT"].replace("E", "2099-12-31"), errors="coerce"
    ).fillna(pd.Timestamp("2099-12-31"))
    lk = lk.dropna(subset=["LINKDT"])

    j = ev.merge(lk[["cik", "gvkey", "LPERMNO", "LINKDT", "LINKENDDT"]],
                 on="cik", how="inner")
    j = j[(j.session_date >= j.LINKDT) & (j.session_date <= j.LINKENDDT)]
    # A handful of accessions resolve to more than one PERMNO; keep the
    # earliest link start, which is the convention in the CCM documentation.
    j = j.sort_values("LINKDT").drop_duplicates("accession", keep="first")
    j = j.rename(columns={"LPERMNO": "permno"})
    j["permno"] = pd.to_numeric(j["permno"], errors="coerce").astype("Int64")
    j = j.dropna(subset=["permno"])
    return j.reset_index(drop=True), n_raw


#WRDS steps, one year at a time


#the CIZ file crsp.dsf_v2 runs to 2025 where the legacy crsp.dsf stops at 2024-12-31,
#so it is the only one that covers the full holdout. The CIZ equivalent of the old shrcd 10/11 and exchcd 1/2/3 screen is spelled out in
#CIZ_COMMON below.
CIZ_COMMON = """
      sharetype = 'NS'
  and securitytype = 'EQTY'
  and securitysubtype = 'COM'
  and usincflg = 'Y'
  and issuertype in ('ACOR', 'CORP')
"""


def nyse_breakpoints(db, years: list[int]) -> pd.DataFrame:
    """20th percentile of NYSE market cap, per year, computed in SQL."""
    q = f"""
        select extract(year from dlycaldt)::int as year,
               percentile_cont(0.20) within group (order by dlycap) as nyse_p20
        from crsp.dsf_v2
        where {CIZ_COMMON}
          and primaryexch = 'N'
          and dlycap is not null
          and extract(year from dlycaldt) between {min(years)} and {max(years)}
        group by 1
        order by 1
    """
    return db.raw_sql(q)


def trading_calendar(db, years: list[int]) -> pd.DatetimeIndex:
    #we retrieve all US equity trading dates over the sample period to correctly assign
    #after-close releases to the next trading session. Without this adjustment, 5,869
    #events (1.8% of the matched sample), mainly on Fridays and before holidays, would
    #be incorrectly excluded. We retrieve the calendar with a single query.
    q = f"""
        select distinct dlycaldt as date
        from crsp.dsf_v2
        where dlycaldt between date '{min(years) - 1}-11-01'
                           and date '{max(years)}-12-31'
        order by 1
    """
    return pd.DatetimeIndex(db.raw_sql(q, date_cols=["date"])["date"])


def roll_to_session(dates: pd.Series, cal: pd.DatetimeIndex) -> pd.Series:
    #we assign each release to the first trading session on or after its date.
    #we always move forward to avoid assigning returns before the news was public.
    i = cal.searchsorted(pd.DatetimeIndex(dates), side="left")
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    ok = i < len(cal)
    out[ok] = cal[i[ok]]
    return out


def crsp_for_year(db, year: int, permnos: list[int]) -> pd.DataFrame:
    lst = ",".join(str(p) for p in permnos)
    q = f"""
        select permno, dlycaldt as date,
               abs(dlyprc) as prc, dlyprc as prc_signed, dlycap as mktcap
        from crsp.dsf_v2
        where {CIZ_COMMON}
          and primaryexch in ('N', 'A', 'Q')
          and permno in ({lst})
          and dlycaldt between date '{year - 1}-11-01' and date '{year}-12-31'
    """
    return db.raw_sql(q, date_cols=["date"])


def ibes_crsp_bridge(db) -> pd.DataFrame:
    #we use the WRDS IBES ticker-to-PERMNO link with its validity dates, keeping reliable
    #matches (scores 1–3). This replaces the slower CUSIP-based matching approach.
    q = """
        select ticker, permno, sdate, edate
        from wrdsapps.ibcrsphist
        where score <= 3
    """
    return db.raw_sql(q, date_cols=["sdate", "edate"])


def analyst_estimates(db, year: int, tickers: list[str]) -> pd.DataFrame:
    lst = ",".join("'" + t.replace("'", "''") + "'" for t in tickers)
    q = f"""
        select distinct d.ticker, d.anndats, d.analys
        from ibes.detu_epsus as d
        where d.measure = 'EPS'
          and d.fpi = '6'
          and d.analys is not null
          and d.ticker in ({lst})
          and d.anndats between date '{year - 1}-09-01' and date '{year}-12-31'
    """
    return db.raw_sql(q, date_cols=["anndats"])



class Funnel:
    def __init__(self, n0: int):
        self.rows = [("raw: 8-K Item 2.02 filings 2006-2025", n0, 0, 0.0)]
        self.n = n0

    def step(self, name: str, n_after: int) -> None:
        lost = self.n - n_after
        pct = round(100.0 * lost / self.n, 2) if self.n else 0.0
        self.rows.append((name, n_after, lost, pct))
        self.n = n_after

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows,
                            columns=["step", "remaining", "removed", "pct_removed"])


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=data_path("event_table.csv"))
    ap.add_argument("--link", default=data_path("ccm_link.csv"))
    ap.add_argument("--out-sample", default=data_path("matched_sample.csv"))
    ap.add_argument("--out-funnel", default=data_path("attrition_funnel.csv"))
    ap.add_argument("--min-analysts", type=int, default=3)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--user", default=None,
                    help="WRDS username. The password is never passed here; "
                         "it is read from the local pgpass file.")
    args = ap.parse_args()

    ev, n_raw = matched_events(args.events, args.link)
    funnel = Funnel(n_raw)
    funnel.step("matched to a PERMNO, link window respected", len(ev))
    print(f"{n_raw:,} raw  ->  {len(ev):,} matched", flush=True)

    db = wrds.Connection(wrds_username=args.user) if args.user else wrds.Connection()

    years = sorted(ev["session_date"].dt.year.unique())
    years = [y for y in years if 2006 <= y <= 2025]

    cal = trading_calendar(db, years)
    rolled = roll_to_session(ev["session_date"], cal)
    moved = int((rolled.notna() & (rolled != ev["session_date"])).sum())
    off_calendar = int(rolled.isna().sum())
    ev["session_date_raw"] = ev["session_date"]
    ev["session_date"] = rolled
    ev = ev.dropna(subset=["session_date"])
    print(f"rolled to the next session: {moved:,} events "
          f"({100 * moved / max(len(rolled), 1):.2f}%), "
          f"{off_calendar:,} past the end of the calendar", flush=True)
    funnel.step("release session rolled onto the trading calendar", len(ev))

    years = sorted(ev["session_date"].dt.year.unique())
    years = [y for y in years if 2006 <= y <= 2025]

    bp = nyse_breakpoints(db, years)
    print(f"NYSE breakpoints for {len(bp)} years", flush=True)

    bridge = ibes_crsp_bridge(db)
    bridge["edate"] = bridge["edate"].fillna(pd.Timestamp("2099-12-31"))
    print(f"IBES-CRSP bridge: {len(bridge):,} rows", flush=True)

    kept = []
    #exact per-stage accounting, summed across years, so that the bottom of
    #the funnel equals the number of rows actually written.
    cnt = {"in": 0, "cal": 0, "px": 0, "price": 0, "cap": 0, "cov": 0}

    for year in years:
        sub = ev[ev["session_date"].dt.year == year].copy()
        permnos = sorted(sub["permno"].dropna().unique().tolist())
        if not permnos:
            continue

        cnt["in"] += len(sub)
        px = crsp_for_year(db, year, permnos)
        if px.empty:
            continue

        #t-2 on the market's trading calendar, not on calendar days and not on the subset of dates this year's permnos happened to trade.
        #A calendar built from the holdings themselves shifts whenever the
        #holdings change, which would make the reference date depend on the sample rather than on the market.
        sub["i"] = cal.searchsorted(pd.DatetimeIndex(sub["session_date"]),
                                    side="left")
        sub = sub[(sub["i"] >= 2) & (sub["i"] < len(cal))]
        cnt["cal"] += len(sub)
        sub["ref_date"] = cal[sub["i"].to_numpy() - 2]

        sub = sub.merge(px, left_on=["permno", "ref_date"],
                        right_on=["permno", "date"], how="inner")
        cnt["px"] += len(sub)

        sub = sub[sub["prc"] > args.min_price]
        cnt["price"] += len(sub)

        cut = bp.loc[bp["year"] == year, "nyse_p20"]
        cut = float(cut.iloc[0]) if len(cut) else 0.0
        sub = sub[sub["mktcap"] > cut]
        cnt["cap"] += len(sub)

        if sub.empty:
            continue

        #permno -> IBES ticker, honouring the bridge's validity window
        need = sub[["accession", "permno", "session_date"]].merge(
            bridge, on="permno", how="inner")
        need = need[(need.session_date >= need.sdate)
                    & (need.session_date <= need.edate)]
        tickers = sorted(need["ticker"].dropna().unique().tolist())
        if not tickers:
            continue

        cov = analyst_estimates(db, year, tickers)
        if cov.empty:
            continue
        c = need.merge(cov, on="ticker", how="left")
        c = c[(c.anndats < c.session_date)
              & (c.anndats >= c.session_date - pd.Timedelta(days=90))]
        #distinct people over the whole window, not the sum of the daily counts, so a revision does not buy a second analyst.
        agg = c.groupby("accession")["analys"].nunique().rename("n_analysts")

        sub = sub.merge(agg, on="accession", how="left")
        sub["n_analysts"] = sub["n_analysts"].fillna(0).astype(int)
        sub = sub[sub["n_analysts"] >= args.min_analysts]
        cnt["cov"] += len(sub)

        kept.append(sub)
        print(f"  {year}: {len(sub):,}", flush=True)

    out = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()

    funnel.step("inside the t-2 window of the trading calendar", cnt["cal"])
    funnel.step("has a CRSP price at t-2", cnt["px"])
    funnel.step(f"price at t-2 above ${args.min_price:g}", cnt["price"])
    funnel.step("market cap above the NYSE 20th percentile", cnt["cap"])
    funnel.step(f"at least {args.min_analysts} analysts in the prior 90 days",
                cnt["cov"])

    keep_cols = ["accession", "cik", "permno", "gvkey", "filing_date",
                 "session_date", "session_date_raw", "when", "timing_flag",
                 "prc", "mktcap", "n_analysts"]
    if not out.empty:
        out[keep_cols].sort_values("session_date").to_csv(args.out_sample,
                                                          index=False)

    f = funnel.frame()
    f.to_csv(args.out_funnel, index=False)
    print()
    print(f.to_string(index=False))
    print()
    print(f"sample -> {args.out_sample}  ({len(out):,} rows)")
    print(f"funnel -> {args.out_funnel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
