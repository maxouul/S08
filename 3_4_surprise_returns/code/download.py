"""
01 Download: the only code in parts 3-4 that talks to WRDS.

Every pull is written to cache/ as parquet, one file per table (and per year
for the big ones). A file that already exists is skipped, so after a Colab
disconnect you rerun the same cell and only the missing years are pulled.
Each finished pull is logged in cache/manifest.csv with its row count, the
hash of its SQL and the time it was pulled.

Nothing here depends on Mahaut's files. The event list is built later from
the I/B/E/S actuals pulled here.

cache/ holds WRDS data: it stays in the team's private storage and never goes
into the submitted package.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import time
from dataclasses import dataclass, field

import pandas as pd

from common import PROJECT_DIR

CACHE = PROJECT_DIR / "cache"
MANIFEST = CACHE / "manifest.csv"

# One year of look-back before the first 2006 announcement (estimates,
# beta estimation), through the end of CRSP.
YEARS = list(range(2005, 2026))

# Ordinary common shares of US-incorporated operating companies (CIZ fields),
# the same filter Mahaut used for the Nov 2 funnel.
COMMON_STOCK = (
    "sharetype = 'NS' and securitytype = 'EQTY' and securitysubtype = 'COM' "
    "and usincflg = 'Y' and issuertype in ('ACOR', 'CORP')"
)

# I/B/E/S tickers and CRSP permnos with a reliable link (score 1-3). Filtering
# on the server keeps every pull to the firms we can actually use.
LINKED_TICKERS = "select ticker from wrdsapps.ibcrsphist where score <= 3"
LINKED_PERMNOS = "select permno from wrdsapps.ibcrsphist where score <= 3"


@dataclass
class Pull:
    sql: str                      # may contain {lo} and {hi} for yearly pulls
    per_year: bool = False
    date_cols: list[str] = field(default_factory=list)
    note: str = ""


CORE = {
    # --- links and reference tables (small, one file each) -----------------
    "link_ibes_crsp": Pull(
        "select ticker, permno, ncusip, sdate, edate, score from wrdsapps.ibcrsphist",
        date_cols=["sdate", "edate"],
        note="I/B/E/S ticker to CRSP permno, with validity dates"),
    "link_ccm": Pull(
        "select gvkey, lpermno as permno, lpermco as permco, linktype, linkprim, liid, "
        "linkdt, linkenddt from crsp.ccmxpf_lnkhist "
        "where linktype in ('LU', 'LC', 'LS') and lpermno is not null",
        date_cols=["linkdt", "linkenddt"],
        note="Compustat gvkey to CRSP permno (usable link types only)"),
    "link_cik_gvkey": Pull(
        "select cik, gvkey, source, link_desc, sec_start_date, sec_end_date, "
        "link_start_date, link_end_date from wrdssec.wciklink_gvkey",
        date_cols=["sec_start_date", "sec_end_date", "link_start_date", "link_end_date"],
        note="EDGAR CIK to gvkey: joins our events to Mahaut's filings later"),
    "crsp_names": Pull(
        "select permno, permco, namedt, nameenddt, cusip, ticker, issuernm, primaryexch, "
        "sharetype, securitytype, securitysubtype, usincflg, issuertype, siccd "
        "from crsp.stocknames_v2",
        date_cols=["namedt", "nameenddt"]),
    "crsp_delists": Pull(
        "select * from crsp.stkdelists where delistingdt >= date '2005-01-01'",
        date_cols=["delistingdt"],
        note="delisting returns; checked against dlyret before use"),
    "ff_daily": Pull(
        "select date, mktrf, smb, hml, rf, umd from ff.factors_daily "
        "where date between date '2004-01-01' and date '2026-12-31'",
        date_cols=["date"],
        note="Fama-French three factors plus momentum, daily"),

    # --- yearly pulls ----------------------------------------------------
    "ibes_actuals": Pull(
        "select ticker, cusip, oftic, cname, pends, measure, pdicity, anndats, anntims, "
        "actdats, acttims, value, curr_act, usfirm from ibes.actu_epsus "
        "where measure = 'EPS' and pdicity = 'QTR' "
        "and anndats between {lo} and {hi} "
        f"and ticker in ({LINKED_TICKERS})",
        per_year=True, date_cols=["pends", "anndats", "actdats"],
        note="quarterly EPS actuals, unadjusted: the event list comes from here"),
    "ibes_detail": Pull(
        "select ticker, cusip, estimator, analys, pdf, fpi, measure, value, curr, currfl, "
        "report_curr, usfirm, fpedats, anndats, anntims, actdats, acttims, revdats, revtims "
        "from ibes.detu_epsus "
        "where measure = 'EPS' and fpi in ('6', '7') "
        "and anndats between {lo} and {hi} "
        f"and ticker in ({LINKED_TICKERS})",
        per_year=True, date_cols=["fpedats", "anndats", "actdats", "revdats"],
        note="individual quarterly EPS estimates, unadjusted, one and two quarters ahead"),
    "crsp_daily": Pull(
        "select permno, dlycaldt as date, dlyprc, dlyopen, dlyclose, dlyret, dlyretx, "
        "dlyretmissflg, dlydelflg, dlycap, shrout, dlycumfacpr, dlycumfacshr, dlyvol, "
        "primaryexch, siccd from crsp.dsf_v2 "
        "where dlycaldt between {lo} and {hi} "
        f"and {COMMON_STOCK} "
        f"and permno in ({LINKED_PERMNOS})",
        per_year=True, date_cols=["date"],
        note="daily prices, opens, returns and adjustment factors for linked common stocks"),
    "ibes_stopped": Pull(
        "select ticker, estimator, pdicity, measure, fpedats, astpdats, astptims "
        "from ibes.stopu_epsus where measure = 'EPS' "
        "and astpdats between {lo} and {hi} "
        f"and ticker in ({LINKED_TICKERS})",
        per_year=True, date_cols=["fpedats", "astpdats"],
        note="brokers that stopped estimating a firm: their earlier estimates are stale"),
    "ibes_surprise_summary": Pull(
        "select * from ibes.surpsumu where measure = 'EPS' and fiscalp = 'QTR' "
        "and anndats between {lo} and {hi} "
        f"and ticker in ({LINKED_TICKERS})",
        per_year=True, date_cols=["anndats"],
        note="I/B/E/S's own surprise figures: an independent check on our SUE"),
    "nyse_breakpoints": Pull(
        "with month_end as ("
        "  select max(dlycaldt) as d from crsp.dsf_v2"
        "  where permno = 14593 and dlycaldt between {lo} and {hi}"
        "  group by date_trunc('month', dlycaldt)) "
        "select dlycaldt as date, "
        "percentile_cont(0.2) within group (order by dlycap) as nyse_p20, "
        "percentile_cont(0.5) within group (order by dlycap) as nyse_p50, "
        "count(*) as n_nyse "
        "from crsp.dsf_v2 "
        "where dlycaldt in (select d from month_end) and primaryexch = 'N' "
        f"and {COMMON_STOCK} and dlycap > 0 "
        "group by 1 order by 1",
        per_year=True, date_cols=["date"],
        note="NYSE month-end size breakpoints, for the size filter decision"),
}

# Needed for the wider baseline and the GAAP check, not for the checkpoint.
LATER = {
    "ibes_rev_actuals": Pull(
        # Revenue is only in the adjusted non-EPS tables (the unadjusted ones
        # hold per-share items only). Split adjustment does not change a total
        # like revenue, so adjusted is correct here.
        "select * from ibes.act_xepsus where measure = 'SAL' and pdicity = 'QTR' "
        "and anndats between {lo} and {hi} "
        f"and ticker in ({LINKED_TICKERS})",
        per_year=True, date_cols=["pends", "anndats", "actdats"],
        note="quarterly revenue actuals"),
    "ibes_rev_detail": Pull(
        "select * from ibes.det_xepsus where measure = 'SAL' and fpi in ('6', '7') "
        "and anndats between {lo} and {hi} "
        f"and ticker in ({LINKED_TICKERS})",
        per_year=True, date_cols=["fpedats", "anndats", "actdats", "revdats"],
        note="individual quarterly revenue estimates"),
    "ibes_guidance": Pull(
        "select * from ibes.det_guidance where measure in ('EPS', 'SAL') "
        "and anndats between {lo} and {hi} "
        f"and ticker in ({LINKED_TICKERS})",
        per_year=True, date_cols=["anndats", "actdats"],
        note="company guidance for EPS and revenue"),
    "comp_fundq": Pull(
        "select gvkey, datadate, fyearq, fqtr, fyr, rdq, tic, cusip, cik, epspxq, epsfxq, "
        "epspiq, epsfiq, opepsq, oepsxq, cshprq, ajexq, saleq, revtq from comp.fundq "
        "where indfmt = 'INDL' and datafmt = 'STD' and popsrc = 'D' and consol = 'C' "
        "and datadate between date '2004-01-01' and date '2026-06-30'",
        date_cols=["datadate", "rdq"],
        note="GAAP EPS and report dates, for the GAAP versus street comparison"),
}

# Tables whose columns we have not seen yet: we only record their schema.
PEEK = ["ibes.stopu_epsus", "ibes.excu_epsus", "ibes.surpsumu", "ibes.statsumu_epsus",
        "ibes.actu_xepsus", "ibes.detu_xepsus"]


def _rollback(db) -> None:
    try:
        db.connection.rollback()
    except Exception:
        pass


def _rows(path) -> int:
    try:
        import pyarrow.parquet as pq
        return pq.read_metadata(path).num_rows
    except Exception:
        return len(pd.read_parquet(path))


def _log(row: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(MANIFEST, mode="a", header=not MANIFEST.exists(), index=False)


def run(db, specs: dict[str, Pull], user: str, years: list[int] | None = None,
        only: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    """Pull every spec not yet cached. Returns a table of what happened."""
    years = years or YEARS
    done = []
    for name, spec in specs.items():
        if only and name not in only:
            continue
        targets = ([(y, CACHE / name / f"{y}.parquet") for y in years]
                   if spec.per_year else [(None, CACHE / f"{name}.parquet")])
        for year, path in targets:
            label = f"{name}{'' if year is None else ' ' + str(year)}"
            if path.exists() and not force:
                if _rows(path) > 0:
                    done.append({"pull": label, "status": "cached"})
                    continue
                print(f"retry   {label}: cached file is empty")
            sql = spec.sql
            if year is not None:
                sql = sql.format(lo=f"date '{year}-01-01'", hi=f"date '{year}-12-31'")
            t0 = time.time()
            try:
                df = db.raw_sql(sql, date_cols=spec.date_cols or None)
            except Exception as e:
                _rollback(db)
                msg = str(e).splitlines()[0][:160]
                print(f"FAILED  {label}: {msg}")
                done.append({"pull": label, "status": "failed", "error": msg})
                continue
            secs = round(time.time() - t0, 1)
            if df.empty:
                # An empty year is almost always a wrong filter or a missing
                # entitlement, not a fact. Not saving it means the next run
                # tries again instead of skipping it.
                print(f"EMPTY   {label}: 0 rows, not saved")
                done.append({"pull": label, "status": "empty"})
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temporary name first, so a disconnect mid-write never
            # leaves a half file that the next run would mistake for finished.
            tmp = path.with_name(path.stem + ".partial.parquet")
            df.to_parquet(tmp, index=False)
            tmp.replace(path)
            _log({"pull": name, "year": year, "rows": len(df), "seconds": secs,
                  "sql_sha1": hashlib.sha1(sql.encode()).hexdigest()[:12],
                  "pulled_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  "wrds_user": user})
            print(f"ok      {label:<26} {len(df):>10,} rows  {secs:>6.1f}s")
            done.append({"pull": label, "status": "pulled", "rows": len(df)})
    return pd.DataFrame(done)


def peek(db, tables: list[str] = PEEK) -> dict[str, list[str]]:
    """Record the columns of tables we have not used yet."""
    out = {}
    for t in tables:
        try:
            out[t] = list(db.raw_sql(f"select * from {t} limit 1").columns)
        except Exception as e:
            _rollback(db)
            out[t] = [f"error: {str(e).splitlines()[0][:120]}"]
        print(f"{t}: {', '.join(out[t])}")
    return out


def summary() -> pd.DataFrame:
    """What is in the cache: rows and disk size per pull."""
    if not MANIFEST.exists():
        print("nothing pulled yet")
        return pd.DataFrame()
    m = pd.read_csv(MANIFEST)
    m = m.drop_duplicates(["pull", "year"], keep="last")
    sizes = {}
    for p in CACHE.rglob("*.parquet"):
        if ".partial" in p.name:
            continue
        key = p.parent.name if p.parent != CACHE else p.stem
        sizes[key] = sizes.get(key, 0) + p.stat().st_size
    s = m.groupby("pull").agg(files=("rows", "size"), rows=("rows", "sum"),
                              minutes=("seconds", lambda x: round(x.sum() / 60, 1)))
    s["mb_on_disk"] = s.index.map(lambda k: round(sizes.get(k, 0) / 1e6, 1))
    return s.sort_values("rows", ascending=False)


def measure_counts(db, since: str = "2024-01-01") -> None:
    """
    Why did the revenue pulls come back empty? Count rows by measure in the
    unadjusted and adjusted non-EPS tables. Revenue is a total, not a per-share
    number, so the adjusted tables are a valid source if the unadjusted are empty.
    """
    for table in ["ibes.actu_xepsus", "ibes.act_xepsus", "ibes.detu_xepsus", "ibes.det_xepsus"]:
        try:
            q = (f"select measure, count(*) as n from {table} "
                 f"where anndats >= date '{since}' group by 1 order by 2 desc limit 8")
            r = db.raw_sql(q)
            print(f"{table}: {dict(zip(r['measure'], r['n']))}")
        except Exception as e:
            _rollback(db)
            print(f"{table}: error {str(e).splitlines()[0][:120]}")
