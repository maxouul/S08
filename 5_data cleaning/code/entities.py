"""
Step 5.3: Collect, for each accession, the names the masking step will have to hide.

For every event of the release clock we gather:
- the company name EDGAR gives (company in event_table.csv);
- every CRSP name and ticker the security carried up to the event date, so former
  names are masked too, and the ones valid on that date (crsp.stocknames_v2);
- the I/B/E/S ticker valid on that date (wrdsapps.ibcrsphist, score <= 3).

The CRSP names and the I/B/E/S link are read from the WRDS pulls part 3-4 already
cached (3_4_surprise_returns/cache: crsp_names.parquet, link_ibes_crsp.parquet), so
this script does not query WRDS. The event date is the filing date, which the timing corrections do not move.

The output holds WRDS data: it stays on our private drive and must never be pushed
to a public repository.

Output: data/entities.parquet

Usage: python code/entities.py
"""

from __future__ import annotations

import argparse

import pandas as pd

import settings as S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock", default=str(S.CLOCK))
    ap.add_argument("--events", default=str(S.EVENT_TABLE))
    ap.add_argument("--names", default=str(S.CRSP_NAMES))
    ap.add_argument("--ibes-link", default=str(S.IBES_LINK))
    ap.add_argument("--out", default=str(S.ENTITIES))
    args = ap.parse_args()

    ms = pd.read_csv(args.clock, usecols=["accession", "permno", "filing_date"],
                     dtype={"accession": str})
    ms = ms.drop_duplicates("accession")
    ms["permno"] = pd.to_numeric(ms.permno, errors="coerce").astype("Int64")
    ms["d"] = pd.to_datetime(ms.filing_date, errors="coerce")
    ev = pd.read_csv(args.events, usecols=["accession", "company"], dtype=str)
    ms = ms.merge(ev.drop_duplicates("accession"), on="accession", how="left")
    print(f"{len(ms):,} events, {ms.permno.nunique():,} permnos", flush=True)

    names = pd.read_parquet(args.names)
    name_col = next(c for c in ("comnam", "issuernm", "securitynm") if c in names.columns)
    names = names[["permno", "namedt", "nameenddt", "ticker", name_col]] \
        .rename(columns={name_col: "comnam"})
    bridge = pd.read_parquet(args.ibes_link)
    bridge = bridge[pd.to_numeric(bridge.score, errors="coerce") <= 3] \
        .rename(columns={"ticker": "ibes_ticker"})
    print(f"crsp name rows {len(names):,} ({name_col}), ibes link rows {len(bridge):,}",
          flush=True)

    for df in (names, bridge):
        df["permno"] = pd.to_numeric(df.permno, errors="coerce").astype("Int64")
    for c in ("namedt", "nameenddt"):
        names[c] = pd.to_datetime(names[c], errors="coerce")
    for c in ("sdate", "edate"):
        bridge[c] = pd.to_datetime(bridge[c], errors="coerce")
    names["nameenddt"] = names.nameenddt.fillna(pd.Timestamp("2099-12-31"))
    bridge["edate"] = bridge.edate.fillna(pd.Timestamp("2099-12-31"))
    for c in ("ticker", "comnam"):
        names[c] = names[c].astype("string").str.strip()
    bridge["ibes_ticker"] = bridge.ibes_ticker.astype("string").str.strip()

    # Names up to the event: every row that had started by then.
    j = ms[["accession", "permno", "d"]].merge(names, on="permno", how="left")
    upto = j[j.namedt <= j.d]
    hist = upto.groupby("accession").agg(
        crsp_names=("comnam", lambda s: sorted(set(s.dropna()))),
        crsp_tickers=("ticker", lambda s: sorted(set(s.dropna()))))
    now = upto[upto.nameenddt >= upto.d].sort_values("namedt").groupby("accession").last()
    now = now[["comnam", "ticker"]].rename(columns={"comnam": "crsp_name_at_event",
                                                    "ticker": "crsp_ticker_at_event"})

    b = ms[["accession", "permno", "d"]].merge(bridge, on="permno", how="inner")
    b = b[(b.sdate <= b.d) & (b.d <= b.edate)].sort_values("sdate")
    ib = b.groupby("accession").ibes_ticker.last()

    out = (ms[["accession", "permno", "filing_date", "company"]]
           .merge(hist, on="accession", how="left")
           .merge(now, on="accession", how="left")
           .merge(ib, on="accession", how="left"))
    for c in ("crsp_names", "crsp_tickers"):
        out[c] = out[c].apply(lambda v: v if isinstance(v, list) else [])
    out.to_parquet(args.out, index=False)

    print(f"\nevents                      : {len(out):,}")
    print(f"  with an EDGAR company name : {out.company.notna().sum():,}")
    print(f"  with a CRSP name history   : {(out.crsp_names.str.len() > 0).sum():,}")
    print(f"  with a CRSP ticker at date : {out.crsp_ticker_at_event.notna().sum():,}")
    print(f"  with an I/B/E/S ticker     : {out.ibes_ticker.notna().sum():,}")
    print(f"  with more than one name    : {(out.crsp_names.str.len() > 1).sum():,}")
    print(f"-> {args.out}  (WRDS data: private drive only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
