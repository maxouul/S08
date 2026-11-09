"""
15.S08 - the wire time, which is what the release time actually is.

The feedback is explicit that release time R is the wire time, that EDGAR is
the fallback when no wire carried the release, and that taking the earliest of
several noisy dates is not acceptable. We had been using EDGAR acceptance for
every event, which is the fallback applied everywhere.

RavenPack's Dow Jones press-release file turns out to be in our entitlement,
one table per year, with a UTC timestamp on every story and an entity
taxonomy. That gives a genuine wire time for the subset of releases Dow Jones
carried, so EDGAR becomes the fallback it was meant to be rather than the only
source.

Three points of care.

  The timestamp is UTC and the session rule is in New York time, so every
  story is converted before anything is compared. This is the same trap the
  feedback flags for the EDGAR submissions JSON.
  Only the earliest story per firm and day is kept. A release generates
  dozens of follow-ups and summaries, and the last of them is not the release.
  The entity is mapped to PERMNO through CUSIP with validity windows, because
  an entity identifier reused after a merger would otherwise attach one firm's
  wire time to another firm's filing.

    python wire_times.py --years 2006-2025
"""

from __future__ import annotations

from paths import BASE, src, dst

import argparse

import pandas as pd
import wrds

# A release can be tagged with any of the earnings categories. Restricting to
# the bare "earnings" tag would drop the releases RavenPack classified by
# direction, which are exactly the informative ones.
EARNINGS_LIKE = "earnings%%"

# Relevance 100 means the story is about this entity rather than mentioning
# it. Anything lower attaches a peer's release time to our filing.
MIN_RELEVANCE = 100


def wire_for_year(db: wrds.Connection, year: int) -> pd.DataFrame:
    """Earliest Dow Jones earnings story per entity and UTC day."""
    sql = f"""
        select rp_entity_id,
               min(timestamp_utc) as wire_utc
        from ravenpack_dj.rpa_djpr_equities_{year}
        where relevance >= {MIN_RELEVANCE}
          and category like '{EARNINGS_LIKE}'
          and news_type = 'PRESS-RELEASE'
        group by rp_entity_id, rpa_date_utc
    """
    try:
        return db.raw_sql(sql)
    except Exception as e:
        # Some vintages lack news_type. Fall back rather than lose the year,
        # and say so, because the looser filter admits wire stories that are
        # not the press release itself.
        print(f"  {year}: retrying without news_type ({str(e).splitlines()[0][:60]})",
              flush=True)
        return db.raw_sql(sql.replace(
            "and news_type = 'PRESS-RELEASE'", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2006-2025")
    ap.add_argument("--out", default=dst("wire_times.csv"))
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.years.split("-"))

    db = wrds.Connection(wrds_username="mahaut22")

    maps = db.raw_sql("""
        select rp_entity_id, cusip, range_start, range_end
        from ravenpack_common.wrds_rpa_all_mappings
        where cusip is not null
    """, date_cols=["range_start", "range_end"])
    maps["cusip8"] = maps.cusip.str.strip().str[:8]
    print(f"entity to cusip rows: {len(maps):,}", flush=True)

    # CRSP historical CUSIP, so a firm that changed CUSIP is matched on the
    # one it carried at the time rather than its last one.
    names = db.raw_sql("""
        select permno, cusip as cusip8, namedt, nameenddt
        from crsp.stocknames_v2
        where cusip is not null
    """, date_cols=["namedt", "nameenddt"])
    names["cusip8"] = names.cusip8.str.strip().str[:8]
    print(f"crsp cusip rows: {len(names):,}", flush=True)

    frames = []
    for y in range(lo, hi + 1):
        w = wire_for_year(db, y)
        w["wire_utc"] = pd.to_datetime(w.wire_utc, errors="coerce", utc=True)
        w = w.dropna(subset=["wire_utc"])
        frames.append(w)
        print(f"  {y}: {len(w):,} entity-days", flush=True)
    db.close()

    wire = pd.concat(frames, ignore_index=True)
    wire["wire_et"] = wire.wire_utc.dt.tz_convert("America/New_York")
    wire["wire_date"] = wire.wire_et.dt.date.astype("string")

    m = wire.merge(maps[["rp_entity_id", "cusip8", "range_start", "range_end"]],
                   on="rp_entity_id", how="inner")
    d = m.wire_et.dt.tz_localize(None)
    m = m[(d >= m.range_start) & (d <= m.range_end.fillna(pd.Timestamp("2100-01-01")))]

    m = m.merge(names, on="cusip8", how="inner")
    d = m.wire_et.dt.tz_localize(None)
    m = m[(d >= m.namedt) & (d <= m.nameenddt)]
    m["permno"] = m.permno.astype("Int64")

    out = (m.groupby(["permno", "wire_date"], as_index=False)
             .wire_et.min()
             .rename(columns={"wire_et": "wire_time_et"}))
    # Written with the offset, not as a bare clock reading. A naive
    # "16:05:00" is Eastern here and UTC to the next reader, and reading it
    # as UTC moves a release from after the close to the middle of the
    # session, which is a look-ahead of four hours on exactly the events
    # the project is about. The offset makes that mistake impossible.
    out["wire_time_et"] = out.wire_time_et.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    out.to_csv(args.out, index=False)

    print(f"\n{len(out):,} firm-days with a Dow Jones wire time")
    print(f"distinct permnos: {out.permno.nunique():,}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
