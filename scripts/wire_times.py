#we use RavenPack Dow Jones timestamps as the primary release time, with EDGAR
#acceptance as a fallback when no wire timestamp is available.
#we convert UTC timestamps to New York time and keep the earliest story per firm
#and day to exclude follow-ups. We match entities to PERMNO using CUSIP and
#validity windows to avoid incorrect matches after mergers.

#python wire_times.py --years 2006-2025

from __future__ import annotations

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
        #some vintages lack news_type. Fall back rather than lose the year,
        #and say so, because the looser filter admits wire stories that are not the press release itself.
        print(f"  {year}: retrying without news_type ({str(e).splitlines()[0][:60]})",
              flush=True)
        return db.raw_sql(sql.replace(
            "and news_type = 'PRESS-RELEASE'", ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2006-2025")
    ap.add_argument("--out", default="wire_times.csv")
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

    #CRSP historical CUSIP, so a firm that changed CUSIP is matched on the one it carried at the time rather than its last one.
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

    out["wire_time_et"] = out.wire_time_et.dt.strftime("%Y-%m-%d %H:%M:%S%z")
    out.to_csv(args.out, index=False)

    print(f"\n{len(out):,} firm-days with a Dow Jones wire time")
    print(f"distinct permnos: {out.permno.nunique():,}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
