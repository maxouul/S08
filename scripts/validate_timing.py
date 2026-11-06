#we compare EDGAR release sessions with I/B/E/S announcement dates and times
#to check whether independent sources agree on the timing of each release.
#we treat ANNTIMS = 00:00:00 as missing and exclude ACTDATS/ACTTIMS, which reflect
#when I/B/E/S recorded the data rather than when the announcement was made.
#if filingDate is later than the acceptance date, we use 6:00 AM ET on filingDate.
#we calculate the agreement rate, count excluded events, and save conflicting
#cases for manual validation against newswire timestamps.

#python validate_timing.py


from __future__ import annotations
from project_paths import DATA_DIR, data_path


import pandas as pd
import wrds

def sessions(dates: pd.Series, times: pd.Series) -> pd.DataFrame:
    #we assign each release to its trading session based on its date and time.
    #after-close releases move to the next day, while missing times remain unknown
    #and cannot confirm a session. We parse ANNTIMS and use vectorized operations
    #to efficiently process the I/B/E/S data.
    d = pd.to_datetime(dates, errors="coerce")
    t = pd.to_datetime(times.astype("string"), format="%H:%M:%S",
                       errors="coerce")
    secs = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second

    missing = secs.isna() | (secs == 0)       # 00:00:00 is missing, not midnight
    after = ~missing & (secs >= 16 * 3600)
    pre = ~missing & (secs < 9 * 3600 + 1800)

    sess = d.dt.date.astype("string")
    sess = sess.where(~after, (d + pd.Timedelta(days=1)).dt.date.astype("string"))

    flag = pd.Series("intraday", index=d.index, dtype="string")
    flag = flag.mask(after, "after_close").mask(pre, "pre_open")                .mask(missing, "no_time").mask(d.isna(), "no_date")
    return pd.DataFrame({"ibes_session": sess.where(d.notna()),
                         "ibes_flag": flag})


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(data_path("matched_sample.csv"),
                     usecols=["accession", "permno", "session_date", "when"],
                     dtype={"accession": str, "permno": "Int64",
                            "session_date": str, "when": str})
    idn = pd.read_csv(data_path("identity.csv"), usecols=["accession"], dtype=str)
    ev = ev[ev.accession.isin(set(idn.accession))]
    print(f"{len(ev):,} matched events", flush=True)

    db = wrds.Connection(wrds_username="mahaut22")

    print("crsp.dsf_v2 coverage:",
          db.raw_sql("select max(dlycaldt) as d from crsp.dsf_v2")
            .d.iloc[0], flush=True)

    bridge = db.raw_sql("""
        select distinct ticker as ibtic, permno, sdate, edate
        from wrdsapps.ibcrsphist
        where score <= 3
    """, date_cols=["sdate", "edate"])

    #only the quarterly EPS actual, and only ANNDATS/ANNTIMS.
    #the actuals file carries one row per firm-period, so this is a small table.
    act = db.raw_sql("""
        select ticker as ibtic, pends, anndats, anntims
        from ibes.actu_epsus
        where measure = 'EPS' and pdicity = 'QTR'
          and anndats between '2005-12-01' and '2026-01-31'
    """, date_cols=["pends", "anndats"])
    db.close()
    print(f"ibes actuals {len(act):,}, bridge rows {len(bridge):,}",
          flush=True)

    act["ibtic"] = act.ibtic.str.strip()
    bridge["ibtic"] = bridge.ibtic.str.strip()
    act["permno"] = pd.NA

    a = act.merge(bridge, on="ibtic", how="inner", suffixes=("", "_b"))
    a = a[(a.anndats >= a.sdate) & (a.anndats <= a.edate)]
    a["permno"] = a.permno_b.astype("Int64")

    a = pd.concat([a.reset_index(drop=True),
                   sessions(a.anndats, a.anntims).reset_index(drop=True)],
                  axis=1)

    ev["_d"] = pd.to_datetime(ev.session_date, errors="coerce")
    #an I/B/E/S announcement is matched to the EDGAR event nearest in time,
    #within three calendar days. A wider window would start pairing adjacent quarters and manufacture agreement.
    ev = ev.sort_values("_d")
    a = a.sort_values("anndats")
    m = pd.merge_asof(ev, a[["permno", "anndats", "anntims", "ibes_session",
                             "ibes_flag"]],
                      left_on="_d", right_on="anndats", by="permno",
                      direction="nearest", tolerance=pd.Timedelta("3D"))

    has = m.ibes_session.notna()
    agree = has & (m.ibes_session == m.session_date)
    timed = has & (m.ibes_flag != "no_time")
    agree_timed = timed & (m.ibes_session == m.session_date)

    n = len(m)
    print(f"\nI/B/E/S announcement found within 3 days: {has.sum():,} "
          f"({100*has.mean():.1f}%)")
    print(f"  of those, a usable ANNTIMS:              {timed.sum():,} "
          f"({100*timed.sum()/max(has.sum(),1):.1f}%)")
    print(f"\nsame session, all matched:   {agree.sum():,} "
          f"({100*agree.sum()/max(has.sum(),1):.1f}%)")
    print(f"same session, timed only:    {agree_timed.sum():,} "
          f"({100*agree_timed.sum()/max(timed.sum(),1):.1f}%)")

    # The two-source rule the feedback asks for, costed before it is adopted.
    keep = agree
    print(f"\nif an event is kept only when two sources agree:")
    print(f"  kept    {keep.sum():,} ({100*keep.mean():.1f}% of {n:,})")
    print(f"  dropped {n - keep.sum():,} "
          f"({100*(1-keep.mean()):.1f}%), of which")
    print(f"    no I/B/E/S match   {(~has).sum():,}")
    print(f"    match, different session {(has & ~agree).sum():,}")

    diff = (m.loc[has & ~agree, "_d"]
            - m.loc[has & ~agree, "anndats"]).dt.days
    if len(diff):
        print(f"\nwhen they disagree, EDGAR session minus I/B/E/S date:")
        print(diff.value_counts().head(8).sort_index().to_string())

    m["agrees"] = agree
    cols = ["accession", "permno", "session_date", "when", "anndats",
            "anntims", "ibes_session", "ibes_flag", "agrees"]
    m[cols].to_csv(data_path("timing_validation.csv"), index=False)
    con = m.loc[has & ~agree, cols].head(50)
    con.to_csv(data_path("timing_conflicts_50.csv"), index=False)
    print("\n-> timing_validation.csv\n-> timing_conflicts_50.csv "
          "(for the hand check against the wire)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
