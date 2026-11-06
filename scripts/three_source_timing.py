#we compare trading sessions from EDGAR, I/B/E/S, and Dow Jones, keeping events
#only when at least two sources agree. We use wire timestamps when available
#and EDGAR as a fallback, without selecting the earliest timestamp.
#we report agreement patterns, count excluded events, and save conflicts for manual validation.

#python three_source_timing.py

from __future__ import annotations

import argparse

import pandas as pd

CLOSE_S = 16 * 3600
OPEN_S = 9 * 3600 + 1800


OFFSET_RE = r"[+-]\d{2}:?\d{2}$|Z$"


def parse_et(s: pd.Series) -> pd.Series:
    #we convert timestamps to New York time, handling both older files without UTC
    #offsets and newer files with offsets. We check the format before parsing to
    #avoid timezone errors and look-ahead bias, including daylight saving changes.
    txt = s.astype("string").str.strip()
    has_offset = txt.dropna().str.contains(OFFSET_RE, regex=True, na=False)
    if len(has_offset) and has_offset.any():
        t = pd.to_datetime(txt, errors="coerce", utc=True)
        return t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return pd.to_datetime(txt, errors="coerce")


def to_et(ts: pd.Series) -> pd.Series:
    #we convert timezone-aware timestamps to New York time and assume naive timestamps
    #are already in New York time. This ensures correct trading session assignment
    #and avoids classifying after-close releases as intraday.
    t = pd.to_datetime(ts, errors="coerce")
    if getattr(t.dtype, "tz", None) is not None:
        t = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return t


def session(dates: pd.Series, secs: pd.Series) -> pd.Series:
    #we assign releases to sessions based on their date and time, moving after-close
    #releases to the next calendar day. Missing times result in missing sessions
    #rather than estimated ones.
    d = to_et(dates)
    missing = secs.isna() | (secs == 0)
    after = ~missing & (secs >= CLOSE_S)
    s = d.dt.date.astype("string")
    s = s.where(~after, (d + pd.Timedelta(days=1)).dt.date.astype("string"))
    return s.where(d.notna() & ~missing)


def secs_of(ts: pd.Series) -> pd.Series:
    t = to_et(ts)
    return t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-all", action="store_true",
                    help="write every event, corroborated or not. "
                         "The default keeps only the events two "
                         "sources place in the same session.")
    args = ap.parse_args()

    t = pd.read_csv("timing_validation.csv", dtype={"accession": str})
    w = pd.read_csv("wire_times.csv")

    #we match wire stories using the exact release timestamp rather than the session
    #date to avoid incorrect matches. We use release_et when available, or fall back to acceptance_et for older event tables.

    head = pd.read_csv("event_table.csv", nrows=0).columns
    col = "release_et" if "release_et" in head else "acceptance_et"
    if col != "release_et":
        print("event_table.csv has no release_et column, so the 06:00 rule "
              "for filings disseminated the next morning is not applied "
              "here. Rebuild it with the current build_event_table.py.")
    ev = pd.read_csv("event_table.csv", dtype={"accession": str},
                     usecols=["accession", col]).rename(
                         columns={col: "release_et"})
    ev = ev.drop_duplicates("accession")
    t = t.merge(ev, on="accession", how="left", validate="many_to_one")
    print(f"{len(t):,} events, {len(w):,} firm-days with a wire time, "
          f"{t.release_et.notna().sum():,} with a release timestamp")

    t["permno"] = pd.to_numeric(t.permno, errors="coerce").astype("Int64")
    w["permno"] = pd.to_numeric(w.permno, errors="coerce").astype("Int64")

    #each event is matched to the closest wire story for the same firm within three days,
    #using the same tolerance as I/B/E/S. Both timestamps are parsed as UTC using their
    #stored timezone offsets to avoid incorrect matches and timing errors.

    w["wire_dt"] = parse_et(w.wire_time_et)
    t["edgar_dt"] = parse_et(t.release_et)
    t = t.dropna(subset=["edgar_dt"])
    t = t.sort_values("edgar_dt")
    w = w.sort_values("wire_dt").dropna(subset=["wire_dt"])

    m = pd.merge_asof(t, w[["permno", "wire_dt"]], left_on="edgar_dt",
                      right_on="wire_dt", by="permno", direction="nearest",
                      tolerance=pd.Timedelta("3D"))

    m["wire_session"] = session(m.wire_dt, secs_of(m.wire_dt))
    m["ibes_session_c"] = m.ibes_session
    m["edgar_session"] = m.session_date

    has_w = m.wire_session.notna()
    has_i = m.ibes_session_c.notna()
    n = len(m)

    print(f"\ncoverage")
    print(f"  EDGAR session            {n:>8,} (100.0%)")
    print(f"  I/B/E/S session          {has_i.sum():>8,} "
          f"({100*has_i.mean():.1f}%)")
    print(f"  Dow Jones wire session   {has_w.sum():>8,} "
          f"({100*has_w.mean():.1f}%)")
    print(f"  all three                {(has_i & has_w).sum():>8,} "
          f"({100*(has_i & has_w).mean():.1f}%)")

    ei = has_i & (m.edgar_session == m.ibes_session_c)
    ew = has_w & (m.edgar_session == m.wire_session)
    iw = has_i & has_w & (m.ibes_session_c == m.wire_session)

    print(f"\npairwise agreement, among events where both sources exist")
    print(f"  EDGAR with I/B/E/S       {ei.sum():>8,} of {has_i.sum():,} "
          f"({100*ei.sum()/max(has_i.sum(),1):.1f}%)")
    print(f"  EDGAR with the wire      {ew.sum():>8,} of {has_w.sum():,} "
          f"({100*ew.sum()/max(has_w.sum(),1):.1f}%)")
    print(f"  I/B/E/S with the wire    {iw.sum():>8,} of "
          f"{(has_i & has_w).sum():,} "
          f"({100*iw.sum()/max((has_i & has_w).sum(),1):.1f}%)")

    two = (ei.astype(int) + ew.astype(int) + iw.astype(int)) >= 1
    m["agree_two"] = two
    print(f"\nkeeping an event only when at least two sources agree")
    print(f"  kept     {two.sum():>8,} ({100*two.mean():.1f}%)")
    print(f"  dropped  {n - two.sum():>8,} ({100*(1-two.mean()):.1f}%)")

    only_edgar = ~has_i & ~has_w
    print(f"    of the dropped, with no second source at all: "
          f"{(only_edgar).sum():,}")
    print(f"    of the dropped, with a second source that disagrees: "
          f"{(n - two.sum() - only_edgar.sum()):,}")

    #where the wire speaks, it is the release time. This is the column the return windows should key on, with EDGAR as the stated fallback.

    m["session_final"] = m.wire_session.where(has_w, m.edgar_session)
    m["session_source"] = pd.Series("edgar", index=m.index).where(
        ~has_w, "wire")
    moved = (m.session_final != m.edgar_session).sum()
    print(f"\nadopting the wire where it exists moves {moved:,} events "
          f"({100*moved/n:.1f}%) to another session")

    print("\nhow far the wire moves the session, in sessions")
    delta = (pd.to_datetime(m.session_final, errors="coerce")
             - pd.to_datetime(m.edgar_session, errors="coerce")).dt.days
    print(delta[delta != 0].value_counts().head(8).sort_index().to_string())

    cols = ["accession", "permno", "edgar_session", "when", "ibes_session_c",
            "ibes_flag", "wire_dt", "wire_session", "session_final",
            "session_source", "agree_two"]


    m[cols].to_csv("three_source_timing_all.csv", index=False)
    kept = m if args.keep_all else m[two]
    kept[cols].to_csv("three_source_timing.csv", index=False)
    print(f"\nthree_source_timing.csv holds {len(kept):,} events"
          + ("" if args.keep_all else
             f", {n - len(kept):,} dropped for want of a second source"))


    conflict = m[has_w & has_i & ~(ei & ew & iw)]
    take = conflict.sample(min(50, len(conflict)), random_state=20261104)
    hand = take[cols].copy()
    hand["edgar_url"] = [
        "https://www.sec.gov/Archives/edgar/data/"
        + f"{a.replace('-', '')}/{a}-index.htm" for a in hand.accession]
    for c in ["session_you_read", "source_you_trust", "note"]:
        hand[c] = ""
    hand.to_csv("timing_handcheck_50.csv", index=False,
                encoding="utf-8-sig")
    print("-> three_source_timing_all.csv\n-> three_source_timing.csv"
          "\n-> timing_handcheck_50.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
