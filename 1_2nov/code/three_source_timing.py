"""
15.S08 - which session does each release belong to, according to all three sources?

EDGAR acceptance, I/B/E/S ANNDATS/ANNTIMS and the Dow Jones wire each imply a
trading session. The feedback asks for an event to be kept only when two
sources agree, and for the drop count to be reported. It also says the release
time is the wire time, so where the wire speaks it decides and EDGAR is the
fallback.

This script does three things and refuses to do a fourth. It computes the
session each source implies, it reports every pattern of agreement rather than
only the headline rate, and it writes the sample of conflicts to hand-check.
It does not pick a winner by taking the earliest timestamp, which is the
specific thing we were told not to do.

    python three_source_timing.py
"""

from __future__ import annotations

from paths import BASE, src, dst

import argparse
import os

import pandas as pd

CLOSE_S = 16 * 3600
OPEN_S = 9 * 3600 + 1800


OFFSET_RE = r"[+-]\d{2}:?\d{2}$|Z$"


def parse_et(s: pd.Series) -> pd.Series:
    """Read a timestamp column and return New York time, naive.

    The column may or may not carry a UTC offset. Older files written by
    wire_times.py do not; current ones do. Passing utc=True unconditionally
    would read a bare "16:05:00" as UTC and move the release to 12:05 New
    York, which is the look-ahead this project exists to avoid. Passing it
    never would fail on a column whose offset changes with daylight saving.
    So the column is inspected before it is parsed.
    """
    txt = s.astype("string").str.strip()
    has_offset = txt.dropna().str.contains(OFFSET_RE, regex=True, na=False)
    if len(has_offset) and has_offset.any():
        t = pd.to_datetime(txt, errors="coerce", utc=True)
        return t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return pd.to_datetime(txt, errors="coerce")


def to_et(ts: pd.Series) -> pd.Series:
    """New York time, whatever zone the caller handed over.

    The session rule is written in New York hours. Anything tz-aware is
    converted; anything naive is assumed to be New York already, which is
    what the files in this project carry. Without this the hour read off a
    UTC timestamp is four or five hours ahead and a release after the close
    is filed as intraday.
    """
    t = pd.to_datetime(ts, errors="coerce")
    if getattr(t.dtype, "tz", None) is not None:
        t = t.dt.tz_convert("America/New_York").dt.tz_localize(None)
    return t


def session(dates: pd.Series, secs: pd.Series) -> pd.Series:
    """The session implied by a date and a seconds-past-midnight time.

    After the close belongs to the next calendar day, before the open to the
    same one. A missing time yields a missing session rather than a guess,
    because a release with no time cannot corroborate anything.
    """
    d = to_et(dates)
    missing = secs.isna() | (secs == 0)
    after = ~missing & (secs >= CLOSE_S)
    s = d.dt.date.astype("string")
    s = s.where(~after, (d + pd.Timedelta(days=1)).dt.date.astype("string"))
    return s.where(d.notna() & ~missing)


def secs_of(ts: pd.Series) -> pd.Series:
    t = to_et(ts)
    return t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second


def roll_to_calendar(sess: pd.Series, path: str) -> pd.Series:
    """The first trading session on or after each session date.

    An after-close release is moved to the next calendar day, so a Friday
    evening release lands on a Saturday, while the EDGAR session it is
    compared with has been rolled to Monday by wrds_match_and_funnel.py.
    Rolled forward on the same CRSP calendar (forward, never backward), the
    sources are comparable and a weekend no longer reads as a disagreement.
    """
    if not os.path.exists(path):
        print("no trading_calendar.csv yet (wrds_match_and_funnel.py writes "
              "it): sessions are left on calendar days")
        return sess
    cal = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path).date)).sort_values()
    d = pd.to_datetime(sess, errors="coerce")
    i = cal.searchsorted(d, side="left")
    ok = d.notna().to_numpy() & (i < len(cal))
    out = pd.Series(pd.NA, index=sess.index, dtype="string")
    out[ok] = cal[i[ok]].strftime("%Y-%m-%d")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-json-times", action="store_true",
                    help="drop events whose EDGAR time still comes from "
                         "the submissions JSON. On by default once "
                         "fetch_acceptance.py has covered everything; "
                         "until then it would throw away most of the "
                         "sample, so it is a flag.")
    ap.add_argument("--keep-all", action="store_true",
                    help="write every event, corroborated or not. "
                         "The default keeps only the events two "
                         "sources place in the same session.")
    args = ap.parse_args()

    t = pd.read_csv(src("timing_validation.csv"), dtype={"accession": str})
    w = pd.read_csv(src("wire_times.csv"))

    # we match the wire on the release timestamp, not on the session date.
    # a date rounds everything to midnight, so "nearest" would pick the
    # story closest to midnight instead of the one closest to the release.
    # release_et falls back to acceptance_et on an older event table.
    head = pd.read_csv(src("event_table.csv"), nrows=0).columns
    col = "release_et" if "release_et" in head else "acceptance_et"
    if col != "release_et":
        print("event_table.csv has no release_et column, so the 06:00 rule "
              "for filings disseminated the next morning is not applied "
              "here. Rebuild it with the current build_event_table.py.")
    # acceptance_source says where each EDGAR time came from. The ones from
    # the submissions JSON are wrong for two filings in three and they are
    # carried here so that nothing downstream has to take them on trust.
    want = ["accession", col]
    if "acceptance_source" in head:
        want.append("acceptance_source")
    ev = pd.read_csv(src("event_table.csv"), dtype={"accession": str},
                     usecols=want).rename(columns={col: "release_et"})
    ev = ev.drop_duplicates("accession")
    t = t.merge(ev, on="accession", how="left", validate="many_to_one")
    if "acceptance_source" not in t.columns:
        t["acceptance_source"] = "json"
    print(f"{len(t):,} events, {len(w):,} firm-days with a wire time, "
          f"{t.release_et.notna().sum():,} with a release timestamp")

    from_json = (t.acceptance_source == "json")
    print(f"EDGAR time read from the filing header: "
          f"{(~from_json).sum():,}, still from the JSON: {from_json.sum():,}")
    if args.drop_json_times:
        t = t[~from_json]
        print(f"dropping the JSON ones leaves {len(t):,} events")
    elif from_json.any():
        print("those rows are kept and flagged. Pass --drop-json-times to "
              "exclude them, or finish fetch_acceptance.py first.")

    t["permno"] = pd.to_numeric(t.permno, errors="coerce").astype("Int64")
    w["permno"] = pd.to_numeric(w.permno, errors="coerce").astype("Int64")

    # nearest story on the same firm within three days, same tolerance as
    # I/B/E/S. parse_et reads the offset if the column has one: reading a
    # bare "16:05" as UTC would move the release to 12:05 New York.
    w["wire_dt"] = parse_et(w.wire_time_et)
    t["edgar_dt"] = parse_et(t.release_et)
    t = t.dropna(subset=["edgar_dt"])
    t = t.sort_values("edgar_dt")
    w = w.sort_values("wire_dt").dropna(subset=["wire_dt"])

    m = pd.merge_asof(t, w[["permno", "wire_dt"]], left_on="edgar_dt",
                      right_on="wire_dt", by="permno", direction="nearest",
                      tolerance=pd.Timedelta("3D"))

    cal_path = src("trading_calendar.csv")
    m["wire_session"] = roll_to_calendar(session(m.wire_dt, secs_of(m.wire_dt)),
                                         cal_path)
    m["ibes_session_c"] = roll_to_calendar(m.ibes_session, cal_path)
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

    # the wire is the release time where it speaks, EDGAR is the fallback.
    # we settle this first because the rule below is about this column.
    m["session_final"] = m.wire_session.where(has_w, m.edgar_session)
    m["session_source"] = pd.Series("edgar", index=m.index).where(
        ~has_w, "wire")

    # two sources must confirm the session we keep, not just agree with
    # each other. if EDGAR and I/B/E/S say Monday while we keep the wire's
    # Tuesday, that is two sources agreeing and none confirming.
    c_e = m.edgar_session == m.session_final
    c_i = has_i & (m.ibes_session_c == m.session_final)
    c_w = has_w & (m.wire_session == m.session_final)
    confirm = c_e.astype(int) + c_i.astype(int) + c_w.astype(int)
    two = confirm >= 2
    m["n_confirm"] = confirm
    m["agree_two"] = two

    pairs_only = ((ei.astype(int) + ew.astype(int) + iw.astype(int)) >= 1)
    waved = (pairs_only & ~two).sum()

    print(f"\nkeeping an event only when two sources confirm the session we "
          f"keep")
    print(f"  kept     {two.sum():>8,} ({100*two.mean():.1f}%)")
    print(f"  dropped  {n - two.sum():>8,} ({100*(1-two.mean()):.1f}%)")
    print(f"  of which {waved:>8,} would have passed on pairwise agreement, "
          f"with the pair agreeing on a session we do not keep")

    only_edgar = ~has_i & ~has_w
    print(f"    of the dropped, with no second source at all: "
          f"{(only_edgar).sum():,}")
    print(f"    of the dropped, with a second source that disagrees: "
          f"{(n - two.sum() - only_edgar.sum()):,}")

    moved = (m.session_final != m.edgar_session).sum()
    print(f"\nadopting the wire where it exists moves {moved:,} events "
          f"({100*moved/n:.1f}%) to another session")

    print("\nhow far the wire moves the session, in sessions")
    delta = (pd.to_datetime(m.session_final, errors="coerce")
             - pd.to_datetime(m.edgar_session, errors="coerce")).dt.days
    print(delta[delta != 0].value_counts().head(8).sort_index().to_string())

    cols = ["accession", "permno", "edgar_session", "when", "ibes_session_c",
            "ibes_flag", "wire_dt", "wire_session", "session_final",
            "session_source", "n_confirm", "agree_two",
            "acceptance_source"]

    # The corroborated set is what the rest of the project reads. Keeping
    # everything and offering the agreeing subset as an option puts the
    # burden the wrong way round: an event no second source confirms is an
    # event whose session we are guessing, and the guess would reach the
    # return windows unless somebody remembered to filter.
    m[cols].to_csv(dst("three_source_timing_all.csv"), index=False)
    kept = m if args.keep_all else m[two]
    kept[cols].to_csv(dst("three_source_timing.csv"), index=False)
    print(f"\nthree_source_timing.csv holds {len(kept):,} events"
          + ("" if args.keep_all else
             f", {n - len(kept):,} dropped for want of a second source"))

    # Fifty events for the hand check against the wire, drawn from the
    # conflicts rather than at random: a random fifty would be almost all
    # agreements and would tell us nothing we do not already know. Seeded,
    # so the same fifty come back on a rerun.
    conflict = m[has_w & has_i & ~(ei & ew & iw)]
    take = conflict.sample(min(50, len(conflict)), random_state=20261104)
    hand = take[cols].copy()
    hand["edgar_url"] = [
        "https://www.sec.gov/Archives/edgar/data/"
        + f"{a.replace('-', '')}/{a}-index.htm" for a in hand.accession]
    for c in ["session_you_read", "source_you_trust", "note"]:
        hand[c] = ""
    hand.to_csv(dst("timing_handcheck_50.csv"), index=False,
                encoding="utf-8-sig")
    print("-> three_source_timing_all.csv\n-> three_source_timing.csv"
          "\n-> timing_handcheck_50.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
