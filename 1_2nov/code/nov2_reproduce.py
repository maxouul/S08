"""Recompute every number in the data report from the event-level files, and
check each one against the value printed in the report.

The report is the revision of 8 November, written on the acceptance times
read from the filing headers. Each file is read in its rebuilt version when
there is one and in its base version otherwise, the same rule paths.py
applies, so the check sees exactly what the rest of the pipeline sees.

The version submitted on 2 November was written on the submissions-JSON
times. It is not checked here any more; its figures and the data behind them
are kept beside it under their own names.
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BASE, BUILT  # 1_2nov/data and files csv, unless overridden


def pick(name):
    """The rebuilt file when there is one, the base file otherwise."""
    built = os.path.join(BUILT, name)
    return built if os.path.exists(built) else os.path.join(BASE, name)

ok, bad = 0, []


def check(label, got, want, pct=None):
    global ok
    hit = (round(got, 4) == round(want, 4))
    if hit:
        ok += 1
    else:
        bad.append(f"{label}: got {got}, report says {want}")
    tail = f"  ({pct:.1f}%)" if pct is not None else ""
    print(f"  {'ok ' if hit else 'NO '}{label:<52}{got:>12,}{tail}")


print("1. raw counts")
rc = pd.read_csv(pick("raw_counts.csv"))
# The file carries its own TOTAL row; the yearly rows are the data.
total = rc[rc.year == "TOTAL"].iloc[0]
rc = rc[rc.year != "TOTAL"].astype({"year": int})
check("TOTAL row agrees with the sum of the years",
      int(rc.item_2_02_filings.sum()), int(total.item_2_02_filings))
check("Item 2.02 filings, 2006-2025", int(rc.item_2_02_filings.sum()), 383153)
check("all 8-K filings", int(rc.all_8k_filings.sum()), 1549567)
check("fewest distinct filers in a year",
      int(rc.distinct_filers_with_2_02.min()), 4277)
check("most distinct filers in a year",
      int(rc.distinct_filers_with_2_02.max()), 6003)
check("years covered", int(rc.year.nunique()), 20)
first, last = rc.set_index("year").item_2_02_filings.loc[[2006, 2025]]
check("fall in Item 2.02 count, 2006 to 2025, in percent",
      round(100 * (1 - last / first)), 27)

print("\n2. match attrition")
fn = pd.read_csv(pick("attrition_funnel.csv"))
ms = pd.read_csv(pick("matched_sample.csv"))
check("events after the last stage", int(fn.remaining.iloc[-1]), 115561)
check("matched sample rows", len(ms), 115561)
rem = fn.remaining.to_numpy()
check("stages where removed equals the drop in remaining",
      int(sum(int(fn.removed.iloc[i]) == rem[i - 1] - rem[i]
              for i in range(1, len(fn)))), len(fn) - 1)
check("largest single removal, the size cut", int(fn.removed.max()), 81231)
check("median market capitalisation, $bn",
      round(ms.mktcap.median() / 1e6, 1), 3.5)
check("median number of analysts", int(ms.n_analysts.median()), 7)
check("events in 2025, lost with the legacy CRSP file",
      int((pd.to_datetime(ms.session_date).dt.year == 2025).sum()), 5664)

print("\n3. release text")
ex = pd.read_csv(pick("extraction_report.csv"))
src = ex.source.fillna("")
got = src.ne("none") & src.ne("index fetch failed")
check("events with a release recovered", int(got.sum()), 114578,
      100 * got.mean())
v1 = src.str.startswith("EX-99.1")
other = src.str.startswith("EX-99") & ~v1
check("EX-99.1 and its variants", int(v1.sum()), 101688, 100 * v1.mean())
check("other EX-99 exhibits", int(other.sum()), 10970, 100 * other.mean())
body = src.eq("8-K body")
check("the 8-K body itself", int(body.sum()), 1920, 100 * body.mean())
check("no prose document found", int(src.eq("none").sum()), 981,
      100 * src.eq("none").mean())
check("index page unreachable", int(src.eq("index fetch failed").sum()), 2)
check("share lost by taking EX-99.1 alone, in percent",
      round(100 * (got.sum() - v1.sum()) / got.sum(), 1), 11.2)
w = ex.loc[got, "words"]
check("median release, words", int(w.median()), 3161)
check("tenth percentile, words", int(w.quantile(0.10)), 1425)
check("ninetieth percentile, words", round(w.quantile(0.90)), 5978)
check("median tables dropped", int(ex.loc[got, "tables_dropped"].median()), 3)

print("\n4. release timing")
CLOSE_S, OPEN_S = 16 * 3600, 9 * 3600 + 1800


def session(dates, secs):
    d = pd.to_datetime(dates, errors="coerce")
    missing = secs.isna() | (secs == 0)
    after = ~missing & (secs >= CLOSE_S)
    s = d.dt.date.astype("string")
    s = s.where(~after, (d + pd.Timedelta(days=1)).dt.date.astype("string"))
    return s.where(d.notna() & ~missing)


def secs_of(ts):
    t = pd.to_datetime(ts, errors="coerce")
    return t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second


t = pd.read_csv(pick("timing_validation.csv"))
wi = pd.read_csv(pick("wire_times.csv"))
n = len(t)
check("firm-days with a Dow Jones wire time", len(wi), 444915)
check("distinct securities on the wire file", int(wi.permno.nunique()), 8853)

has_i_all = t.ibes_flag.notna()
check("events with an I/B/E/S announcement", int(has_i_all.sum()), 101706,
      100 * has_i_all.mean())
check("EDGAR and I/B/E/S on the same session", int(t.agrees.sum()), 96827,
      100 * t.agrees.sum() / has_i_all.sum())
check("kept if two sources must agree, in percent",
      round(100 * t.agrees.mean(), 1), 85.6)
check("dropped by that rule", int(n - t.agrees.sum()), 16253)
check("  of those, no I/B/E/S match", int((~has_i_all).sum()), 11374)
dis = (~t.agrees) & has_i_all
check("  of those, matched to a different session", int(dis.sum()), 4879)

d = t[dis].copy()
d["lag"] = (pd.to_datetime(d.session_date)
            - pd.to_datetime(d.anndats)).dt.days
same = d.lag == 0
check("same date, I/B/E/S after the close, EDGAR in session",
      int((same & d.when.eq("intraday")).sum()), 199)
check("same date, I/B/E/S after the close, EDGAR pre-open",
      int((same & d.when.eq("pre_open")).sum()), 187)
check("same date, both after the close, different session",
      int((same & d.when.eq("after_close")).sum()), 36)
check("same date, differing only on the side of the close",
      int(same.sum()), 422)
check("EDGAR session one to three days after the I/B/E/S date",
      int(d.lag.between(1, 3).sum()), 4211)
check("EDGAR session before the I/B/E/S date", int((d.lag < 0).sum()), 246)

t["permno"] = pd.to_numeric(t.permno, errors="coerce").astype("Int64")
wi["permno"] = pd.to_numeric(wi.permno, errors="coerce").astype("Int64")
wi["wire_dt"] = pd.to_datetime(wi.wire_time_et, errors="coerce")
t["edgar_dt"] = pd.to_datetime(t.session_date, errors="coerce")
m = pd.merge_asof(t.sort_values("edgar_dt"),
                  wi.sort_values("wire_dt").dropna(subset=["wire_dt"])
                    [["permno", "wire_dt"]],
                  left_on="edgar_dt", right_on="wire_dt", by="permno",
                  direction="nearest", tolerance=pd.Timedelta("3D"))
m["wire_session"] = session(m.wire_dt, secs_of(m.wire_dt))
has_w, has_i = m.wire_session.notna(), m.ibes_session.notna()
check("events given a wire time", int(has_w.sum()), 86538, 100 * has_w.mean())

ei = has_i & (m.session_date == m.ibes_session)
ew = has_w & (m.session_date == m.wire_session)
iw = has_i & has_w & (m.ibes_session == m.wire_session)
check("I/B/E/S agrees with the wire", int(iw.sum()), 82650,
      100 * iw.sum() / (has_i & has_w).sum())
check("  out of events where both exist", int((has_i & has_w).sum()), 83829)
check("EDGAR agrees with the wire", int(ew.sum()), 82363,
      100 * ew.sum() / has_w.sum())
check("EDGAR agrees with I/B/E/S", int(ei.sum()), 96827,
      100 * ei.sum() / has_i.sum())

two = (ei.astype(int) + ew.astype(int) + iw.astype(int)) >= 1
check("retained when two of three sources agree", int(two.sum()), 102726,
      100 * two.mean())
check("dropped by that rule", int(n - two.sum()), 10354)
only_e = ~has_i & ~has_w
check("  of those, with no second source", int(only_e.sum()), 8665)
check("  of those, with a second source that disagrees",
      int(n - two.sum() - only_e.sum()), 1689)

m["session_final"] = m.wire_session.where(has_w, m.session_date)
moved = m.session_final != m.session_date
check("events the wire moves to another session", int(moved.sum()), 4175,
      100 * moved.mean())
delta = (pd.to_datetime(m.session_final, errors="coerce")
         - pd.to_datetime(m.session_date, errors="coerce")).dt.days
check("  EDGAR one to three sessions too early", int((delta > 0).sum()), 574)
check("  EDGAR one to three sessions too late", int((delta < 0).sum()), 3601)

print(f"\n{ok} of {ok + len(bad)} figures reproduce the report")
for line in bad:
    print("  MISMATCH " + line)
