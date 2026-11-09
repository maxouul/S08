"""fill the 50-event timing check from the filings themselves.

the sheet asks which source is wrong when EDGAR, I/B/E/S and the wire
disagree about the session. that is not an opinion: the header carries
ACCEPTANCE-DATETIME in Eastern time and the session follows from it.

so we refetch the header, recompute the session, and mark every source that
matches. a human still owns the verdict column and can overwrite a row.

what we cannot settle is a release the wire carried before the filing was
accepted. there the header says nothing about when the company spoke, so
the row is marked and left.

    python adjudicate_timing_50.py
"""

from __future__ import annotations

import argparse
import bisect
import os
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from paths import src, dst

ET = ZoneInfo("America/New_York")
USER_AGENT = "MIT 15.S08 project marcusjean501@gmail.com"
RATE = 9.0
HEADER_RE = re.compile(r"<ACCEPTANCE-DATETIME>\s*(\d{14})")
CLOSE = 16 * 3600
OPEN = 9 * 3600 + 1800


class RateLimiter:
    def __init__(self, rate):
        self.gap = 1.0 / rate
        self.lock = threading.Lock()
        self.next_at = 0.0

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.gap
        if wait:
            time.sleep(wait)


def head_of(cik, acc, limiter, retries=3):
    url = (f"https://www.sec.gov/Archives/edgar/data/{str(cik).lstrip('0')}/"
           f"{acc.replace('-', '')}/{acc}.txt")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        limiter.acquire()
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read(4096).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 20))
                continue
            return None
        except Exception:
            time.sleep(min(2 ** attempt, 20))
    return None


def session_of(ts):
    """same rule as the pipeline: after the close belongs to the next day."""
    if ts is None:
        return None
    secs = ts.hour * 3600 + ts.minute * 60 + ts.second
    if secs >= CLOSE:
        return (ts + timedelta(days=1)).date().isoformat()
    return ts.date().isoformat()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=src("timing_handcheck_50.csv"))
    ap.add_argument("--out", default=dst("timing_handcheck_50.csv"))
    args = ap.parse_args()

    d = pd.read_csv(args.sheet, dtype={"accession": str})
    # a sheet settled by an earlier run carries the columns this run adds;
    # they are dropped so the merge below does not duplicate cik.
    d = d.drop(columns=[c for c in ("cik", "header_acceptance_et",
                                    "sources_matching_header", "judged_by")
                        if c in d.columns])
    # the header session is rolled onto the trading calendar, like every
    # source it is compared with: a Friday evening filing belongs to Monday.
    cal_path = src("trading_calendar.csv")
    days = (sorted(pd.read_csv(cal_path, dtype=str).date)
            if os.path.exists(cal_path) else [])

    def roll(day):
        if not day or not days:
            return day
        i = bisect.bisect_left(days, day)
        return days[i] if i < len(days) else day
    ev = pd.read_csv(src("event_table.csv"), dtype={"accession": str},
                     usecols=["accession", "cik"]).drop_duplicates("accession")
    d = d.merge(ev, on="accession", how="left")
    print(f"{len(d)} events to settle", flush=True)

    limiter = RateLimiter(RATE)
    rows = []
    for i, r in d.iterrows():
        head = head_of(r.cik, r.accession, limiter) if pd.notna(r.cik) else None
        m = HEADER_RE.search(head or "")
        hdr = (datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=ET)
               if m else None)
        truth = roll(session_of(hdr))

        matches = []
        for label, col in (("edgar", "edgar_session"),
                           ("ibes", "ibes_session_c"),
                           ("wire", "wire_session")):
            v = r.get(col)
            if isinstance(v, str) and truth and v[:10] == truth:
                matches.append(label)

        # a wire story that predates the filing is not a source being wrong
        # wire_dt is New York time already when it carries no offset (as
        # three_source_timing.py writes it); reading it as UTC would move
        # the story four or five hours early and make the wire look first.
        wire_dt = pd.to_datetime(r.get("wire_dt"), errors="coerce")
        if pd.notna(wire_dt) and wire_dt.tzinfo is not None:
            wire_dt = wire_dt.tz_convert(ET).tz_localize(None)
        early = bool(hdr is not None and pd.notna(wire_dt)
                     and wire_dt < hdr.replace(tzinfo=None))

        if truth is None:
            verdict, note = "", "no header could be read"
        elif early:
            verdict = "wire"
            note = ("the wire carried it before EDGAR accepted the filing, "
                    "so the wire is the release time and EDGAR is late")
        elif matches:
            verdict = matches[0]
            note = f"header says {truth}; matches {', '.join(matches)}"
        else:
            verdict = "none"
            note = f"header says {truth}; no source agrees"

        rows.append({
            "header_acceptance_et": hdr.isoformat() if hdr else "",
            "session_you_read": truth or "",
            "source_you_trust": verdict,
            "sources_matching_header": ",".join(matches),
            "note": note,
            "judged_by": "code, from the filing header",
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(d)}", flush=True)

    add = pd.DataFrame(rows)
    for c in add.columns:
        d[c] = add[c].values
    d.to_csv(args.out, index=False, encoding="utf-8-sig")

    print()
    print(d.source_you_trust.value_counts().to_string())
    got = d[d.session_you_read != ""]
    print(f"\nsettled from the header: {len(got)}/{len(d)}")
    for label, col in (("edgar", "edgar_session"), ("ibes", "ibes_session_c"),
                       ("wire", "wire_session")):
        ok = got.sources_matching_header.str.contains(label).sum()
        print(f"  {label:<6} matches the header in {ok:>3} of {len(got)}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
