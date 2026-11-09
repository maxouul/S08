"""is the acceptance time we stored the one in the filing header?

refetching 383,153 headers is twelve hours, so we check a sample first.
this draws 300 filings across the twenty years, reads ACCEPTANCE-DATETIME
out of each header, and classifies the gap with what we stored.

result: 99 agree, 201 do not. the JSON is Eastern labelled Z in 128 cases
and converted to UTC twice in 73. three regimes inside the same year.

    python check_acceptance_header.py --n 300
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from paths import src, dst

ET = ZoneInfo("America/New_York")
USER_AGENT = "MIT 15.S08 project marcusjean501@gmail.com"
RATE = 9.0
EVENTS = src("event_table.csv")             # files csv, else 1_2nov/data
OUT = dst("acceptance_header_check.csv")    # always files csv
SEED = 20261104

HEADER_RE = re.compile(r"<ACCEPTANCE-DATETIME>\s*(\d{14})")


class RateLimiter:
    def __init__(self, rate: float):
        self.min_gap = 1.0 / rate
        self.lock = threading.Lock()
        self.next_at = 0.0

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            wait = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.min_gap
        if wait:
            time.sleep(wait)


def fetch(url: str, limiter: RateLimiter, retries: int = 3) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        limiter.acquire()
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                # The header sits in the first few hundred bytes of the
                # submission file, so there is no reason to read the rest.
                return r.read(4096).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 20))
                continue
            return None
        except Exception:
            time.sleep(min(2 ** attempt, 20))
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--events", default=EVENTS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    d = pd.read_csv(args.events,
                    usecols=["accession", "cik", "acceptance_et",
                             "acceptance_utc", "filing_date"],
                    dtype={"accession": str})
    d = d.dropna(subset=["acceptance_et"]).drop_duplicates("accession")
    d["year"] = pd.to_datetime(d.filing_date, errors="coerce").dt.year
    d = d[d.year.between(2006, 2025)]

    per_year = max(1, args.n // d.year.nunique())
    s = (d.groupby("year", group_keys=False)
          .sample(n=per_year, random_state=SEED)
          .sort_values(["year", "accession"]).reset_index(drop=True))
    print(f"checking {len(s)} filings, {per_year} a year", flush=True)

    limiter = RateLimiter(RATE)
    rows = []
    for i, r in s.iterrows():
        cik = str(int(r.cik))
        acc = r.accession
        url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
               f"{acc.replace('-', '')}/{acc}.txt")
        head = fetch(url, limiter)
        m = HEADER_RE.search(head or "")
        if not m:
            rows.append({"accession": acc, "year": r.year,
                         "json_raw": r.acceptance_utc,
                         "stored_et": r.acceptance_et, "header_et": None,
                         "delta_hours": None, "regime": "no header"})
            continue
        hdr = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=ET)
        stored = pd.Timestamp(r.acceptance_et).to_pydatetime()
        # Which of the three things the JSON field actually is, decided
        # per filing rather than guessed from the sign of the gap.
        json_clock = datetime.fromisoformat(
            str(r.acceptance_utc).replace("Z", "+00:00")).replace(tzinfo=None)
        hdr_clock = hdr.replace(tzinfo=None)
        gap = (json_clock - hdr_clock).total_seconds() / 3600
        off = -hdr.utcoffset().total_seconds() / 3600
        if gap == 0:
            regime = "Eastern labelled Z"
        elif gap == off:
            regime = "true UTC"
        elif gap == 2 * off:
            regime = "converted to UTC twice"
        else:
            regime = f"other, {gap:+.0f} h from the header clock"
        rows.append({
            "accession": acc, "year": r.year,
            "json_raw": r.acceptance_utc,
            "stored_et": stored.isoformat(),
            "header_et": hdr.isoformat(),
            "delta_hours": round((stored - hdr).total_seconds() / 3600, 2),
            "regime": regime,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(s)}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    got = out.dropna(subset=["delta_hours"])
    print(f"\nheader read for {len(got)} of {len(out)}")
    if not len(got):
        print("no header could be read; nothing can be concluded")
        return 1

    same = (got.delta_hours == 0)
    print(f"identical to the header : {same.sum():>4}  "
          f"{100 * same.mean():5.1f}%")
    print(f"different               : {(~same).sum():>4}  "
          f"{100 * (~same).mean():5.1f}%")
    if (~same).any():
        print("\nsize of the gap, in hours")
        print(got.loc[~same, "delta_hours"].value_counts()
                 .sort_index().to_string())
        print("\nby year")
        print(got.assign(off=~same).groupby("year").off
                 .agg(["size", "sum"]).to_string())
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
