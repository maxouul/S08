"""read ACCEPTANCE-DATETIME from each filing's own header.

the submissions JSON is wrong about the acceptance time for two filings in
three (see check_acceptance_header.py), and in three different ways, so no
rule fixes it. the header is the field the instructors say does not move.

one request per filing, under the SEC limit of ten a second, and it resumes
from whatever is already in the output file.

    python fetch_acceptance.py
"""

from __future__ import annotations

from paths import BASE, src, dst

import argparse
import csv
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

USER_AGENT = "MIT 15.S08 project marcusjean501@gmail.com"
RATE = 9.0
WORKERS = 8
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


def fetch_head(url: str, limiter: RateLimiter, retries: int = 3) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        limiter.acquire()
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                # The header is the first thing in the submission file.
                return r.read(4096).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 20))
                continue
            return None
        except Exception:
            time.sleep(min(2 ** attempt, 20))
    return None


def already_done(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["accession"] for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=src("event_table.csv"))
    ap.add_argument("--out", default=dst("acceptance_times.csv"))
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    seen, rows = set(), []
    with open(args.events, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = r["accession"]
            if a and a not in seen:
                seen.add(a)
                rows.append((a, r["cik"]))

    done = already_done(args.out)
    fresh = not done
    if done:
        before = len(rows)
        rows = [r for r in rows if r[0] not in done]
        print(f"resuming: {before - len(rows):,} stored, {len(rows):,} to go",
              flush=True)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("nothing to do")
        return 0

    limiter = RateLimiter(RATE)
    lock = threading.Lock()
    t0 = time.monotonic()
    n = {"done": 0, "missing": 0}

    out = open(args.out, "a", newline="", encoding="utf-8")
    w = csv.writer(out)
    if fresh:
        w.writerow(["accession", "acceptance_header_et"])

    def work(item) -> None:
        acc, cik = item
        url = (f"https://www.sec.gov/Archives/edgar/data/{str(cik).lstrip('0')}/"
               f"{acc.replace('-', '')}/{acc}.txt")
        head = fetch_head(url, limiter)
        m = HEADER_RE.search(head or "")
        stamp = m.group(1) if m else ""
        with lock:
            w.writerow([acc, stamp])
            n["done"] += 1
            if not stamp:
                n["missing"] += 1
            k = n["done"]
            if k % 2000 == 0:
                out.flush()
                el = time.monotonic() - t0
                rate = k / el
                left = (len(rows) - k) / rate / 3600 if rate else 0
                print(f"  {k:,}/{len(rows):,}  {rate:.1f}/s  "
                      f"{left:.1f} h left  {n['missing']:,} without a header",
                      flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, rows))
    out.close()

    print(f"\n{n['done']:,} fetched in {(time.monotonic() - t0) / 3600:.2f} h")
    print(f"{n['missing']:,} with no ACCEPTANCE-DATETIME in the header")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
