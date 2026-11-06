#we download earnings release texts for the 117,926 matched filings.
#we first read each filing's index to identify and download only the relevant exhibit,
#reducing data transfer from about 190 GB to roughly one-seventh of that amount.
#we use parallel workers with a shared SEC rate limit of 10 requests per second.
#results are saved in yearly ZIP files, and interrupted downloads can be resumed.

#python fetch_releases.py --events matched_sample.csv --out-zip-dir releases

from __future__ import annotations

import argparse
import csv
import html
import os
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

USER_AGENT = "MIT 15.S08 project mbonnetderoovere@gmail.com"
RATE = 9.0          #below the SEC's ceiling of 10 requests a second
WORKERS = 8
MIN_WORDS = 500

ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
HREF_RE = re.compile(r'href="([^"]+)"', re.I)
SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t\r\f\v]+")
NL_RE = re.compile(r"\n{3,}")


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


def fetch(url: str, limiter: RateLimiter, retries: int = 4) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        limiter.acquire()
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 20))
                continue
            return None
        except Exception:
            time.sleep(min(2 ** attempt, 20))
    return None


def _strip(fragment: str) -> str:
    return html.unescape(TAG_RE.sub(" ", fragment)).replace("\xa0", " ")


def to_text(raw: str) -> tuple[str, int]:

    body = SCRIPT_RE.sub(" ", raw)

    dropped = 0

    def decide(m: re.Match) -> str:
        nonlocal dropped
        inner = _strip(m.group(0))
        letters = sum(c.isalpha() for c in inner)
        digits = sum(c.isdigit() for c in inner)
        if digits > 0.30 * max(letters + digits, 1):
            dropped += 1
            return " "
        return " " + inner + " "

    body = TABLE_RE.sub(decide, body)
    body = _strip(body)
    body = NL_RE.sub("\n\n", WS_RE.sub(" ", body))
    return body.strip(), dropped


def looks_like_prose(text: str) -> bool:
    words = text.split()
    if len(words) < MIN_WORDS or text.count(". ") < 10:
        return False
    return sum(c.isdigit() for c in text) / max(len(text), 1) < 0.25


def rank(doc_type: str) -> int:
    t = doc_type.upper()
    if t == "EX-99.1":
        return 0
    if t == "EX-99.2":
        return 1
    if t.startswith("EX-99"):
        return 2
    if t == "8-K":
        return 3
    return 9


def parse_index(body: str, base: str) -> list[tuple[str, str]]:
    """Return [(type, absolute url)] for the documents worth trying."""
    out = []
    for row in ROW_RE.findall(body):
        cells = CELL_RE.findall(row)
        if len(cells) < 4:
            continue
        doc_type = re.sub(r"<[^>]+>", " ", cells[3]).strip()
        if rank(doc_type) == 9:
            continue
        m = HREF_RE.search(cells[2])
        if not m:
            continue
        href = html.unescape(m.group(1))
        # EDGAR serves some documents through an "ix?doc=" viewer wrapper.
        if "ix?doc=" in href:
            href = href.split("ix?doc=", 1)[1]
        url = href if href.startswith("http") else \
            "https://www.sec.gov" + (href if href.startswith("/") else "/" + href)
        out.append((doc_type, url))
    out.sort(key=lambda p: rank(p[0]))
    return out


def one_filing(row: dict, limiter: RateLimiter) -> tuple[dict, str | None]:
    acc = row["accession"]
    cik = str(row["cik"]).lstrip("0")
    base = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{acc.replace('-', '')}/")
    rec = {"accession": acc, "cik": cik,
           "filing_date": row.get("filing_date", ""),
           "source": "", "filename": "", "words": 0, "tables_dropped": 0}

    idx = fetch(base + f"{acc}-index.htm", limiter)
    if idx is None:
        rec["source"] = "index fetch failed"
        return rec, None

    for doc_type, url in parse_index(idx.decode("utf-8", "replace"), base):
        body = fetch(url, limiter)
        if body is None:
            continue
        text, n_tables = to_text(body.decode("utf-8", "replace"))
        if looks_like_prose(text):
            rec["source"] = doc_type if doc_type.upper().startswith("EX-99") \
                else "8-K body"
            rec["filename"] = url.rsplit("/", 1)[-1]
            rec["words"] = len(text.split())
            rec["tables_dropped"] = n_tables
            return rec, text

    rec["source"] = "none"
    return rec, None


def already_done(zip_dir: str) -> set[str]:
    done: set[str] = set()
    if not os.path.isdir(zip_dir):
        return done
    for name in sorted(os.listdir(zip_dir)):
        if not name.endswith(".zip"):
            continue
        try:
            with zipfile.ZipFile(os.path.join(zip_dir, name)) as z:
                done.update(n[:-4] for n in z.namelist() if n.endswith(".txt"))
        except zipfile.BadZipFile:
            print(f"  warning: {name} is corrupt, ignoring", file=sys.stderr)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="matched_sample.csv")
    ap.add_argument("--out-zip-dir", default="releases")
    ap.add_argument("--report", default="extraction_report_full.csv")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.events, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    os.makedirs(args.out_zip_dir, exist_ok=True)
    done = already_done(args.out_zip_dir)
    if done:
        before = len(rows)
        rows = [r for r in rows if r["accession"] not in done]
        print(f"resuming: {before - len(rows):,} stored, {len(rows):,} to go",
              flush=True)
    if args.limit:
        rows = rows[:args.limit]

    limiter = RateLimiter(RATE)
    writer_q: queue.Queue = queue.Queue(maxsize=2000)
    report_rows: list[dict] = []
    report_lock = threading.Lock()

    def writer() -> None:
        """Single thread owns the archives; zipfile is not thread safe."""
        zips: dict[str, zipfile.ZipFile] = {}
        while True:
            item = writer_q.get()
            if item is None:
                break
            year, acc, text = item
            if year not in zips:
                zips[year] = zipfile.ZipFile(
                    os.path.join(args.out_zip_dir, f"releases_{year}.zip"),
                    "a", zipfile.ZIP_DEFLATED, compresslevel=6)
            zips[year].writestr(f"{acc}.txt", text)
            writer_q.task_done()
        for z in zips.values():
            z.close()
        writer_q.task_done()

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    t0 = time.monotonic()
    counter = {"n": 0}

    def work(row: dict) -> None:
        rec, text = one_filing(row, limiter)
        if text is not None:
            year = (row.get("session_date") or row.get("filing_date") or "")[:4]
            writer_q.put((year or "unknown", rec["accession"], text))
        with report_lock:
            report_rows.append(rec)
            counter["n"] += 1
            n = counter["n"]
        if n % 500 == 0:
            el = time.monotonic() - t0
            rate = n / el
            left = (len(rows) - n) / rate / 3600 if rate else 0
            print(f"  {n:,}/{len(rows):,}  {rate:.1f} filings/s  "
                  f"{left:.1f} h left", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, rows))

    writer_q.put(None)
    wt.join()

    if not report_rows:
        print("nothing processed")
        return 0

    with open(args.report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w.writeheader()
        w.writerows(report_rows)

    n = len(report_rows)
    counts: dict[str, int] = {}
    for r in report_rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print(f"\nfilings processed: {n:,} in {(time.monotonic()-t0)/3600:.2f} h")
    for src, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {src:<22} {c:>7,}  {100*c/n:5.1f}%")
    ok = sorted(r["words"] for r in report_rows if r["words"])
    if ok:
        print(f"word count: median {ok[len(ok)//2]:,}  "
              f"p10 {ok[len(ok)//10]:,}  p90 {ok[(9*len(ok))//10]:,}")
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
