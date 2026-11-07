"""
Step 5.1: Cache the raw bytes of every release document, once.

fetch_releases.py cleaned each document while downloading it and kept only the text,
so the HTML is gone. This script downloads the chosen document of every event again
and stores its bytes undecoded, so that cleaning becomes a separate, versioned step
(clean_release.py) that never has to go back to EDGAR.

Two ways to know which document to take:
- report mode, the default: the report of fetch_releases.py (files csv/
  extraction_report.csv, or extraction_report_full.csv after a rerun) names the
  document of each filing (accession, cik, filename). One request per document,
  straight to its Archives URL: about 117,000 requests, 3 to 4 hours at 9 a second.
  The 943 filings where fetch_releases.py found no prose document are not fetched;
- index mode, only if no such report is found: for each filing of the release clock,
  the index page is read again and the document is chosen with fetch_releases.py's own
  rules (parse_index, to_text, looks_like_prose), so the choice is the same. About two
  requests per filing, 7 to 8 hours.

Output: cache/raw/raw_{year}.zip holding {accession}__{filename} (year of the filing date,
which the timing corrections do not move), and raw_manifest.parquet (accession, cik,
kind, source, url, filename, http_status, n_bytes, sha256, fetched_at_utc), in cache/.

The archives are committed every --commit events, so a crash loses at most that
many. On restart, an archive that does not open is set aside and its documents are
fetched again; events already stored, or already found to have no prose document,
are skipped. A download that fails for a passing reason (no answer, 403, 429, 5xx)
is recorded as "fetch failed", never as "none", and is tried again on the next run.

Usage: python code/cache_raw.py --limit 200
       python code/cache_raw.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import settings as S  # first: it puts scripts/ on the path for fetch_releases
from fetch_releases import (RATE, USER_AGENT, WORKERS, RateLimiter, looks_like_prose,
                            parse_index, to_text)

JOURNAL = "_journal.csv"
# Errors that say the document is not there; anything else (no answer, 403, 429,
# 5xx after the retries) may pass, so the event is tried again on the next run.
PERMANENT = {400, 404, 410}
FIELDS = ["accession", "cik", "kind", "source", "url", "filename", "http_status",
          "n_bytes", "sha256", "fetched_at_utc"]
REPORT_COLUMNS = S.FETCH_REPORT_COLUMNS


def fetch_status(url: str, limiter: RateLimiter, retries: int = 4) -> tuple[int, bytes | None]:
    """fetch() from fetch_releases.py, but it returns the HTTP status for the manifest."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    status = 0
    for attempt in range(retries):
        limiter.acquire()
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            status = e.code
            if e.code in (403, 429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 20))
                continue
            return status, None
        except Exception:
            time.sleep(min(2 ** attempt, 20))
    return status, None


def base_url(acc: str, cik: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"


def from_report(job: dict, limiter: RateLimiter):
    """Report mode: the filename is known."""
    url = base_url(job["accession"], job["cik"]) + job["filename"]
    status, body = fetch_status(url, limiter)
    # a document that did not come back is marked as such: it is fetched again on the
    # next run, and the failed count shows it
    source = job["source"] if body is not None else "fetch failed"
    return [("doc", source, url, job["filename"], status, body)]


def from_index(job: dict, limiter: RateLimiter, keep_index: bool):
    """Index mode: the same choice as fetch_releases.one_filing, keeping the bytes."""
    acc, base = job["accession"], base_url(job["accession"], job["cik"])
    out = []
    status, idx = fetch_status(base + f"{acc}-index.htm", limiter)
    if idx is None:
        return [("doc", "index fetch failed", base + f"{acc}-index.htm", "", status, None)]
    if keep_index:
        out.append(("index", "", base + f"{acc}-index.htm", "index.htm", status, idx))
    for doc_type, url in parse_index(idx.decode("utf-8", "replace"), base):
        status, body = fetch_status(url, limiter)
        if body is None and status not in PERMANENT:
            # A download that failed for a passing reason says nothing about the
            # filing: stop rather than settle for a lower-ranked document or "none",
            # and leave the event to the next run.
            out.append(("doc", "fetch failed", url, url.rsplit("/", 1)[-1], status, None))
            return out
        if body is None:
            continue
        text, _ = to_text(body.decode("utf-8", "replace"))
        if looks_like_prose(text):
            source = doc_type if doc_type.upper().startswith("EX-99") else "8-K body"
            out.append(("doc", source, url, url.rsplit("/", 1)[-1], status, body))
            return out
    out.append(("doc", "none", base, "", 200, None))
    return out


def stored(raw_dir: str) -> set[str]:
    """Members of every archive that opens. An archive that does not is renamed, so
    its documents are fetched again rather than trusted."""
    names: set[str] = set()
    for f in sorted(os.listdir(raw_dir)):
        if not f.endswith(".zip"):
            continue
        p = os.path.join(raw_dir, f)
        try:
            with zipfile.ZipFile(p) as z:
                names.update(z.namelist())
        except zipfile.BadZipFile:
            bad = p + f".corrupt-{int(time.time())}"
            os.replace(p, bad)
            print(f"  warning: {f} does not open, moved to {bad}", file=sys.stderr)
    return names


def read_journal(raw_dir: str):
    import pandas as pd
    jp = os.path.join(raw_dir, JOURNAL)
    if not os.path.exists(jp):
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(jp, dtype=str, keep_default_na=False)


def build_manifest(raw_dir: str, out: str, present: set[str]) -> int:
    """Last journal line per (accession, kind); a fetch counted as stored only if its
    bytes are in an archive that opens."""
    import pandas as pd
    j = read_journal(raw_dir).drop_duplicates(["accession", "kind"], keep="last")
    member = j.accession + "__" + j.filename
    has_bytes = j.sha256 != ""
    j = j[~has_bytes | member.isin(present)].copy()
    j["http_status"] = pd.to_numeric(j.http_status, errors="coerce").astype("Int64")
    j["n_bytes"] = pd.to_numeric(j.n_bytes, errors="coerce").astype("Int64")
    j.to_parquet(out, index=False)
    return len(j)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(S.fetch_report() or ""),
                    help="report of fetch_releases.py (accession, cik, filename, source); "
                         "index mode if absent")
    ap.add_argument("--events", default=str(S.CLOCK),
                    help="filings for index mode only: accession, cik, filing_date")
    ap.add_argument("--raw-dir", default=str(S.RAW_DIR))
    ap.add_argument("--manifest", default=str(S.RAW_MANIFEST))
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="first N events only, for a test")
    ap.add_argument("--commit", type=int, default=500)
    ap.add_argument("--with-index", action="store_true",
                    help="also keep the filing index page")
    args = ap.parse_args()

    import pandas as pd

    report_ok = False
    if args.report and os.path.exists(args.report):
        cols = set(pd.read_csv(args.report, nrows=0).columns)
        report_ok = REPORT_COLUMNS <= cols
        if not report_ok:
            print(f"{args.report} lacks {sorted(REPORT_COLUMNS - cols)}: index mode")
    if report_ok:
        mode = "report"
        ev = pd.read_csv(args.report, dtype=str, keep_default_na=False)
        ev = ev[~ev.source.isin(["none", "index fetch failed", ""]) & ev.filename.ne("")]
    else:
        mode = "index"
        ev = pd.read_csv(args.events, usecols=["accession", "cik", "filing_date"], dtype=str)
    ev = ev.dropna(subset=["accession", "cik"]).drop_duplicates("accession")
    ev["cik"] = ev.cik.str.lstrip("0")
    print(f"mode: {mode}; {len(ev):,} events from "
          f"{args.report if mode == 'report' else args.events}", flush=True)

    os.makedirs(args.raw_dir, exist_ok=True)
    present = stored(args.raw_dir)
    done = {m.split("__", 1)[0] for m in present if not m.endswith("__index.htm")}
    j = read_journal(args.raw_dir)
    done |= set(j.loc[j.source == "none", "accession"])      # no prose document: final
    jobs = [r for r in ev.to_dict("records") if r["accession"] not in done]
    print(f"already done: {len(done):,}; to fetch: {len(jobs):,}", flush=True)
    if args.limit:
        jobs = jobs[:args.limit]

    limiter = RateLimiter(RATE)
    q: queue.Queue = queue.Queue(maxsize=2000)

    def writer() -> None:
        """Single thread owns the archives (zipfile is not thread safe). Journal lines
        are written only once the archive holding the bytes has been closed."""
        zips: dict[str, zipfile.ZipFile] = {}
        pending: list[dict] = []
        jpath = os.path.join(args.raw_dir, JOURNAL)
        new = not os.path.exists(jpath)
        with open(jpath, "a", newline="", encoding="utf-8") as jf:
            jw = csv.DictWriter(jf, fieldnames=FIELDS)
            if new:
                jw.writeheader()

            def commit() -> None:
                for z in zips.values():
                    z.close()
                zips.clear()
                jw.writerows(pending)
                jf.flush()
                pending.clear()

            while True:
                item = q.get()
                if item is None:
                    break
                year, recs = item
                for rec, body in recs:
                    if body is not None:
                        if year not in zips:
                            zips[year] = zipfile.ZipFile(
                                os.path.join(args.raw_dir, f"raw_{year}.zip"), "a",
                                zipfile.ZIP_DEFLATED, compresslevel=6)
                        zips[year].writestr(f"{rec['accession']}__{rec['filename']}", body)
                    pending.append(rec)
                if len(pending) >= args.commit:
                    commit()
            commit()

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()
    t0 = time.monotonic()
    counter = {"n": 0, "ok": 0, "failed": 0}
    every = max(10, min(1000, len(jobs) // 20))    # about twenty progress lines a run
    lock = threading.Lock()

    def work(job: dict) -> None:
        got = from_report(job, limiter) if mode == "report" \
            else from_index(job, limiter, args.with_index)
        if mode == "report" and args.with_index:
            base = base_url(job["accession"], job["cik"])
            st, idx = fetch_status(base + f"{job['accession']}-index.htm", limiter)
            got.append(("index", "", base + f"{job['accession']}-index.htm",
                        "index.htm", st, idx))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        recs = [({"accession": job["accession"], "cik": job["cik"], "kind": kind,
                  "source": source, "url": url, "filename": filename, "http_status": status,
                  "n_bytes": len(body) if body is not None else 0,
                  "sha256": hashlib.sha256(body).hexdigest() if body is not None else "",
                  "fetched_at_utc": now}, body)
                for kind, source, url, filename, status, body in got]
        q.put(((job.get("filing_date") or "")[:4] or "unknown", recs))
        with lock:
            counter["n"] += 1
            counter["ok"] += any(r["kind"] == "doc" and b is not None for r, b in recs)
            counter["failed"] += any(r["source"] in ("fetch failed", "index fetch failed")
                                     for r, _ in recs)
            n, ok, failed = counter["n"], counter["ok"], counter["failed"]
        if n % every == 0:
            rate = n / (time.monotonic() - t0)
            print(f"  {n:,}/{len(jobs):,} events  {ok:,} documents  {failed:,} failed  "
                  f"{rate:.1f} events/s  {(len(jobs) - n) / rate / 3600:.1f} h left",
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))
    q.put(None)
    wt.join()

    present = stored(args.raw_dir)
    n_man = build_manifest(args.raw_dir, args.manifest, present)
    man = pd.read_parquet(args.manifest)
    docs = man[man.kind == "doc"]
    print(f"\nevents fetched this run: {counter['n']:,} ({counter['ok']:,} with a document, "
          f"{counter['failed']:,} failed, left for the next run)")
    print(f"documents stored       : {int((docs.sha256 != '').sum()):,}")
    print(docs.source.value_counts().to_string())
    print(f"manifest rows          : {n_man:,} -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
