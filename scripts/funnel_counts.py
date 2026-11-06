#we count all 8-K Item 2.02 filings using the SEC bulk submissions archive.
#we process the ZIP without extracting it and save the raw counts to CSV.
#this is the first step of the attrition funnel, before CRSP matching and filtering.

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict

BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
USER_AGENT = os.environ.get("SEC_USER_AGENT", "MIT 15.S08 project mbonnetderoovere@gmail.com")

START_YEAR = 2006
END_YEAR = 2025


def download(url: str, dest: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r  {done/1e9:.2f} / {total/1e9:.2f} GB  ({pct:.1f}%)",
                      end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)


def iter_filing_tables(zf: zipfile.ZipFile):
    for info in zf.infolist():
        name = info.filename
        if not name.endswith(".json") or not name.startswith("CIK"):
            continue
        try:
            with zf.open(info) as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, zipfile.BadZipFile, OSError):
            yield None, name  # counted as unreadable, not silently dropped
            continue

        if isinstance(doc, dict) and "filings" in doc:
            recent = doc.get("filings", {}).get("recent")
            if recent:
                yield recent, name
        elif isinstance(doc, dict) and "form" in doc:
            yield doc, name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="path to submissions.zip")
    ap.add_argument("--out", default="raw_counts.csv")
    args = ap.parse_args()

    if not os.path.exists(args.zip):
        print(f"downloading {BULK_URL}", file=sys.stderr)
        download(BULK_URL, args.zip)

    by_year = Counter()            # Item 2.02 filings per year
    all_8k_by_year = Counter()     # every 8-K, for context
    filers_by_year = defaultdict(set)
    unreadable = 0
    members = 0

    with zipfile.ZipFile(args.zip) as zf:
        for table, name in iter_filing_tables(zf):
            members += 1
            if table is None:
                unreadable += 1
                continue

            forms = table.get("form", [])
            items = table.get("items", [])
            dates = table.get("filingDate", [])
            cik = name.split("-")[0][3:].removesuffix(".json").lstrip("0")

            for i, form in enumerate(forms):
                if form != "8-K":
                    continue
                date = dates[i] if i < len(dates) else ""
                if len(date) < 4:
                    continue
                year = int(date[:4])
                if not (START_YEAR <= year <= END_YEAR):
                    continue
                all_8k_by_year[year] += 1

                codes = (items[i] if i < len(items) else "").split(",")
                if "2.02" in [c.strip() for c in codes]:
                    by_year[year] += 1
                    filers_by_year[year].add(cik)

            if members % 20000 == 0:
                print(f"\r  {members:,} filers read", end="",
                      file=sys.stderr, flush=True)

    print(file=sys.stderr)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "item_2_02_filings", "all_8k_filings",
                    "distinct_filers_with_2_02"])
        for year in range(START_YEAR, END_YEAR + 1):
            w.writerow([year, by_year[year], all_8k_by_year[year],
                        len(filers_by_year[year])])
        w.writerow(["TOTAL", sum(by_year.values()), sum(all_8k_by_year.values()), ""])

    print(f"filers read      : {members:,}")
    print(f"unreadable members: {unreadable:,}")
    print(f"Item 2.02 total  : {sum(by_year.values()):,}")
    print(f"all 8-K total    : {sum(all_8k_by_year.values()):,}")
    print(f"written          : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
