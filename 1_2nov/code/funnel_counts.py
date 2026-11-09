"""
15.S08 — Step 1 of the work plan: raw counts of 8-K Item 2.02 filings.

Produces the first column of the attrition funnel the instructors asked for
at the 2 November milestone. This script answers "how many Item 2.02 filings
exist", nothing else. Matching to CRSP and the per-filter attrition come next,
once the WRDS pull is in place.

Source: the SEC bulk submissions archive, which carries the filing metadata
for every EDGAR filer in one download. The per-CIK API would need thousands of
requests for the same information.

    https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip

The archive is read member by member without extracting it, so the only disk
cost is the 1.6 GB zip itself.

Usage:
    python funnel_counts.py --zip /path/to/submissions.zip --out raw_counts.csv

If --zip points at a file that does not exist, the script downloads it first.
"""

from __future__ import annotations

from paths import BASE, src, dst

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

# The SEC requires a descriptive User-Agent carrying a contact address and
# caps requests at 10 per second. One bulk download stays well inside that.
USER_AGENT = os.environ.get(
    "SEC_USER_AGENT", "MIT 15.S08 project marcusjean501@gmail.com"
)

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
    """Yield one parallel-array filing table per archive member.

    Two shapes occur. The primary CIK##########.json nests the table under
    filings.recent. The overflow files CIK##########-submissions-00N.json hold
    the table at the top level. Both are handled.
    """
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
    ap.add_argument("--out", default=dst("raw_counts.csv"))
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
                # items is a comma-separated string, e.g. "2.02,9.01".
                # Match on the exact code so 12.02 can never match 2.02.
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
