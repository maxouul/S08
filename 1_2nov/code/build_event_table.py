"""
15.S08 — event-table skeleton for every 8-K Item 2.02 filing, 2006-2025.

Everything here comes out of the SEC bulk submissions archive, so no crawling
is needed at this stage. One row per filing, carrying what we need before the
WRDS join: identifiers, the acceptance timestamp converted to New York time,
and the trading session the release belongs to.

The text fetch is deliberately NOT done here. It should run after the CRSP
match, so that we only download exhibits for filings that survive the filters.

Session rule, matching the instructor feedback:
  - acceptance at or after 16:00 ET  -> the release is public for the next session
  - acceptance before 09:30 ET       -> same session, pre-open
  - otherwise                        -> same session, intraday
  - filingDate later than the acceptance date -> treated as public at 06:00 ET
    on the filingDate, flagged in `timing_flag`

Usage:
    python build_event_table.py --zip <submissions.zip> --out event_table.csv
"""

from __future__ import annotations

from paths import BASE, src, dst

import argparse
import csv
import json
import os
import sys
import zipfile
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo

UTC = ZoneInfo("UTC")
ET = ZoneInfo("America/New_York")

START_YEAR = 2006
END_YEAR = 2025

CLOSE = time(16, 0)
OPEN = time(9, 30)


def parse_acceptance(raw: str):
    """Acceptance as the submissions JSON gives it, read as true UTC.

    This is the fallback and it is known to be unreliable. On a sample of
    300 filings the field agrees with the filing header in 99 cases and
    disagrees in 201: in 128 it is Eastern time with a "Z" on it, and in 73
    it is eight hours ahead of Eastern. The three regimes are mixed inside
    the same year, so nothing here can tell them apart. Use
    `load_header_times` wherever the header has been fetched.
    """
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ET)


def load_header_times(path: str) -> dict[str, datetime]:
    """ACCEPTANCE-DATETIME per accession, from fetch_acceptance.py.

    The header field is Eastern time written as YYYYMMDDHHMMSS with no
    zone, and the instructors say it is the one that does not move.
    """
    if not path or not os.path.exists(path):
        return {}
    out: dict[str, datetime] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = (r.get("acceptance_header_et") or "").strip()
            if len(s) != 14 or not s.isdigit():
                continue
            out[r["accession"]] = datetime.strptime(
                s, "%Y%m%d%H%M%S").replace(tzinfo=ET)
    return out


def session_for(acc_et: datetime | None, filing_date: str):
    """Return (release_et, session_date, when, flag).

    `release_et` is the moment we treat the filing as public, which is the
    acceptance time in every ordinary case. `session_date` is a calendar
    date: an after-close release is rolled one calendar day here, and
    wrds_match_and_funnel.py then rolls it forward onto the trading
    calendar, because a Friday evening release lands on a Saturday and
    Saturday is not a session.
    """
    if acc_et is None:
        return None, filing_date, "unknown", "no_acceptance_time"

    acc_date = acc_et.date().isoformat()
    if filing_date and filing_date > acc_date:
        # EDGAR took the filing but disseminated it the following business
        # morning. The report says we treat that as public at 06:00 ET on
        # the filing date, so the timestamp has to say 06:00 and not the
        # acceptance time, which is the previous evening and would date the
        # release before anybody could see it.
        y, m, d = (int(x) for x in filing_date.split("-"))
        return (datetime(y, m, d, 6, 0, tzinfo=ET), filing_date,
                "pre_open", "filed_after_acceptance")

    t = acc_et.timetz().replace(tzinfo=None)
    if t >= CLOSE:
        nxt = (acc_et + timedelta(days=1)).date().isoformat()
        return acc_et, nxt, "after_close", ""
    if t < OPEN:
        return acc_et, acc_date, "pre_open", ""
    return acc_et, acc_date, "intraday", ""


def iter_tables(zf: zipfile.ZipFile):
    for info in zf.infolist():
        name = info.filename
        if not name.endswith(".json") or not name.startswith("CIK"):
            continue
        try:
            with zf.open(info) as fh:
                doc = json.load(fh)
        except Exception:
            continue
        if isinstance(doc, dict) and "filings" in doc:
            t = doc.get("filings", {}).get("recent")
            if t:
                yield t, name, doc.get("name", "")
        elif isinstance(doc, dict) and "form" in doc:
            yield doc, name, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default=dst("event_table.csv"))
    ap.add_argument("--acceptance", default=src("acceptance_times.csv"),
                    help="ACCEPTANCE-DATETIME per accession, from "
                         "fetch_acceptance.py. Used in preference to "
                         "the submissions JSON wherever it has a row.")
    args = ap.parse_args()

    header_et = load_header_times(args.acceptance)
    if header_et:
        print(f"header acceptance times: {len(header_et):,}", flush=True)
    else:
        print("no header file; falling back to the submissions JSON, "
              "which is wrong for about two filings in three",
              file=sys.stderr)

    rows = 0
    counted = {"header": 0, "json": 0}
    flags = {"": 0, "no_acceptance_time": 0, "filed_after_acceptance": 0}
    when = {"after_close": 0, "pre_open": 0, "intraday": 0, "unknown": 0}

    with zipfile.ZipFile(args.zip) as zf, \
         open(args.out, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow([
            "cik", "company", "accession", "filing_date", "report_date",
            "acceptance_utc", "acceptance_et", "acceptance_source",
            "release_et", "session_date", "when",
            "timing_flag", "items", "primary_document", "filing_dir_url",
        ])

        for table, member, company in iter_tables(zf):
            cik_raw = member.split("-")[0][3:].removesuffix(".json")
            cik = cik_raw.lstrip("0")
            forms = table.get("form", [])
            items = table.get("items", [])
            fdates = table.get("filingDate", [])
            rdates = table.get("reportDate", [])
            accs = table.get("accessionNumber", [])
            accept = table.get("acceptanceDateTime", [])
            prim = table.get("primaryDocument", [])

            for i, form in enumerate(forms):
                if form != "8-K":
                    continue
                fd = fdates[i] if i < len(fdates) else ""
                if len(fd) < 4:
                    continue
                year = int(fd[:4])
                if not (START_YEAR <= year <= END_YEAR):
                    continue
                codes = [c.strip() for c in
                         (items[i] if i < len(items) else "").split(",")]
                if "2.02" not in codes:
                    continue

                acc = accs[i] if i < len(accs) else ""
                raw_acc_dt = accept[i] if i < len(accept) else ""
                acc_et = header_et.get(acc)
                if acc_et is not None:
                    counted["header"] += 1
                else:
                    acc_et = parse_acceptance(raw_acc_dt)
                    counted["json"] += 1
                rel_et, sess, wh, flag = session_for(acc_et, fd)

                nodash = acc.replace("-", "")
                url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                       f"{nodash}/")

                w.writerow([
                    cik, company, acc, fd,
                    rdates[i] if i < len(rdates) else "",
                    raw_acc_dt,
                    acc_et.isoformat() if acc_et else "",
                    "header" if acc in header_et else "json",
                    rel_et.isoformat() if rel_et else "",
                    sess, wh, flag,
                    items[i] if i < len(items) else "",
                    prim[i] if i < len(prim) else "",
                    url,
                ])
                rows += 1
                flags[flag] = flags.get(flag, 0) + 1
                when[wh] = when.get(wh, 0) + 1

    print(f"rows written            : {rows:,}")
    print(f"  from the filing header: {counted['header']:,}")
    print(f"  from the JSON         : {counted['json']:,}")
    print(f"  after close           : {when['after_close']:,}")
    print(f"  pre-open              : {when['pre_open']:,}")
    print(f"  intraday              : {when['intraday']:,}")
    print(f"  unknown time          : {when['unknown']:,}")
    print(f"flagged filed>acceptance: {flags['filed_after_acceptance']:,}")
    print(f"flagged no timestamp    : {flags['no_acceptance_time']:,}")
    print(f"written                 : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
