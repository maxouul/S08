"""put the header acceptance times into the event table.

we cannot rebuild from scratch: build_event_table.py needs the 1.6 GB SEC
archive and it is not on this machine any more. we do not need it either.
only the time changes, and fetch_acceptance.py has it, so we rewrite the
four columns that depend on it and leave the rest of the row alone.

the session rule is imported from build_event_table.py, not copied, so
there is one rule in the project.

    python apply_header_times.py
"""

from __future__ import annotations

from paths import BASE, src, dst

import argparse
import csv
import importlib.util
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def load_rule():
    """session_for, from build_event_table.py, without running its main."""
    spec = importlib.util.spec_from_file_location(
        "bet", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "build_event_table.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.session_for, mod.parse_acceptance


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=os.path.join(BASE,
                                                 "event_table.csv"),
                    help="always the base table: this rebuilds it, so "
                         "reading its own output would compound.")
    ap.add_argument("--acceptance", default=src("acceptance_times.csv"))
    ap.add_argument("--out", default=dst("event_table.csv"))
    ap.add_argument("--require-complete", action="store_true",
                    help="refuse to write unless every filing has a header")
    args = ap.parse_args()

    session_for, parse_acceptance = load_rule()

    header: dict[str, datetime] = {}
    with open(args.acceptance, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = (r.get("acceptance_header_et") or "").strip()
            if len(s) == 14 and s.isdigit():
                header[r["accession"]] = datetime.strptime(
                    s, "%Y%m%d%H%M%S").replace(tzinfo=ET)
    print(f"{len(header):,} header timestamps", flush=True)

    with open(args.events, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        for extra in ("acceptance_source", "release_et"):
            if extra not in cols:
                cols.insert(cols.index("session_date"), extra)

        n = {"rows": 0, "header": 0, "json": 0, "moved": 0, "resession": 0}
        with open(args.out, "w", newline="", encoding="utf-8") as out:
            w = csv.DictWriter(out, fieldnames=cols)
            w.writeheader()
            for row in reader:
                n["rows"] += 1
                acc = row["accession"]
                was_session = row.get("session_date", "")
                was_acc = row.get("acceptance_et", "")

                acc_et = header.get(acc)
                if acc_et is not None:
                    n["header"] += 1
                    row["acceptance_source"] = "header"
                else:
                    acc_et = parse_acceptance(row.get("acceptance_utc", ""))
                    n["json"] += 1
                    row["acceptance_source"] = "json"

                rel, sess, when, flag = session_for(acc_et,
                                                    row.get("filing_date", ""))
                row["acceptance_et"] = acc_et.isoformat() if acc_et else ""
                row["release_et"] = rel.isoformat() if rel else ""
                row["session_date"] = sess
                row["when"] = when
                row["timing_flag"] = flag

                if row["acceptance_et"] != was_acc:
                    n["moved"] += 1
                if sess != was_session:
                    n["resession"] += 1
                w.writerow(row)

    if args.require_complete and n["json"]:
        print(f"{n['json']:,} filings still have no header. Finish "
              f"fetch_acceptance.py first, or drop --require-complete.",
              file=sys.stderr)
        return 1

    print(f"rows                      {n['rows']:>9,}")
    print(f"  time from the header    {n['header']:>9,}  "
          f"{100 * n['header'] / max(n['rows'], 1):5.1f}%")
    print(f"  still from the JSON     {n['json']:>9,}")
    print(f"timestamp changed         {n['moved']:>9,}  "
          f"{100 * n['moved'] / max(n['rows'], 1):5.1f}%")
    print(f"session changed           {n['resession']:>9,}  "
          f"{100 * n['resession'] / max(n['rows'], 1):5.1f}%")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
