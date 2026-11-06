"""
Step 5.7: Assign each event to a sample split by the year of its reaction date.

The split rule is not restated here: SPLITS and split_of are taken from part 3-4's
common.py (dev 2006-2016, validation 2017-2019, excluded_2020, holdout 2021-2025),
so the texts and the returns are always split the same way.

The date is reaction_date from the release clock (3_4_surprise_returns/data/
clock_provisional.csv): the wire time first, EDGAR without a wire, never the earliest
of noisy dates. Where the clock has no date yet, session_final from
three_source_timing.csv stands in and date_source says so. The clock is provisional,
so by default the script only writes a draft, data/splits_draft.parquet, carrying the
clock status for later filtering.

--freeze writes data/holdout_ids.txt, the sorted holdout accessions, and its sha256.
It is a separate, deliberate act: run it once, when the sample and the clock are
final, and never after looking at holdout results. It refuses to overwrite an
existing freeze.

Usage: python code/splits.py                  (draft only)
       python code/splits.py --freeze         (once, at the end)
"""

from __future__ import annotations

import argparse
import hashlib
import os

import pandas as pd

import settings as S


def label(dates: pd.Series, common) -> pd.Series:
    """common.split_of, with the dates it leaves blank named."""
    out = common.split_of(dates).astype("string")
    out = out.mask(out.isna() & dates.notna(), "outside")
    return out.mask(dates.isna(), "no_date")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock", default=str(S.CLOCK))
    ap.add_argument("--timing", default=str(S.TIMING))
    ap.add_argument("--out", default=str(S.SPLITS_DRAFT))
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    common = S.common_3_4()

    ck = pd.read_csv(args.clock, usecols=["accession", "reaction_date", "status",
                                          "primary_eligible"], dtype=str)
    ck = ck.drop_duplicates("accession").rename(columns={"status": "clock_status"})
    ck["date_used"] = pd.to_datetime(ck.reaction_date, errors="coerce")
    ck["date_source"] = "clock_reaction_date"
    ck.loc[ck.date_used.isna(), "date_source"] = "none"

    if os.path.exists(args.timing):
        t = pd.read_csv(args.timing, usecols=["accession", "session_final"], dtype=str)
        t["f"] = pd.to_datetime(t.session_final, errors="coerce")
        t = t.dropna(subset=["f"]).drop_duplicates("accession")
        ck = ck.merge(t[["accession", "f"]], on="accession", how="left")
        fill = ck.date_used.isna() & ck.f.notna()
        ck.loc[fill, "date_used"] = ck.loc[fill, "f"]
        ck.loc[fill, "date_source"] = "timing_session_final"
        ck = ck.drop(columns="f")

    ck["year"] = ck.date_used.dt.year.astype("Int64")
    ck["split"] = label(ck.date_used, common)
    ck["primary_eligible"] = ck.primary_eligible.eq("True")
    cols = ["accession", "date_used", "date_source", "clock_status", "primary_eligible",
            "year", "split"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ck[cols].to_parquet(args.out, index=False)

    print(f"{len(ck):,} events")
    print(ck.date_source.value_counts().to_string())
    print()
    for s in list(common.SPLITS) + ["outside", "no_date"]:
        print(f"  {s:<14} {int((ck.split == s).sum()):>8,}")
    print(f"-> {args.out} (draft)")

    if args.freeze:
        ids_path = S.HOLDOUT_IDS
        if ids_path.exists():
            print(f"{ids_path} exists already: the holdout is frozen, nothing written")
            return 1
        ids = sorted(ck.loc[ck.split == "holdout", "accession"])
        body = "\n".join(ids) + "\n"
        digest = hashlib.sha256(body.encode()).hexdigest()
        ids_path.write_text(body, encoding="utf-8")
        ids_path.with_suffix(".sha256").write_text(f"{digest}  {ids_path.name}\n",
                                                    encoding="utf-8")
        print(f"FROZEN: {len(ids):,} holdout accessions, sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
