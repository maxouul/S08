"""
Step 5.7: Chronological splits, and the freeze of the test set.

The split rule is not restated here: SPLITS and split_of come from part 3-4's
common.py (dev 2006-2016, validation 2017-2019, excluded_2020, holdout 2021-2025, by
reaction date), so texts, returns and evaluation are always split the same way.

Each accession takes the reaction date of its event in part 4's events_final.parquet,
the table parts 7-8 evaluate on, and the split computed from it must equal the split
part 4 wrote there. A filing no event is linked to takes the reaction date of part
3-4's release clock instead; date_source says which. The clock is still being
corrected (part 1: EDGAR acceptance times against the Dow Jones wire), so dates and
splits are a draft until it is final: data/splits_draft.parquet, rewritten every run.

--freeze writes the test set, once: data/holdout_ids.txt, the sorted accessions of the
holdout events of part 4's text sample (in_text_sample: every event with a filing,
whatever its timing status; the primary sample is a subset of it), its sha256 in
holdout_ids.sha256, and holdout_freeze.json saying what it was built from. Run it when
the clock and the sample are final and before anyone looks at a holdout result. It
needs events_final.parquet and refuses to overwrite an existing freeze.

In another script:
    from splits import load_event_dates     # accession, permno, date_used, ...

Usage: python code/splits.py                  (draft only)
       python code/splits.py --freeze         (once, at the end)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import settings as S

EVENT_COLUMNS = ["accession", "event_id", "permno", "reaction_date", "split", "clock_status",
                 "in_text_sample", "in_primary", "sample_version"]
DRAFT_COLUMNS = ["accession", "event_id", "permno", "date_used", "date_source", "clock_status",
                 "in_text_sample", "in_primary", "year", "split"]


def label(dates: pd.Series, common) -> pd.Series:
    """common.split_of, with the dates it leaves blank named."""
    out = common.split_of(dates).astype("string")
    out = out.mask(out.isna() & dates.notna(), "outside")
    return out.mask(dates.isna(), "no_date")


def load_event_dates(clock: str | Path = S.CLOCK,
                     events: str | Path = S.EVENTS_FINAL) -> pd.DataFrame:
    """One row per accession: permno, reaction date (date_used) and where it comes from.
    Part 4's events_final first, then the release clock for the filings no event is
    linked to. clock_status keeps each source's own vocabulary (events_final:
    wire_ibes_agree ...; clock: two_source_agree ...)."""
    frames = []
    if Path(events).exists():
        ev = pd.read_parquet(events)
        ev = ev[[c for c in EVENT_COLUMNS if c in ev.columns]].rename(columns={"split": "split_part4"})
        ev = ev[ev.accession.notna()].copy()
        ev["accession"] = ev.accession.astype(str).str.strip()
        n_dup = int(ev.accession.duplicated().sum())
        if n_dup:
            print(f"  warning: {n_dup} accessions linked to more than one event in "
                  f"{Path(events).name}; the first is kept")
        ev = ev.drop_duplicates("accession")
        ev["date_source"] = "events_final"
        frames.append(ev)
    if Path(clock).exists():
        ck = pd.read_csv(clock, usecols=["accession", "permno", "reaction_date", "status"],
                         dtype={"accession": str})
        ck = ck.drop_duplicates("accession").rename(columns={"status": "clock_status"})
        if frames:
            ck = ck[~ck.accession.isin(frames[0].accession)]
        ck["date_source"] = "clock"
        frames.append(ck)
    if not frames:
        raise FileNotFoundError(f"neither {events} nor {clock} exists: run part 3-4 first")

    out = pd.concat(frames, ignore_index=True)
    out["permno"] = pd.to_numeric(out.permno, errors="coerce").astype("Int64")
    out["date_used"] = pd.to_datetime(out.reaction_date, errors="coerce")
    out.loc[out.date_used.isna(), "date_source"] = "none"
    for c in ("in_text_sample", "in_primary"):
        out[c] = (out[c] if c in out else pd.Series(False, index=out.index)) \
            .fillna(False).astype(bool)
    for c in ("event_id", "clock_status", "split_part4", "sample_version"):
        if c not in out:
            out[c] = pd.NA
    return out.drop(columns="reaction_date")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(S.REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock", default=str(S.CLOCK))
    ap.add_argument("--events", default=str(S.EVENTS_FINAL))
    ap.add_argument("--out", default=str(S.SPLITS_DRAFT))
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--freeze-sample", default="in_text_sample",
                    choices=["in_text_sample", "in_primary"],
                    help="part 4 sample whose holdout events are frozen")
    args = ap.parse_args()

    common = S.common_3_4()
    d = load_event_dates(args.clock, args.events)
    d["year"] = d.date_used.dt.year.astype("Int64")
    d["split"] = label(d.date_used, common)

    # The split must be part 4's: same rule, same date.
    has4 = d.split_part4.notna()
    differ = has4 & (d.split_part4.astype("string") != d.split)
    if not has4.any():
        print(f"no {Path(args.events).name}: dates from the release clock only (draft)")
    elif differ.any():
        print(f"WARNING: {int(differ.sum()):,} accessions whose split differs from part 4's "
              f"(common.py or the dates changed since events_final was built: rerun 04_sample)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    d[DRAFT_COLUMNS].to_parquet(args.out, index=False)

    print(f"{len(d):,} filings")
    print(d.date_source.value_counts().to_string())
    print()
    print(f"  {'split':<14} {'all filings':>12} {'text sample':>12} {'primary':>9}")
    for s in list(common.SPLITS) + ["outside", "no_date"]:
        m = d.split == s
        print(f"  {s:<14} {int(m.sum()):>12,} {int((m & d.in_text_sample).sum()):>12,} "
              f"{int((m & d.in_primary).sum()):>9,}")
    print(f"-> {args.out} (draft)")

    if not args.freeze:
        return 0
    if S.HOLDOUT_IDS.exists():
        print(f"{S.HOLDOUT_IDS} exists already: the holdout is frozen, nothing written")
        return 1
    if not has4.any():
        print(f"--freeze needs part 4's {args.events}: the test set is frozen on the "
              f"events parts 7-8 evaluate, not on the provisional clock")
        return 1
    if differ.any():
        print("--freeze refused: the splits disagree with part 4's (see the warning above)")
        return 1
    sel = d[(d.date_source == "events_final") & d[args.freeze_sample] & (d.split == "holdout")]
    ids = sorted(sel.accession)
    body = "\n".join(ids) + "\n"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    # bytes, not text: on Windows text mode would write CRLF and break the sha256
    S.HOLDOUT_IDS.write_bytes(body.encode("utf-8"))
    S.HOLDOUT_IDS.with_suffix(".sha256").write_bytes(
        f"{digest}  {S.HOLDOUT_IDS.name}\n".encode("utf-8"))
    record = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accessions": len(ids),
        "sha256": digest,
        "sample": args.freeze_sample,
        "holdout_rule": [str(x) for x in common.SPLITS["holdout"]],
        "date": "reaction_date of part 4's events_final.parquet",
        "sample_version": sorted(map(str, sel.sample_version.dropna().unique())),
        "git_commit": git_commit(),
    }
    S.HOLDOUT_FREEZE.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"FROZEN: {len(ids):,} holdout accessions ({args.freeze_sample}), sha256 {digest}")
    print(f"-> {S.HOLDOUT_IDS}\n-> {S.HOLDOUT_FREEZE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
