"""
Step 5.5: Flag duplicate and overlapping releases. Nothing is removed here; the flags
are applied, or not, when the final sample is built.

Three kinds of flag:
- exact: the same cleaned text (same sha256_text) under several accessions;
- near: Jaccard similarity of word 5-grams of text_body of at least 0.9, estimated by
  MinHash with LSH (datasketch, 128 permutations), then confirmed on the estimate;
- close_events: the same PERMNO with two events at most 5 trading sessions apart, as
  when a preliminary release is followed by the final one. The date is reaction_date
  from the release clock (part 3-4) and sessions are counted on its trading calendar.
  Events the clock has not dated yet get no such flag; the clock is provisional, so
  rerun this script when it is replaced.

Output: data/dedup_flags.parquet (accession, dup_group, flag_type); an accession can carry
several flags. dup_group is unique within a flag_type.

Usage: python code/dedup.py --threshold 0.9
"""

from __future__ import annotations

import argparse
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

import settings as S

NUM_PERM = 128
WORD_RE = re.compile(r"\w+")


def minhash(text: str):
    from datasketch import MinHash
    w = WORD_RE.findall(text.lower())
    shingles = {" ".join(w[i:i + 5]).encode() for i in range(max(len(w) - 4, 1))}
    m = MinHash(num_perm=NUM_PERM)
    m.update_batch(list(shingles))
    return m


class Groups:
    """Union-find, to turn similar pairs into groups."""

    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, a: str) -> str:
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: str, b: str) -> None:
        self.p[self.find(a)] = self.find(b)

    def labelled(self) -> pd.DataFrame:
        df = pd.DataFrame({"accession": list(self.p)})
        df["root"] = df.accession.map(self.find)
        df = df[df.groupby("root").accession.transform("size") > 1]
        df["dup_group"] = df.root.factorize()[0]
        return df[["accession", "dup_group"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", default=str(S.CLEAN_DIR))
    ap.add_argument("--clock", default=str(S.CLOCK))
    ap.add_argument("--calendar", default=str(S.CALENDAR))
    ap.add_argument("--out", default=str(S.DEDUP_FLAGS))
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--max-gap", type=int, default=5, help="trading sessions")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    from datasketch import MinHashLSH

    parts = [pd.read_parquet(os.path.join(args.clean_dir, f),
                             columns=["accession", "sha256_text", "text_body"])
             for f in sorted(os.listdir(args.clean_dir))
             if f.startswith("clean_") and f.endswith(".parquet")]
    if not parts:
        print(f"no clean_YYYY.parquet in {args.clean_dir}: run clean_release.py first")
        return 1
    df = pd.concat(parts, ignore_index=True).dropna(subset=["sha256_text"])
    print(f"{len(df):,} cleaned texts", flush=True)
    flags = []

    # exact
    g = df[df.groupby("sha256_text").accession.transform("size") > 1].copy()
    g["dup_group"] = g.sha256_text.factorize()[0]
    g["flag_type"] = "exact"
    flags.append(g[["accession", "dup_group", "flag_type"]])
    print(f"exact duplicates: {len(g):,} texts in {g.dup_group.nunique():,} groups",
          flush=True)

    # near: one representative per exact group, so exact copies are not counted twice
    rep = df.drop_duplicates("sha256_text")
    with Pool(args.workers) as pool:
        mh = dict(zip(rep.accession,
                      pool.map(minhash, rep.text_body.fillna("").tolist(), chunksize=64)))
    lsh = MinHashLSH(threshold=args.threshold, num_perm=NUM_PERM)
    for acc, m in mh.items():
        lsh.insert(acc, m)
    near = Groups()
    for acc, m in mh.items():
        for other in lsh.query(m):
            if other != acc and m.jaccard(mh[other]) >= args.threshold:
                near.union(acc, other)
    n = near.labelled()
    n["flag_type"] = "near"
    flags.append(n)
    print(f"near duplicates (Jaccard >= {args.threshold}): {len(n):,} texts in "
          f"{n.dup_group.nunique():,} groups", flush=True)

    # close events of the same security, in trading sessions
    ck = pd.read_csv(args.clock, usecols=["accession", "permno", "reaction_date"],
                     dtype={"accession": str})
    ck = ck[ck.accession.isin(set(df.accession))].drop_duplicates("accession")
    ck["d"] = pd.to_datetime(ck.reaction_date, errors="coerce")
    undated = int(ck.d.isna().sum())
    ck = ck.dropna(subset=["d", "permno"]).sort_values(["permno", "d"])
    cal = pd.to_datetime(pd.read_csv(args.calendar, usecols=["date"]).date).sort_values()
    ck["pos"] = np.searchsorted(cal.values, ck.d.values)        # session index
    ck["prev_acc"] = ck.groupby("permno").accession.shift()
    ck["prev_pos"] = ck.groupby("permno").pos.shift()
    p = ck.dropna(subset=["prev_pos"])
    close = Groups()
    for a, b in p.loc[p.pos - p.prev_pos <= args.max_gap,
                      ["prev_acc", "accession"]].itertuples(index=False):
        close.union(a, b)
    c = close.labelled()
    c["flag_type"] = "close_events"
    flags.append(c)
    print(f"same permno within {args.max_gap} sessions: {len(c):,} events in "
          f"{c.dup_group.nunique():,} groups ({undated:,} texts not dated by the clock)",
          flush=True)

    out = pd.concat(flags, ignore_index=True)
    out.to_parquet(args.out, index=False)
    print(f"\nflags written: {len(out):,} rows, {out.accession.nunique():,} accessions")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
