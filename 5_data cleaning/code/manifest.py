"""
Step 5.8: Describe every file part 5 produced, in manifest.json at the folder root.

For each output under cache/, data/ and results/: the script that writes it, the
cleaning version where it applies, the modification date, the number of rows (or
archive members) and the sha256. Comparing two manifests tells which outputs
changed and which steps need rerunning.

Usage: python code/manifest.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone

import settings as S

OUTPUTS = {
    "cache/raw/raw_*.zip": "cache_raw.py",
    "cache/raw_manifest.parquet": "cache_raw.py",
    "cache/model_outputs.sqlite": "model_cache.py",
    "data/clean/clean_*.parquet": "clean_release.py",
    "data/entities.parquet": "entities.py",
    "data/chunks/*/chunks_*.parquet": "token_stats.py",
    "data/token_stats.parquet": "token_stats.py",
    "results/token_quantiles.csv": "token_stats.py",
    "data/dedup_flags.parquet": "dedup.py",
    "data/splits_draft.parquet": "splits.py",
    "data/holdout_ids.txt": "splits.py --freeze",
}
VERSIONED = {"clean_release.py", "token_stats.py", "dedup.py"}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def n_rows(path: str):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return len(z.namelist())
    if path.endswith((".csv", ".txt")):
        with open(path, "rb") as f:
            n = sum(1 for _ in f)
        return n - 1 if path.endswith(".csv") else n
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(S.MANIFEST))
    args = ap.parse_args()

    from clean_release import CLEAN_VERSION

    files = []
    for pattern, script in OUTPUTS.items():
        for p in sorted(S.STEP5_DIR.glob(pattern)):
            files.append({
                "path": p.relative_to(S.STEP5_DIR).as_posix(),
                "script": script,
                "version": CLEAN_VERSION if script in VERSIONED else "",
                "modified_utc": datetime.fromtimestamp(os.path.getmtime(p), timezone.utc)
                                        .isoformat(timespec="seconds"),
                "rows": n_rows(str(p)),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })
    doc = {"written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "files": files}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    for e in files:
        print(f"  {e['path']:<36} {e['script']:<20} {str(e['rows']):>9}")
    print(f"{len(files)} files -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
