"""
Step 5.6: Cache of model outputs, so that no model call is ever paid for twice.

The key is sha256(accession | sha256_text | clean_version | variant | model_id |
model_revision | prompt_id), with variant one of raw, mask, placebo. Changing the text,
the cleaning, the model or the prompt therefore gives a new key, and an old answer is
never served for a new question.

Storage is one SQLite file. Every write is its own transaction (WAL journal), so a
crash leaves either the whole record or nothing, and a rerun picks up where it stopped.
For each call we keep the raw model response, the score extracted from it (JSON), the
time and the duration.

In a scoring script:
    from model_cache import ModelCache
    mc = ModelCache()
    rec = mc.get_or_compute(dict(accession=..., sha256_text=..., clean_version="v1",
                                 variant="mask", model_id=..., model_revision=...,
                                 prompt_id="themes_v1"),
                            lambda: call_model(text))   # returns (raw_response, score)

Usage: python code/model_cache.py     (creates cache/model_outputs.sqlite if needed, prints counts)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone

import settings as S

KEY_FIELDS = ("accession", "sha256_text", "clean_version", "variant", "model_id",
              "model_revision", "prompt_id")
VARIANTS = {"raw", "mask", "placebo"}

SCHEMA = """
create table if not exists outputs (
    key            text primary key,
    accession      text not null,
    sha256_text    text not null,
    clean_version  text not null,
    variant        text not null,
    model_id       text not null,
    model_revision text not null,
    prompt_id      text not null,
    raw_response   text,
    score_json     text,
    created_at_utc text not null,
    duration_s     real
);
create index if not exists outputs_model on outputs (model_id, prompt_id, variant);
"""


def cache_key(parts: dict) -> str:
    missing = [f for f in KEY_FIELDS if not parts.get(f)]
    if missing:
        raise ValueError(f"cache key needs {missing}")
    if parts["variant"] not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    return hashlib.sha256("|".join(str(parts[f]) for f in KEY_FIELDS).encode()).hexdigest()


class ModelCache:
    def __init__(self, path: str | None = None):
        path = path or str(S.MODEL_CACHE)
        self.db = sqlite3.connect(path, timeout=60, isolation_level=None)
        self.db.execute("pragma journal_mode=wal")
        self.db.execute("pragma synchronous=normal")
        self.db.executescript(SCHEMA)

    def get(self, parts: dict) -> dict | None:
        cur = self.db.execute("select * from outputs where key = ?", (cache_key(parts),))
        row = cur.fetchone()
        if row is None:
            return None
        rec = dict(zip([c[0] for c in cur.description], row))
        rec["score"] = json.loads(rec.pop("score_json")) if rec["score_json"] else None
        return rec

    def put(self, parts: dict, raw_response: str, score, duration_s: float) -> str:
        key = cache_key(parts)
        # insert or ignore: a key already present is never overwritten
        self.db.execute(
            "insert or ignore into outputs values (?,?,?,?,?,?,?,?,?,?,?,?)",
            (key, *(str(parts[f]) for f in KEY_FIELDS), raw_response,
             json.dumps(score), datetime.now(timezone.utc).isoformat(timespec="seconds"),
             duration_s))
        return key

    def get_or_compute(self, parts: dict, compute) -> dict:
        rec = self.get(parts)
        if rec is not None:
            return rec
        t0 = time.monotonic()
        raw_response, score = compute()
        self.put(parts, raw_response, score, time.monotonic() - t0)
        return self.get(parts)

    def counts(self):
        return self.db.execute("""
            select model_id, prompt_id, variant, count(*), round(avg(duration_s), 2)
            from outputs group by 1, 2, 3 order by 1, 2, 3""").fetchall()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(S.MODEL_CACHE))
    args = ap.parse_args()
    mc = ModelCache(args.db)
    rows = mc.counts()
    total = sum(r[3] for r in rows)
    print(f"{args.db}: {total:,} cached outputs")
    for model, prompt, variant, n, dur in rows:
        print(f"  {model:<40} {prompt:<14} {variant:<8} {n:>8,}  {dur} s avg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
