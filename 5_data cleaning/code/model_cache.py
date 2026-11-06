"""
Step 5.6: Cache of model outputs, so that no model call is ever paid for twice.

The key is sha256(accession | sha256_text | clean_version | variant | model_id |
model_revision | prompt_id), followed by | config_sha256 when the call passes a
prompt_config. variant is one of raw, mask, placebo. prompt_config holds whatever else
changes the answer: the prompt template and system prompt, the decoding parameters
(greedy, max new tokens), the masking protocol version, the window. It is stored once
in table configs and its sha256 enters the key, so a template edited without renaming
prompt_id still gets a new key. Changing the text, the cleaning, the model, the prompt
or its configuration therefore gives a new key, and an old answer is never served for
a new question.

Storage is one SQLite file. Every write is its own transaction (WAL journal), so a
crash leaves either the whole record or nothing, and a rerun picks up where it stopped.
For each call we keep the raw model response, the score extracted from it (JSON), the
time and the duration.

Parts 7-8 read table outputs (accession, clean_version, variant, model_id,
model_revision, prompt_id, score_json, created_at_utc: 7_8_baselines_evaluation/docs/
INTERFACES.md). score_json is a number, or an object with a numeric "score" field;
higher means better news for the firm. Part 9 reads the placebo variant.

In a scoring script (part 6):
    from model_cache import ModelCache
    mc = ModelCache()
    config = {"template": PROMPT, "max_new_tokens": 200, "do_sample": False,
              "mask_protocol": "none"}
    jobs = [dict(accession=a, sha256_text=h, clean_version="v2", variant="raw",
                 model_id=MODEL, model_revision=REV, prompt_id="tone_v1",
                 prompt_config=config) for a, h in texts]
    for parts in mc.missing(jobs):               # only what is not cached yet
        mc.get_or_compute(parts, lambda: call_model(...))   # -> (raw_response, score)

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
    duration_s     real,
    config_sha256  text not null default ''
);
create index if not exists outputs_model on outputs (model_id, prompt_id, variant);
create table if not exists configs (
    config_sha256  text primary key,
    prompt_id      text not null,
    config_json    text not null,
    created_at_utc text not null
);
"""
OUTPUT_COLUMNS = ("key",) + KEY_FIELDS + ("raw_response", "score_json", "created_at_utc",
                                          "duration_s", "config_sha256")


def config_json(config) -> str:
    """The canonical JSON of a prompt configuration ('' when there is none)."""
    return json.dumps(config, sort_keys=True, ensure_ascii=False) if config else ""


def config_sha256(config) -> str:
    text = config_json(config)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def cache_key(parts: dict) -> str:
    missing = [f for f in KEY_FIELDS if not parts.get(f)]
    if missing:
        raise ValueError(f"cache key needs {missing}")
    if parts["variant"] not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    base = "|".join(str(parts[f]) for f in KEY_FIELDS)
    cfg = config_sha256(parts.get("prompt_config"))
    if cfg:
        base += "|" + cfg
    return hashlib.sha256(base.encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _plain(o):
    """numpy scalars in a score become plain numbers in score_json."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


class ModelCache:
    def __init__(self, path: str | None = None):
        path = path or str(S.MODEL_CACHE)
        self.db = sqlite3.connect(path, timeout=60, isolation_level=None)
        self.db.execute("pragma journal_mode=wal")
        self.db.execute("pragma synchronous=normal")
        self.db.executescript(SCHEMA)
        # a cache created before config_sha256 existed gets the column; its keys
        # (no config) stay valid
        cols = {r[1] for r in self.db.execute("pragma table_info(outputs)")}
        if "config_sha256" not in cols:
            self.db.execute("alter table outputs add column config_sha256 text not null default ''")

    def close(self) -> None:
        self.db.close()

    def get(self, parts: dict) -> dict | None:
        cur = self.db.execute("select * from outputs where key = ?", (cache_key(parts),))
        row = cur.fetchone()
        if row is None:
            return None
        rec = dict(zip([c[0] for c in cur.description], row))
        rec["score"] = json.loads(rec.pop("score_json")) if rec["score_json"] else None
        return rec

    def missing(self, parts_list: list[dict]) -> list[dict]:
        """The calls of parts_list not in the cache yet, in order: what a batch job still
        has to run after a restart."""
        keys = [cache_key(p) for p in parts_list]
        have: set[str] = set()
        for lo in range(0, len(keys), 900):           # SQLite's limit on bound variables
            chunk = keys[lo:lo + 900]
            q = f"select key from outputs where key in ({','.join('?' * len(chunk))})"
            have.update(r[0] for r in self.db.execute(q, chunk))
        return [p for p, k in zip(parts_list, keys) if k not in have]

    def put(self, parts: dict, raw_response: str, score, duration_s: float) -> str:
        key = cache_key(parts)
        cfg = parts.get("prompt_config")
        cfg_sha = config_sha256(cfg)
        if cfg_sha:
            self.db.execute("insert or ignore into configs values (?,?,?,?)",
                            (cfg_sha, str(parts["prompt_id"]), config_json(cfg), _now()))
        # insert or ignore: a key already present is never overwritten
        self.db.execute(
            f"insert or ignore into outputs ({', '.join(OUTPUT_COLUMNS)}) "
            f"values ({', '.join('?' * len(OUTPUT_COLUMNS))})",
            (key, *(str(parts[f]) for f in KEY_FIELDS), raw_response,
             json.dumps(score, default=_plain), _now(), duration_s, cfg_sha))
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
            select model_id, prompt_id, variant, count(*), count(distinct config_sha256),
                   round(avg(duration_s), 2)
            from outputs group by 1, 2, 3 order by 1, 2, 3""").fetchall()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(S.MODEL_CACHE))
    args = ap.parse_args()
    mc = ModelCache(args.db)
    rows = mc.counts()
    total = sum(r[3] for r in rows)
    n_cfg = mc.db.execute("select count(*) from configs").fetchone()[0]
    print(f"{args.db}: {total:,} cached outputs, {n_cfg:,} prompt configurations")
    for model, prompt, variant, n, n_conf, dur in rows:
        print(f"  {model:<40} {prompt:<14} {variant:<8} {n:>8,}  {n_conf} config(s)  "
              f"{dur} s avg")
    mc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
