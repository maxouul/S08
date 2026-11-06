"""
Baseline 2 (task 7): Loughran-McDonald (LM) dictionary tone.

Why LM and not a general sentiment lexicon
------------------------------------------
Loughran and McDonald (2011, Journal of Finance) show that general-purpose word
lists (Harvard-IV) misclassify financial text: "liability", "tax", "cost",
"capital" are "negative" in Harvard-IV but neutral in a 10-K. Their lists were
built from SEC filings, they are the standard text baseline in accounting and
finance, and they need no training, so they cannot overfit and cannot leak.

What we compute (per release, on part 5's ``text_body``)
-------------------------------------------------------
counts of Positive, Negative, Uncertainty, Litigious, Strong/Weak modal and
Constraining words, the number of words, and four signals:

* ``lm_tone``     = (pos - neg) / (pos + neg)   in [-1, 1]; 0 when neither occurs.
                    Main LM signal: it does not depend on document length.
* ``lm_net``      = (pos - neg) / words
* ``lm_negshare`` = - neg / words  (sign flipped so that higher = better news,
                    like every other signal in this repository; Tetlock 2007
                    finds the negative share is what matters)
* ``lm_unc``      = uncertainty / words (not a direction; used as a control)

Negation: following LM (2011), a positive word preceded within three words by
"no, not, none, neither, never, nobody" is not counted as positive ("did not
achieve"). Negated negatives are left as negatives (LM do the same: "not
impaired" is rare and ambiguous).

Caching
-------
Counts are keyed on (accession, sha256_text, dictionary file sha256). They are
stored in data/text_features/lm_counts.parquet and only new or changed texts are
counted on a rerun, in the same spirit as part 5's model-output cache.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S
from text_utils import words

CATEGORIES = ["Negative", "Positive", "Uncertainty", "Litigious", "Strong_Modal",
              "Weak_Modal", "Constraining"]
NEGATORS = {"no", "not", "none", "neither", "never", "nobody"}
NEGATION_WINDOW = 3
LM_VERSION = "lm_v1"


def find_dictionary(lm_dir: Path | None = None) -> Path | None:
    """The LM master dictionary CSV placed in resources/lm/ (newest file wins)."""
    lm_dir = lm_dir or S.LM_DIR
    files = sorted(lm_dir.glob("*MasterDictionary*.csv")) + sorted(lm_dir.glob("*master_dictionary*.csv"))
    return files[-1] if files else None


def load_dictionary(path: Path | str | None = None) -> tuple[dict[str, set[str]], str]:
    """
    Word sets per category and the file's sha256 (part of the cache key).

    In the master dictionary each category column holds the year a word was added
    (e.g. 2009), 0 if the word is not in the category, and a negative year if it
    was removed. Only positive values count as membership.
    """
    path = Path(path) if path else find_dictionary()
    if path is None:
        raise FileNotFoundError(
            "Loughran-McDonald master dictionary not found in resources/lm/. Download the CSV "
            "from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ (see resources/lm/README.md).")
    raw = path.read_bytes()
    df = pd.read_csv(path, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    lists = {}
    for c in CATEGORIES:
        if c not in df.columns:
            raise KeyError(f"{path.name} has no column {c}: is it the LM master dictionary?")
        lists[c] = set(df.loc[pd.to_numeric(df[c], errors="coerce").fillna(0) > 0, "Word"]
                       .astype(str).str.lower())
    return lists, hashlib.sha256(raw).hexdigest()[:16]


def count_text(text: str, lists: dict[str, set[str]]) -> dict:
    """Category counts for one text, with LM negation for positive words."""
    toks = words(text)
    out = {c: 0 for c in CATEGORIES}
    out["n_words"] = len(toks)
    out["n_pos_negated"] = 0
    for i, w in enumerate(toks):
        for c in CATEGORIES:
            if w in lists[c]:
                if c == "Positive" and any(t in NEGATORS for t in toks[max(0, i - NEGATION_WINDOW):i]):
                    out["n_pos_negated"] += 1
                    continue
                out[c] += 1
    return out


def signals_from_counts(c: pd.DataFrame) -> pd.DataFrame:
    """The four LM signals from the count columns."""
    pos, neg, n = c["Positive"], c["Negative"], c["n_words"].replace(0, np.nan)
    s = pd.DataFrame(index=c.index)
    s["lm_tone"] = ((pos - neg) / (pos + neg).replace(0, np.nan)).fillna(0.0)
    s["lm_net"] = (pos - neg) / n
    s["lm_negshare"] = -neg / n
    s["lm_unc"] = c["Uncertainty"] / n
    return s


def lm_features(texts: pd.DataFrame, dict_path: Path | str | None = None,
                cache_path: Path | str | None = None) -> pd.DataFrame:
    """
    LM counts and signals for every text, using and extending the cache.

    texts: accession, sha256_text, text_body (io_inputs.load_texts).
    Returns one row per accession with counts, signals and the dictionary id.
    """
    lists, dict_id = load_dictionary(dict_path)
    cache_path = Path(cache_path) if cache_path else S.TEXT_FEATURES / "lm_counts.parquet"
    key_cols = ["accession", "sha256_text", "dict_id"]
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=key_cols)

    t = texts[["accession", "sha256_text", "text_body"]].copy()
    t["dict_id"] = dict_id
    have = t.merge(cached[key_cols], on=key_cols, how="left", indicator=True)["_merge"].eq("both").to_numpy()
    todo = t.loc[~have]
    print(f"LM: {int(have.sum()):,} texts from cache, {len(todo):,} to count (dictionary {dict_id})")
    if len(todo):
        rows = [count_text(x, lists) for x in todo["text_body"]]
        new = pd.concat([todo[key_cols].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
        new["lm_version"] = LM_VERSION
        cached = pd.concat([cached, new], ignore_index=True).drop_duplicates(key_cols, keep="last")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_parquet(cache_path, index=False)

    feats = t[key_cols].merge(cached, on=key_cols, how="left")
    feats = pd.concat([feats, signals_from_counts(feats)], axis=1)
    return feats
