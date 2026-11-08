"""
Text normalisation for the *baselines* (dictionary and TF-IDF).

Part 5 already did the heavy cleaning (HTML, iXBRL, page furniture, repeated
paragraphs, boiler-plate sections, financial tables set aside). What remains here
is the light, model-specific normalisation a bag-of-words needs, kept separate
from part 5 on purpose: the LLMs of part 6 read part 5's text as is, and nothing
we do here should change what they see.

Choices
-------
* **Lower-case, letters only for the dictionary.** The Loughran-McDonald word
  lists are upper-case single words; matching is case-insensitive.
* **Numbers become placeholder tokens for TF-IDF** (``_num_``, ``_pct_``,
  ``_usd_``). A bag-of-words cannot read "$1.23 vs $1.10", and leaving raw numbers
  in would create millions of one-off features (every EPS value, every year).
  Keeping a placeholder preserves the *presence* of figures, which is informative
  ("record", "_pct_ increase").
* **Years are masked too** (``_year_``): otherwise "2008" becomes a crisis
  feature, a pure look-ahead / regime artefact.
* **No stemming.** The LM lists are already inflected ("losses", "lost"), and
  stemming would merge words LM keeps apart.
"""
from __future__ import annotations

import re

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
USD_RE = re.compile(r"\$\s*\(?\d[\d,]*(?:\.\d+)?\)?(?:\s*(?:million|billion|thousand))?", re.I)
PCT_RE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?\s*(?:%|percent\b|per cent\b)", re.I)
NUM_RE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")
SPACE_RE = re.compile(r"\s+")


def words(text: str) -> list[str]:
    """Lower-cased word tokens (letters and inner apostrophes only)."""
    return WORD_RE.findall((text or "").lower())


def mask_numbers(text: str) -> str:
    """Replace years, dollar amounts, percentages and other numbers by tokens."""
    t = text or ""
    t = YEAR_RE.sub(" _year_ ", t)
    t = USD_RE.sub(" _usd_ ", t)
    t = PCT_RE.sub(" _pct_ ", t)
    t = NUM_RE.sub(" _num_ ", t)
    return SPACE_RE.sub(" ", t).strip().lower()


# Token pattern for scikit-learn's vectorizer: words, and the placeholders above.
TFIDF_TOKEN_PATTERN = r"(?u)\b_?[a-z][a-z']+_?\b"
