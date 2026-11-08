"""
Synthetic data with a *known* answer, for the tests and for the demo run.

Why
---
1. The real inputs are licensed WRDS data and cannot leave the team Drive, so a
   reviewer (or a teammate without WRDS) needs a way to run every notebook.
2. More importantly, it is the only way to check that the statistics recover an
   effect we planted, and find nothing where we planted nothing. A pipeline that
   "finds" text information in the placebo outcome is broken, whatever the real
   data say.

The data-generating process
---------------------------
For firm i, quarter t:

    z_sue  ~ N(0, 1)      the earnings news, seen by I/B/E/S
    z_text ~ N(0, 1)      news only in the text (guidance, one-offs), independent of z_sue
    tone   = 0.6 * z_sue + 0.8 * z_text + noise    what the release "sounds" like

    car_0_1      = 0.030 * tanh(z_sue) + TEXT_EFFECT * z_text + season shock + e,  e ~ N(0, 0.05)
    post_car_60  = DRIFT_EFFECT * z_text + e'
    ar_m1        = pure noise (the placebo: nothing should predict it)

So: SUE alone explains part of the reaction (non-linearly, through tanh); the
text tone also correlates with SUE (as real releases do), but only its z_text
part adds information beyond SUE. A correct pipeline must find (i) a positive
text coefficient controlling for SUE, (ii) a positive text effect in both rows of
the 2x2, (iii) nothing on ar_m1.

Texts are drawn from a small vocabulary: LM-style positive and negative words
in proportion to ``tone``, neutral filler, and a firm-specific token (the
"firm name") that the TF-IDF firm filter must remove.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import settings as S

TEXT_EFFECT = 0.012
DRIFT_EFFECT = 0.006

POS = ["growth", "record", "strong", "exceeded", "improved", "gains", "achieved", "success"]
NEG = ["decline", "loss", "weak", "impairment", "adverse", "challenging", "losses", "decreased"]
UNC = ["uncertain", "may", "approximately", "risk"]
FILLER = ("the company reported results for the quarter revenue operating segment customers "
          "products services margin net income per share compared with prior year period "
          "management said continue business market fiscal outlook").split()


def make_events(n_firms: int = 240, years=range(2006, 2026), seed: int = S.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    common = S.common_3_4()
    rows = []
    firm_size = rng.lognormal(8, 1.3, n_firms)
    for y in years:
        for q, month in enumerate((1, 4, 7, 10)):
            season_shock = rng.normal(0, 0.01)
            for i in range(n_firms):
                if rng.random() < 0.08:      # not every firm reports every quarter in the sample
                    continue
                day = pd.Timestamp(y, month, 1) + pd.Timedelta(days=int(rng.integers(14, 45)))
                day = day + pd.offsets.BDay(0)
                rows.append((i, y, q, day, season_shock))
    ev = pd.DataFrame(rows, columns=["firm", "year", "q", "reaction_date", "season_shock"])
    n = len(ev)
    z_sue, z_text = rng.normal(size=n), rng.normal(size=n)
    ev["z_sue"], ev["z_text"] = z_sue, z_text
    ev["tone_latent"] = 0.6 * z_sue + 0.8 * z_text + rng.normal(0, 0.3, n)

    ev["permno"] = 10000 + ev["firm"]
    ev["ticker"] = "F" + ev["firm"].astype(str)
    ev["event_id"] = ev["ticker"] + "|" + ev["reaction_date"].dt.strftime("%Y-%m-%d")
    ev["accession"] = [f"0000{f:06d}-{d:%y}-{k:06d}" for k, (f, d) in
                       enumerate(zip(ev["firm"], ev["reaction_date"]))]
    ev["price_t2"] = rng.lognormal(3.3, 0.6, n)
    ev["consensus_median"] = rng.normal(0.8, 0.5, n).round(2)
    # surprise in cents: some exact meets, as in real data
    surprise = np.round(0.03 * z_sue * ev["price_t2"] / 20, 2)
    surprise[rng.random(n) < 0.08] = 0.0
    ev["surprise"] = surprise
    ev["actual"] = ev["consensus_median"] + ev["surprise"]
    ev["sue_price"] = ev["surprise"] / ev["price_t2"]
    ev["n_analysts"] = rng.integers(3, 25, n).astype(float)
    ev["dispersion"] = np.abs(rng.normal(0.03, 0.02, n))
    ev["sue_disp"] = ev["surprise"] / ev["dispersion"].clip(lower=0.005)
    ev["mktcap_t2"] = firm_size[ev["firm"]] * rng.lognormal(0, 0.1, n)

    e = lambda s: rng.normal(0, s, n)
    ev["car_0_1_ff4"] = 0.03 * np.tanh(z_sue) + TEXT_EFFECT * z_text + ev["season_shock"] + e(0.05)
    ev["post_car_1_ff4"] = 0.3 * (0.03 * np.tanh(z_sue) + TEXT_EFFECT * z_text) + e(0.03)
    ev["post_car_60_ff4"] = DRIFT_EFFECT * z_text + e(0.12)
    ev["car_1_60_ff4"] = DRIFT_EFFECT * z_text + e(0.12)
    ev["ar_m1"] = e(0.02)
    for a, b in [("car_0_1_mkt", "car_0_1_ff4"), ("post_car_1_mkt", "post_car_1_ff4"),
                 ("post_car_60_mkt", "post_car_60_ff4")]:
        ev[a] = ev[b] + e(0.01)
    ev["ret_prior_1m"] = e(0.08)

    ev["R_et"] = (ev["reaction_date"] - pd.Timedelta(hours=2)).dt.tz_localize("America/New_York").astype(str)
    ev["session_type"] = "pre_open"
    ev["clock_status"] = np.where(rng.random(n) < 0.8, "wire_ibes_agree", "wire_ibes_conflict")
    ev["R_source"] = "synthetic"
    ev["two_source"] = ev["clock_status"].eq("wire_ibes_agree")
    ev["timing_suspect"] = False
    ev["basis_suspect"] = rng.random(n) < 0.03
    ev["f_size"] = ev["mktcap_t2"] > np.quantile(ev["mktcap_t2"], 0.2)
    ev["after_llama_cutoff"] = ev["reaction_date"] > S.LLAMA_CUTOFF
    ev["in_text_sample"] = True
    ev["in_primary"] = ev["two_source"]
    ev["split"] = common.split_of(ev["reaction_date"]).astype(object)
    ev["surprise_version"], ev["returns_version"], ev["sample_version"] = "synthetic", "synthetic", "synthetic"
    return ev.drop(columns=["season_shock"])


def make_texts(ev: pd.DataFrame, words_per_doc: int = 220, seed: int = S.RANDOM_SEED) -> pd.DataFrame:
    """One release per event: tone-dependent mix of positive / negative words."""
    import hashlib

    rng = np.random.default_rng(seed + 1)
    docs = []
    for acc, tone, firm, d in zip(ev["accession"], ev["tone_latent"], ev["firm"], ev["reaction_date"]):
        p_pos = 0.05 / (1 + np.exp(-1.5 * tone)) + 0.01
        p_neg = 0.05 / (1 + np.exp(1.5 * tone)) + 0.01
        k = rng.multinomial(words_per_doc, [p_pos, p_neg, 0.02, 1 - p_pos - p_neg - 0.02])
        toks = (list(rng.choice(POS, k[0])) + list(rng.choice(NEG, k[1])) + list(rng.choice(UNC, k[2]))
                + list(rng.choice(FILLER, k[3])))
        rng.shuffle(toks)
        # firm name token and some figures, as in real releases
        text = (f"acmecorp{firm} reports results for {d.year} quarter. EPS of $1.{firm % 90:02d}, "
                f"up 12% ... " + " ".join(toks) + f" about acmecorp{firm}.")
        docs.append((acc, text))
    t = pd.DataFrame(docs, columns=["accession", "text_body"])
    t["sha256_text"] = t["text_body"].map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    t["clean_version"] = "synthetic"
    t["n_words_body"] = t["text_body"].str.split().str.len()
    t["text_source"] = "synthetic"
    return t


def write_lm_dictionary(path: Path) -> Path:
    """A tiny LM-format master dictionary covering the synthetic vocabulary."""
    from baseline_lm import CATEGORIES

    rows = []
    for w in set(POS + NEG + UNC + FILLER):
        r = {"Word": w.upper(), **{c: 0 for c in CATEGORIES}}
        r["Positive"] = 2009 if w in POS else 0
        r["Negative"] = 2009 if w in NEG else 0
        r["Uncertainty"] = 2009 if w in UNC else 0
        rows.append(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_model_cache(ev: pd.DataFrame, path: Path, seed: int = S.RANDOM_SEED) -> Path:
    """
    A model-output cache in part 5's schema with two fake "LLM" signals:
    a good one (sees z_text) and a useless one (pure noise), so the comparison
    tables can be checked: the first must beat the baselines, the second must not.
    """
    rng = np.random.default_rng(seed + 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("""create table outputs (key text primary key, accession text, sha256_text text,
        clean_version text, variant text, model_id text, model_revision text, prompt_id text,
        raw_response text, score_json text, created_at_utc text, duration_s real)""")
    recs = []
    for acc, zs, zt in zip(ev["accession"], ev["z_sue"], ev["z_text"]):
        good = 0.5 * zs + 0.7 * zt + rng.normal(0, 0.5)
        recs.append((f"g{acc}", acc, "x", "synthetic", "raw", "demo/good-llm", "r1", "tone_v1",
                     "", json.dumps({"score": float(np.clip(good, -3, 3))}), "2026-11-06T00:00:00", 0.1))
        recs.append((f"n{acc}", acc, "x", "synthetic", "raw", "demo/noise-llm", "r1", "tone_v1",
                     "", json.dumps(float(rng.normal())), "2026-11-06T00:00:00", 0.1))
    con.executemany("insert into outputs values (?,?,?,?,?,?,?,?,?,?,?,?)", recs)
    con.commit()
    con.close()
    return path
