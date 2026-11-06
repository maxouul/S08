"""
Task 8: robustness checks across sectors, firm sizes, time periods, event
windows and sample definitions.

What a robustness check should show
------------------------------------
That the main result (the text coefficient of specification (3) in
regressions.py, and the rank IC) is not driven by one corner of the data. Each
check re-estimates the *same* two statistics on a subset or a variant, so the
rows of the output are directly comparable:

* ``coef_text`` / ``t_text``: text rank coefficient controlling for SUE deciles
  and season FE, two-way clustered (regressions.ols_cluster);
* ``mean_ic`` / ``ic_t_nw``: within-season rank IC, Newey-West t.

Dimensions
----------
sector      FF12 industry (point-in-time SIC). Financials and utilities report
            differently (no "revenue" in a bank's release), a classic source of
            heterogeneity.
size        within-season market-cap terciles, and part 4's NYSE-20th-percentile
            flag ``f_size``. Text should matter more where fewer people read it.
period      2006-10 (includes the crisis), 2011-16, 2017-19 (validation); after
            the freeze also 2021-23 and 2024-25.
llm cutoff  before / after Llama 3.1's December 2023 cutoff (feeds part 9).
window      every outcome of part 4 (reaction, post-release +1 and +60 days,
            close-to-+60) and market-adjusted versions.
placebo     day -1 abnormal return. A significant coefficient here means the
            signal "predicts" a return that happened *before* the release: the
            clock is wrong or the signal leaks. It should be zero.
sample      in_primary (two clock sources agree) vs in_text_sample (any timing),
            without basis_suspect events (I/B/E/S actual on the wrong EPS basis),
            without timing_suspect events, alternative SUE (scaled by dispersion).

Subsets with fewer than ``min_n`` events or fewer than 8 seasons are reported
with NaN statistics and a note, never silently dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import settings as S
from metrics import rank_ic
from regressions import ols_cluster


def _one(d: pd.DataFrame, text: str, outcome: str, sue_cat: str = "sue_decile", min_n: int = 300) -> dict:
    y = outcome + "_w" if outcome + "_w" in d.columns else outcome
    dd = d.dropna(subset=[y, text, sue_cat])
    out = {"n": len(dd), "seasons": dd["season"].nunique()}
    if len(dd) < min_n or out["seasons"] < 8:
        out["note"] = "too few events or seasons"
        return out
    try:
        res, _ = ols_cluster(dd, y, [text], [sue_cat, "season"])
        out.update(coef_text=res.params[text], t_text=res.tvalues[text], p_text=res.pvalues[text])
    except Exception as e:                       # singular design in a tiny subset
        out["note"] = f"regression failed: {type(e).__name__}"
    ic = rank_ic(dd, text, outcome)
    out.update(mean_ic=ic.get("mean_ic"), ic_t_nw=ic.get("t_nw"))
    return out


def robustness_grid(df: pd.DataFrame, text: str, outcome: str = S.PRIMARY_OUTCOME,
                    full_events: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Every check for one text signal (already a within-season rank column of df).

    full_events: the panel built on ``in_text_sample`` (a superset of the primary
    sample), to run the sample-definition checks; optional.
    """
    rows = []

    def add(dim, group, d, oc=outcome, sue_cat="sue_decile"):
        rows.append({"dimension": dim, "group": str(group), "outcome": oc, **_one(d, text, oc, sue_cat)})

    add("baseline", "all", df)
    for g, d in df.groupby("ff12"):
        add("sector (FF12)", g, d)
    for g, d in df.groupby("size_tercile"):
        add("size tercile (within season)", {1.0: "small", 2.0: "mid", 3.0: "large"}.get(g, g), d)
    if "f_size" in df:
        for g, d in df.groupby("f_size"):
            add("NYSE 20th pct flag", "above" if g else "below", d)
    for g, d in df.groupby("period"):
        add("period", g, d)
    for g, d in df.groupby("after_llama_cutoff"):
        add("Llama 3.1 cutoff", "after 2023-12" if g else "before 2024", d)
    for oc in S.OUTCOMES + S.MKT_ADJ_OUTCOMES:
        if oc in df.columns and oc != outcome:
            add("event window", oc, df, oc)
    for oc in S.PLACEBO_OUTCOMES:
        add("placebo (should be ~0)", oc, df, oc)
    add("sample", "excluding basis_suspect", df[~df["basis_suspect"].astype(bool)])
    add("sample", "excluding timing_suspect", df[~df["timing_suspect"].astype(bool)])
    if "sue_disp" in df.columns:
        d = df.copy()
        from panel import quantile_bucket
        d["sue_disp_decile"] = quantile_bucket(d["sue_disp"], d["season"], 10)
        add("surprise definition", "SUE scaled by dispersion (3+ analysts)", d, sue_cat="sue_disp_decile")
    if full_events is not None and text in full_events.columns:
        add("sample", "in_text_sample (any clock status)", full_events)
        add("sample", "in_text_sample, wire/I/B/E/S disagree only",
            full_events[~full_events["two_source"].astype(bool)])
    t = pd.DataFrame(rows)
    t.insert(0, "text_signal", text)
    return t
