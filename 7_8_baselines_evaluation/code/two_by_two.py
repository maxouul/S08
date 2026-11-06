"""
Task 8: the 2x2 table, sign of the earnings surprise x sign of the text signal.

Why this table
--------------
It is the most transparent version of the whole project's question. If the
text only repeats the surprise, the two "disagreement" cells (beat with negative
text, miss with positive text) should behave like their surprise row. If the text
carries its own information, returns within each surprise row should move with
the text:

                       text negative        text positive
    SUE > 0 (beat)     "beat, bad tone"     "beat, good tone"
    SUE < 0 (miss)     "miss, bad tone"     "miss, good tone"

The key numbers are the two within-row differences (text effect holding the
surprise sign fixed) and their average.

Definitions
-----------
* Surprise sign: ``surprise`` (actual - median consensus, part 3) > 0 is a beat,
  < 0 a miss. Exact meets (surprise == 0, about 8-10% of events because EPS is
  reported to the cent) are excluded from the table and counted separately: they
  are neither, and folding them into either row would blur it.
* Text sign: a signal is "positive" when it is above its **within-season
  median** (``rule="season_median"``, the default). Raw zero is not a neutral
  point for most signals: LM tone is negative on average in every year (financial
  text uses many LM-negative words), and an LLM's 1-5 grade centres where it
  wants. The median split also gives two text halves of equal size, which makes
  the cells comparable across models. ``rule="zero"`` is available for signals
  whose zero means "neutral" by construction.

Inference
---------
Cell means, their differences and the difference-in-differences come from one
regression of the outcome on the four cell dummies (no intercept), with standard
errors clustered by season (events in the same quarter are correlated); linear
contrasts give the differences. The naive i.i.d. standard error is reported next
to it to show how much clustering matters. 95% confidence intervals use the
clustered errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import settings as S

CELLS = ["beat_textpos", "beat_textneg", "miss_textpos", "miss_textneg"]


def text_sign(signal: pd.Series, season: pd.Series, rule: str = "season_median") -> pd.Series:
    """+1 positive, -1 negative, NaN when missing (or exactly on the threshold)."""
    if rule == "season_median":
        thr = signal.groupby(season).transform("median")
    elif rule == "zero":
        thr = pd.Series(0.0, index=signal.index)
    else:
        raise ValueError(rule)
    s = np.sign(signal - thr)
    return s.where(s != 0)


def two_by_two(df: pd.DataFrame, signal: str, outcome: str = S.PRIMARY_OUTCOME,
               rule: str = "season_median", cluster: str = "season") -> dict:
    """
    Returns {"cells": long table, "matrix": formatted 2x2, "contrasts": tests,
    "excluded": counts of meets and missing}.
    """
    import statsmodels.api as sm

    d = df[[outcome, "surprise", signal, "season", "permno"]].copy()
    d["ts"] = text_sign(d[signal], d["season"], rule)
    excluded = {"meets (surprise == 0)": int((d["surprise"] == 0).sum()),
                "missing outcome / signal / surprise":
                    int(d[[outcome, "surprise", signal]].isna().any(axis=1).sum())}
    d = d.dropna(subset=[outcome, "surprise", "ts"])
    d = d[d["surprise"] != 0]
    beat = d["surprise"] > 0
    pos = d["ts"] > 0
    X = pd.DataFrame({"beat_textpos": (beat & pos), "beat_textneg": (beat & ~pos),
                      "miss_textpos": (~beat & pos), "miss_textneg": (~beat & ~pos)}).astype(float)
    if len(d) < 20 or (X.sum() == 0).any():
        return {"cells": pd.DataFrame(), "matrix": pd.DataFrame(), "contrasts": pd.DataFrame(),
                "excluded": excluded}

    y = d[outcome].astype(float)
    groups = pd.factorize(d[cluster])[0]
    res_cl = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    res_iid = sm.OLS(y, X).fit()

    cells = []
    for c in CELLS:
        m = X[c] == 1
        lo, hi = res_cl.conf_int().loc[c]
        cells.append({"cell": c, "n": int(m.sum()), "share": m.mean(), "mean": res_cl.params[c],
                      "se_naive": res_iid.bse[c], "se_cluster": res_cl.bse[c], "t_cluster": res_cl.tvalues[c],
                      "ci95_lo": lo, "ci95_hi": hi, "median": y[m].median()})
    cells = pd.DataFrame(cells)

    contrasts_def = {
        "text effect | beat  (beat_pos - beat_neg)": {"beat_textpos": 1, "beat_textneg": -1},
        "text effect | miss  (miss_pos - miss_neg)": {"miss_textpos": 1, "miss_textneg": -1},
        "average text effect": {"beat_textpos": .5, "beat_textneg": -.5, "miss_textpos": .5, "miss_textneg": -.5},
        "diff-in-diff (beat effect - miss effect)": {"beat_textpos": 1, "beat_textneg": -1,
                                                       "miss_textpos": -1, "miss_textneg": 1},
        "surprise effect | text pos (beat_pos - miss_pos)": {"beat_textpos": 1, "miss_textpos": -1},
        "surprise effect | text neg (beat_neg - miss_neg)": {"beat_textneg": 1, "miss_textneg": -1},
        "beat, bad text vs miss, good text": {"beat_textneg": 1, "miss_textpos": -1},
    }
    rows = []
    for name, w in contrasts_def.items():
        R = np.array([[w.get(c, 0.0) for c in CELLS]])
        t = res_cl.t_test(R)
        lo, hi = np.asarray(t.conf_int()).ravel()
        rows.append({"contrast": name, "estimate": float(np.ravel(t.effect)[0]),
                     "se_cluster": float(np.ravel(t.sd)[0]), "t_cluster": float(np.ravel(t.tvalue)[0]),
                     "p_cluster": float(np.ravel(t.pvalue)[0]), "ci95_lo": lo, "ci95_hi": hi})
    contrasts = pd.DataFrame(rows)

    def fmt(c):
        r = cells.set_index("cell").loc[c]
        return f"{100 * r['mean']:.2f}% [{100 * r['ci95_lo']:.2f}, {100 * r['ci95_hi']:.2f}]  n={int(r['n']):,}"

    matrix = pd.DataFrame({"text negative": [fmt("beat_textneg"), fmt("miss_textneg")],
                           "text positive": [fmt("beat_textpos"), fmt("miss_textpos")]},
                          index=["SUE > 0 (beat)", "SUE < 0 (miss)"])
    for t in (cells, contrasts):
        t.insert(0, "signal", signal)
        t.insert(1, "outcome", outcome)
        t.insert(2, "text_rule", rule)
    return {"cells": cells, "matrix": matrix, "contrasts": contrasts, "excluded": excluded}
