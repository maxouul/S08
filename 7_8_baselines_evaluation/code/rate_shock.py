"""
Task 8: does the information in the text change around the 2022 rate shock?

Why the rate shock is interesting here
--------------------------------------
From March 2022 the Federal Reserve raised rates by 525 bp in 16 months, the
fastest tightening in four decades. Two things changed for earnings releases:
(i) the discount-rate news dominated price moves, so the cash-flow news in a
release may explain less of the reaction; (ii) the *content* of releases changed
(pricing power, input costs, interest expense, guidance cuts), vocabulary that a
model trained on 2006-2019 releases has rarely seen. A text signal whose power
falls in 2022-23 and recovers afterwards points to (i); one that stays weak
points to (ii) or to a model that relied on patterns of the past (and, for an
LLM, possibly on memorised outcomes of pre-cutoff releases, task 9).

Design
------
Regimes are defined by FOMC dates (settings.RATE_SHOCK): pre_shock 2021 to
15 March 2022, hiking 16 March 2022 to 26 July 2023 (the last hike),
post_hiking afterwards. For each regime we report the rank IC and the text
coefficient of specification (3), then one pooled regression with text x regime
interactions (and SUE deciles x regime, so the surprise response may also
change) to test whether the text effect differs between regimes.

All three regimes are in the holdout (2021-2025). Nothing here runs until the
holdout is unlocked: ``rate_shock_analysis`` returns an explanatory empty
result otherwise. This is by design, not a missing feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import settings as S
from metrics import rank_ic
from regressions import ols_cluster, wald_p


def label_regime(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    out = pd.Series(pd.NA, index=d.index, dtype=object)
    for name, (lo, hi) in S.RATE_SHOCK.items():
        out[(d >= lo) & (d <= hi)] = name
    return out


def rate_shock_analysis(df: pd.DataFrame, text: str, outcome: str = S.PRIMARY_OUTCOME) -> dict:
    d = df.copy()
    d["regime"] = label_regime(d["reaction_date"])
    d = d[d["regime"].notna()]
    if d.empty:
        msg = ("no 2021-2025 events in the panel: the rate-shock analysis runs only after the "
               "freeze, with UNLOCK_HOLDOUT=1 and build_panel(include_holdout=True)")
        print("rate shock:", msg)
        return {"by_regime": pd.DataFrame({"note": [msg]}), "interaction": pd.DataFrame(), "fit": None}

    y = outcome + "_w" if outcome + "_w" in d.columns else outcome
    rows = []
    for g in S.RATE_SHOCK:
        dd = d[d["regime"] == g].dropna(subset=[y, text, "sue_decile"])
        r = {"regime": g, "start": S.RATE_SHOCK[g][0], "end": S.RATE_SHOCK[g][1], "n": len(dd),
             "seasons": dd["season"].nunique()}
        if len(dd) >= 300:
            res, _ = ols_cluster(dd, y, [text], ["sue_decile", "season"])
            ic = rank_ic(dd, text, outcome)
            r.update(coef_text=res.params[text], t_text=res.tvalues[text], mean_ic=ic["mean_ic"],
                     ic_t_nw=ic["t_nw"], mean_abs_car=dd[outcome].abs().mean())
        rows.append(r)
    by_regime = pd.DataFrame(rows)

    # Pooled interaction model: base regime = pre_shock.
    dd = d.dropna(subset=[y, text, "sue_decile"]).copy()
    xs = [text]
    for g in list(S.RATE_SHOCK)[1:]:
        dd[f"{text}_x_{g}"] = dd[text] * (dd["regime"] == g)
        xs.append(f"{text}_x_{g}")
    # The SUE response may differ by regime: deciles 2-10 x regime dummies. Decile 1
    # is the omitted level *within* each regime, because the regime level itself is
    # already absorbed by the season fixed effects (seasons nest in regimes up to
    # the boundary quarters); full decile x regime dummies would be collinear with them.
    sue_x = []
    for g in S.RATE_SHOCK:
        for k in range(2, 11):
            col = f"sue_d{k}_x_{g}"
            dd[col] = ((dd["sue_decile"] == k) & (dd["regime"] == g)).astype(float)
            sue_x.append(col)
    res, _ = ols_cluster(dd, y, xs + sue_x, ["season"])
    inter = pd.DataFrame({"coef": res.params[xs], "se": res.bse[xs], "t": res.tvalues[xs],
                          "p": res.pvalues[xs]}).reset_index(names="term")
    inter.loc[len(inter)] = {"term": "joint: regime interactions = 0", "coef": np.nan, "se": np.nan,
                             "t": np.nan, "p": wald_p(res, xs[1:])}
    inter.insert(0, "text_signal", text)
    by_regime.insert(0, "text_signal", text)
    return {"by_regime": by_regime, "interaction": inter, "fit": res}
