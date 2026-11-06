"""
Task 7's comparison: surprise-only vs text-only vs combined, for every text
signal, with the same events, outcome, metrics and chronological split.

How the three approaches are defined
------------------------------------
For a text signal T (LM tone, a TF-IDF prediction, an LLM score, ...):

    surprise-only   y = sum_d b_d * 1[SUE decile = d]                 (10 dummies)
    text-only       y = a + c * rank(T)
    combined        y = sum_d b_d * 1[SUE decile = d] + c * rank(T)

``rank(T)`` is T's within-season rank in [-0.5, 0.5] (panel.cs_rank). Putting
every signal on the same rank scale is what makes the comparison fair across
models whose raw outputs live on different scales (an LLM's 1-5 grade, a tone in
[-1, 1], a predicted return). The SUE part is non-linear (deciles) so the text
gets no credit for curvature the surprise already explains.

The three models are *linear stacks fitted by OLS on past data only*
(crossfit.py), evaluated on the evaluation split. Simple stacks on purpose: the
question is whether the text carries information, not which learner squeezes it
best, and a flexible stacker fitted on few seasons would overfit.

Fair comparison rules
---------------------
* **Common support.** By default all approaches and all text signals are
  evaluated on the events where *every* compared signal exists. Otherwise a model
  that failed on hard documents (long, table-heavy) would be graded on an easier
  sample.
* **Same metrics** (metrics.evaluate_signal) and Clark-West for nested pairs
  (combined vs surprise-only).
* Signals that need training (TF-IDF) must already be out-of-sample when they
  enter here (they are: baseline_tfidf cross-fits them).

A note on within-season ranks
-----------------------------
Ranking an event against the other events of its calendar quarter uses releases
published later in the same quarter. That is standard for cross-sectional tests
(Fama-MacBeth) and harmless for the statistical question, but it is not a
tradable strategy. A trading-rule version would rank against the trailing
distribution; see docs/METHODOLOGY.md, section 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import settings as S
from baseline_surprise import decile_design
from crossfit import crossfit_predict
from metrics import clark_west, evaluate_signal, oos_r2
from panel import cs_rank


def _fit_stack(y: str, use_sue: bool, text_col: str | None):
    def design(df):
        parts = []
        if use_sue:
            parts.append(decile_design(df["sue_decile"]))
        else:
            parts.append(np.ones((len(df), 1)))
        if text_col is not None:
            parts.append(df[[text_col]].to_numpy(float))
        return np.column_stack(parts)

    def fit(train):
        need = [y] + (["sue_decile"] if use_sue else []) + ([text_col] if text_col else [])
        t = train.dropna(subset=need)
        beta, *_ = np.linalg.lstsq(design(t), t[y].to_numpy(float), rcond=None)

        def predict(test):
            ok = test["sue_decile"].notna() if use_sue else pd.Series(True, index=test.index)
            if text_col is not None:
                ok &= test[text_col].notna()
            X = np.nan_to_num(design(test))
            return np.where(ok, X @ beta, np.nan)
        return predict
    return fit


def compare_models(panel: pd.DataFrame, signals: pd.DataFrame, text_signals: list[str],
                   outcome: str = S.PRIMARY_OUTCOME, eval_split: str = S.TUNE_SPLIT,
                   common_support: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (table, predictions).

    table: one row per (text signal, approach) with every metric on the
           evaluation split, plus a Clark-West test of combined vs surprise-only.
    predictions: the out-of-sample predictions, indexed like panel.
    """
    if eval_split == "holdout" and (panel["split"] == "holdout").sum() == 0:
        raise ValueError("no holdout rows in the panel: build it with include_holdout=True "
                         "after the freeze")
    target = outcome + "_w" if outcome + "_w" in panel.columns else outcome
    df = panel.copy()
    for t in text_signals:
        df[t + "_rk"] = cs_rank(signals[t], df["season"])

    fit_splits = ("dev", eval_split) if eval_split != "dev" else ("dev",)
    preds = pd.DataFrame(index=df.index)
    preds["surprise_only"] = crossfit_predict(df, _fit_stack(target, True, None), fit_splits)["pred"]
    for t in text_signals:
        preds[f"text_only__{t}"] = crossfit_predict(df, _fit_stack(target, False, t + "_rk"), fit_splits)["pred"]
        preds[f"combined__{t}"] = crossfit_predict(df, _fit_stack(target, True, t + "_rk"), fit_splits)["pred"]

    ev = df["split"].eq(eval_split) & df[outcome].notna() & df["sue_decile"].notna()
    if common_support:
        for t in text_signals:
            ev &= df[t + "_rk"].notna()
        ev &= preds.notna().all(axis=1)
    e = df.loc[ev].join(preds.loc[ev])
    print(f"compare [{outcome}, {eval_split}]: {len(e):,} events on common support, "
          f"{e['season'].nunique()} seasons")
    if len(e) == 0:
        return pd.DataFrame(), preds

    rows = []
    base = evaluate_signal(e, "surprise_only", outcome, prediction_in_return_units=True)
    rows.append({"text_signal": "(none)", "approach": "surprise_only", **base})
    rows.append({"text_signal": "(none)", "approach": "sue_rank (raw signal)",
                 **evaluate_signal(e, "sue_rank", outcome)})
    for t in text_signals:
        rows.append({"text_signal": t, "approach": "text (raw signal)",
                     **evaluate_signal(e, t + "_rk", outcome)})
        for appr in ("text_only", "combined"):
            col = f"{appr}__{t}"
            r = {"text_signal": t, "approach": appr,
                 **evaluate_signal(e, col, outcome, prediction_in_return_units=True)}
            if appr == "combined":
                r.update(clark_west(e[outcome], e["surprise_only"], e[col], e["season"]))
                # OOS R^2 of combined *relative to* surprise-only: the share of the
                # surprise model's squared error the text removes.
                r["oos_r2_vs_surprise"] = oos_r2(e[outcome], e[col], e["surprise_only"])
            rows.append(r)
    tab = pd.DataFrame(rows)
    tab.insert(0, "eval_split", eval_split)
    tab.insert(1, "n_events", len(e))
    return tab, preds
