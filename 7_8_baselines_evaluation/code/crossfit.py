"""
Chronological cross-fitting: how every *trained* signal gets out-of-sample
predictions.

The problem
-----------
A TF-IDF regression (or the "combined" model of compare.py) fitted on the
development years and evaluated on the same years is in-sample: its IC would be
meaningless. Random K-fold is no better, because it lets a 2015 release teach the
model about 2009 (look-ahead) and because neighbouring releases of the same firm
share boiler-plate, so random folds leak firm identity.

The rule
--------
Every prediction for an event is made by a model fitted **only on events whose
reaction date is strictly earlier, minus an embargo**:

* ``eval_split="dev"``: expanding window by calendar year. For each dev year y
  after the first ``min_train_years``, fit on dev events before 1 January of y
  (minus the embargo), predict year y. This gives pseudo-out-of-sample
  predictions for most of the development period, used to *tune*
  hyper-parameters without touching validation.
* ``eval_split="validation"``: fit on all dev events (minus the embargo before
  the first validation day), predict 2017-2019. This is the honest comparison set
  until the freeze.
* ``eval_split="holdout"``: fit on dev + validation, predict 2021-2025. Only with
  the holdout unlocked.

The embargo (default 100 calendar days) removes training events whose 60-day
return window would overlap the evaluation period: without it, the label of a
late-December training event contains January returns of the test year.

Hyper-parameters are chosen on the "dev" cross-fitted predictions only. The
validation years are used once per model to compare models, and the holdout once
at the very end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

EMBARGO_DAYS = 100


@dataclass
class Fold:
    label: str              # e.g. "dev_2012", "validation", "holdout"
    train_idx: pd.Index
    test_idx: pd.Index
    train_end: pd.Timestamp


def chronological_folds(panel: pd.DataFrame, eval_split: str, min_train_years: int = 3,
                        embargo_days: int = EMBARGO_DAYS) -> list[Fold]:
    """Folds for one evaluation split (see the module docstring)."""
    d = pd.to_datetime(panel["reaction_date"])
    split = panel["split"]
    emb = pd.Timedelta(days=embargo_days)
    folds: list[Fold] = []

    if eval_split == "dev":
        dev_years = sorted(panel.loc[split == "dev", "year"].unique())
        for y in dev_years[min_train_years:]:
            start = pd.Timestamp(f"{y}-01-01")
            tr = panel.index[(split == "dev") & (d < start - emb)]
            te = panel.index[(split == "dev") & (panel["year"] == y)]
            if len(tr) and len(te):
                folds.append(Fold(f"dev_{y}", tr, te, start - emb))
    elif eval_split == "validation":
        te = panel.index[split == "validation"]
        if len(te):
            start = d[te].min()
            tr = panel.index[(split == "dev") & (d < start - emb)]
            folds.append(Fold("validation", tr, te, start - emb))
    elif eval_split == "holdout":
        te = panel.index[split == "holdout"]
        if len(te):
            start = d[te].min()
            tr = panel.index[split.isin(["dev", "validation"]) & (d < start - emb)]
            folds.append(Fold("holdout", tr, te, start - emb))
    else:
        raise ValueError(f"unknown eval_split {eval_split!r}")
    return folds


def crossfit_predict(panel: pd.DataFrame, fit: Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]],
                     eval_splits=("dev", "validation"), min_train_rows: int = 200,
                     **fold_kw) -> pd.DataFrame:
    """
    Out-of-sample predictions for every event of ``eval_splits``.

    ``fit(train_df)`` must return a function ``predict(test_df) -> array``.
    Returns a frame indexed like ``panel`` with columns ``pred`` and ``fold``;
    events with no fold (the first ``min_train_years`` dev years) stay NaN.
    """
    out = pd.DataFrame({"pred": np.nan, "fold": pd.Series(pd.NA, index=panel.index, dtype=object)},
                       index=panel.index)
    for es in eval_splits:
        for f in chronological_folds(panel, es, **fold_kw):
            if len(f.train_idx) < min_train_rows:
                print(f"  {f.label}: only {len(f.train_idx)} training rows, skipped")
                continue
            predict = fit(panel.loc[f.train_idx])
            out.loc[f.test_idx, "pred"] = np.asarray(predict(panel.loc[f.test_idx]), dtype=float)
            out.loc[f.test_idx, "fold"] = f.label
    return out


def describe_folds(panel: pd.DataFrame, eval_splits=("dev", "validation"), **fold_kw) -> pd.DataFrame:
    """Table of folds: training window and sizes (for the notebooks)."""
    rows = []
    for es in eval_splits:
        for f in chronological_folds(panel, es, **fold_kw):
            rows.append({"fold": f.label, "train_events": len(f.train_idx),
                         "train_last_date": pd.to_datetime(panel.loc[f.train_idx, "reaction_date"]).max().date()
                         if len(f.train_idx) else None,
                         "embargo_until": f.train_end.date(), "test_events": len(f.test_idx),
                         "test_first_date": pd.to_datetime(panel.loc[f.test_idx, "reaction_date"]).min().date()})
    return pd.DataFrame(rows)
