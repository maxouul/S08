"""
Baseline 1 (task 7): earnings surprise only.

This is the benchmark every text signal must beat. If a language model "predicts"
returns but adds nothing once we know whether the firm beat consensus, the model
has learned to read the headline EPS number, which I/B/E/S already gives us.

Three versions, from the simplest to the most flexible:

* ``sue_rank``: the within-season rank of SUE. No fitting at all; a pure signal
  for IC and long-short statistics.
* ``surp_linear``: OLS of the outcome on sue_rank, cross-fitted (crossfit.py).
  Prediction in return units, for OOS R^2.
* ``surp_deciles``: OLS on SUE-decile dummies, cross-fitted. Allows the familiar
  S-shaped response (Kinney, Burgstahler and Martin 2002; part 4's
  ``sue_deciles_dev.csv`` shows it: the bottom decile moves -4.9%, deciles 4-7
  barely move). This is the benchmark used in the "combined" comparisons, so a
  text model gets no credit for merely bending a straight line.

SUE definition, sign and timing are part 3's (sue_price = (actual - median
consensus) / price at t-2, consensus restricted to estimates issued before the
release). Nothing is recomputed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import settings as S
from crossfit import crossfit_predict


def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def decile_design(dec: pd.Series) -> np.ndarray:
    """Intercept-free design of 10 SUE-decile dummies (missing decile -> all zeros)."""
    D = np.zeros((len(dec), 10))
    v = dec.to_numpy()
    ok = np.isfinite(v)
    D[np.where(ok)[0], v[ok].astype(int) - 1] = 1.0
    return D


def fit_deciles(outcome: str):
    def fit(train: pd.DataFrame):
        t = train.dropna(subset=[outcome, "sue_decile"])
        beta = _ols_fit(decile_design(t["sue_decile"]), t[outcome].to_numpy())

        def predict(test: pd.DataFrame):
            return np.where(test["sue_decile"].notna(), decile_design(test["sue_decile"]) @ beta, np.nan)
        return predict
    return fit


def fit_linear(outcome: str):
    def fit(train: pd.DataFrame):
        t = train.dropna(subset=[outcome, "sue_rank"])
        X = np.column_stack([np.ones(len(t)), t["sue_rank"]])
        beta = _ols_fit(X, t[outcome].to_numpy())

        def predict(test: pd.DataFrame):
            return beta[0] + beta[1] * test["sue_rank"].to_numpy()
        return predict
    return fit


def surprise_signals(panel: pd.DataFrame, outcome: str = S.PRIMARY_OUTCOME,
                     eval_splits=("dev", "validation")) -> pd.DataFrame:
    """
    All surprise-only signals for one outcome. The fitted versions are trained on
    the winsorised outcome (``<outcome>_w``) so a handful of +300% days do not
    set the coefficients; they are evaluated on the raw outcome.
    """
    target = outcome + "_w" if outcome + "_w" in panel.columns else outcome
    out = pd.DataFrame(index=panel.index)
    out["sue_rank"] = panel["sue_rank"]
    out[f"surp_linear__{outcome}"] = crossfit_predict(panel, fit_linear(target), eval_splits)["pred"]
    out[f"surp_deciles__{outcome}"] = crossfit_predict(panel, fit_deciles(target), eval_splits)["pred"]
    return out
