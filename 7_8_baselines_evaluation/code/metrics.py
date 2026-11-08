"""
Evaluation metrics shared by every baseline and every model (task 7, "consistent
metrics"), and by the statistical tests of task 8.

Why these metrics
-----------------
We evaluate a *signal* s (one number per event, higher = better news) against an
*outcome* y (an abnormal return). Four questions, four metrics:

1. **Does the ranking of s line up with the ranking of y?** Rank information
   coefficient (IC): Spearman correlation between s and y *within* each
   announcement season, averaged over seasons. Within-season, because a model
   that only learned "2008 was bad" should get no credit. Ranks, because returns
   have fat tails and an IC is not driven by three events. Inference: t-stat of
   the mean IC across seasons, with Newey-West errors (part 4's function, reused).

2. **Is the spread economically large?** Long-short return: mean y of the top
   decile of s minus the bottom decile, within season, averaged. Reads as "basis
   points per event", which a finance audience understands.

3. **Does s forecast y in levels, out of sample?** Out-of-sample R^2 against a
   zero forecast (abnormal returns have mean zero by construction, so zero is the
   natural naive forecast, as in Campbell and Thompson 2008 with a 0 benchmark).
   Only for predictions that are in return units (fitted regressions).

4. **Does a bigger model beat a smaller nested one?** Clark-West (2007) test of
   equal predictive accuracy for nested models: MSPE comparisons are biased
   against the larger model under the null, CW corrects the bias. We compute the
   CW statistic per event, average it within season and Newey-West the season
   series, so cross-sectional correlation within a season does not inflate t.

Classification metrics (hit rate, AUC on the sign of y) are added for the
classifiers, and because part 6's LLMs may output a direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import settings as S

try:                                           # reuse part 4's implementation
    _R34 = S.returns_3_4()
    newey_west_se = _R34.newey_west_se
    _rank_ic_by_season = _R34.rank_ic_by_season
except Exception:                              # pragma: no cover - only if part 4 is absent
    _R34 = None

    def newey_west_se(x: np.ndarray, lags: int = 2) -> float:
        """Standard error of the mean of x, robust to autocorrelation up to `lags`
        (Bartlett kernel). Same formula as part 4's returns.newey_west_se."""
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        n = len(x)
        if n < 3:
            return np.nan
        e = x - x.mean()
        s = e @ e / n
        for lag in range(1, min(lags, n - 1) + 1):
            s += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
        return float(np.sqrt(s / n))

    _rank_ic_by_season = None


# ---------------------------------------------------------------------------
# Season-level aggregation: the backbone of all inference here
# ---------------------------------------------------------------------------
def season_mean_test(values: pd.Series, season: pd.Series, min_n: int = 1,
                     lags: int = S.NW_LAGS) -> dict:
    """
    Average a per-event quantity within each season, then test the mean of the
    season averages with Newey-West errors.

    This treats the season (not the event) as the independent unit. Events in the
    same quarter share macro news, so treating 10,000 events as 10,000
    independent draws would overstate precision by an order of magnitude.
    """
    d = pd.DataFrame({"v": values, "s": season}).dropna()
    g = d.groupby("s")["v"]
    m = g.mean()[g.size() >= min_n].sort_index()
    if len(m) < 3:
        return {"mean": float(m.mean()) if len(m) else np.nan, "se_nw": np.nan, "t_nw": np.nan,
                "seasons": len(m), "n": int(len(d))}
    se = newey_west_se(m.to_numpy(), lags)
    return {"mean": float(m.mean()), "se_nw": se, "t_nw": float(m.mean() / se) if se else np.nan,
            "seasons": int(len(m)), "n": int(len(d))}


def rank_ic(df: pd.DataFrame, signal: str, outcome: str, min_n: int = S.MIN_SEASON_N) -> dict:
    """Mean within-season Spearman IC with Newey-West t (part 4's rank_ic_by_season)."""
    if _rank_ic_by_season is not None:
        return _rank_ic_by_season(df, signal, outcome, min_n=min_n, nw_lags=S.NW_LAGS)
    d = df[[signal, outcome, "reaction_date"]].dropna()
    d = d.assign(season=pd.to_datetime(d["reaction_date"]).dt.to_period("Q"))
    ic = d.groupby("season").apply(lambda g: g[signal].rank().corr(g[outcome].rank())
                                   if len(g) >= min_n else np.nan).dropna()
    if ic.empty:
        return {"seasons": 0, "mean_ic": np.nan, "t_stat": np.nan, "t_nw": np.nan,
                "share_positive": np.nan}
    se = ic.std(ddof=1) / np.sqrt(len(ic))
    nw = newey_west_se(ic.to_numpy(), S.NW_LAGS)
    return {"seasons": len(ic), "mean_ic": round(float(ic.mean()), 4),
            "t_stat": round(float(ic.mean() / se), 2), "t_nw": round(float(ic.mean() / nw), 2),
            "share_positive": round(float((ic > 0).mean()), 2)}


def long_short(df: pd.DataFrame, signal: str, outcome: str, q: int = 10,
               min_n: int = S.MIN_SEASON_N) -> dict:
    """Top-minus-bottom q-tile mean outcome, within season, NW across seasons."""
    d = df[[signal, outcome, "season"]].dropna()
    rows = []
    for s, g in d.groupby("season"):
        if len(g) < max(min_n, 2 * q):
            continue
        b = pd.qcut(g[signal].rank(method="first"), q, labels=False)
        rows.append((s, g.loc[b == q - 1, outcome].mean() - g.loc[b == 0, outcome].mean()))
    if len(rows) < 3:
        return {"ls_mean": np.nan, "ls_t_nw": np.nan, "ls_seasons": len(rows)}
    v = pd.Series([r[1] for r in rows])
    se = newey_west_se(v.to_numpy(), S.NW_LAGS)
    return {"ls_mean": float(v.mean()), "ls_t_nw": float(v.mean() / se), "ls_seasons": len(rows)}


def oos_r2(y: np.ndarray, yhat: np.ndarray, benchmark: np.ndarray | float = 0.0) -> float:
    """1 - SSE(model) / SSE(benchmark). Negative = worse than the benchmark."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    b = np.broadcast_to(np.asarray(benchmark, float), y.shape)
    ok = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(1 - ((y[ok] - yhat[ok]) ** 2).sum() / ((y[ok] - b[ok]) ** 2).sum())


def clark_west(y: pd.Series, f_small: pd.Series, f_big: pd.Series, season: pd.Series) -> dict:
    """
    Clark-West (2007) test that the larger (nested) model has no extra
    predictive power. H1 (one-sided): the larger model forecasts better.

        f_t = (y - f_small)^2 - [ (y - f_big)^2 - (f_small - f_big)^2 ]

    Mean of f > 0 favours the big model. Here f is averaged within season and the
    season series is tested with Newey-West errors.
    """
    from scipy import stats

    f = (y - f_small) ** 2 - ((y - f_big) ** 2 - (f_small - f_big) ** 2)
    r = season_mean_test(f, season)
    # Small-sample p-value: Student t with (seasons - 1) degrees of freedom, not
    # the normal. With the 12 validation seasons the normal would call t = 2.0
    # "p = 2.3%" when the t(11) answer is 3.5%. Newey-West with few seasons is
    # itself optimistic, so treat |t| < 3 on validation as weak evidence.
    df_t = max(r["seasons"] - 1, 1)
    p = float(1 - stats.t.cdf(r["t_nw"], df_t)) if np.isfinite(r["t_nw"]) else np.nan
    return {"cw_mean": r["mean"], "cw_t_nw": r["t_nw"], "cw_p": p, "cw_seasons": r["seasons"]}


def hit_rate(y: pd.Series, s: pd.Series) -> float:
    """Share of events where sign(s - median s) matches sign(y). Ignores zeros."""
    d = pd.DataFrame({"y": y, "s": s}).dropna()
    d = d[(d["y"] != 0)]
    if d.empty:
        return np.nan
    centred = d["s"] - d["s"].median()
    d = d[centred != 0]
    return float((np.sign(centred.loc[d.index]) == np.sign(d["y"])).mean())


def auc(y_up: pd.Series, s: pd.Series) -> float:
    """Area under the ROC curve of s for the event "y > 0"."""
    from sklearn.metrics import roc_auc_score

    d = pd.DataFrame({"y": y_up, "s": s}).dropna()
    if d["y"].nunique() < 2:
        return np.nan
    return float(roc_auc_score(d["y"].astype(int), d["s"]))


def evaluate_signal(df: pd.DataFrame, signal: str, outcome: str,
                    prediction_in_return_units: bool = False) -> dict:
    """Every metric of one signal against one outcome, on the rows of df."""
    ic = rank_ic(df, signal, outcome)
    ls = long_short(df, signal, outcome)
    out = {"signal": signal, "outcome": outcome,
           "n": int(df[[signal, outcome]].dropna().shape[0]),
           "seasons": ic.get("seasons"), "mean_ic": ic.get("mean_ic"), "ic_t_nw": ic.get("t_nw"),
           "ic_share_positive": ic.get("share_positive"), **ls,
           "hit_rate": hit_rate(df[outcome], df[signal]),
           "auc": auc((df[outcome] > 0).astype(float).where(df[outcome].notna()), df[signal])}
    out["oos_r2_vs0"] = (oos_r2(df[outcome], df[signal]) if prediction_in_return_units else np.nan)
    return out
