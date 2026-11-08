"""
Task 8: regressions of abnormal returns on text signals, controlling for the
earnings surprise and firm characteristics.

The question each regression answers
------------------------------------
"Holding the earnings surprise fixed, do events with more positive text have
higher abnormal returns?"  If yes, the text carries information the EPS number
does not; if the coefficient vanishes once SUE is controlled for, the model is
only re-reading the headline number.

Specification ladder (``incremental_info``)
-------------------------------------------
    (1) y = SUE deciles + season FE                         surprise only
    (2) y = text + season FE                                text only
    (3) y = SUE deciles + text + season FE                  the main test
    (4) (3) + controls + FF12 sector FE                     + firm characteristics
    (5) (4) + text x 1[miss]                                does text matter more after a miss?

* ``text`` is the within-season rank of the signal in [-0.5, 0.5]: its
  coefficient is the abnormal-return difference between the most and the least
  positive release of a season, holding the rest fixed. Same scale for every
  model, so coefficients are comparable across models.
* SUE enters as **decile dummies** (flexible, non-linear), never as a single
  linear term, for the reason given in baseline_surprise.py.
* **Season fixed effects** absorb everything common to a quarter (market moves
  not captured by the factor model, macro news). Identification is purely
  cross-sectional, within quarter.
* Controls: log market cap, log number of analysts, dispersion / price (+ an
  indicator when it is missing), prior one-month return, and sector FE. Size and
  coverage proxy for the information environment (the same text should matter
  less for a mega-cap followed by 30 analysts); prior return controls for
  momentum and for any pre-release leakage the text may pick up. Book-to-market is
  **not yet** included: it needs Compustat ceqq, which part 3-4 has not pulled
  (see docs/METHODOLOGY.md, open items).
* Outcomes are winsorised at 1/99% within season (``*_w``).

Standard errors
---------------
Two-way clustered by firm (permno) and season (Cameron, Gelbach and Miller
2011; Petersen 2009). Firm clustering handles a firm's repeated releases (its
writing style and its return persistence); season clustering handles the
cross-sectional correlation of events in the same quarter, which is the larger of
the two problems here. As a second, independent inference method we report
Fama-MacBeth by season with Newey-West errors (``fama_macbeth``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import settings as S
from metrics import newey_west_se

CONTROLS = ["log_mktcap", "log_analysts", "disp_price", "disp_missing", "ret_prior_1m_w"]


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def ols_cluster(df: pd.DataFrame, y: str, x: list[str], cat: list[str] = (),
                cluster: list[str] = ("permno", "season")):
    """
    OLS of y on x plus dummies for each categorical in ``cat`` (fixed effects),
    with one- or two-way clustered standard errors.

    statsmodels computes two-way clustered covariance when ``groups`` has two
    columns (Cameron-Gelbach-Miller: V_firm + V_season - V_intersection).
    """
    import statsmodels.api as sm

    cat, cluster = list(cat), list(cluster)
    need = [y] + list(x) + cat + cluster
    d = df.dropna(subset=need).copy()
    # A regressor that does not vary is collinear with the constant. Typical case:
    # disp_missing in the primary sample, where 3+ analysts guarantee a dispersion.
    # Drop it (and say so) rather than let the design matrix go singular.
    const_x = [c for c in x if d[c].nunique() <= 1]
    if const_x:
        print(f"  ols: dropped constant regressors {const_x}")
    x = [c for c in x if c not in const_x]
    X = d[list(x)].astype(float)
    for c in cat:
        X = X.join(pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float))
    X = sm.add_constant(X, has_constant="add")
    groups = np.column_stack([pd.factorize(d[c])[0] for c in cluster])
    if groups.shape[1] == 1:
        groups = groups[:, 0]
    # The season fixed-effect dummies have no meaningful season-clustered variance
    # (each dummy is one cluster), and the two-way formula can return a negative
    # number for them, hence NaN standard errors and a numpy warning. Only the
    # fixed effects are affected, never the coefficients we report; the warning is
    # silenced here for that reason alone.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in sqrt", RuntimeWarning)
        res = sm.OLS(d[y].astype(float), X).fit(cov_type="cluster", cov_kwds={"groups": groups})
        _ = res.bse                      # computed (and cached) inside the filter
    bad = [c for c in x if not np.isfinite(res.bse.get(c, np.nan))]
    if bad:
        print(f"  ols: WARNING non-finite standard errors for {bad}: two-way clustered covariance "
              f"not positive definite; report the Fama-MacBeth result instead")
    return res, d


def fama_macbeth(df: pd.DataFrame, y: str, x: list[str], cat: list[str] = (), by: str = "season",
                 min_n: int = S.MIN_SEASON_N, lags: int = S.NW_LAGS) -> pd.DataFrame:
    """
    One cross-sectional OLS per season, then the time-series mean of each
    coefficient with Newey-West errors. Robust to any within-season correlation by
    construction, and shows whether the effect is stable across seasons
    (``share_positive``).
    """
    coefs = []
    for s, g in df.dropna(subset=[y] + list(x) + list(cat)).groupby(by):
        if len(g) < max(min_n, len(x) + 15):
            continue
        X = g[list(x)].astype(float)
        for c in cat:
            X = X.join(pd.get_dummies(g[c].astype(str), prefix=c, drop_first=True, dtype=float))
        X.insert(0, "const", 1.0)
        X = X.loc[:, X.std() > 0].assign(const=1.0) if len(X.columns) > 1 else X
        beta, *_ = np.linalg.lstsq(X.to_numpy(), g[y].to_numpy(float), rcond=None)
        coefs.append(pd.Series(beta, index=X.columns).reindex(["const"] + list(x)).rename(s))
    if len(coefs) < 3:
        return pd.DataFrame()
    C = pd.DataFrame(coefs)
    rows = []
    for v in ["const"] + list(x):
        c = C[v].dropna()
        se = newey_west_se(c.to_numpy(), lags)
        rows.append({"variable": v, "coef": c.mean(), "se_nw": se, "t_nw": c.mean() / se if se else np.nan,
                     "share_positive": (c > 0).mean(), "seasons": len(c)})
    return pd.DataFrame(rows)


def wald_p(res, names: list[str]) -> float:
    """p-value of the joint test that the named coefficients are all zero."""
    names = [n for n in names if n in res.params.index]
    if not names:
        return np.nan
    R = np.zeros((len(names), len(res.params)))
    for i, n in enumerate(names):
        R[i, res.params.index.get_loc(n)] = 1.0
    return float(res.wald_test(R, scalar=True).pvalue)


# ---------------------------------------------------------------------------
# The specification ladder
# ---------------------------------------------------------------------------
def incremental_info(df: pd.DataFrame, text: str, outcome: str = S.PRIMARY_OUTCOME,
                     controls: list[str] = CONTROLS, sector: str = "ff12") -> dict:
    """
    Fit specifications (1)-(5) on the same events and return the fitted models
    plus a compact summary row for the text coefficient.

    ``df`` must contain ``text`` (a within-season rank), the winsorised outcome,
    ``sue_decile``, ``season``, ``permno`` and the controls.
    """
    y = outcome + "_w" if outcome + "_w" in df.columns else outcome
    ctrl = [c for c in controls if c in df.columns]
    has_sector = sector in df.columns and df[sector].nunique() > 1
    # Every specification on the same events: the intersection of what each needs.
    need = [y, text, "sue_decile", "season", "permno"] + ctrl + ([sector] if has_sector else [])
    d = df.dropna(subset=need).copy()
    d["miss"] = (d["surprise"] < 0).astype(float)
    d["text_x_miss"] = d[text] * d["miss"]

    specs = {
        "(1) SUE": ([], ["sue_decile", "season"]),
        "(2) text": ([text], ["season"]),
        "(3) SUE + text": ([text], ["sue_decile", "season"]),
        "(4) + controls": ([text] + ctrl, ["sue_decile", "season"] + ([sector] if has_sector else [])),
        "(5) + text x miss": ([text, "text_x_miss", "miss"] + ctrl,
                              ["sue_decile", "season"] + ([sector] if has_sector else [])),
    }
    fits = {}
    for name, (x, cat) in specs.items():
        fits[name], _ = ols_cluster(d, y, x, cat)
    r2_base = fits["(1) SUE"].rsquared
    summary = []
    for name, res in fits.items():
        row = {"spec": name, "outcome": outcome, "text_signal": text, "n": int(res.nobs),
               "r2": res.rsquared, "delta_r2_vs_sue": res.rsquared - r2_base}
        if text in res.params:
            row.update(coef_text=res.params[text], se_text=res.bse[text], t_text=res.tvalues[text],
                       p_text=res.pvalues[text])
        if "text_x_miss" in res.params:
            row.update(coef_text_x_miss=res.params["text_x_miss"], t_text_x_miss=res.tvalues["text_x_miss"],
                       p_joint_text=wald_p(res, [text, "text_x_miss"]))
        summary.append(row)
    fm = fama_macbeth(d, y, [text] + ctrl, cat=["sue_decile"])
    if len(fm):
        r = fm.set_index("variable").loc[text]
        summary.append({"spec": "(FM) SUE + text + controls, Fama-MacBeth", "outcome": outcome,
                        "text_signal": text, "n": len(d), "coef_text": r["coef"], "se_text": r["se_nw"],
                        "t_text": r["t_nw"], "fm_share_positive": r["share_positive"],
                        "fm_seasons": r["seasons"]})
    return {"fits": fits, "summary": pd.DataFrame(summary), "fama_macbeth": fm, "n": len(d)}


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def regression_table(fits: dict, keep: list[str], stars: bool = True, digits: int = 4) -> pd.DataFrame:
    """
    Journal-style table: coefficients with t-statistics in parentheses below,
    one column per specification, and N, R^2 and the fixed effects at the bottom.
    """
    def star(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    cols = {}
    for name, res in fits.items():
        cells = {}
        for v in keep:
            if v in res.params:
                cells[v] = f"{res.params[v]:.{digits}f}{star(res.pvalues[v]) if stars else ''}"
                cells[v + " (t)"] = f"({res.tvalues[v]:.2f})"
            else:
                cells[v], cells[v + " (t)"] = "", ""
        names = " ".join(res.params.index)
        cells["SUE deciles"] = "yes" if "sue_decile_" in names else "no"
        cells["season FE"] = "yes" if "season_" in names else "no"
        cells["sector FE"] = "yes" if "ff12_" in names else "no"
        cells["N"] = f"{int(res.nobs):,}"
        cells["R2"] = f"{res.rsquared:.4f}"
        cols[name] = cells
    order = [r for v in keep for r in (v, v + " (t)")] + ["SUE deciles", "season FE", "sector FE", "N", "R2"]
    t = pd.DataFrame(cols).reindex(order)
    t.index = [i if not i.endswith(" (t)") else "" for i in t.index]
    return t
