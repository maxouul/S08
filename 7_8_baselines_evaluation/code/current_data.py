"""
What parts 7-8 can say with the inputs that exist today (6 Nov 2026).

The main pipeline (pipeline.py) evaluates a text signal only when it covers at
least 5% of the evaluation split, and today no text signal does: part 5 has
cleaned 2006 only and part 6 has not scored anything. This module produces the
results that do not need text coverage, plus a clearly-labelled pilot on the few
texts we have. It reads the panel and signals written by ``pipeline.py --step all``
and writes only summary tables (results/tables/cd_*).

    python code/current_data.py            # after: python code/pipeline.py --step all

Tables
------
cd_surprise_regressions__{split}   the benchmark regression the text must beat:
                                   reaction on SUE deciles + season FE, then
                                   + controls (incl. book-to-market) + FF12 FE,
                                   two-way clustered; Fama-MacBeth alongside
cd_surprise_sign__{split}          the one-axis version of the 2x2: mean CAR for
                                   beats / meets / misses, CIs clustered by season
cd_surprise_robustness__{split}    SUE rank IC by sector, size, period, window, sample
cd_placebo_diagnostic              SUE rank IC with the day -1 return, by session,
                                   period, split and sample flags (dev + validation)
cd_lm_pilot                        Loughran-McDonald tone on the ~115 texts we have:
                                   pooled correlations and a SUE + tone regression.
                                   Exploratory only: n is far below the 5% rule.
cd_lm_pilot_2x2                    the 2x2 cells on the same texts (illustration)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import settings as S
import outputs as OUT
from metrics import rank_ic, long_short
from regressions import CONTROLS, ols_cluster, fama_macbeth, regression_table
from two_by_two import two_by_two

SPLITS = ("dev", "validation")


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_parquet(S.PANEL)
    sig = pd.read_parquet(S.SIGNALS)
    if "split" in p and set(p["split"].dropna()) - {"dev", "validation"}:
        raise RuntimeError("panel contains holdout rows: current_data.py runs on dev/validation only")
    ctx = {c: ",".join(sorted(map(str, p[c].dropna().unique())))
           for c in ("surprise_version", "returns_version", "sample_version", "clean_version") if c in p}
    OUT.set_context(sample=S.SAMPLE, **ctx)
    return p, sig


def _y(p: pd.DataFrame, oc: str) -> str:
    return oc + "_w" if oc + "_w" in p.columns else oc


# ---------------------------------------------------------------------------
# 1. The benchmark regression
# ---------------------------------------------------------------------------
def surprise_regressions(p: pd.DataFrame, split: str, outcome: str = S.PRIMARY_OUTCOME):
    d = p[p["split"] == split]
    y = _y(d, outcome)
    ctrl = [c for c in CONTROLS if c in d.columns]
    need = [y, "sue_rank", "sue_decile", "season", "permno", "ff12"] + ctrl
    d = d.dropna(subset=need)
    specs = {
        "(1) SUE deciles": ([], ["sue_decile", "season"]),
        "(2) SUE rank": (["sue_rank"], ["season"]),
        "(3) SUE rank + controls": (["sue_rank"] + ctrl, ["season", "ff12"]),
        "(4) SUE deciles + controls": (ctrl, ["sue_decile", "season", "ff12"]),
    }
    fits = {name: ols_cluster(d, y, x, cat)[0] for name, (x, cat) in specs.items()}
    tab = regression_table(fits, ["sue_rank"] + ctrl)
    fm = fama_macbeth(d, y, ["sue_rank"] + ctrl)
    for name, res in fits.items():
        OUT.save_fit(res, f"cd_surprise__{outcome}__{split}__{name.split(')')[0].strip('(')}", name)
    OUT.save_table(tab, f"cd_surprise_regressions__{split}",
                   f"{outcome} on the earnings surprise and controls, {split} split "
                   f"(two-way clustered by firm and season; t in parentheses)", index=True)
    if len(fm):
        OUT.save_table(fm, f"cd_surprise_fama_macbeth__{split}",
                       f"Fama-MacBeth by season (Newey-West), {outcome} on SUE rank + controls, {split}")
    return tab, fm, len(d)


# ---------------------------------------------------------------------------
# 2. Beat / meet / miss
# ---------------------------------------------------------------------------
def surprise_sign(p: pd.DataFrame, split: str, outcome: str = S.PRIMARY_OUTCOME,
                  save: bool = True) -> pd.DataFrame:
    import statsmodels.api as sm
    d = p[p["split"] == split].dropna(subset=[outcome, "surprise"])
    cat = np.select([d["surprise"] > 0, d["surprise"] < 0], ["beat", "miss"], "meet")
    X = pd.get_dummies(pd.Series(cat, index=d.index), dtype=float)
    res = sm.OLS(d[outcome].astype(float), X).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d["season"])[0]})
    rows = []
    for c in ["beat", "meet", "miss"]:
        if c not in X:
            continue
        lo, hi = res.conf_int().loc[c]
        rows.append({"group": c, "n": int(X[c].sum()), "share": X[c].mean(), "mean_car": res.params[c],
                     "se_cluster": res.bse[c], "ci95_lo": lo, "ci95_hi": hi,
                     "median_car": d.loc[X[c] == 1, outcome].median()})
    diff = res.t_test("beat - miss")
    rows.append({"group": "beat - miss", "mean_car": float(diff.effect[0]),
                 "se_cluster": float(diff.sd[0][0]), "ci95_lo": float(diff.conf_int()[0][0]),
                 "ci95_hi": float(diff.conf_int()[0][1])})
    t = pd.DataFrame(rows)
    t["n"] = t["n"].astype("Int64")
    t.insert(0, "outcome", outcome)
    if not save:
        return t
    OUT.save_table(t, f"cd_surprise_sign__{split}",
                   f"Mean {outcome} by sign of the surprise, {split} (95% CI clustered by season)")
    return t


# ---------------------------------------------------------------------------
# 3. Robustness of the benchmark
# ---------------------------------------------------------------------------
def _ic_row(dim, group, d, outcome):
    ic = rank_ic(d, "sue_rank", outcome)
    ls = long_short(d, "sue_rank", outcome)
    return {"dimension": dim, "group": str(group), "outcome": outcome,
            "n": int(d[["sue_rank", outcome]].dropna().shape[0]), "seasons": ic.get("seasons"),
            "mean_ic": ic.get("mean_ic"), "ic_t_nw": ic.get("t_nw"),
            "ls_mean": ls.get("ls_mean"), "ls_t_nw": ls.get("ls_t_nw")}


def surprise_robustness(p: pd.DataFrame, split: str, outcome: str = S.PRIMARY_OUTCOME) -> pd.DataFrame:
    d = p[p["split"] == split]
    rows = [_ic_row("baseline", "all", d, outcome)]
    for g, x in d.groupby("ff12"):
        rows.append(_ic_row("sector (FF12)", g, x, outcome))
    for g, x in d.groupby("size_tercile"):
        rows.append(_ic_row("size tercile", {1.0: "small", 2.0: "mid", 3.0: "large"}.get(g, g), x, outcome))
    for g, x in d.groupby("year"):
        rows.append(_ic_row("year", int(g), x, outcome))
    for g, x in d.groupby("session_type"):
        if len(x) >= 300:
            rows.append(_ic_row("release session", g, x, outcome))
    for oc in S.OUTCOMES + S.MKT_ADJ_OUTCOMES + S.PLACEBO_OUTCOMES:
        if oc in d.columns and oc != outcome:
            rows.append(_ic_row("event window", oc, d, oc))
    rows.append(_ic_row("sample", "excluding basis_suspect", d[~d["basis_suspect"].astype(bool)], outcome))
    rows.append(_ic_row("sample", "excluding timing_suspect", d[~d["timing_suspect"].astype(bool)], outcome))
    t = pd.DataFrame(rows)
    OUT.save_table(t, f"cd_surprise_robustness__{split}",
                   f"SUE rank IC with {outcome} across subsets, {split} (Newey-West t across seasons)")
    return t


# ---------------------------------------------------------------------------
# 4. The day -1 placebo
# ---------------------------------------------------------------------------
def placebo_diagnostic(p: pd.DataFrame) -> pd.DataFrame:
    """
    SUE should not rank the return of the day *before* the reaction day. Where it
    does, either the clock puts day 0 one day too late for some events, or news
    leaks before the release (pre-announcements, other outlets). Breaking the IC
    down by session and by period tells the two apart: a clock problem concentrates
    in one session type or source; a leak / pre-announcement effect is diffuse.
    """
    oc = "ar_m1"
    rows = []
    for split in SPLITS:
        d = p[p["split"] == split]
        rows.append(_ic_row(split, "all", d, oc))
        for g, x in d.groupby("session_type"):
            if len(x) >= 300:
                rows.append(_ic_row(f"{split}: session", g, x, oc))
        for g, x in d.groupby("period"):
            rows.append(_ic_row(f"{split}: period", g, x, oc))
        for flag, label in (("link_same_session", "8-K in same session as wire"),
                            ("timing_suspect", "timing_suspect")):
            if flag in d and d[flag].nunique() > 1:      # constant in in_primary: says nothing
                for g, x in d.groupby(d[flag].astype(str)):
                    if len(x) >= 300:
                        rows.append(_ic_row(f"{split}: {label}", g, x, oc))
        for g, x in d.groupby("size_tercile"):
            rows.append(_ic_row(f"{split}: size tercile", {1.0: "small", 2.0: "mid", 3.0: "large"}.get(g, g), x, oc))
    t = pd.DataFrame(rows)
    OUT.save_table(t, "cd_placebo_diagnostic",
                   "SUE rank IC with the day -1 abnormal return (placebo), by subset; should be ~0")
    return t


# ---------------------------------------------------------------------------
# 5. Loughran-McDonald pilot on the texts we have
# ---------------------------------------------------------------------------
def lm_pilot(p: pd.DataFrame, sig: pd.DataFrame, outcome: str = S.PRIMARY_OUTCOME):
    import statsmodels.api as sm
    from scipy import stats

    cols = [c for c in ("lm_tone", "lm_net", "lm_negshare", "lm_unc") if c in sig]
    d = p.merge(sig[["event_id"] + cols], on="event_id", how="inner").dropna(subset=["lm_tone", outcome])
    if len(d) < 30:
        print(f"lm pilot: only {len(d)} events with LM tone, skipped")
        return None
    rows = []
    for c in cols:
        for target in (outcome, "post_car_60_ff4", "ar_m1", "sue_price"):
            x = d[[c, target]].dropna()
            rho, pv = stats.spearmanr(x[c], x[target])
            rows.append({"signal": c, "with": target, "n": len(x), "spearman": rho, "p_value": pv})
    corr = pd.DataFrame(rows)

    # SUE rank + tone, standardised tone so the coefficient is per 1 SD; HC1 (too few
    # seasons to cluster: the texts are almost all from 2006).
    d = d.assign(lm_tone_z=(d["lm_tone"] - d["lm_tone"].mean()) / d["lm_tone"].std())
    regs = []
    for name, x in {"SUE rank": ["sue_rank"], "LM tone": ["lm_tone_z"],
                    "SUE rank + LM tone": ["sue_rank", "lm_tone_z"],
                    "+ size, B/M": ["sue_rank", "lm_tone_z", "log_mktcap", "log_bm"]}.items():
        dd = d.dropna(subset=[outcome] + x)
        res = sm.OLS(dd[outcome].astype(float), sm.add_constant(dd[x].astype(float))).fit(cov_type="HC1")
        row = {"spec": name, "n": int(res.nobs), "r2": res.rsquared}
        for v in ("sue_rank", "lm_tone_z"):
            if v in res.params:
                row[f"coef_{v}"], row[f"t_{v}"] = res.params[v], res.tvalues[v]
        regs.append(row)
    regs = pd.DataFrame(regs)
    by = d.groupby("split").size().to_dict()
    cap = (f"EXPLORATORY: LM tone on the {len(d)} events with a text "
           f"({', '.join(f'{k} {v}' for k, v in by.items())}); below the 5% coverage rule, not a result")
    OUT.save_table(corr, "cd_lm_pilot_correlations", cap)
    OUT.save_table(regs, "cd_lm_pilot_regressions", cap + "; HC1 standard errors")

    # 2x2 cells, text sign = above/below the pilot median (seasons are too thin)
    d["_one"] = "pilot"
    tb = two_by_two(d.assign(season="pilot"), "lm_tone", outcome, rule="season_median", cluster="permno")
    if len(tb["cells"]):
        OUT.save_table(tb["cells"], "cd_lm_pilot_2x2", cap + "; text sign = above / below median tone")
    return corr, regs, tb.get("cells")


def main() -> int:
    p, sig = _load()
    for split in SPLITS:
        surprise_regressions(p, split)
        surprise_sign(p, split)
        surprise_robustness(p, split)
    placebo_diagnostic(p)
    lm_pilot(p, sig)
    print(f"done: results/tables/cd_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
