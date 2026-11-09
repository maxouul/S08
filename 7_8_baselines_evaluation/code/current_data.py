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


# ---------------------------------------------------------------------------
# 6. Where the surprise gets the sign wrong (error cases)
# ---------------------------------------------------------------------------
def surprise_errors(p: pd.DataFrame, split: str = "validation", outcome: str = S.PRIMARY_OUTCOME,
                    n_cases: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    How often the sign of the surprise and the sign of the reaction disagree,
    by size of the surprise, plus the largest disagreements among large firms.

    The summary goes to results/; the list of cases names firms and dates
    (row-level, WRDS-derived), so it is written to data/ only.
    """
    d = p[p["split"] == split].dropna(subset=[outcome, "surprise", "sue_rank"]).copy()
    d = d[d["surprise"] != 0]
    d["beat"] = d["surprise"] > 0
    d["wrong_sign"] = np.where(d["beat"], d[outcome] < 0, d[outcome] > 0)
    d["big_wrong"] = np.where(d["beat"], d[outcome] < -0.05, d[outcome] > 0.05)
    d["sue_quintile"] = pd.qcut(d["sue_rank"], 5, labels=["Q1 (worst)", "Q2", "Q3", "Q4", "Q5 (best)"])
    rows = []
    for (label, g) in [("all beats", d[d["beat"]]), ("all misses", d[~d["beat"]])] + \
                      [(f"SUE {q}", g) for q, g in d.groupby("sue_quintile", observed=True)]:
        rows.append({"group": label, "n": len(g), "share_wrong_sign": g["wrong_sign"].mean(),
                     "share_wrong_by_5pct": g["big_wrong"].mean(), "mean_car": g[outcome].mean()})
    summary = pd.DataFrame(rows)
    summary.insert(0, "split", split)
    OUT.save_table(summary, f"cd_surprise_errors__{split}",
                   f"How often the surprise sign and the {outcome} sign disagree, {split}")

    # strongest-signal disagreements: top-quintile beats that fell, bottom-quintile misses that rose
    big = d[d["size_tercile"] == 3]
    cases = pd.concat([
        big[(big["sue_rank"] > 0.3) & (big[outcome] < 0)].nsmallest(n_cases, outcome),
        big[(big["sue_rank"] < -0.3) & (big[outcome] > 0)].nlargest(n_cases, outcome)])
    keep = ["cname", "ticker", "accession", "reaction_date", "session_type", "actual",
            "consensus_median", "n_analysts", "sue_rank", outcome, "post_car_60_ff4", "mktcap_t2", "ff12"]
    cases = cases[[c for c in keep if c in cases.columns]]
    S.DATA.mkdir(parents=True, exist_ok=True)
    cases.to_csv(S.DATA / f"error_cases__{split}.csv", index=False)     # private: names firms
    return summary, cases


# ---------------------------------------------------------------------------
# 7. Sample sizes and minimum detectable effects
# ---------------------------------------------------------------------------
def sample_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Events per split and sample, and after Llama 3.1's cutoff. Counts only: no outcome is read."""
    e = events.copy()
    e["year"] = pd.to_datetime(e["reaction_date"]).dt.year
    rows = []
    for split, g in e.groupby("split"):
        rows.append({"split": split, "years": f"{g['year'].min()}-{g['year'].max()}",
                     "events_all": len(g), "in_text_sample": int(g["in_text_sample"].sum()),
                     "in_primary": int(g["in_primary"].sum()),
                     "primary_after_llama_cutoff": int((g["in_primary"] & g["after_llama_cutoff"]).sum()),
                     "quarters": g["year"].astype(str).str.cat(
                         pd.to_datetime(g["reaction_date"]).dt.quarter.astype(str)).nunique()})
    t = pd.DataFrame(rows)
    order = {"dev": 0, "validation": 1, "excluded_2020": 2, "holdout": 3}
    t = t.sort_values("split", key=lambda s: s.map(order)).reset_index(drop=True)
    tot = {"split": "total", "years": f"{e['year'].min()}-{e['year'].max()}", "events_all": len(e),
           "in_text_sample": int(e["in_text_sample"].sum()), "in_primary": int(e["in_primary"].sum()),
           "primary_after_llama_cutoff": int((e["in_primary"] & e["after_llama_cutoff"]).sum())}
    t = pd.concat([t, pd.DataFrame([tot])], ignore_index=True)
    t["quarters"] = t["quarters"].astype("Int64")
    OUT.save_table(t, "cd_sample_counts", "Events by split and sample (counts only; holdout outcomes not read)")
    return t


def mde_tables(p: pd.DataFrame, counts: pd.DataFrame, outcome: str = S.PRIMARY_OUTCOME,
               n_sim: int = 200, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two power calculations, both two-sided 5% with 80% power.

    (a) Incremental IC on the evaluation split. The test statistic is the mean of
        quarterly ICs, so its noise depends on the number of quarters and the
        events per quarter. Two estimates of the quarterly IC standard deviation:
        - "noise floor": a random signal on the real validation outcomes
          (simulated n_sim times), which is the best case;
        - "SUE-like": the observed quarter-to-quarter SD of the SUE IC, which is
          what a real signal with time-varying strength looks like.
        MDE = (t_{0.975,S-1} + t_{0.80,S-1}) * sd / sqrt(S). For the holdout
        (outcomes locked) the validation SDs are scaled to its quarter count.
    (b) The checkpoint's paired-arm formula for the contamination comparison,
        SE = sqrt(2 (1 - r) (1/n_a + 1/n_b)), MDE = 2.80 SE, with the actual
        pre- and post-cutoff holdout counts and r in {0.5, 0.7, 0.9}.
    """
    from scipy import stats
    rng = np.random.default_rng(seed)
    v = p[p["split"] == "validation"].dropna(subset=[outcome, "sue_rank"])
    ic_sue = v.groupby("season").apply(lambda g: g["sue_rank"].corr(g[outcome], method="spearman"))
    sims = []
    for _ in range(n_sim):
        r = pd.Series(rng.standard_normal(len(v)), index=v.index)
        sims.append(v.assign(_r=r).groupby("season").apply(
            lambda g: g["_r"].corr(g[outcome], method="spearman")).std(ddof=1))
    sd_noise, sd_sue = float(np.mean(sims)), float(ic_sue.std(ddof=1))

    def mde(sd, S):
        k = stats.t.ppf(0.975, S - 1) + stats.t.ppf(0.80, S - 1)
        return k * sd / np.sqrt(S)

    c = counts.set_index("split")
    rows = []
    for split, S in (("validation", int(c.loc["validation", "quarters"])),
                     ("holdout", int(c.loc["holdout", "quarters"]))):
        n = int(c.loc[split, "in_primary"])
        for label, sd in (("noise floor (random signal)", sd_noise), ("SUE-like time variation", sd_sue)):
            rows.append({"split": split, "events": n, "quarters": S, "quarterly_ic_sd": label,
                         "sd": sd, "mde_mean_ic": mde(sd, S), "clears_0.02_bar": mde(sd, S) <= 0.02})
    mde_ic = pd.DataFrame(rows)
    OUT.save_table(mde_ic, "cd_mde_incremental_ic",
                   "Minimum detectable mean IC (two-sided 5%, 80% power), quarter-clustered")

    h = int(c.loc["holdout", "in_primary"])
    n_post = int(c.loc["holdout", "primary_after_llama_cutoff"])
    n_pre = h - n_post
    rows = []
    for arm, (na, nb) in {"pre- vs post-cutoff holdout (actual)": (n_pre, n_post),
                          "checkpoint example": (2500, 2500), "checkpoint example, large": (20000, 20000)}.items():
        for r in (0.5, 0.7, 0.9):
            se = np.sqrt(2 * (1 - r) * (1 / na + 1 / nb))
            rows.append({"comparison": arm, "n_a": na, "n_b": nb, "r": r, "se": se, "mde_80": 2.80 * se})
    mde_pair = pd.DataFrame(rows)
    OUT.save_table(mde_pair, "cd_mde_contamination",
                   "Paired-arm MDE, SE = sqrt(2(1-r)(1/n_a+1/n_b)), MDE = 2.80 SE (checkpoint formula)")
    return mde_ic, mde_pair


# ---------------------------------------------------------------------------
# 8. Figure for the presentations
# ---------------------------------------------------------------------------
# Default validated categorical palette (dataviz skill), slots 1-3, light mode.
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
_INK, _INK2, _GRID, _SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def figure_surprise_response(p: pd.DataFrame, split: str = "validation"):
    """(a) mean reaction and 60-day drift by SUE decile; (b) beat / meet / miss by window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = p[p["split"] == split]
    dec = d.groupby("sue_decile").agg(react=("car_0_1_ff4", "mean"), drift=("post_car_60_ff4", "mean"))
    grp = np.select([d["surprise"] > 0, d["surprise"] < 0], ["Beat", "Miss"], "Meet")
    win = {"Day −1": "ar_m1", "Days 0 to +1": "car_0_1_ff4", "Next 60 days": "post_car_60_ff4"}
    bars = d.assign(g=grp).groupby("g")[list(win.values())].mean().loc[["Beat", "Meet", "Miss"]] * 100

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=_SURF)
    for ax in (a, b):
        ax.set_facecolor(_SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(_GRID)
        ax.tick_params(colors=_INK2, labelsize=9)
        ax.yaxis.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.axhline(0, color=_INK2, linewidth=0.8)

    x = dec.index.astype(int) + (1 if dec.index.min() == 0 else 0)
    for col, name, colr in (("react", "Days 0 to +1", _SERIES[0]), ("drift", "Next 60 days", _SERIES[1])):
        a.plot(x, dec[col] * 100, color=colr, linewidth=2, marker="o", markersize=5,
               markeredgecolor=_SURF, markeredgewidth=1.5, label=name)
        a.annotate(name, (x[-1], dec[col].iloc[-1] * 100), xytext=(6, 0), textcoords="offset points",
                   va="center", fontsize=9, color=_INK2)
    a.set_xticks(list(x))
    a.set_xlabel("SUE decile (1 = most negative surprise)", color=_INK2, fontsize=9)
    a.set_ylabel("Mean abnormal return (%)", color=_INK2, fontsize=9)
    a.set_title("The surprise is priced at once, with no drift", loc="left", fontsize=11, color=_INK)
    a.legend(frameon=False, fontsize=9, labelcolor=_INK2, loc="upper left")
    a.set_xlim(x.min() - 0.5, x.max() + 2.2)

    w = 0.26
    xs = np.arange(len(bars.index))
    # colour follows the window in both panels: reaction blue, drift orange, day -1 aqua
    colour = {"ar_m1": _SERIES[2], "car_0_1_ff4": _SERIES[0], "post_car_60_ff4": _SERIES[1]}
    for i, (lab, col) in enumerate(win.items()):
        b.bar(xs + (i - 1) * w, bars[col], width=w - 0.03, color=colour[col], label=lab)
    lo, hi = bars.min().min(), bars.max().max()
    b.set_ylim(lo - 0.15 * (hi - lo), hi + 0.12 * (hi - lo))
    b.set_xticks(xs, bars.index)
    b.set_ylabel("Mean abnormal return (%)", color=_INK2, fontsize=9)
    b.set_title("Beat, meet, miss", loc="left", fontsize=11, color=_INK)
    b.legend(frameon=False, fontsize=9, labelcolor=_INK2, loc="lower left")
    for i, g in enumerate(bars.index):
        v = bars.loc[g, "car_0_1_ff4"]
        b.annotate(f"{v:+.1f}%", (i, v), xytext=(0, 3 if v > 0 else -11), textcoords="offset points",
                   ha="center", fontsize=8, color=_INK)
    fig.text(0.01, 0.01, f"{split.capitalize()} split, {len(d):,} events in the primary sample; FF4 abnormal returns.",
             fontsize=8, color=_INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = OUT.save_figure(fig, f"cd_surprise_response__{split}")
    plt.close(fig)
    return path


def main() -> int:
    p, sig = _load()
    for split in SPLITS:
        surprise_regressions(p, split)
        surprise_sign(p, split)
        surprise_robustness(p, split)
    placebo_diagnostic(p)
    lm_pilot(p, sig)
    surprise_errors(p, "validation")
    ev = pd.read_parquet(S.EVENTS, columns=["split", "reaction_date", "in_primary", "in_text_sample",
                                            "after_llama_cutoff"])
    counts = sample_counts(ev)
    mde_tables(p, counts)
    figure_surprise_response(p, "validation")
    print(f"done: results/tables/cd_*, results/figures/cd_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
