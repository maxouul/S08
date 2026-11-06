"""
03 Returns (part 4): abnormal returns around each event.

Reads data/surprise.parquet (events, with reaction_date and entry_field from
the release clock), cache/crsp_daily, cache/crsp_delists, cache/ff_daily and
data/trading_calendar.csv. No WRDS connection.

Day 0 is the reaction session from the clock: the first trading session that
can react to the release. All windows count trading days from there.

Pre-specified definitions
-------------------------
Expected return   rf + b1*MKTRF + b2*SMB + b3*HML + b4*UMD. Betas from OLS of
                  excess returns on the four factors over days -280..-31 (about
                  a year, ending 30 trading days before day 0), with at least
                  120 returns. The intercept is estimated but not used.
AR_t              ret_t - expected_t.  Market-adjusted: ret_t - (MKTRF_t + RF_t),
                  a robustness check that needs no estimation.
car_0_1           AR on days 0 and +1: the market's reaction (descriptive).
post-release      "Not yet priced": enter at the first CRSP price after the
                  release - the open of day 0, or the close of day 0 for
                  intraday releases - and hold to the close of day +1 or +60.
                    open entry : (open-to-close return on day 0 - expected_0)
                                 + sum of AR on days 1..k
                    close entry: sum of AR on days 1..k
                  expected_0 is a full-day figure applied to part of a day: an
                  approximation, and one that is common to all firms that day.
                  post_ret_* compound the same legs, raw.
ret_prior_1m      raw return over days -22..-2 (a baseline control).
Delistings        CRSP's daily return on the delisting date excludes the
                  delisting return (checked: they match in 207 of 4,306 cases),
                  so the two are compounded; with no daily return that day the
                  delisting return stands alone. After delisting the stock
                  earns no abnormal return (sums run over available days).
Data end          a 60-day window running past CRSP's last date is left missing
                  (drift_complete False), never shortened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NS = "datetime64[ns]"
EST_START, EST_END = -280, -31
MIN_EST_OBS = 120
PRIOR_START, PRIOR_END = -22, -2
MIN_PRIOR_OBS = 15
DRIFT = 60
BATCH = 5000
VERSION = "returns_v3_placebo_day_m1"


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def factor_arrays(ff: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, str]:
    """Factors aligned to the trading days. Detects percent versus decimal units."""
    f = ff.copy()
    f["date"] = pd.to_datetime(f["date"]).astype(NS)
    f = f.set_index("date").reindex(dates)
    cols = ["mktrf", "smb", "hml", "umd"]
    med = f["mktrf"].abs().median()
    scale = 100.0 if med > 0.05 else 1.0
    note = f"factor units: {'percent, divided by 100' if scale == 100 else 'decimal'} (median |MKTRF| {med:.4f})"
    F = f[cols].to_numpy(dtype=float) / scale
    rf = f["rf"].to_numpy(dtype=float) / scale
    missing = int(np.isnan(F).any(axis=1).sum())
    if missing:
        note += f"; {missing} trading days without factors"
    return F, rf, note


def build_panel(px: pd.DataFrame, permnos, dates: pd.DatetimeIndex,
                delists: pd.DataFrame | None = None) -> tuple[np.ndarray, pd.Index, dict]:
    """Stock-by-day matrix of daily returns, with delisting returns filled in."""
    pidx = pd.Index(np.sort(np.unique(np.asarray(permnos, dtype="int64"))))
    p = px[px["permno"].isin(pidx)]
    i = pidx.get_indexer(p["permno"].astype("int64"))
    j = dates.get_indexer(pd.to_datetime(p["date"]).astype(NS))
    ok = (i >= 0) & (j >= 0)
    R = np.full((len(pidx), len(dates)), np.nan)
    R[i[ok], j[ok]] = p["dlyret"].to_numpy(dtype=float)[ok]

    info = {"delisting returns available": 0, "delisting returns used alone (no daily return)": 0,
            "delisting returns compounded with the day's return": 0,
            "delisting returns already in the day's return": 0}
    if delists is not None and len(delists):
        d = delists[delists["permno"].isin(pidx) & delists["delret"].notna()].copy()
        d["delistingdt"] = pd.to_datetime(d["delistingdt"]).astype(NS)
        d = d.drop_duplicates(["permno", "delistingdt"])
        i2 = pidx.get_indexer(d["permno"].astype("int64"))
        j2 = dates.get_indexer(d["delistingdt"])
        ok2 = (i2 >= 0) & (j2 >= 0)
        rows, cols = i2[ok2], j2[ok2]
        cur = R[rows, cols]
        dr = d["delret"].to_numpy(dtype=float)[ok2]
        # CRSP's daily return on the delisting date is the last trading return;
        # the delisting return runs from that price to what holders received.
        # The day's total is the two compounded. Where they are equal, CRSP has
        # already put the delisting return in that day, so it is left as is.
        missing = np.isnan(cur)
        same = ~missing & (np.abs(cur - dr) < 1e-6)
        comp = ~missing & ~same
        R[rows, cols] = np.where(missing, dr, np.where(same, cur, (1 + cur) * (1 + dr) - 1))
        info = {"delisting returns available": int(ok2.sum()),
                "delisting returns used alone (no daily return)": int(missing.sum()),
                "delisting returns compounded with the day's return": int(comp.sum()),
                "delisting returns already in the day's return": int(same.sum())}
    return R, pidx, info


def delisting_check(px: pd.DataFrame, delists: pd.DataFrame) -> dict:
    """Does CRSP's daily return on the delisting date already include delret?"""
    d = delists[delists["delret"].notna()][["permno", "delistingdt", "delret"]].copy()
    d = d.astype({"permno": "int64"})
    d["delistingdt"] = pd.to_datetime(d["delistingdt"]).astype(NS)
    m = d.merge(px[["permno", "date", "dlyret"]].astype({"permno": "int64"})
                .rename(columns={"date": "delistingdt"}), on=["permno", "delistingdt"], how="left")
    has = m["dlyret"].notna()
    same = has & (m["dlyret"] - m["delret"]).abs().lt(1e-6)
    return {"delistings with a delisting return": len(m),
            "with a daily return on the delisting date": int(has.sum()),
            "where that return equals the delisting return": int(same.sum())}


def day0_open_to_close(ev: pd.DataFrame, px: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """
    Return from the open to the close of day 0, dividends included:
    (1 + ret_0) * |P_-1| / Open_0 - 1, when no split falls on day 0;
    otherwise |P_0| / Open_0 - 1.
    """
    t0 = dates.get_indexer(ev["reaction_date"])
    prev = pd.Series(pd.NaT, index=ev.index, dtype=NS)
    okp = t0 > 0
    prev[okp] = dates[t0[okp] - 1]
    k = pd.DataFrame({"permno": ev["permno"].astype("int64"),
                      "d0": ev["reaction_date"].astype(NS), "dm1": prev})
    cols = ["permno", "date", "dlyopen", "dlyprc", "dlyret", "dlycumfacpr"]
    p = px[cols].astype({"permno": "int64"})
    p["date"] = pd.to_datetime(p["date"]).astype(NS)
    k = k.merge(p.rename(columns={"date": "d0", "dlyopen": "o0", "dlyprc": "p0", "dlyret": "r0",
                                  "dlycumfacpr": "f0"}), on=["permno", "d0"], how="left")
    k = k.merge(p[["permno", "date", "dlyprc", "dlycumfacpr"]]
                .rename(columns={"date": "dm1", "dlyprc": "pm1", "dlycumfacpr": "fm1"}),
                on=["permno", "dm1"], how="left")
    k.index = ev.index
    o0 = k["o0"].where(k["o0"] > 0)
    no_split = (k["f0"] - k["fm1"]).abs().lt(1e-9)
    with_div = (1 + k["r0"]) * k["pm1"].abs() / o0 - 1
    simple = k["p0"].abs() / o0 - 1
    return with_div.where(no_split & with_div.notna(), simple)


# --------------------------------------------------------------------------
# main computation
# --------------------------------------------------------------------------
def compute_returns(ev: pd.DataFrame, px: pd.DataFrame, ff: pd.DataFrame,
                    cal: pd.DataFrame, delists: pd.DataFrame | None = None
                    ) -> tuple[pd.DataFrame, dict]:
    ev = ev.copy()
    ev["reaction_date"] = pd.to_datetime(ev["reaction_date"]).astype(NS)
    ev["permno"] = ev["permno"].astype("int64")

    last_crsp = pd.to_datetime(px["date"]).max()
    dates = pd.DatetimeIndex(cal.loc[cal["date"] <= last_crsp, "date"]).astype(NS)
    T = len(dates)
    F, rf, fnote = factor_arrays(ff, dates)
    R, pidx, dinfo = build_panel(px, ev["permno"], dates, delists)
    info = {"trading days in panel": T, "stocks in panel": len(pidx), "factors": fnote, **dinfo}

    oc0 = day0_open_to_close(ev, px, dates).to_numpy(dtype=float)
    open_entry = ev["entry_field"].eq("open").to_numpy()

    prow = pidx.get_indexer(ev["permno"])
    t0 = dates.get_indexer(ev["reaction_date"])
    n = len(ev)

    out = {k: np.full(n, np.nan) for k in [
        "alpha", "b_mkt", "b_smb", "b_hml", "b_umd", "n_est_obs",
        "ar_m1", "ar_m1_mkt", "ar_0", "ar_1", "car_0_1_ff4", "car_0_1_mkt", "ret_0_1_raw",
        "oc_0", "post_ar_0", "post_car_1_ff4", "post_car_60_ff4",
        "post_car_1_mkt", "post_car_60_mkt", "post_ret_1_raw", "post_ret_60_raw",
        "n_days_60", "ret_prior_1m"]}
    drift_complete = np.zeros(n, dtype=bool)

    o_est = np.arange(EST_START, EST_END + 1)
    o_post = np.arange(0, DRIFT + 1)
    o_prior = np.arange(PRIOR_START, PRIOR_END + 1)

    def window(pi, t, offs):
        tt = t[:, None] + offs
        inside = (tt >= 0) & (tt < T)
        ttc = np.clip(tt, 0, T - 1)
        vals = R[np.clip(pi, 0, None)[:, None], ttc]
        vals = np.where(inside, vals, np.nan)
        return vals, ttc, inside

    n_singular = 0
    for b in range(0, n, BATCH):
        sl = slice(b, min(b + BATCH, n))
        pi, t = prow[sl], t0[sl]
        valid = (pi >= 0) & (t >= 0)
        m_ = len(pi)

        # ---- betas -------------------------------------------------------
        Y, tte, _ = window(pi, t, o_est)
        Y = Y - rf[tte]
        X = np.concatenate([np.ones((m_, len(o_est), 1)), F[tte]], axis=2)
        use = np.isfinite(Y) & np.isfinite(X).all(axis=2) & valid[:, None]
        Xm = np.where(use[..., None], X, 0.0)
        Ym = np.where(use, Y, 0.0)
        XtX = np.einsum("nti,ntj->nij", Xm, Xm)
        Xty = np.einsum("nti,nt->ni", Xm, Ym)
        nobs = use.sum(axis=1)
        ok = nobs >= MIN_EST_OBS
        beta = np.full((m_, 5), np.nan)
        if ok.any():
            try:
                beta[ok] = np.linalg.solve(XtX[ok], Xty[ok][..., None])[..., 0]
            except np.linalg.LinAlgError:
                # A degenerate window would stop the whole batch: solve the
                # full-rank ones and leave the others without betas.
                A, bvec = XtX[ok], Xty[ok]
                full = np.linalg.matrix_rank(A) == A.shape[-1]
                sol = np.full((len(A), 5), np.nan)
                if full.any():
                    sol[full] = np.linalg.solve(A[full], bvec[full][..., None])[..., 0]
                beta[ok] = sol
                n_singular += int((~full).sum())
        out["n_est_obs"][sl] = np.where(valid, nobs, np.nan)
        for c, name in enumerate(["alpha", "b_mkt", "b_smb", "b_hml", "b_umd"]):
            out[name][sl] = beta[:, c]

        # ---- day 0 .. +60 ------------------------------------------------
        Rp, ttp, inside = window(pi, t, o_post)
        Rp = np.where(valid[:, None], Rp, np.nan)
        Fp, rfp = F[ttp], rf[ttp]
        exp_ff = rfp + np.einsum("ntk,nk->nt", Fp, beta[:, 1:])
        exp_mkt = rfp + Fp[..., 0]
        AR = Rp - exp_ff
        ARm = Rp - exp_mkt

        two = np.isfinite(AR[:, 0]) & np.isfinite(AR[:, 1])
        out["ar_0"][sl] = AR[:, 0]
        out["ar_1"][sl] = AR[:, 1]
        out["car_0_1_ff4"][sl] = np.where(two, AR[:, 0] + AR[:, 1], np.nan)
        twom = np.isfinite(ARm[:, 0]) & np.isfinite(ARm[:, 1])
        out["car_0_1_mkt"][sl] = np.where(twom, ARm[:, 0] + ARm[:, 1], np.nan)
        out["ret_0_1_raw"][sl] = np.where(two | twom, (1 + Rp[:, 0]) * (1 + Rp[:, 1]) - 1, np.nan)

        # post-release legs
        oe = open_entry[sl]
        oc = oc0[sl]
        out["oc_0"][sl] = np.where(oe, oc, np.nan)
        leg0_ff = np.where(oe, oc - exp_ff[:, 0], 0.0)
        leg0_mkt = np.where(oe, oc - exp_mkt[:, 0], 0.0)
        leg0_raw = np.where(oe, oc, 0.0)
        leg_ok = ~oe | np.isfinite(oc)
        out["post_ar_0"][sl] = np.where(leg_ok, leg0_ff, np.nan)

        alive1 = np.isfinite(Rp[:, 1])
        complete = (t + DRIFT < T) & valid
        drift_complete[sl] = complete
        n60 = np.isfinite(Rp[:, 1:]).sum(axis=1)
        out["n_days_60"][sl] = np.where(complete, n60, np.nan)

        ok1 = leg_ok & alive1
        out["post_car_1_ff4"][sl] = np.where(ok1, leg0_ff + AR[:, 1], np.nan)
        out["post_car_1_mkt"][sl] = np.where(ok1, leg0_mkt + ARm[:, 1], np.nan)
        out["post_ret_1_raw"][sl] = np.where(ok1, (1 + leg0_raw) * (1 + Rp[:, 1]) - 1, np.nan)

        ok60 = ok1 & complete
        out["post_car_60_ff4"][sl] = np.where(ok60, leg0_ff + np.nansum(AR[:, 1:], axis=1), np.nan)
        out["post_car_60_mkt"][sl] = np.where(ok60, leg0_mkt + np.nansum(ARm[:, 1:], axis=1), np.nan)
        gross = np.prod(1 + np.nan_to_num(Rp[:, 1:]), axis=1)
        out["post_ret_60_raw"][sl] = np.where(ok60, (1 + leg0_raw) * gross - 1, np.nan)

        # ---- day -1: a placebo. The day before the reaction session; if the
        # clock were late, the real reaction would land here.
        Rb, ttb, _ = window(pi, t, np.array([-1]))
        Rb = np.where(valid, Rb[:, 0], np.nan)
        tb = ttb[:, 0]
        out["ar_m1"][sl] = Rb - (rf[tb] + np.einsum("nk,nk->n", F[tb], beta[:, 1:]))
        out["ar_m1_mkt"][sl] = Rb - (rf[tb] + F[tb, 0])

        # ---- prior month ---------------------------------------------------
        Rq, _, _ = window(pi, t, o_prior)
        Rq = np.where(valid[:, None], Rq, np.nan)
        nq = np.isfinite(Rq).sum(axis=1)
        out["ret_prior_1m"][sl] = np.where(nq >= MIN_PRIOR_OBS,
                                           np.prod(1 + np.nan_to_num(Rq), axis=1) - 1, np.nan)

    res = pd.DataFrame(out, index=ev.index)
    res["drift_complete"] = drift_complete
    # FF4 measures need betas; without them they are missing, not market-adjusted.
    no_beta = res["b_mkt"].isna()
    for c in ["car_0_1_ff4", "post_ar_0", "post_car_1_ff4", "post_car_60_ff4", "ar_0", "ar_1", "ar_m1"]:
        res.loc[no_beta, c] = np.nan

    # delisted within the 60-day window
    res["delisted_in_window"] = False
    if delists is not None and len(delists):
        d = delists[["permno", "delistingdt"]].astype({"permno": "int64"}).copy()
        d["delistingdt"] = pd.to_datetime(d["delistingdt"]).astype(NS)
        end = pd.Series(pd.NaT, index=ev.index, dtype=NS)
        okt = t0 >= 0
        end[okt] = dates[np.minimum(t0[okt] + DRIFT, T - 1)]
        k = pd.DataFrame({"permno": ev["permno"], "d0": ev["reaction_date"], "end": end,
                          "_row": np.arange(n)})
        k = k.merge(d, on="permno", how="inner")
        hit = k.loc[(k["delistingdt"] > k["d0"]) & (k["delistingdt"] <= k["end"]), "_row"].unique()
        res.iloc[hit, res.columns.get_loc("delisted_in_window")] = True

    keep = ["event_id", "permno", "reaction_date", "session_type", "entry_field"]
    keep = [c for c in keep if c in ev.columns]
    result = pd.concat([ev[keep], res], axis=1)
    result["returns_version"] = VERSION
    info["events"] = n
    info["events with a singular estimation window"] = n_singular
    info["events with betas"] = int(res["b_mkt"].notna().sum())
    return result, info


# --------------------------------------------------------------------------
# checks for the notebook (development years only)
# --------------------------------------------------------------------------
def newey_west_se(x: np.ndarray, lags: int = 2) -> float:
    """Standard error of the mean of a series, robust to autocorrelation up to `lags`."""
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


def rank_ic_by_season(df: pd.DataFrame, signal: str, outcome: str, min_n: int = 30,
                      nw_lags: int = 2) -> dict:
    """
    Rank correlation per announcement season (calendar quarter), then averaged.
    t_stat treats seasons as independent; t_nw allows for correlation between
    neighbouring seasons (60-day windows spill into the next season).
    """
    d = df[[signal, outcome, "reaction_date"]].dropna()
    if d.empty:
        return {"seasons": 0, "mean_ic": np.nan, "t_stat": np.nan, "t_nw": np.nan, "share_positive": np.nan}
    d = d.assign(season=pd.to_datetime(d["reaction_date"]).dt.to_period("Q"))
    ic = d.groupby("season").apply(
        lambda g: g[signal].rank().corr(g[outcome].rank()) if len(g) >= min_n else np.nan,
        include_groups=False)
    ic = pd.to_numeric(ic, errors="coerce").dropna() if isinstance(ic, pd.Series) else pd.Series(dtype=float)
    if ic.empty:
        return {"seasons": 0, "mean_ic": np.nan, "t_stat": np.nan, "t_nw": np.nan, "share_positive": np.nan}
    se = ic.std(ddof=1) / np.sqrt(len(ic)) if len(ic) > 1 else np.nan
    nw = newey_west_se(ic.to_numpy(), nw_lags)
    return {"seasons": len(ic), "mean_ic": round(float(ic.mean()), 4),
            "t_stat": round(float(ic.mean() / se), 2), "t_nw": round(float(ic.mean() / nw), 2),
            "share_positive": round(float((ic > 0).mean()), 2)}


def decile_means(df: pd.DataFrame, signal: str, outcomes: list[str]) -> pd.DataFrame:
    """Average outcomes by signal decile, deciles formed within each season."""
    d = df.dropna(subset=[signal]).copy()
    d["season"] = pd.to_datetime(d["reaction_date"]).dt.to_period("Q")
    d["decile"] = d.groupby("season")[signal].transform(
        lambda x: pd.qcut(x.rank(method="first"), 10, labels=False) + 1)
    t = d.groupby("decile")[outcomes].mean()
    t.insert(0, "events", d.groupby("decile").size())
    return t
