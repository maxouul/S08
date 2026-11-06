"""
02 Surprise (part 3): event list, point-in-time consensus and SUE.

Reads only cache/ (from 01_download) and data/trading_calendar.csv: no WRDS
connection and no dependency on Mahaut's files.

Event: one I/B/E/S quarterly EPS announcement (I/B/E/S ticker, fiscal quarter
end), linked to a CRSP permno.

Release time R: the Dow Jones wire time where one exists (apply_wire_clock),
otherwise I/B/E/S ANNDATS + ANNTIMS as a stand-in until EDGAR header times
exist. I/B/E/S is the check: clock_status says whether the two agree.

Point-in-time consensus (instructors' feedback):
  - an estimate counts only if its ANNDATS and ACTDATS are both before R's date;
  - it is for the same fiscal quarter as the actual (FPEDATS = PENDS);
  - estimates from a broker that stopped covering the firm before R are dropped;
  - each analyst's latest remaining estimate, issued within WINDOW_DAYS;
  - estimates are unadjusted, so each is put on the actual's share basis with
    CRSP's cumulative share factor (a split between estimate and release
    would otherwise create a fake surprise).

SUE, primary:     (actual - median consensus) / price at t-2
SUE, alternative: (actual - median consensus) / dispersion, if 3+ analysts
t is the reaction session; t-2 is two trading days earlier, before any release.
No winsorising here: that is an analysis choice, made once, in step 04.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from clock import assign_session
from common import ET, PROJECT_DIR, split_of

CACHE = PROJECT_DIR / "cache"
DATA = PROJECT_DIR / "data"

FIRST, LAST = pd.Timestamp("2006-01-01"), pd.Timestamp("2025-12-31")
WINDOW_DAYS = 90
LINK_MAX_SCORE = 3
VERSION = "surprise_v2_wire_first"
NS = "datetime64[ns]"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_cache(name: str, columns: list[str] | None = None) -> pd.DataFrame | None:
    """One cached pull, all years. None if it has not been pulled."""
    folder = CACHE / name
    if folder.is_dir():
        files = [f for f in sorted(folder.glob("*.parquet")) if ".partial" not in f.name]
        if not files:
            return None
        return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    single = CACHE / f"{name}.parquet"
    return pd.read_parquet(single, columns=columns) if single.exists() else None


def _dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_datetime(df[c]).astype(NS)
    return df


def _strip(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


class Funnel:
    """Exact counts after each step, written from the data, never typed."""

    def __init__(self):
        self.rows: list[tuple[str, int]] = []

    def step(self, name: str, n: int) -> None:
        self.rows.append((name, int(n)))

    def frame(self) -> pd.DataFrame:
        f = pd.DataFrame(self.rows, columns=["step", "remaining"])
        f["removed"] = (-f["remaining"].diff()).fillna(0).astype(int)
        f["removed_pct"] = (100 * f["removed"] / f["remaining"].shift(1)).round(1).fillna(0)
        return f


# --------------------------------------------------------------------------
# CRSP lookups
# --------------------------------------------------------------------------
def value_asof(keys: pd.DataFrame, table: pd.DataFrame, col: str) -> pd.Series:
    """
    For each (permno, date) in keys, the value of `col` on the last CRSP day
    at or before that date. Used for share factors, which only change on
    distribution days, so the last known value is the right one.
    """
    left = keys[["permno", "date"]].copy()
    left["_row"] = np.arange(len(left))
    left = left.dropna(subset=["permno", "date"]).astype({"permno": "int64", "date": NS})
    right = table[["permno", "date", col]].dropna().astype({"permno": "int64", "date": NS})
    m = pd.merge_asof(left.sort_values("date"), right.sort_values("date"),
                      on="date", by="permno", direction="backward")
    out = pd.Series(np.nan, index=range(len(keys)))
    out.loc[m["_row"].to_numpy()] = m[col].to_numpy()
    return pd.Series(out.to_numpy(), index=keys.index)


# --------------------------------------------------------------------------
# step 1: events
# --------------------------------------------------------------------------
def build_events(act: pd.DataFrame, link: pd.DataFrame, cal: pd.DataFrame,
                 funnel: Funnel) -> pd.DataFrame:
    a = act.copy()
    a["ticker"] = _strip(a["ticker"])
    a = _dates(a, ["pends", "anndats", "actdats"])
    a = a[a["anndats"].between(FIRST, LAST)]
    funnel.step("I/B/E/S quarterly EPS actuals announced 2006-2025", len(a))

    # One announcement per firm-quarter. If I/B/E/S holds more than one row,
    # keep the first announced, first recorded: what the market saw first.
    a = (a.sort_values(["ticker", "pends", "anndats", "actdats"])
          .drop_duplicates(["ticker", "pends"], keep="first"))
    funnel.step("one announcement per firm-quarter", len(a))

    a = a[a["value"].notna()]
    funnel.step("actual EPS value present", len(a))
    a = a[a["curr_act"].astype("string").eq("USD")]
    funnel.step("actual reported in US dollars", len(a))

    lk = link.copy()
    lk["ticker"] = _strip(lk["ticker"])
    lk = _dates(lk, ["sdate", "edate"])
    lk["edate"] = lk["edate"].fillna(pd.Timestamp("2099-12-31"))
    lk = lk[lk["score"] <= LINK_MAX_SCORE]
    m = a.merge(lk[["ticker", "permno", "sdate", "edate", "score"]], on="ticker", how="inner")
    m = m[(m["anndats"] >= m["sdate"]) & (m["anndats"] <= m["edate"])]
    # Several permnos for one ticker-date (share classes): best score first.
    m = (m.sort_values(["score", "sdate"])
          .drop_duplicates(["ticker", "pends"], keep="first"))
    m["permno"] = m["permno"].astype("int64")
    funnel.step(f"linked to a CRSP permno on the announcement date (score <= {LINK_MAX_SCORE})", len(m))

    # I/B/E/S announcement time: replaced by the wire time where one exists
    # (apply_wire_clock); kept as the stand-in elsewhere. 00:00:00 means
    # unknown, not midnight.
    tims = m["anntims"].astype("string").str.strip()
    unknown = tims.isna() | tims.eq("00:00:00")
    # An explicit format: left to guess, pandas infers it from the first row
    # and can misread the whole column.
    txt = (m["anndats"].dt.strftime("%Y-%m-%d") + " " + tims).where(~unknown)
    naive = pd.to_datetime(txt, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    m["R_et"] = naive.dt.tz_localize(ET, ambiguous="NaT", nonexistent="NaT")
    m = m[m["R_et"].notna()].copy()
    funnel.step("announcement time known (I/B/E/S)", len(m))

    sess = assign_session(m["R_et"], cal)
    m = pd.concat([m, sess], axis=1)
    m["reaction_date"] = pd.to_datetime(m["reaction_date"]).astype(NS)
    m["R_date"] = m["R_et"].dt.tz_localize(None).dt.normalize().astype(NS)
    m["R_source"] = "ibes_standin"

    # t-2: two trading sessions before the reaction session.
    days = cal["date"].to_numpy(dtype=NS)
    pos = np.searchsorted(days, m["reaction_date"].to_numpy(dtype=NS))
    m["t2_date"] = pd.to_datetime(days[pos - 2])

    m["event_id"] = m["ticker"] + "|" + m["pends"].dt.strftime("%Y-%m-%d")
    m = m.rename(columns={"value": "actual"})
    keep = ["event_id", "ticker", "permno", "cname", "pends", "anndats", "anntims",
            "actual", "R_et", "R_date", "R_source", "session_type", "reaction_date",
            "entry_field", "t2_date", "score"]
    return m[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# step 1b: wire time first
# --------------------------------------------------------------------------
def apply_wire_clock(ev: pd.DataFrame, clk: pd.DataFrame | None, cal: pd.DataFrame,
                     max_days: int = 4) -> tuple[pd.DataFrame, dict]:
    """
    Instructors' rule: the release time is the wire time; I/B/E/S is a check.
    For each event, the nearest wire time for the same stock within `max_days`
    replaces the I/B/E/S time. clock_status records the agreement rule:
      wire_ibes_agree     wire and I/B/E/S put the release in the same session
      wire_ibes_conflict  they do not; the wire is used
      ibes_only           no wire time; I/B/E/S stands in until EDGAR times exist
    The release date also sets which estimates count, so this changes the
    consensus as well as the return windows.
    """
    out = ev.copy()
    out["ibes_R_et"] = out["R_et"]
    out["ibes_reaction_date"] = out["reaction_date"]
    if clk is None or clk.empty:
        out["clock_status"] = "ibes_only"
        return out, {"events": len(out), "matched to a wire time": 0}

    w = clk[clk["R_source"].eq("wire") & clk["R_et"].notna()][["permno", "R_et"]].copy()
    w["permno"] = w["permno"].astype("int64")
    w["wire_R"] = pd.to_datetime(w["R_et"], utc=True).dt.tz_convert(ET)
    w = w[["permno", "wire_R"]].drop_duplicates()

    k = out[["event_id", "permno", "R_et"]].merge(w, on="permno", how="inner")
    k["hours"] = (k["wire_R"] - k["R_et"]).dt.total_seconds().abs() / 3600
    k = k[k["hours"] <= max_days * 24]
    k = k.loc[k.groupby("event_id")["hours"].idxmin(), ["event_id", "wire_R"]]

    out = out.merge(k, on="event_id", how="left")
    has = out["wire_R"].notna()
    out["R_et"] = out["wire_R"].where(has, out["R_et"])
    out = out.drop(columns="wire_R")

    sess = assign_session(out["R_et"], cal)
    out["session_type"] = sess["session_type"]
    out["entry_field"] = sess["entry_field"]
    out["reaction_date"] = pd.to_datetime(sess["reaction_date"]).astype(NS)
    out["R_date"] = out["R_et"].dt.tz_localize(None).dt.normalize().astype(NS)
    out["R_source"] = np.where(has, "wire", "ibes_standin")
    agree = has & out["reaction_date"].eq(out["ibes_reaction_date"])
    out["clock_status"] = np.select([agree, has & ~agree],
                                    ["wire_ibes_agree", "wire_ibes_conflict"], "ibes_only")
    days = cal["date"].to_numpy(dtype=NS)
    pos = np.searchsorted(days, out["reaction_date"].to_numpy(dtype=NS))
    out["t2_date"] = pd.to_datetime(days[pos - 2])

    counts = {"events": len(out), "matched to a wire time": int(has.sum())}
    counts.update(out["clock_status"].value_counts().to_dict())
    moved = has & ~agree
    gap = (out.loc[moved, "reaction_date"] - out.loc[moved, "ibes_reaction_date"]).dt.days
    counts["conflicts: wire session later than I/B/E/S"] = int((gap > 0).sum())
    counts["conflicts: wire session earlier than I/B/E/S"] = int((gap < 0).sum())
    return out, counts


# --------------------------------------------------------------------------
# step 2: point-in-time consensus
# --------------------------------------------------------------------------
def build_consensus(ev: pd.DataFrame, det: pd.DataFrame, fac: pd.DataFrame,
                    stops: pd.DataFrame | None, window_days: int = WINDOW_DAYS
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns (consensus per event, the estimates actually used, counts).
    `fac` holds permno, date, dlycumfacshr.
    """
    d = det.copy()
    d["ticker"] = _strip(d["ticker"])
    d = _dates(d, ["fpedats", "anndats", "actdats"])
    d = d.rename(columns={"fpedats": "pends", "anndats": "est_date",
                          "actdats": "est_actdats", "value": "est"})
    e = ev[["event_id", "ticker", "pends", "permno", "R_date"]]
    x = e.merge(d, on=["ticker", "pends"], how="inner")
    counts = {"estimates for the event's quarter": len(x)}

    before = (x["est_date"] < x["R_date"]) & (x["est_actdats"] < x["R_date"])
    counts["dropped: issued or recorded on or after R's date"] = int((~before).sum())
    x = x[before]

    fresh = x["est_date"] >= x["R_date"] - pd.Timedelta(days=window_days)
    counts[f"dropped: older than {window_days} days"] = int((~fresh).sum())
    x = x[fresh & x["est"].notna()]

    # Currency: `curr` came back empty in the pull, so we also read report_curr.
    cur = x["curr"].astype("string")
    rep = x["report_curr"].astype("string")
    usd = (cur.isna() | cur.eq("USD")) & (rep.isna() | rep.eq("USD"))
    counts["dropped: estimate not in US dollars"] = int((~usd).sum())
    x = x[usd]

    x = x.copy()
    x["estimator"] = _strip(x["estimator"])
    if stops is not None and len(stops):
        s = stops.copy()
        s["ticker"] = _strip(s["ticker"])
        s["estimator"] = _strip(s["estimator"])
        s = _dates(s, ["fpedats", "astpdats"]).rename(columns={"fpedats": "pends"})
        k = (x[["ticker", "estimator", "pends", "est_date", "R_date"]].reset_index()
             .merge(s[["ticker", "estimator", "pends", "astpdats"]],
                    on=["ticker", "estimator", "pends"], how="inner"))
        stale = k.loc[(k["astpdats"] >= k["est_date"]) & (k["astpdats"] < k["R_date"]), "index"].unique()
        counts["dropped: broker stopped covering before R"] = len(stale)
        x = x.drop(index=stale)
    else:
        counts["dropped: broker stopped covering before R"] = "not checked (no stopped-estimates file)"

    # Each analyst's latest estimate.
    x["analyst"] = x["estimator"] + "/" + x["analys"].astype("string")
    x = (x.sort_values(["event_id", "analyst", "est_date", "est_actdats"])
          .drop_duplicates(["event_id", "analyst"], keep="last"))
    counts["estimates used (latest per analyst)"] = len(x)

    # Share basis: estimate x factor(R) / factor(estimate date).
    x["f_est"] = value_asof(x.rename(columns={"est_date": "date"}), fac, "dlycumfacshr")
    x["f_R"] = value_asof(x.rename(columns={"R_date": "date"}), fac, "dlycumfacshr")
    ratio = (x["f_R"] / x["f_est"]).where(x["f_est"].gt(0) & x["f_R"].gt(0))
    x["share_adj"] = ratio.fillna(1.0)
    x["est_adj"] = x["est"] * x["share_adj"]
    counts["estimates rescaled for a split or distribution"] = int((x["share_adj"].sub(1).abs() > 1e-9).sum())
    counts["estimates with no CRSP factor (left unscaled)"] = int(ratio.isna().sum())

    g = x.groupby("event_id")["est_adj"]
    cons = pd.DataFrame({
        "consensus_median": g.median(),
        "consensus_mean": g.mean(),
        "dispersion": g.std(),
        "n_analysts": g.size(),
        "est_latest_date": x.groupby("event_id")["est_date"].max(),
    }).reset_index()
    return cons, x, counts


# --------------------------------------------------------------------------
# step 3: price and SUE
# --------------------------------------------------------------------------
def add_sue(ev: pd.DataFrame, cons: pd.DataFrame, px: pd.DataFrame, funnel: Funnel) -> pd.DataFrame:
    s = ev.merge(cons, on="event_id", how="inner")
    funnel.step(f"at least one point-in-time estimate within {WINDOW_DAYS} days", len(s))

    p = px[["permno", "date", "dlyprc", "dlycumfacshr"]].copy()
    p = p.astype({"permno": "int64", "date": NS})
    # The CRSP pull holds US common stocks only, so this step removes ADRs,
    # foreign-incorporated firms, funds and other share types.
    s = s[s["permno"].isin(p["permno"].unique())]
    funnel.step("stock is a US common share in CRSP", len(s))
    s = s.merge(p.rename(columns={"date": "t2_date", "dlyprc": "prc_t2", "dlycumfacshr": "f_t2"}),
                on=["permno", "t2_date"], how="left")
    s["f_R"] = value_asof(s.rename(columns={"R_date": "date"}), p, "dlycumfacshr")
    adj = (s["f_R"] / s["f_t2"]).where(s["f_t2"].gt(0) & s["f_R"].gt(0)).fillna(1.0)
    # CRSP marks a bid-ask midpoint with a negative sign; the size is the price.
    s["price_t2"] = s["prc_t2"].abs() * adj
    s = s[s["price_t2"].gt(0)].copy()
    funnel.step("CRSP price at t-2", len(s))

    s["surprise"] = s["actual"] - s["consensus_median"]
    s["sue_price"] = s["surprise"] / s["price_t2"]
    ok = s["n_analysts"].ge(3) & s["dispersion"].gt(0)
    s["sue_disp"] = (s["surprise"] / s["dispersion"]).where(ok)
    s["split"] = split_of(s["reaction_date"])
    s["surprise_version"] = VERSION
    return s.drop(columns=["prc_t2", "f_t2", "f_R"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check_no_lookahead(used: pd.DataFrame) -> None:
    assert (used["est_date"] < used["R_date"]).all(), "an estimate dated on or after R was used"
    assert (used["est_actdats"] < used["R_date"]).all(), "an estimate recorded on or after R was used"
    print(f"look-ahead check passed on {len(used):,} estimates")


def compare_with_ibes(s: pd.DataFrame, surp: pd.DataFrame | None) -> pd.DataFrame | None:
    """Our actual and consensus against I/B/E/S's own surprise summary."""
    if surp is None or surp.empty:
        print("no I/B/E/S surprise summary in the cache yet: rerun the core pulls in 01_download")
        return None
    u = surp.copy()
    u["ticker"] = _strip(u["ticker"])
    u = _dates(u, ["anndats"])
    yr = pd.to_numeric(u["pyear"], errors="coerce")
    mo = pd.to_numeric(u["pmon"], errors="coerce")
    u = u[yr.notna() & mo.notna()].copy()
    u["ym"] = yr.loc[u.index].astype(int) * 100 + mo.loc[u.index].astype(int)
    ours = s.assign(ym=s["pends"].dt.year * 100 + s["pends"].dt.month)
    m = ours.merge(u[["ticker", "ym", "anndats", "actual", "surpmean"]]
                   .rename(columns={"actual": "ibes_actual", "anndats": "ibes_anndats"}),
                   on=["ticker", "ym"], how="inner")
    m = m.drop_duplicates("event_id")
    out = {
        "events matched to the I/B/E/S summary": len(m),
        "share of our events matched": round(len(m) / len(s), 3),
        "actual within 1 cent": round(((m["actual"] - m["ibes_actual"]).abs() <= 0.01).mean(), 3),
        "consensus mean within 1 cent of I/B/E/S mean": round(((m["consensus_mean"] - m["surpmean"]).abs() <= 0.01).mean(), 3),
        "consensus mean within 5 cents": round(((m["consensus_mean"] - m["surpmean"]).abs() <= 0.05).mean(), 3),
        "same sign of surprise": round((np.sign(m["actual"] - m["consensus_mean"])
                                        == np.sign(m["ibes_actual"] - m["surpmean"])).mean(), 3),
    }
    for k, v in out.items():
        print(f"  {k:<48} {v}")
    m["gap"] = (m["consensus_mean"] - m["surpmean"]).abs()
    return m.sort_values("gap", ascending=False)[
        ["event_id", "cname", "reaction_date", "actual", "ibes_actual",
         "consensus_mean", "surpmean", "n_analysts", "gap"]].head(10)


def add_crsp_names(df: pd.DataFrame, names: pd.DataFrame | None,
                   date_col: str = "anndats") -> pd.DataFrame:
    """
    The exchange ticker and company name from CRSP on the event date. The
    I/B/E/S ticker is I/B/E/S's own code (TGT is Pactiv, not Target), so anyone
    looking up a release needs these instead.
    """
    out = df.drop(columns=[c for c in ["crsp_ticker", "crsp_name"] if c in df.columns])
    if names is None:
        out["crsp_ticker"], out["crsp_name"] = pd.NA, pd.NA
        return out
    nm = names[["permno", "namedt", "nameenddt", "ticker", "issuernm"]].copy()
    nm = _dates(nm, ["namedt", "nameenddt"])
    nm["nameenddt"] = nm["nameenddt"].fillna(pd.Timestamp("2099-12-31"))
    nm["permno"] = nm["permno"].astype("int64")
    k = out[["permno", date_col]].copy()
    k["_row"] = np.arange(len(k))
    k["permno"] = k["permno"].astype("int64")
    k[date_col] = pd.to_datetime(k[date_col]).astype(NS)
    m = k.merge(nm, on="permno", how="inner")
    m = m[(m[date_col] >= m["namedt"]) & (m[date_col] <= m["nameenddt"])]
    m = m.sort_values("namedt").drop_duplicates("_row", keep="last").set_index("_row")
    out["crsp_ticker"] = m["ticker"].reindex(range(len(out))).to_numpy()
    out["crsp_name"] = m["issuernm"].reindex(range(len(out))).to_numpy()
    return out


HANDCHECK_FIELDS = ["eps_in_release", "gaap_or_adjusted", "matches_ibes", "notes"]


def handcheck_sample(s: pd.DataFrame, n: int = 50, seed: int = 15,
                     names: pd.DataFrame | None = None) -> pd.DataFrame:
    """A fixed random sample for checking reported EPS against the releases."""
    cols = ["event_id", "ticker", "cname", "permno", "pends", "anndats", "R_et",
            "actual", "consensus_median", "n_analysts", "sue_price"]
    out = s.sample(n=min(n, len(s)), random_state=seed)[cols].sort_values("anndats")
    out = add_crsp_names(out.reset_index(drop=True), names)
    out = out.rename(columns={"ticker": "ibes_code"})
    first = ["event_id", "crsp_ticker", "crsp_name", "ibes_code", "cname"]
    out = out[first + [c for c in out.columns if c not in first]]
    for c in HANDCHECK_FIELDS:
        out[c] = ""
    return out


def refresh_handcheck(existing: pd.DataFrame, names: pd.DataFrame | None) -> pd.DataFrame:
    """Add CRSP ticker and name to a sample made earlier, keeping what was filled in."""
    out = existing.rename(columns={"ticker": "ibes_code"})
    out = add_crsp_names(out.reset_index(drop=True), names)
    first = ["event_id", "crsp_ticker", "crsp_name", "ibes_code", "cname"]
    return out[[c for c in first if c in out.columns] + [c for c in out.columns if c not in first]]
