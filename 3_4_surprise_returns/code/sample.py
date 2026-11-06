"""
04 Sample: the event-level table every other part of the project uses.

Joins data/surprise.parquet and data/returns.parquet, links each event to
Mahaut's 8-K filing (the accession number the text is keyed on), and adds
every sample rule as a flag rather than a deletion, so a teammate can change
a rule without rebuilding anything. Reads only data/ and cache/; no WRDS.

Flags
-----
f_analysts3     3+ analysts in the consensus
f_price5        price above $5 at t-2 (instructors' feedback)
f_size          market cap at t-2 at or above the NYSE 20th percentile of the
                latest month-end (a flag: the team has not decided to cut on it)
two_source      wire and I/B/E/S put the release in the same session
timing_suspect  I/B/E/S-only events timed intraday: I/B/E/S marks 6.3% of
                them intraday against 0.5% where the wire confirms the time,
                so their post-release returns are unreliable
has_filing      linked to an 8-K in Mahaut's filing sample (text available)
basis_suspect   the I/B/E/S actual equals GAAP EPS, GAAP differs from
                operating EPS by 5+ cents, and the consensus sits far closer to
                operating EPS: the actual is probably on the wrong basis

Samples
-------
in_primary      f_analysts3 & f_price5 & two_source & has_filing & a reaction
                return: the pre-registered primary sample
in_text_sample  f_analysts3 & f_price5 & has_filing: every event with text,
                whatever its timing status (for robustness)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NS = "datetime64[ns]"
LLAMA_CUTOFF = pd.Timestamp("2023-12-31")
LINK_MAX_DAYS = 4
VERSION = "sample_v1"


def _d(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).astype(NS)


def _num(s: pd.Series) -> pd.Series:
    """Plain float64. WRDS pulls can arrive as nullable types (Float64, Int64),
    whose comparisons give true/false/missing and break numpy steps."""
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _flag(s: pd.Series) -> pd.Series:
    """Plain True/False, with missing counted as False."""
    return s.astype("boolean").fillna(False).astype(bool)


# --------------------------------------------------------------------------
# link to Mahaut's filings
# --------------------------------------------------------------------------
def link_filings(ev: pd.DataFrame, clk: pd.DataFrame | None, max_days: int = LINK_MAX_DAYS) -> pd.DataFrame:
    """
    Each event's 8-K: same stock, filed within `max_days` of the release.
    Preference: the filing whose clock gives the same reaction session, then
    the nearest filing date. An Item 2.02 8-K carries the press release
    itself, so a filing a few days after the release adds no later news.
    """
    out = ev[["event_id"]].copy()
    for c in ["accession", "cik", "filing_date", "link_gap_days", "link_same_session"]:
        out[c] = pd.NA
    if clk is None or clk.empty:
        return out
    c = clk[["accession", "permno", "cik", "filing_date", "reaction_date"]].dropna(subset=["accession", "permno"]).copy()
    c["permno"] = c["permno"].astype("int64")
    c["filing_date"] = _d(c["filing_date"])
    c["clk_reaction"] = pd.to_datetime(c["reaction_date"]).astype(NS)
    k = ev[["event_id", "permno", "R_date", "reaction_date"]].astype({"permno": "int64"}).merge(
        c.drop(columns="reaction_date"), on="permno", how="inner")
    k["link_gap_days"] = (k["filing_date"] - _d(k["R_date"])).dt.days
    k = k[k["link_gap_days"].between(-1, max_days)]
    k["link_same_session"] = k["clk_reaction"].eq(_d(k["reaction_date"]))
    k = (k.assign(_abs=k["link_gap_days"].abs())
          .sort_values(["event_id", "link_same_session", "_abs"], ascending=[True, False, True])
          .drop_duplicates("event_id"))
    # one filing should serve one event: if two events claim it, keep the closer
    k = k.sort_values(["accession", "link_same_session", "_abs"], ascending=[True, False, True]).drop_duplicates("accession")
    out = out[["event_id"]].merge(
        k[["event_id", "accession", "cik", "filing_date", "link_gap_days", "link_same_session"]],
        on="event_id", how="left")
    return out


# --------------------------------------------------------------------------
# size
# --------------------------------------------------------------------------
def add_size(ev: pd.DataFrame, cap: pd.DataFrame, bp: pd.DataFrame) -> pd.DataFrame:
    """Market cap at t-2 against the NYSE 20th percentile of the latest month-end."""
    k = ev[["event_id", "permno", "t2_date"]].copy()
    k["permno"] = k["permno"].astype("int64")
    k["t2_date"] = _d(k["t2_date"])
    c = cap[["permno", "date", "dlycap"]].astype({"permno": "int64"}).copy()
    c["date"] = _d(c["date"])
    k = k.merge(c.rename(columns={"date": "t2_date", "dlycap": "mktcap_t2"}), on=["permno", "t2_date"], how="left")
    b = bp[["date", "nyse_p20"]].copy()
    b["date"] = _d(b["date"])
    k = pd.merge_asof(k.sort_values("t2_date"), b.sort_values("date"),
                      left_on="t2_date", right_on="date", direction="backward").drop(columns="date")
    k["mktcap_t2"] = _num(k["mktcap_t2"])
    k["nyse_p20"] = _num(k["nyse_p20"])
    k["f_size"] = _flag(k["mktcap_t2"].ge(k["nyse_p20"]))
    return k[["event_id", "mktcap_t2", "nyse_p20", "f_size"]]


# --------------------------------------------------------------------------
# GAAP versus street
# --------------------------------------------------------------------------
def add_gaap(ev: pd.DataFrame, ccm: pd.DataFrame, fundq: pd.DataFrame, tol: float = 0.01,
             special_min: float = 0.05) -> pd.DataFrame:
    """
    Compares the I/B/E/S actual with Compustat's diluted GAAP EPS (epsfxq) and
    diluted operating EPS, which excludes special items (oepsxq), for the same
    fiscal quarter. actual_basis:
      gaap_and_operating  no special items: both agree with the actual
      operating_only      actual excludes special items (the usual street basis)
      gaap_only           actual equals GAAP while GAAP and operating differ
      neither             actual excludes more than special items (common for
                          street EPS, e.g. stock compensation, amortisation)
      no_compustat        no matching Compustat quarter
    """
    k = ev[["event_id", "permno", "R_date", "pends", "actual", "consensus_median"]].copy()
    k["permno"] = k["permno"].astype("int64")
    k["R_date"] = _d(k["R_date"])
    k["ym"] = _d(k["pends"]).dt.year * 100 + _d(k["pends"]).dt.month

    l = ccm[["gvkey", "permno", "linkprim", "linkdt", "linkenddt"]].copy()
    l["permno"] = l["permno"].astype("int64")
    l["linkdt"] = _d(l["linkdt"])
    l["linkenddt"] = _d(l["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))
    m = k.merge(l, on="permno", how="inner")
    m = m[(m["R_date"] >= m["linkdt"]) & (m["R_date"] <= m["linkenddt"])]
    m["_prim"] = m["linkprim"].map({"P": 0, "C": 1}).fillna(2)
    m = m.sort_values(["event_id", "_prim"]).drop_duplicates("event_id")[["event_id", "gvkey"]]

    f = fundq[["gvkey", "datadate", "epsfxq", "oepsxq"]].copy()
    f["ym"] = _d(f["datadate"]).dt.year * 100 + _d(f["datadate"]).dt.month
    f = f.dropna(subset=["epsfxq"]).drop_duplicates(["gvkey", "ym"], keep="last")
    k = k.merge(m, on="event_id", how="left").merge(f[["gvkey", "ym", "epsfxq", "oepsxq"]],
                                                    on=["gvkey", "ym"], how="left")
    k = k.rename(columns={"epsfxq": "gaap_eps", "oepsxq": "operating_eps"})
    for c in ["actual", "consensus_median", "gaap_eps", "operating_eps"]:
        k[c] = _num(k[c])

    g = (k["actual"] - k["gaap_eps"]).abs() <= tol
    o = (k["actual"] - k["operating_eps"]).abs() <= tol
    has = k["gaap_eps"].notna()
    k["actual_basis"] = np.select(
        [~has, g & o, o & ~g, g & ~o], ["no_compustat", "gaap_and_operating", "operating_only", "gaap_only"],
        "neither")
    special = (k["gaap_eps"] - k["operating_eps"]).abs()
    off_actual = (k["actual"] - k["consensus_median"]).abs()
    off_oper = (k["operating_eps"] - k["consensus_median"]).abs()
    k["basis_suspect"] = _flag(k["actual_basis"].eq("gaap_only") & special.ge(special_min)
                               & off_actual.ge(0.05) & (off_actual > 2 * off_oper))
    return k[["event_id", "gvkey", "gaap_eps", "operating_eps", "actual_basis", "basis_suspect"]]


# --------------------------------------------------------------------------
# assemble
# --------------------------------------------------------------------------
def build_final(s: pd.DataFrame, ret: pd.DataFrame, links: pd.DataFrame,
                size: pd.DataFrame, gaap: pd.DataFrame) -> pd.DataFrame:
    r = ret.drop(columns=[c for c in ["permno", "reaction_date", "session_type", "entry_field"] if c in ret.columns])
    e = s.merge(r, on="event_id", how="inner", validate="one_to_one")
    for t in (links, size, gaap):
        e = e.merge(t, on="event_id", how="left", validate="one_to_one")

    # The stricter "not yet priced" window: from day 0's close to day +60,
    # so it does not depend on the opening price at all.
    e["car_1_60_ff4"] = e["post_car_60_ff4"] - e["post_ar_0"].fillna(0)

    e["f_analysts3"] = _flag(_num(e["n_analysts"]).ge(3))
    e["f_price5"] = _flag(_num(e["price_t2"]).gt(5))
    e["two_source"] = _flag(e["clock_status"].eq("wire_ibes_agree"))
    e["timing_suspect"] = _flag(e["clock_status"].eq("ibes_only") & e["session_type"].eq("intraday"))
    e["has_filing"] = _flag(e["accession"].notna())
    e["basis_suspect"] = _flag(e["basis_suspect"])
    e["f_size"] = _flag(e["f_size"])
    e["after_llama_cutoff"] = _flag(_d(e["reaction_date"]) > LLAMA_CUTOFF)

    base = e["f_analysts3"] & e["f_price5"] & e["has_filing"]
    e["in_text_sample"] = base
    e["in_primary"] = base & e["two_source"] & _flag(_num(e["car_0_1_ff4"]).notna())
    e["sample_version"] = VERSION
    return e


def funnel(e: pd.DataFrame) -> pd.DataFrame:
    steps = [
        ("events with SUE and returns (02-03)", pd.Series(True, index=e.index)),
        ("3+ analysts", e["f_analysts3"]),
        ("price above $5 at t-2", e["f_price5"]),
        ("linked to an 8-K in the filing sample (text available)", e["has_filing"]),
        ("wire and I/B/E/S agree on the session (two sources)", e["two_source"]),
        ("reaction return available", e["car_0_1_ff4"].notna()),
    ]
    keep = pd.Series(True, index=e.index)
    rows = []
    for name, cond in steps:
        keep = keep & cond
        rows.append((name, int(keep.sum())))
    f = pd.DataFrame(rows, columns=["step", "remaining"])
    f["removed"] = (-f["remaining"].diff()).fillna(0).astype(int)
    f["removed_pct"] = (100 * f["removed"] / f["remaining"].shift(1)).round(1).fillna(0)
    assert f["remaining"].iloc[-1] == int(e["in_primary"].sum()), "funnel and in_primary disagree"
    return f


def by_split(e: pd.DataFrame, mask_cols: list[str]) -> pd.DataFrame:
    rows = {}
    for c in mask_cols:
        g = e[e[c]]
        row = g["split"].value_counts().to_dict()
        row["after Llama cutoff (2024-25)"] = int(g["after_llama_cutoff"].sum())
        row["total"] = len(g)
        rows[c] = row
    cols = ["dev", "validation", "excluded_2020", "holdout", "after Llama cutoff (2024-25)", "total"]
    return pd.DataFrame(rows).T.reindex(columns=cols).fillna(0).astype(int)


HANDOFF_COLUMNS = ["event_id", "accession", "cik", "permno", "filing_date", "R_et", "R_source",
                   "clock_status", "reaction_date", "split", "after_llama_cutoff",
                   "in_primary", "in_text_sample", "sample_version"]
