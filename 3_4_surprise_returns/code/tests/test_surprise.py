"""Synthetic test for surprise.py: every rule has a hand-computed answer."""
import os, sys
os.environ.setdefault("PROJECT_DIR", "/tmp/proj")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import surprise as S

sys.path.insert(0, os.path.dirname(__file__))
from make_test_calendar import make as make_test_calendar
cal = make_test_calendar()          # approximate NYSE calendar, for tests only
T = pd.Timestamp

act = pd.DataFrame([
    # ticker, pends, anndats, anntims, actdats, value, curr
    ("AAA", "2015-03-31", "2015-04-20", "16:30:00", "2015-04-20", 1.25, "USD"),
    ("AAA", "2015-03-31", "2015-04-25", "09:00:00", "2015-04-25", 9.99, "USD"),   # duplicate, later: dropped
    ("BBB", "2015-03-31", "2015-04-22", "08:00:00", "2015-04-22", 1.05, "USD"),
    ("CCC", "2015-03-31", "2015-04-23", "00:00:00", "2015-04-23", 0.50, "USD"),   # time unknown
    ("DDD", "2015-03-31", "2015-04-23", "16:05:00", "2015-04-23", 0.70, "CAD"),   # not USD
    ("FFF", "2015-03-31", "2015-04-23", "16:05:00", "2015-04-23", 0.80, "USD"),   # weak link only
], columns=["ticker", "pends", "anndats", "anntims", "actdats", "value", "curr_act"])
act["cname"] = act["ticker"] + " Corp"
for c in ["cusip", "oftic", "measure", "pdicity", "acttims", "usfirm"]:
    act[c] = None

link = pd.DataFrame([("AAA", 1, 1), ("BBB", 2, 1), ("CCC", 3, 1), ("DDD", 4, 1), ("FFF", 6, 6)],
                    columns=["ticker", "permno", "score"])
link["sdate"], link["edate"], link["ncusip"] = "2000-01-01", None, None

E = lambda t, est, an, v, d, act_d=None, rep="USD": dict(
    ticker=t, estimator=est, analys=an, value=v, fpedats="2015-03-31", anndats=d,
    actdats=act_d or d, curr=None, report_curr=rep, pdf="D", fpi="6")
det = pd.DataFrame([
    E("AAA", 10, 100, 1.00, "2015-02-01"),
    E("AAA", 10, 100, 1.10, "2015-04-01"),                 # revision: latest wins -> 1.10
    E("AAA", 20, 200, 1.20, "2015-04-20"),                 # dated on R's date: out
    E("AAA", 30, 300, 0.90, "2015-03-01", "2015-04-21"),   # recorded after R: out
    E("AAA", 40, 400, 1.00, "2014-12-01"),                 # older than 90 days: out
    E("AAA", 50, 500, 1.30, "2015-03-15"),                 # broker stopped 2015-04-01: out
    E("AAA", 60, 600, 1.40, "2015-03-20", rep="CAD"),      # not USD: out
    E("BBB", 10, 101, 2.00, "2015-02-15"),                 # pre-split basis -> 1.00
    E("BBB", 20, 201, 1.10, "2015-03-10"),
    E("BBB", 30, 301, 0.90, "2015-04-10"),
])
stops = pd.DataFrame([dict(ticker="AAA", estimator=50, fpedats="2015-03-31", astpdats="2015-04-01")])

days = cal["date"][(cal["date"] >= "2014-10-01") & (cal["date"] <= "2015-05-31")]
px = pd.concat([
    pd.DataFrame({"permno": 1, "date": days, "dlyprc": -50.0, "dlycumfacshr": 1.0}),
    pd.DataFrame({"permno": 2, "date": days, "dlyprc": 40.0,
                  "dlycumfacshr": np.where(days < "2015-03-02", 2.0, 1.0)}),
], ignore_index=True)

f = S.Funnel()
ev = S.build_events(act, link, cal, f)
cons, used, counts = S.build_consensus(ev, det, px, stops)
s = S.add_sue(ev, cons, px, f)
S.check_no_lookahead(used)
print(f.frame().to_string(index=False)); print(counts)

a = s.set_index("ticker")
# events and funnel
assert list(f.frame()["remaining"]) == [6, 5, 5, 4, 3, 2, 2, 2, 2], f.frame()
assert set(a.index) == {"AAA", "BBB"}
# AAA: after close Monday -> reaction Tuesday, t-2 = previous Friday
assert a.loc["AAA", "session_type"] == "after_close"
assert a.loc["AAA", "reaction_date"] == T("2015-04-21") and a.loc["AAA", "t2_date"] == T("2015-04-17")
assert a.loc["AAA", "actual"] == 1.25                       # first announcement kept, not 9.99
assert a.loc["AAA", "n_analysts"] == 1 and np.isclose(a.loc["AAA", "consensus_median"], 1.10)
assert np.isclose(a.loc["AAA", "price_t2"], 50.0)           # sign of bid-ask midpoint removed
assert np.isclose(a.loc["AAA", "sue_price"], 0.15 / 50)
assert np.isnan(a.loc["AAA", "sue_disp"])                   # fewer than 3 analysts
# BBB: pre-open -> same day; split-adjusted estimate; three analysts
assert a.loc["BBB", "session_type"] == "pre_open" and a.loc["BBB", "reaction_date"] == T("2015-04-22")
assert a.loc["BBB", "n_analysts"] == 3
assert np.isclose(a.loc["BBB", "consensus_median"], 1.00)  # 2.00 pre-split -> 1.00
assert np.isclose(a.loc["BBB", "dispersion"], 0.1)
assert np.isclose(a.loc["BBB", "sue_price"], 0.05 / 40)
assert np.isclose(a.loc["BBB", "sue_disp"], 0.5)
assert counts["dropped: broker stopped covering before R"] == 1
assert counts["estimates rescaled for a split or distribution"] == 1

# wire first: AAA's wire says pre-open on the 20th (I/B/E/S: after close),
# BBB's wire agrees, and a wire for an unrelated date is ignored
clk = pd.DataFrame([
    dict(permno=1, R_et="2015-04-20T08:00:00-04:00", R_source="wire"),
    dict(permno=2, R_et="2015-04-22T07:58:00-04:00", R_source="wire"),
    dict(permno=2, R_et="2015-01-21T07:58:00-05:00", R_source="wire"),
    dict(permno=3, R_et=None, R_source="ibes_standin"),
])
ev2, wc = S.apply_wire_clock(ev, clk, cal)
w = ev2.set_index("ticker")
assert w.loc["AAA", "clock_status"] == "wire_ibes_conflict" and w.loc["AAA", "R_source"] == "wire"
assert w.loc["AAA", "reaction_date"] == T("2015-04-20") and w.loc["AAA", "session_type"] == "pre_open"
assert w.loc["AAA", "ibes_reaction_date"] == T("2015-04-21") and w.loc["AAA", "t2_date"] == T("2015-04-16")
assert w.loc["BBB", "clock_status"] == "wire_ibes_agree" and w.loc["BBB", "reaction_date"] == T("2015-04-22")
assert wc["conflicts: wire session earlier than I/B/E/S"] == 1 and wc["matched to a wire time"] == 2
# without a clock file nothing changes
ev3, _ = S.apply_wire_clock(ev, None, cal)
assert (ev3["clock_status"] == "ibes_only").all() and ev3["reaction_date"].equals(ev["reaction_date"])

surp = pd.DataFrame([dict(ticker="AAA", pyear=2015, pmon=3, anndats="2015-04-20", actual=1.25, surpmean=1.10),
                     dict(ticker="BBB", pyear=2015, pmon=3, anndats="2015-04-22", actual=1.05, surpmean=1.02)])
S.compare_with_ibes(s, surp)
names = pd.DataFrame([
    dict(permno=1, namedt="2000-01-01", nameenddt="2010-12-31", ticker="OLD", issuernm="AAA OLD NAME"),
    dict(permno=1, namedt="2011-01-01", nameenddt=None, ticker="AAAX", issuernm="AAA HOLDINGS"),
    dict(permno=2, namedt="2000-01-01", nameenddt=None, ticker="BBQ", issuernm="BBB INC")])
hc = S.handcheck_sample(s, n=50, names=names)
assert len(hc) == 2 and "eps_in_release" in hc.columns
assert dict(zip(hc.ibes_code, hc.crsp_ticker)) == {"AAA": "AAAX", "BBB": "BBQ"}, hc
# an older sample, partly filled in, keeps its work and gains the names
old = hc.drop(columns=["crsp_ticker", "crsp_name"]).rename(columns={"ibes_code": "ticker"}).astype(str)
old.loc[0, "eps_in_release"] = "1.25"
new = S.refresh_handcheck(old, names)
assert new.loc[0, "eps_in_release"] == "1.25" and new["crsp_ticker"].notna().all()
assert list(new.columns[:3]) == ["event_id", "crsp_ticker", "crsp_name"]
print("\nALL SURPRISE TESTS PASSED")
