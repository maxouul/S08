"""Synthetic test for returns.py: known betas, injected shocks, hand-computed legs."""
import os, sys
os.environ.setdefault("PROJECT_DIR", "/tmp/proj")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import returns as Rm

sys.path.insert(0, os.path.dirname(__file__))
from make_test_calendar import make as make_test_calendar
cal = make_test_calendar()          # approximate NYSE calendar, for tests only
days = cal["date"][(cal["date"] >= "2010-01-01") & (cal["date"] <= "2012-12-31")].reset_index(drop=True)
T = len(days)
rng = np.random.default_rng(0)
fac = pd.DataFrame({"date": days, "mktrf": rng.normal(0.0004, 0.01, T), "smb": rng.normal(0, 0.005, T),
                    "hml": rng.normal(0, 0.005, T), "umd": rng.normal(0, 0.007, T), "rf": 0.0001})
B = np.array([1.2, 0.5, -0.3, 0.1])
base = fac["rf"].to_numpy() + fac[["mktrf", "smb", "hml", "umd"]].to_numpy() @ B

def stock(permno, start=0, end=T, shocks=None, noise=0.001):
    r = base + rng.normal(0, noise, T)
    for k, v in (shocks or {}).items():
        r[k] += v
    r = r[start:end]
    p = 100 * np.cumprod(1 + r)
    return pd.DataFrame({"permno": permno, "date": days[start:end].to_numpy(), "dlyret": r, "dlyprc": p,
                         "dlyopen": p / (1 + r) * 1.0, "dlycumfacpr": 1.0})

t_a = 400                                   # event day index for stock 1
s1 = stock(1, shocks={t_a: 0.05, t_a + 1: 0.02})
# opening gap on day 0: the open is 4% above the previous close
pm1 = s1.loc[t_a - 1, "dlyprc"]
s1.loc[t_a, "dlyopen"] = pm1 * 1.04
s2 = stock(2, end=t_a + 11)                 # trades until day +10, then delists
s2.loc[t_a + 10, "dlyret"] = np.nan         # no return on the delisting date
s3 = stock(3)                               # intraday release: enters at the close
t_d = T - 20                                # too close to the data end for 60 days
s4 = stock(4)
s5 = stock(5, start=t_a - 50)               # listed 50 days before the event: no betas
s6 = stock(6, end=t_a + 11)                 # trades on its delisting date too
s6.loc[t_a + 10, "dlyret"] = 0.01
s7 = stock(7, end=t_a + 11)                 # CRSP already put the delisting return in
s7.loc[t_a + 10, "dlyret"] = -0.20
px = pd.concat([s1, s2, s3, s4, s5, s6, s7], ignore_index=True)
delists = pd.DataFrame({"permno": [2, 6, 7], "delistingdt": [days[t_a + 10]] * 3,
                        "delret": [-0.30, -0.50, -0.20]})

ev = pd.DataFrame({
    "event_id": ["A", "B", "C", "D", "E", "F", "G"],
    "permno": [1, 2, 3, 4, 5, 6, 7],
    "reaction_date": [days[t_a], days[t_a], days[t_a], days[t_d], days[t_a], days[t_a], days[t_a]],
    "session_type": ["pre_open", "after_close", "intraday", "after_close", "pre_open", "pre_open", "pre_open"],
    "entry_field": ["open", "open", "close", "open", "open", "open", "open"],
})

# factors supplied in percent: must be detected and rescaled
ff_pct = fac.copy()
for c in ["mktrf", "smb", "hml", "umd", "rf"]:
    ff_pct[c] = ff_pct[c] * 100
out, info = Rm.compute_returns(ev, px, ff_pct, cal, delists)
print(info)
o = out.set_index("event_id")

assert "percent" in info["factors"]
# A: betas recovered, reaction = injected shocks, open-to-close leg
assert np.allclose(o.loc["A", ["b_mkt", "b_smb", "b_hml", "b_umd"]].astype(float), B, atol=0.05), o.loc["A"]
assert o.loc["A", "n_est_obs"] == 250
assert abs(o.loc["A", "car_0_1_ff4"] - 0.07) < 0.005
r0 = s1.loc[t_a, "dlyret"]
oc = (1 + r0) / 1.04 - 1
assert np.isclose(o.loc["A", "oc_0"], oc)
exp0 = fac["rf"][t_a] + fac.loc[t_a, ["mktrf", "smb", "hml", "umd"]].to_numpy() @ o.loc["A", ["b_mkt", "b_smb", "b_hml", "b_umd"]].to_numpy(dtype=float)
assert np.isclose(o.loc["A", "post_ar_0"], oc - exp0)
assert np.isclose(o.loc["A", "post_car_1_ff4"], o.loc["A", "post_ar_0"] + o.loc["A", "ar_1"])
assert o.loc["A", "drift_complete"] and o.loc["A", "n_days_60"] == 60
raw60 = (1 + oc) * np.prod(1 + s1.loc[t_a + 1:t_a + 60, "dlyret"]) - 1
assert np.isclose(o.loc["A", "post_ret_60_raw"], raw60)
prior = np.prod(1 + s1.loc[t_a - 22:t_a - 2, "dlyret"]) - 1
assert np.isclose(o.loc["A", "ret_prior_1m"], prior)
# day -1 placebo: the plain abnormal return on the day before the event
bA = o.loc["A", ["b_mkt", "b_smb", "b_hml", "b_umd"]].to_numpy(dtype=float)
exp_m1 = fac["rf"][t_a - 1] + fac.loc[t_a - 1, ["mktrf", "smb", "hml", "umd"]].to_numpy() @ bA
assert np.isclose(o.loc["A", "ar_m1"], s1.loc[t_a - 1, "dlyret"] - exp_m1)
assert abs(o.loc["A", "ar_m1"]) < 0.005      # no shock planted there
# Newey-West: for independent noise it is close to the plain standard error
x = rng.normal(size=400)
assert abs(Rm.newey_west_se(x, 2) / (x.std(ddof=1) / np.sqrt(400)) - 1) < 0.15
# and larger for a positively autocorrelated series
y = np.convolve(rng.normal(size=403), np.ones(4) / 4, mode="valid")
assert Rm.newey_west_se(y, 3) > 1.3 * y.std(ddof=1) / np.sqrt(len(y))
# B: delisting return filled on day +10; drift sums over available days; flagged
assert info["delisting returns used alone (no daily return)"] == 1
assert info["delisting returns compounded with the day's return"] == 1
assert info["delisting returns already in the day's return"] == 1
R6, pidx6, _ = Rm.build_panel(px, [6, 7], pd.DatetimeIndex(days).astype("datetime64[ns]"), delists)
assert np.isclose(R6[0, t_a + 10], 1.01 * 0.50 - 1)     # compounded: -0.495
assert np.isclose(R6[1, t_a + 10], -0.20)               # already included: unchanged
assert o.loc["B", "delisted_in_window"] and o.loc["B", "n_days_60"] == 10
assert o.loc["B", "post_car_60_ff4"] < -0.25
# C: intraday release enters at the close: no day-0 leg
assert o.loc["C", "post_ar_0"] == 0 and np.isnan(o.loc["C", "oc_0"])
assert np.isclose(o.loc["C", "post_car_1_ff4"], o.loc["C", "ar_1"])
# D: 60-day drift would run past the data: missing, not shortened; reaction still there
assert not o.loc["D", "drift_complete"] and np.isnan(o.loc["D", "post_car_60_ff4"])
assert np.isfinite(o.loc["D", "car_0_1_ff4"])
# E: too little history for betas: factor-model measures missing, market-adjusted present
assert np.isnan(o.loc["E", "b_mkt"]) and np.isnan(o.loc["E", "car_0_1_ff4"])
assert np.isfinite(o.loc["E", "car_0_1_mkt"])

# helpers run
big = pd.DataFrame({"sig": rng.normal(size=4000), "reaction_date": np.repeat(days[:40].to_numpy(), 100)})
big["y"] = 0.3 * big["sig"] + rng.normal(size=4000)
print(Rm.rank_ic_by_season(big, "sig", "y"))
print(Rm.decile_means(big, "sig", ["y"]).round(3).T)
# a degenerate factor window must not stop the batch
ff0 = fac.copy(); ff0[["smb", "hml", "umd"]] = 0.0
out0, info0 = Rm.compute_returns(ev, px, ff0, cal, delists)
assert info0["events with a singular estimation window"] == 6 and out0["b_mkt"].isna().all()
assert np.isfinite(out0.set_index("event_id").loc["A", "car_0_1_mkt"])
print("\nALL RETURNS TESTS PASSED")
