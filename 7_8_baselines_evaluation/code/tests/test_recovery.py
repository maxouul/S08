"""
End-to-end recovery test on synthetic data with a planted effect (synthetic.py):

* an informative text signal must add information beyond SUE (positive,
  significant coefficient controlling for SUE deciles; Clark-West > 0);
* a pure-noise signal must not;
* nothing may predict the day -1 placebo return.

If this test fails after a change, the change broke the statistics, whatever the
real-data results look like.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings as S  # noqa: E402

S.redirect_outputs("tests")

import synthetic as SY  # noqa: E402
from compare import compare_models  # noqa: E402
from panel import build_panel, cs_rank  # noqa: E402
from regressions import incremental_info  # noqa: E402


@pytest.fixture(scope="module")
def data():
    ev = SY.make_events(n_firms=200, years=range(2006, 2020))
    p = build_panel(ev, sample=None)
    rng = np.random.default_rng(5)
    zt, zs = ev.set_index("event_id").loc[p["event_id"], ["z_text", "z_sue"]].to_numpy().T
    sig = pd.DataFrame({"good": 0.5 * zs + 0.7 * zt + rng.normal(0, 0.5, len(p)),
                        "noise": rng.normal(size=len(p))}, index=p.index)
    return p, sig


def test_incremental_regression_recovers_planted_effect(data):
    p, sig = data
    d = p[p["split"] == "validation"].copy()
    for c in sig:
        d[c] = cs_rank(sig.loc[d.index, c], d["season"])
    good = incremental_info(d, "good", "car_0_1_ff4")["summary"].set_index("spec")
    noise = incremental_info(d, "noise", "car_0_1_ff4")["summary"].set_index("spec")
    assert good.loc["(3) SUE + text", "t_text"] > 4
    assert abs(noise.loc["(3) SUE + text", "t_text"]) < 3
    placebo = incremental_info(d, "good", "ar_m1")["summary"].set_index("spec")
    assert abs(placebo.loc["(3) SUE + text", "t_text"]) < 3


def test_compare_combined_beats_surprise_only_for_informative_text(data):
    p, sig = data
    tab, _ = compare_models(p, sig, ["good"], "car_0_1_ff4", "validation")
    comb = tab[tab["approach"] == "combined"].set_index("text_signal")
    assert comb.loc["good", "cw_t_nw"] > 3 and comb.loc["good", "oos_r2_vs_surprise"] > 0


def test_clark_west_size_under_the_null(data):
    """
    With a pure-noise signal the Clark-West t must be centred near zero. One draw
    can exceed 2 or even 3 with only 12 validation seasons (it does for some
    seeds), so the test looks at the distribution over 15 independent noise
    signals, not at one draw.
    """
    p, _ = data
    ts = []
    for seed in range(15):
        noise = pd.DataFrame({"n": np.random.default_rng(1000 + seed).normal(size=len(p))}, index=p.index)
        tab, _ = compare_models(p, noise, ["n"], "car_0_1_ff4", "validation")
        ts.append(tab.loc[tab["approach"] == "combined", "cw_t_nw"].iloc[0])
    ts = np.array(ts)
    assert np.median(ts) < 1.5 and (ts > 2.5).mean() <= 0.2


def test_rate_shock_runs_on_unlocked_synthetic_holdout(monkeypatch):
    """The rate-shock module only runs on the holdout: exercise it on synthetic data."""
    from rate_shock import rate_shock_analysis

    monkeypatch.setenv("UNLOCK_HOLDOUT", "1")
    ev = SY.make_events(n_firms=150, years=range(2018, 2026))
    p = build_panel(ev, sample=None, include_holdout=True)
    zt = ev.set_index("event_id").loc[p["event_id"], "z_text"].to_numpy()
    p["txt"] = cs_rank(pd.Series(zt, index=p.index), p["season"])
    r = rate_shock_analysis(p, "txt", "car_0_1_ff4")
    by = r["by_regime"].set_index("regime")
    assert list(by.index) == ["pre_shock", "hiking", "post_hiking"]
    assert (by["t_text"] > 3).all()                  # the planted effect is in every regime
    joint = r["interaction"].set_index("term").loc["joint: regime interactions = 0", "p"]
    assert joint > 0.01                              # and it does not change between regimes


def test_rate_shock_is_inert_without_holdout():
    from rate_shock import rate_shock_analysis

    p = build_panel(SY.make_events(n_firms=30, years=[2015, 2018]), sample=None)
    p["txt"] = 0.0
    assert "note" in rate_shock_analysis(p, "txt")["by_regime"].columns
