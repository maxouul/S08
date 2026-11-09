"""
Unit tests with hand-computed answers. Run from the repository root:

    python -m pytest code/tests -q         (or: python code/tests/run_all.py)

They use only synthetic data, never the WRDS-derived files.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings as S  # noqa: E402

S.redirect_outputs("tests")

from crossfit import chronological_folds  # noqa: E402
from metrics import clark_west, newey_west_se, oos_r2, season_mean_test  # noqa: E402
from panel import cs_rank, quantile_bucket  # noqa: E402
from text_utils import mask_numbers, words  # noqa: E402


# ---------------------------------------------------------------- transforms
def test_cs_rank_bounds_ties_and_nan():
    x = pd.Series([1.0, 2.0, 2.0, 3.0, np.nan, 10.0, 20.0])
    g = pd.Series(["a", "a", "a", "a", "a", "b", "b"])
    r = cs_rank(x, g)
    assert r.iloc[0] == -0.5 and r.iloc[3] == 0.5          # extremes of group a
    assert r.iloc[1] == r.iloc[2] == 0.0                     # ties share the middle rank
    assert np.isnan(r.iloc[4])                               # NaN stays NaN
    assert list(r.iloc[5:]) == [-0.5, 0.5]


def test_quantile_bucket_equal_sizes():
    x = pd.Series(np.arange(100.0))
    b = quantile_bucket(x, pd.Series(0, index=x.index), 10)
    assert b.value_counts().eq(10).all() and b.min() == 1 and b.max() == 10


# ---------------------------------------------------------------- text
def test_mask_numbers_removes_years_amounts_percentages():
    t = mask_numbers("In 2008 EPS was $1.23, up 12% from 1,050 units")
    assert "2008" not in t and "1.23" not in t and "12" not in t
    assert {"_year_", "_usd_", "_pct_", "_num_"} <= set(t.split())


def test_words_lowercase_letters_only():
    assert words("Net LOSS of $3.2m; didn't grow.") == ["net", "loss", "of", "m", "didn't", "grow"]


def test_lm_negation_only_for_positive_words(tmp_path):
    from baseline_lm import CATEGORIES, count_text, load_dictionary

    rows = [{"Word": w.upper(), **{c: 0 for c in CATEGORIES}} for w in ["strong", "loss"]]
    rows[0]["Positive"] = 2009
    rows[1]["Negative"] = 2009
    p = tmp_path / "Mini_MasterDictionary.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    lists, _ = load_dictionary(p)
    c = count_text("results were not very strong. margins were strong. no loss", lists)
    assert c["Positive"] == 1 and c["n_pos_negated"] == 1   # "not very strong" is negated
    assert c["Negative"] == 1                                # negated negatives are kept


def test_lm_removed_words_are_not_members(tmp_path):
    from baseline_lm import CATEGORIES, load_dictionary

    rows = [{"Word": "GOOD", **{c: 0 for c in CATEGORIES}}]
    rows[0]["Positive"] = -2012                              # removed from the list in 2012
    p = tmp_path / "X_MasterDictionary.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    lists, _ = load_dictionary(p)
    assert "good" not in lists["Positive"]


# ---------------------------------------------------------------- chronology
def _toy_panel():
    d = pd.date_range("2006-01-10", "2019-12-20", freq="21D")
    common = S.common_3_4()
    p = pd.DataFrame({"reaction_date": d})
    p["year"] = p["reaction_date"].dt.year
    p["split"] = common.split_of(p["reaction_date"]).astype(object)
    return p


@pytest.mark.parametrize("es", ["dev", "validation"])
def test_folds_never_train_on_the_future(es):
    p = _toy_panel()
    for f in chronological_folds(p, es, embargo_days=100):
        tr_max = p.loc[f.train_idx, "reaction_date"].max()
        te_min = p.loc[f.test_idx, "reaction_date"].min()
        assert tr_max < te_min - pd.Timedelta(days=100), f.label
        assert set(p.loc[f.test_idx, "split"]) == {es}


def test_holdout_fold_trains_on_dev_and_validation_only():
    p = _toy_panel()
    p = pd.concat([p, pd.DataFrame({"reaction_date": pd.to_datetime(["2022-05-01"]), "year": [2022],
                                    "split": ["holdout"]})], ignore_index=True)
    (f,) = chronological_folds(p, "holdout")
    assert set(p.loc[f.train_idx, "split"]) <= {"dev", "validation"}


# ---------------------------------------------------------------- inference
def test_newey_west_matches_part4():
    x = np.random.default_rng(0).normal(size=50)
    r34 = S.returns_3_4()
    assert np.isclose(newey_west_se(x, 2), r34.newey_west_se(x, 2))
    # with no autocorrelation lags it is the plain standard error (ddof = 0)
    assert np.isclose(newey_west_se(x, 0), x.std(ddof=0) / np.sqrt(len(x)))


def test_season_mean_test_uses_seasons_as_units():
    # 4 seasons, each with 100 identical values: the mean of season means is 2.5
    v = pd.Series(np.repeat([1.0, 2.0, 3.0, 4.0], 100))
    s = pd.Series(np.repeat(["q1", "q2", "q3", "q4"], 100))
    r = season_mean_test(v, s)
    assert r["seasons"] == 4 and np.isclose(r["mean"], 2.5)


def test_oos_r2():
    y = np.array([1.0, -1.0, 2.0, -2.0])
    assert oos_r2(y, y) == 1.0
    assert oos_r2(y, np.zeros(4)) == 0.0
    assert oos_r2(y, -y) < 0


def test_clark_west_detects_useful_and_ignores_noise():
    rng = np.random.default_rng(1)
    n = 6000
    x, noise = rng.normal(size=n), rng.normal(size=n)
    y = pd.Series(0.3 * x + rng.normal(size=n))
    season = pd.Series(np.repeat(np.arange(30), n // 30))
    small = pd.Series(np.zeros(n))
    assert clark_west(y, small, pd.Series(0.3 * x), season)["cw_t_nw"] > 5
    assert abs(clark_west(y, small, pd.Series(0.01 * noise), season)["cw_t_nw"]) < 3


def test_two_by_two_cell_means_exact():
    from two_by_two import two_by_two

    # 4 seasons x 4 cells x 25 events, each cell with a known constant mean
    means = {(1, 1): 0.03, (1, -1): 0.01, (-1, 1): -0.01, (-1, -1): -0.03}
    rows = []
    for s in range(4):
        for (sue, txt), m in means.items():
            for k in range(25):
                rows.append({"season": f"q{s}", "permno": k, "surprise": 0.05 * sue,
                             "sig": txt * (1 + k), "y": m + (0.001 if k % 2 else -0.001)})
    df = pd.DataFrame(rows)
    out = two_by_two(df, "sig", "y")
    c = out["cells"].set_index("cell")["mean"]
    assert np.isclose(c["beat_textpos"], 0.03, atol=1e-4) and np.isclose(c["miss_textneg"], -0.03, atol=1e-4)
    k = out["contrasts"].set_index("contrast")["estimate"]
    assert np.isclose(k["average text effect"], 0.02, atol=1e-4)
    assert np.isclose(k["diff-in-diff (beat effect - miss effect)"], 0.0, atol=1e-4)


def test_surprise_sign_means_and_difference():
    from current_data import surprise_sign

    rows = []
    for s in range(4):
        for sur, m in ((0.02, 0.03), (0.0, 0.0), (-0.02, -0.02)):
            for k in range(10):
                rows.append({"split": "dev", "season": f"q{s}", "surprise": sur,
                             "y": m + (0.001 if k % 2 else -0.001)})
    t = surprise_sign(pd.DataFrame(rows), "dev", "y", save=False).set_index("group")
    assert np.isclose(t.loc["beat", "mean_car"], 0.03) and np.isclose(t.loc["miss", "mean_car"], -0.02)
    assert t.loc["meet", "n"] == 40
    assert np.isclose(t.loc["beat - miss", "mean_car"], 0.05)


def test_sample_counts_and_mde_reproduce_checkpoint_example():
    from current_data import mde_tables, sample_counts

    rng = np.random.default_rng(0)
    rows = []
    for split, years, n_per_q in (("validation", [2017, 2018, 2019], 60), ("holdout", [2021, 2024], 40)):
        for y in years:
            for q in (1, 2, 3, 4):
                for k in range(n_per_q):
                    rows.append({"split": split, "reaction_date": f"{y}-{3 * q:02d}-15",
                                 "in_primary": True, "in_text_sample": True,
                                 "after_llama_cutoff": y >= 2024})
    ev = pd.DataFrame(rows)
    c = sample_counts(ev).set_index("split")
    assert c.loc["validation", "in_primary"] == 720 and c.loc["validation", "quarters"] == 12
    assert c.loc["holdout", "primary_after_llama_cutoff"] == 160 and c.loc["holdout", "quarters"] == 8

    p = ev[ev["split"] == "validation"].copy()
    p["season"] = pd.to_datetime(p["reaction_date"]).dt.to_period("Q").astype(str)
    p["sue_rank"] = rng.uniform(-0.5, 0.5, len(p))
    p["car_0_1_ff4"] = 0.05 * p["sue_rank"] + rng.normal(0, 0.05, len(p))
    mde_ic, pair = mde_tables(p, sample_counts(ev), n_sim=5)
    ex = pair[(pair["comparison"] == "checkpoint example") & (pair["r"] == 0.7)].iloc[0]
    assert np.isclose(ex["mde_80"], 0.061, atol=0.001)          # the checkpoint's own number
    noise = mde_ic[mde_ic["quarterly_ic_sd"].str.startswith("noise")].set_index("split")
    assert np.isclose(noise.loc["validation", "sd"], 1 / np.sqrt(60), rtol=0.5)


def test_two_by_two_excludes_meets():
    from two_by_two import two_by_two

    df = pd.DataFrame({"season": ["q"] * 8, "permno": range(8), "surprise": [0.0] * 8,
                       "sig": range(8), "y": np.zeros(8)})
    out = two_by_two(df, "sig", "y")
    assert out["excluded"]["meets (surprise == 0)"] == 8 and out["cells"].empty


# ---------------------------------------------------------------- inputs
def test_model_score_parsing(tmp_path):
    import io_inputs as IO

    db = tmp_path / "m.sqlite"
    con = sqlite3.connect(db)
    con.execute("""create table outputs (key text primary key, accession text, sha256_text text,
        clean_version text, variant text, model_id text, model_revision text, prompt_id text,
        raw_response text, score_json text, created_at_utc text, duration_s real)""")
    recs = [("1", "A", "h", "v2", "raw", "meta-llama/Llama-3.1-8B-Instruct", "r", "p1", "", "0.5", "t1", 0),
            ("2", "B", "h", "v2", "raw", "meta-llama/Llama-3.1-8B-Instruct", "r", "p1", "",
             json.dumps({"score": -1}), "t1", 0),
            ("3", "C", "h", "v2", "raw", "meta-llama/Llama-3.1-8B-Instruct", "r", "p1", "", "\"oops\"", "t1", 0),
            # same doc re-scored later: the later record wins
            ("4", "A", "h", "v2", "raw", "meta-llama/Llama-3.1-8B-Instruct", "r", "p1", "", "0.9", "t2", 0)]
    con.executemany("insert into outputs values (?,?,?,?,?,?,?,?,?,?,?,?)", recs)
    con.commit()
    con.close()
    wide, prov = IO.load_model_scores(db)
    col = IO.signal_name("meta-llama/Llama-3.1-8B-Instruct", "p1", "raw")
    w = wide.set_index("accession")[col]
    assert w["A"] == 0.9 and w["B"] == -1.0 and np.isnan(w["C"])
    assert prov.loc[0, "n_scored"] == 3 and prov.loc[0, "n_parsed"] == 2


def test_ff12_mapping():
    import io_inputs as IO

    s = IO.sic_to_ff12(pd.Series([6021, 7372, 2834, 4911, 1311, 9999, np.nan]))
    assert list(s) == ["Money", "BusEq", "Hlth", "Utils", "Enrgy", "Other", "Unknown"]


def test_save_table_refuses_row_level_data():
    import outputs as OUT

    with pytest.raises(ValueError):
        OUT.save_table(pd.DataFrame({"event_id": ["x"], "v": [1]}), "should_fail")


def test_git_commit_found_from_code_dir_even_if_step78_dir_is_not_a_repo(monkeypatch, tmp_path):
    import subprocess

    import outputs as OUT

    here = Path(__file__).resolve().parent
    if subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True).returncode:
        pytest.skip("code is not in a git checkout")
    monkeypatch.setattr(S, "STEP78_DIR", tmp_path)          # e.g. the Drive folder
    OUT._GIT_CACHE.clear()
    try:
        g = OUT._git_commit()
    finally:
        OUT._GIT_CACHE.clear()
    assert g and len(g.split("-")[0]) >= 7


# ---------------------------------------------------------------- book-to-market
def test_book_to_market_uses_only_quarters_public_before_release():
    from book_to_market import compute_book_to_market

    ev = pd.DataFrame({
        "event_id": ["a", "b", "c", "d", "e"],
        "gvkey": ["001", "001", "002", "003", "004"],
        "R_et": ["2015-04-20 07:00:00-04:00",   # Q1 release: rdq of Q1 is this day -> use Q4
                 "2015-07-20 07:00:00-04:00",   # Q2 release -> Q1 (rdq 2015-04-20)
                 "2015-04-20 16:05:00-04:00",   # rdq missing -> datadate + 90 rule
                 "2015-04-20 07:00:00-04:00",   # negative book equity -> NaN
                 "2015-04-20 07:00:00-04:00"],  # only a stale quarter -> NaN
        "mktcap_t2": [2_000_000.0, 2_000_000.0, 500_000.0, 1_000_000.0, 1_000_000.0],  # $000s
    })
    fq = pd.DataFrame({
        "gvkey": ["1", "1", "1", "2", "2", "3", "4"],
        "datadate": pd.to_datetime(["2014-12-31", "2015-03-31", "2015-06-30",
                                    "2014-12-31", "2015-03-31", "2014-12-31", "2013-12-31"]),
        "rdq": pd.to_datetime(["2015-02-01", "2015-04-20", "2015-07-20",
                               None, None, "2015-02-01", "2014-02-01"]),
        "ceqq": [1000.0, 1100.0, 1200.0, 100.0, 999.0, -50.0, 800.0],          # $ millions
    })
    out = compute_book_to_market(ev, fq).set_index("event_id")
    assert out.loc["a", "bm"] == pytest.approx(1000 * 1000 / 2_000_000)        # 0.5, not Q1's 0.55
    assert out.loc["b", "bm"] == pytest.approx(1100 * 1000 / 2_000_000)
    # 2014-12-31 + 90 = 2015-03-31 < 2015-04-20 is usable; 2015-03-31 + 90 is not
    assert out.loc["c", "bm"] == pytest.approx(100 * 1000 / 500_000)
    assert out.loc["c", "bm_source"] == "datadate+90"
    assert np.isnan(out.loc["d", "bm"]) and out.loc["d", "ceqq"] == -50
    assert np.isnan(out.loc["e", "bm"])


def test_panel_log_bm_filled_and_flagged():
    import synthetic as SY
    from panel import build_panel

    ev = SY.make_events(n_firms=40, years=[2015, 2016])
    bm = SY.make_book_to_market(ev)
    assert bm["bm"].notna().mean() > 0.8
    p = build_panel(ev, sample=None, book_to_market=bm)
    assert p["log_bm"].notna().all()
    assert set(p["bm_missing"].unique()) == {0.0, 1.0}
    ok = p["bm_missing"] == 0
    assert np.corrcoef(p.loc[ok, "log_bm"], np.log(p.loc[ok, "bm"]))[0, 1] > 0.95  # up to winsorising
    p0 = build_panel(ev, sample=None)                     # no B/M file: control is inert
    assert (p0["bm_missing"] == 1).all() and (p0["log_bm"] == 0).all()


# ---------------------------------------------------------------- holdout lock
def test_build_panel_holdout_lock(monkeypatch):
    import synthetic as SY
    from panel import build_panel

    ev = SY.make_events(n_firms=30, years=[2015, 2018, 2022])
    monkeypatch.delenv("UNLOCK_HOLDOUT", raising=False)
    p = build_panel(ev, sample=None)
    assert (p["split"] == "holdout").sum() == 0                  # dropped by default
    with pytest.raises(PermissionError):
        build_panel(ev, sample=None, include_holdout=True)       # one key is not enough
    monkeypatch.setenv("UNLOCK_HOLDOUT", "1")
    p = build_panel(ev, sample=None, include_holdout=True)
    assert (p["split"] == "holdout").sum() > 0
