"""
The end-to-end pipeline of parts 7-8, as plain functions the notebooks call
and the command line runs.

    python code/pipeline.py --step all                    # real data, validation split
    python code/pipeline.py --step all --synthetic        # demo with a known answer
    python code/pipeline.py --step baselines              # task 7 signals only
    python code/pipeline.py --step evaluate               # task 7 comparison + task 8
    UNLOCK_HOLDOUT=1 python code/pipeline.py --step all --eval-split holdout   # once, after the freeze

Restartability
--------------
Each step writes its output under data/ (row-level, private) and the next step
reads it, so a crash or a new LLM score only reruns what is needed:

    inputs  ->  data/analysis_panel.parquet          (build_inputs)
            ->  data/signals.parquet                 (build_signals; LM counts are cached
                                                      per text in data/text_features/)
            ->  results/tables/*, results/regressions/*   (evaluate)

When part 6 adds model scores to the cache, rerun ``--step evaluate``: the LLM
signals are read fresh from the cache each time and joined to the saved
baseline signals.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import settings as S
import io_inputs as IO
import outputs as OUT
from panel import build_panel, cs_rank, describe_panel

BASELINE_TEXT_SIGNALS = ["lm_tone", "lm_negshare", f"tfidf_ridge__{S.PRIMARY_OUTCOME}", "tfidf_logit__up"]


# ---------------------------------------------------------------------------
# Step 0: inputs
# ---------------------------------------------------------------------------
def load_all(synthetic: bool = False):
    """events, texts, sectors, model scores (wide), provenance."""
    if synthetic:
        import synthetic as SY
        ev = SY.make_events()
        texts = SY.make_texts(ev)
        SY.write_lm_dictionary(S.DATA / "lm" / "Synthetic_MasterDictionary.csv")
        S.LM_DIR = S.DATA / "lm"
        db = SY.write_model_cache(ev, S.DATA / "model_outputs_synthetic.sqlite")
        scores, prov = IO.load_model_scores(db)
        sectors = ev[["event_id"]].assign(siccd=np.nan,
                                          ff12=np.random.default_rng(0).choice(list(IO.FF12), len(ev)))
        SY.make_book_to_market(ev).to_parquet(S.BOOK_TO_MARKET, index=False)
        return ev, texts, sectors, scores, prov
    ev = IO.load_events()
    texts = IO.load_texts(accessions=ev["accession"].dropna())
    sectors = IO.load_sectors(ev)
    scores, prov = IO.load_model_scores()
    return ev, texts, sectors, scores, prov


def build_inputs(ev, texts, sectors, sample=S.SAMPLE, include_holdout=False, book_to_market=None):
    from book_to_market import load_book_to_market
    bm = book_to_market if book_to_market is not None else load_book_to_market()
    panel = build_panel(ev, sectors=sectors, texts_meta=texts.drop(columns="text_body"), sample=sample,
                        include_holdout=include_holdout, frozen_holdout=IO.load_frozen_holdout(),
                        book_to_market=bm)
    OUT.set_context(sample=sample, **OUT.upstream_versions(ev, texts))
    return panel


# ---------------------------------------------------------------------------
# Step 1 (task 7): baseline signals
# ---------------------------------------------------------------------------
def build_signals(panel: pd.DataFrame, texts: pd.DataFrame, eval_splits=("dev", "validation"),
                  run_tfidf: bool = True, tfidf_cfg=None) -> tuple[pd.DataFrame, dict]:
    """
    Every baseline signal for every event of the panel, out of sample.
    Returns (signals indexed like panel with an event_id column, details).
    """
    from baseline_surprise import surprise_signals

    details = {}
    sig = pd.DataFrame({"event_id": panel["event_id"]}, index=panel.index)

    # Surprise-only (no text needed)
    for oc in S.OUTCOMES:
        s = surprise_signals(panel, oc, eval_splits)
        for c in s.columns:
            if c not in sig:
                sig[c] = s[c]

    tx = texts[texts["accession"].isin(panel["accession"])]
    # Loughran-McDonald (no training)
    try:
        from baseline_lm import lm_features
        lm = lm_features(tx)
        m = panel[["accession"]].merge(lm, on="accession", how="left")
        for c in ["lm_tone", "lm_net", "lm_negshare", "lm_unc", "n_words"]:
            sig[c] = m[c].to_numpy()
        details["lm_coverage"] = float(sig["lm_tone"].notna().mean())
    except FileNotFoundError as e:
        print("LM baseline skipped:", e)

    # TF-IDF (trained, cross-fitted)
    if run_tfidf:
        from baseline_tfidf import TfidfConfig, tfidf_signals
        t0 = time.time()
        out, det = tfidf_signals(panel, tx, S.PRIMARY_OUTCOME, tfidf_cfg or TfidfConfig(),
                                 eval_splits, return_details=True)
        for c in out.columns:
            sig[c] = out[c]
        details["tfidf"] = det
        print(f"TF-IDF done in {time.time() - t0:.0f}s")
    return sig, details


def save_signals(sig: pd.DataFrame) -> None:
    S.ensure_dirs()
    sig.to_parquet(S.SIGNALS, index=False)
    print(f"-> {S.SIGNALS} ({len(sig):,} rows, {sig.shape[1] - 1} signals)")


def load_signals_aligned(panel: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Saved baseline signals + fresh LLM scores, aligned on the panel's index."""
    sig = pd.read_parquet(S.SIGNALS) if S.SIGNALS.exists() else pd.DataFrame({"event_id": panel["event_id"]})
    sig = panel[["event_id", "accession"]].merge(sig, on="event_id", how="left")
    if len(scores) and scores.shape[1] > 1:
        sig = sig.merge(scores, on="accession", how="left")
    sig.index = panel.index
    return sig


def available_text_signals(sig: pd.DataFrame, min_coverage: float = 0.05) -> list[str]:
    """Baseline text signals and every LLM signal with enough coverage."""
    cands = [c for c in BASELINE_TEXT_SIGNALS if c in sig] + [c for c in sig if c.startswith("llm__")]
    return [c for c in cands if sig[c].notna().mean() >= min_coverage]


# ---------------------------------------------------------------------------
# Step 2 (tasks 7 and 8): comparisons, regressions, 2x2, robustness
# ---------------------------------------------------------------------------
def evaluate(panel: pd.DataFrame, sig: pd.DataFrame, eval_split: str = S.TUNE_SPLIT,
             outcomes=None, prov: pd.DataFrame | None = None) -> dict:
    from compare import compare_models
    from regressions import incremental_info, regression_table
    from robustness import robustness_grid
    from two_by_two import two_by_two

    outcomes = outcomes or S.OUTCOMES
    texts = available_text_signals(sig)
    print(f"text signals evaluated: {texts}")
    res = {"text_signals": texts}
    OUT.save_table(describe_panel(panel).reset_index(), "panel_by_split",
                   "Events in the analysis panel, by split")
    if prov is not None and len(prov):
        OUT.save_table(prov, "llm_signal_provenance", "Model, prompt and versions behind each LLM signal")
    if not texts:
        print("no text signal with coverage yet: only the surprise baseline can be evaluated")

    # ---- task 7: comparison table, per outcome ------------------------------
    comp = []
    for oc in outcomes:
        tab, _ = compare_models(panel, sig, texts, oc, eval_split)
        if len(tab):
            comp.append(tab)
            OUT.save_table(tab, f"t7_compare__{oc}__{eval_split}",
                           f"Surprise-only vs text-only vs combined, {oc}, {eval_split} split")
    res["compare"] = pd.concat(comp, ignore_index=True) if comp else pd.DataFrame()

    # ---- task 8: on the evaluation split only -------------------------------
    d = panel[panel["split"] == eval_split].copy()
    for t in texts:
        d[t + "_rk"] = cs_rank(sig.loc[d.index, t], d["season"])
    summ, robs, cells, contr = [], [], [], []
    for t in texts:
        tr = t + "_rk"
        for oc in outcomes + S.PLACEBO_OUTCOMES:
            if d[[tr, oc]].dropna().shape[0] < 300:
                continue
            ii = incremental_info(d, tr, oc)
            summ.append(ii["summary"])
            if oc == S.PRIMARY_OUTCOME:
                keep = [tr, "text_x_miss", "miss", "log_mktcap", "log_analysts", "disp_price", "ret_prior_1m_w",
                        "log_bm"]
                OUT.save_table(regression_table(ii["fits"], keep), f"t8_regressions__{t}__{oc}__{eval_split}",
                               f"Abnormal return on text rank, {t}, {oc}", index=True)
                for name, fit in ii["fits"].items():
                    tag = name.split(")")[0].strip("(")
                    OUT.save_fit(fit, f"t8__{t}__{oc}__{eval_split}__spec{tag}", f"{name}; text={t}")
                if len(ii["fama_macbeth"]):
                    OUT.save_table(ii["fama_macbeth"], f"t8_fama_macbeth__{t}__{oc}__{eval_split}",
                                   "Fama-MacBeth by season, Newey-West")
            tb = two_by_two(d.assign(**{t: sig.loc[d.index, t]}), t, oc)
            if len(tb["cells"]):
                cells.append(tb["cells"])
                contr.append(tb["contrasts"])
                if oc == S.PRIMARY_OUTCOME:
                    OUT.save_table(tb["matrix"], f"t8_2x2_matrix__{t}__{oc}__{eval_split}",
                                   f"Mean {oc} [95% CI], SUE sign x text sign ({t})", index=True)
        robs.append(robustness_grid(d, tr, S.PRIMARY_OUTCOME))

    for name, lst, cap in [("t8_incremental_summary", summ, "Text coefficient across specifications"),
                           ("t8_2x2_cells", cells, "2x2 cells: mean, SEs, CIs"),
                           ("t8_2x2_contrasts", contr, "2x2 contrasts: text effect within surprise sign"),
                           ("t8_robustness", robs, "Robustness grid")]:
        if lst:
            tab = pd.concat(lst, ignore_index=True)
            res[name] = tab
            OUT.save_table(tab, f"{name}__{eval_split}", cap)

    # ---- the surprise benchmark itself, on every outcome ----------------------
    from metrics import evaluate_signal
    sb = pd.DataFrame([evaluate_signal(d, "sue_rank", oc) for oc in outcomes + S.PLACEBO_OUTCOMES])
    OUT.save_table(sb, f"t7_surprise_benchmark__{eval_split}", "SUE rank against every outcome")
    res["surprise_benchmark"] = sb
    return res


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=["baselines", "evaluate", "all"], default="all")
    ap.add_argument("--synthetic", action="store_true", help="demo data with a known answer")
    ap.add_argument("--sample", default=S.SAMPLE)
    ap.add_argument("--eval-split", default=S.TUNE_SPLIT, choices=["dev", "validation", "holdout"])
    ap.add_argument("--no-tfidf", action="store_true", help="skip the (slow) TF-IDF baseline")
    args = ap.parse_args()

    if args.synthetic:
        S.redirect_outputs("synthetic")
    S.ensure_dirs()
    include_holdout = args.eval_split == "holdout"
    ev, texts, sectors, scores, prov = load_all(args.synthetic)
    if args.synthetic and include_holdout:
        os.environ["UNLOCK_HOLDOUT"] = "1"        # synthetic data has no real holdout to protect
    panel = build_inputs(ev, texts, sectors, args.sample, include_holdout)
    panel.to_parquet(S.PANEL, index=False)
    print(describe_panel(panel).to_string())

    splits = ("dev", "validation") + (("holdout",) if include_holdout else ())
    if args.step in ("baselines", "all"):
        sig, det = build_signals(panel, texts, splits, run_tfidf=not args.no_tfidf)
        save_signals(sig)
        if "tfidf" in det and len(det["tfidf"]["tuning"]):
            OUT.save_table(det["tfidf"]["tuning"], "t7_tfidf_tuning", "TF-IDF penalty, dev pseudo-OOS IC")
            if len(det["tfidf"]["top_terms"]):
                OUT.save_table(det["tfidf"]["top_terms"], "t7_tfidf_top_terms", "Largest TF-IDF ridge weights")
    if args.step in ("evaluate", "all"):
        sig = load_signals_aligned(panel, scores)
        evaluate(panel, sig, args.eval_split, prov=prov)
    print(f"done. tables in {S.TABLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
