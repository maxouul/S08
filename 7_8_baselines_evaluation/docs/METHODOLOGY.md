# Methodology: parts 7 and 8

*Priced In or Memorized?* (15.S08, Fall 2026). Owner of parts 7–8: Max Lacroix.

This document explains **why** the baselines and tests are built the way they are. The code
docstrings explain **how**. If you change a choice below, change it here in the same commit and add
a line to `CHANGELOG.md`.

---

## 1. The question parts 7–8 answer

Every text model (FinBERT, ChronoGPT, Llama 3.1 from part 6, and our baselines) produces one
number per earnings release: a *signal*, higher meaning better news. Parts 7–8 ask:

1. **Does the signal line up with the stock's abnormal return?** (rank IC, long-short spread)
2. **Does it add anything beyond the earnings surprise?** (incremental regressions, the 2×2 table,
   Clark–West tests against a surprise-only model)
3. **Does a large language model beat simple text models?** (LM dictionary, TF-IDF)
4. **Is the result robust?** (sectors, size, periods, event windows, sample definitions, a placebo,
   the 2022 rate shock)

Question 2 is the important one. Most of what any text model "knows" about the reaction is
whether the firm beat consensus, and I/B/E/S already gives us that for free. A text model is only
interesting if it moves returns **holding the surprise fixed**.

## 2. Splits and the holdout lock

| Split | Years | Role here |
|---|---|---|
| `dev` | 2006–2016 | fit the trained baselines (TF-IDF), choose their penalty using the dev pseudo-out-of-sample predictions |
| `validation` | 2017–2019 | **the comparison set until the freeze**: every model is compared here once |
| `excluded_2020` | 2020 | never used (COVID: timing and returns abnormal for reasons unrelated to text) |
| `holdout` | 2021–2025 | the frozen test set: looked at **once**, after models, prompts and tests are frozen |

The split rule is not restated here. It comes from `3_4_surprise_returns/code/common.py`
(`SPLITS`, `split_of`), so the texts, the returns and the evaluation always use the same split.

**Holdout lock.** `panel.build_panel` passes every table through part 3-4's `guard_holdout`, which
drops 2021–2025 unless `UNLOCK_HOLDOUT=1`. Opening it here takes *two* keys,
`include_holdout=True` **and** the environment variable, so it cannot happen through a default
argument. When part 5 freezes the holdout (`splits.py --freeze` → `holdout_ids.txt`), the panel checks
that the holdout events are exactly the frozen list.

Final protocol (to be pre-registered):
1. Everything in this repository is run and frozen on dev / validation.
2. Part 5 runs `splits.py --freeze`.
3. One run: `UNLOCK_HOLDOUT=1 python code/pipeline.py --step all --eval-split holdout`.
4. Results are reported as they come out; no model, prompt or test is changed afterwards.

## 3. The analysis panel

One row per event, built once by `panel.build_panel` from part 4's `events_final.parquet`, so that
every baseline and every test sees the same events.

* **Sample:** `in_primary` by default (3+ analysts, price > $5, an 8-K with the release text, wire and
  I/B/E/S agreeing on the session, a reaction return). `in_text_sample` (any timing status) is a
  robustness sample.
* **Seasons:** the calendar quarter of the reaction date. Announcements cluster in four seasons a
  year and events of a season share market-wide shocks. All cross-sectional statistics (ranks,
  deciles, ICs, 2×2 cells) are computed **within season**; all inference treats **seasons**, not
  events, as the independent units (Newey–West across seasons, clustering by season).
* **Ranks:** every signal enters the tests as its within-season rank in [−0.5, 0.5]. Raw outputs
  live on different scales (an LLM's 1–5 grade, LM tone in [−1, 1], a predicted return); ranks put
  them on one scale, are immune to outliers, and a coefficient on a rank reads directly as "return
  difference between the most and least positive release of the quarter".
* **Winsorising:** returns and continuous controls are winsorised at 1%/99% *within season* for OLS
  only. Rank statistics use the raw values.
* **No look-ahead check:** the panel asserts no reaction session precedes its release time.

## 4. Baselines (task 7)

### 4.1 Surprise only (`baseline_surprise.py`)
SUE as defined by part 3: (actual − median point-in-time consensus) / price at t−2. Three versions:
the raw within-season rank, a linear fit, and **10 SUE-decile dummies**. The decile version is the
benchmark everywhere, because the price response to SUE is S-shaped (part 4's
`sue_deciles_dev.csv`: −4.9% in decile 1, about zero in deciles 4–6). A linear SUE benchmark would
let a text model "add information" just by capturing that curvature.

### 4.2 Loughran–McDonald dictionary (`baseline_lm.py`)
LM (2011) built their word lists from 10-Ks because general lexicons misclassify financial words
("liability", "tax", "capital"). It is the standard text baseline in finance and accounting; it is
not trained, so it cannot overfit or leak. Main signal: `lm_tone = (pos − neg) / (pos + neg)`;
also the negative-word share (Tetlock 2007), sign-flipped so that higher means better. LM negation
rule for positive words (three-word window). The dictionary file is not redistributed: see
`resources/lm/README.md`.

### 4.3 TF-IDF (`baseline_tfidf.py`)
The strongest *simple* model: it learns which words matter from past releases and returns.
Uni- and bigrams, sublinear tf, l2 rows, numbers and years replaced by placeholders, and a
**firm filter** (a term must appear in releases of ≥ 20 distinct firms of the training window), so the
model learns news words, not firm names. Ridge regression of the winsorised reaction (and an L2
logit of its sign). Vocabulary, idf weights and filter are fitted **inside each training window**.

### 4.4 Chronological cross-fitting (`crossfit.py`)
Any trained signal must be out of sample for every event it scores. Expanding window by year in
dev (fit on years < y, predict y); dev → validation; dev + validation → holdout. A **100-day
embargo** removes training events whose 60-day return window overlaps the evaluation period.
Random K-fold is not used: it trains on the future and leaks firm boiler-plate across folds.
Hyper-parameters are chosen on the dev pseudo-out-of-sample predictions only, so the validation
years stay a clean comparison set.

### 4.5 Surprise-only vs text-only vs combined (`compare.py`)
For each text signal T, three OLS stacks, all fitted on past data only:

```
surprise-only   y = SUE deciles
text-only       y = a + c · rank(T)
combined        y = SUE deciles + c · rank(T)
```

Evaluated on **common support** (events every compared signal covers), with the same metrics
(`metrics.py`): within-season rank IC with Newey–West t, top-minus-bottom decile spread,
out-of-sample R² against zero, out-of-sample R² of combined **relative to** surprise-only, hit rate,
AUC, and the Clark–West (2007) test of combined vs surprise-only (the correct test for nested
models; plain MSPE comparisons are biased against the larger model). LLM signals from part 6 enter
exactly the same way: nothing in the comparison is model-specific.

## 5. Statistical tests (task 8)

### 5.1 Regressions (`regressions.py`)
```
(1) y = SUE deciles + season FE
(2) y = text + season FE
(3) y = SUE deciles + text + season FE                 ← the main test
(4) (3) + log size, log #analysts, dispersion/price, prior 1-month return + FF12 FE
(5) (4) + text × 1[miss]
(FM) (4) estimated season by season, Fama–MacBeth with Newey–West
```
Two-way clustered standard errors by firm and season (Cameron–Gelbach–Miller 2011; Petersen 2009).
Fama–MacBeth is an independent second inference method and shows how stable the effect is across
seasons (`share_positive`).

### 5.2 The 2×2 table (`two_by_two.py`)
Rows: beat (surprise > 0) / miss (< 0). Exact meets are excluded and counted. Columns: text above /
below its **within-season median** (raw zero is not neutral for LM tone or for an LLM's grade, and a
median split gives equal-sized halves). Cell means, clustered (by season) and naive standard errors,
95% CIs, and contrasts: text effect within beats, within misses, their average, the
difference-in-differences, and "beat with bad text vs miss with good text".

### 5.3 Robustness (`robustness.py`)
The same two statistics (text coefficient in spec (3), rank IC) on: FF12 sectors; within-season size
terciles and part 4's NYSE-20th-percentile flag; periods; before/after the Llama 3.1 cutoff; every
event window (reaction, +1, +60, close-to-+60, market-adjusted); the **day −1 placebo**; without
`basis_suspect` or `timing_suspect` events; SUE scaled by dispersion; and the `in_text_sample`.

**Reading the placebo.** SUE itself has a small positive rank IC with the day −1 return on validation
(0.036, t ≈ 4.7), which fits the known pre-announcement drift. A text signal correlated with SUE would
inherit it, so the placebo regressions control for SUE deciles. A text coefficient on `ar_m1` that is
still significant *after* controlling for SUE means a timing problem or leakage.

### 5.4 The 2022 rate shock (`rate_shock.py`)
Regimes from FOMC dates: pre (2021 to 15 Mar 2022), hiking (16 Mar 2022 to 26 Jul 2023), post. Rank
IC and text coefficient by regime, then a pooled model with text × regime and SUE-decile × regime
interactions and a joint Wald test. **All three regimes are in the holdout**, so this runs only after
the freeze. It also overlaps the Llama cutoff (Dec 2023), which is why part 9 needs it.

## 6. Inference with few seasons: a caution

The validation split has **12 seasons**. Newey–West or season-clustered standard errors estimated from
12 units are noisy and somewhat optimistic. In the synthetic tests a pure-noise signal gets a
Clark–West t above 2.5 in a few percent of draws, and once reached 3.0. So:

* p-values for season-level tests use a Student-t with (seasons − 1) degrees of freedom;
* treat |t| < 3 on validation as weak evidence;
* the decisive evidence is the holdout (20 seasons) together with agreement across methods (two-way
  clustering, Fama–MacBeth, Clark–West, the 2×2).

## 7. Within-season ranks are not a trading strategy

Ranking an event against the other events of its quarter uses releases published later that quarter.
That is standard for cross-sectional tests and harmless for the statistical question. It is not a
tradable rule. A trading version would rank against the trailing distribution of the signal (e.g. the
previous four seasons). If we report a portfolio return, it must use that version.

## 8. Open items

| Item | Waiting on | Effect here |
|---|---|---|
| Release texts for 2007–2025 | part 5 (`clean_release.py` has run on 2006 only) | LM and TF-IDF cover only ~115 events today; TF-IDF needs ≥ 500 training docs per fold |
| LM master dictionary file | someone downloads it into `resources/lm/` | LM baseline skipped until then |
| LLM scores | part 6 (model cache is empty) | appear automatically as `llm__<model>__<prompt>__<variant>` signals |
| Corrected release clock | part 1 → part 3-4 rerun | rerun `pipeline.py`; nothing to change here |
| Book-to-market control | part 3-4 to pull Compustat `ceqq` | add to `regressions.CONTROLS` |
| FactSet consensus | part 3 | an alternative `sue_*` column can be added to the robustness grid |
| Holdout freeze | part 5 `splits.py --freeze` | unlocks the test-set run and the rate-shock analysis |
