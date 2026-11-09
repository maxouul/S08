# Parts 7–8: baselines, model comparisons and financial evaluation

Owner: Max Lacroix. Part of *Priced In or Memorized?* (15.S08, Fall 2026).

This folder takes the event table of parts 3–4, the cleaned texts of part 5 and the model scores of
part 6, and answers: **does the text of an earnings release tell us anything about the stock's
abnormal return that the earnings surprise does not?** It builds the simple benchmarks a language
model has to beat (task 7) and runs the statistical tests (task 8).

* **Why it is built this way:** [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
* **What it reads from the other parts:** [`docs/INTERFACES.md`](docs/INTERFACES.md)
* **What changed and when:** [`docs/CHANGELOG.md`](docs/CHANGELOG.md)
* **Where things stand, with every current number:** [`docs/status_report/status_report.pdf`](docs/status_report/status_report.pdf)

## Layout

```
7_8_baselines_evaluation/
  00_inputs_check.ipynb     what exists upstream, contract checks, the panel by split
  07_baselines.ipynb        task 7: surprise-only, Loughran-McDonald, TF-IDF; the comparison table
  08_evaluation.ipynb       task 8: regressions, 2x2 table, robustness, placebo
  08b_rate_shock.ipynb      task 8: the 2022 rate shock (runs only after the holdout freeze)
  code/                     the logic; notebooks only call it
    settings.py             paths and research-design constants (outcomes, splits' roles, dates)
    io_inputs.py            readers for parts 3-4, 5 and 6, with column contracts
    panel.py                the analysis panel: one row per event, seasons, ranks, controls
    crossfit.py             chronological out-of-sample fitting with an embargo
    baseline_surprise.py    benchmark 1: SUE (rank, linear, deciles)
    text_utils.py           light normalisation for bag-of-words models
    baseline_lm.py          benchmark 2: Loughran-McDonald tone, cached counts
    baseline_tfidf.py       benchmark 3: TF-IDF + ridge / logit, firm filter, tuning on dev
    metrics.py              IC, long-short, OOS R^2, Clark-West, hit rate, AUC
    compare.py              surprise-only vs text-only vs combined, on common support
    book_to_market.py       Compustat ceqq (pulled from WRDS) / market cap, as of the release
    regressions.py          specification ladder, two-way clustering, Fama-MacBeth
    two_by_two.py           SUE sign x text sign, cell means, CIs, contrasts
    robustness.py           sectors, size, periods, windows, placebo, samples
    rate_shock.py           2022 rate-shock regimes and interactions
    outputs.py              saves every table as csv / md / tex and logs its provenance
    synthetic.py            fake data with a planted effect (tests and demo)
    pipeline.py             the whole thing, from the command line
    current_data.py         what today's inputs support: benchmark regression, beat/miss,
                            stability, day -1 placebo, LM pilot, error cases, sample counts,
                            power (MDE), presentation figure (tables and figures cd_*)
    tests/                  29 tests: hand-computed answers + recovery of a planted effect
  resources/lm/             the Loughran-McDonald master dictionary (downloaded, git-ignored, not redistributed)
  data/                     panel, signals, cached LM counts        WRDS-derived: never committed
  results/tables/           every table, as .csv, .md and .tex      summary statistics only
  results/regressions/      full statsmodels output of every regression
  results/_synthetic/       the same outputs on synthetic data (demo)
  docs/status_report/       status report (.tex source and compiled .pdf)
  logs/runs.jsonl           one line per saved result: time, sha256, git commit, upstream versions
```

## Run order

| Step | What | Reads | Writes | Time |
|---|---|---|---|---|
| `00_inputs_check` | check inputs exist and match the contract | parts 3-4, 5, 6 | nothing | < 1 min |
| `07_baselines` | build every baseline signal, compare with surprise | panel, texts | `data/signals.parquet`, `results/tables/t7_*` | 1 min today; ~15 min with all texts (TF-IDF) |
| `08_evaluation` | regressions, 2x2, robustness | panel, signals, model cache | `results/tables/t8_*`, `results/regressions/*` | 1–3 min |
| `08b_rate_shock` | rate-shock regimes | holdout, after the freeze | `results/tables/t8_rate_shock_*` | < 1 min |
| `code/current_data.py` | results that need no text coverage, plus the LM pilot | panel, signals | `results/tables/cd_*` | < 1 min |

Or from a terminal, at the folder root:

```bash
python code/pipeline.py --step all                 # real data, validation split
python code/current_data.py                        # then: benchmark, placebo, errors, power, figure
```

```bash
python code/pipeline.py --step all --synthetic     # demo on synthetic data -> results/_synthetic/
```

```bash
python code/tests/run_all.py                       # the tests (synthetic data only)
```

The notebooks find the checkout themselves (paths come from `scripts/project_paths.py`), so Jupyter or VS Code can be started from any folder of S08. Google Colab is no longer required.
Override with `TEAM_DIR` / `STEP78_DIR` if your layout differs.

## When another part updates its output

Nothing in this folder needs editing. Rerun:

| What changed upstream | Rerun |
|---|---|
| part 5 cleaned more years of texts | `07_baselines` then `08_evaluation` (LM counts are cached: only new texts are counted) |
| part 6 added model scores to the cache | `08_evaluation` only: LLM signals are read fresh from the cache |
| part 1/3/4 corrected the clock, surprise or returns | everything (`--step all`) |
| part 5 froze the holdout | the one final run: `UNLOCK_HOLDOUT=1 python code/pipeline.py --step all --eval-split holdout`, then `08b_rate_shock` |

Every saved table records the upstream versions (`surprise_version`, `returns_version`,
`sample_version`, `clean_version`) in `logs/runs.jsonl`, so you can always tell which inputs a number
came from.

## Status (8 Nov 2026)

**In short:** everything in tasks 7–8 is built and tested. The surprise benchmark and its controls
are final on real data. Every *text* result (LM, TF-IDF, LLMs, the real 2×2 table, the text
regressions) is waiting for upstream inputs: part 5 has cleaned 2006 only (115 events with text)
and part 6 has not scored anything. When those arrive, rerunning the pipeline fills every table
without code changes. Full write-up: [`docs/status_report/status_report.pdf`](docs/status_report/status_report.pdf).

### What runs on real data today (dev 2006–16: 35,508 events; validation 2017–19: 11,375)

| Result | Validation number | Table |
|---|---|---|
| SUE rank IC with the reaction (`car_0_1_ff4`) | 0.314 (t = 22.3); long-short 7.5% per event | `t7_surprise_benchmark__validation` |
| Benchmark regression: SUE rank + controls + sector and season FE | coef 0.080 (t = 19.4), R² 0.102; controls add ≤ 0.003 R² | `cd_surprise_regressions__*` |
| Beat / meet / miss mean reaction | +1.31% / −1.48% / −3.22%; beat − miss 4.53% [4.16, 4.89] | `cd_surprise_sign__*` |
| SUE stability | IC 0.18 (utilities, energy) to 0.41 (shops, finance); 0.29–0.33 across size terciles, years and sessions | `cd_surprise_robustness__*` |
| Post-release drift | none: 60-day IC −0.011 (t = −0.7) | `t7_surprise_benchmark__validation` |
| Day −1 placebo | SUE IC 0.036 (t = 4.7); dev 0.052 (t = 9.1); diffuse across sessions, sizes and periods, which looks like pre-announcement drift, not a clock error. Flagged to parts 1 and 3–4 | `cd_placebo_diagnostic` |
| LM tone pilot (115 texts, **exploratory**) | Spearman with reaction 0.16 (p = 0.09); +0.73% per SD beyond SUE (t = 1.6) | `cd_lm_pilot_*` |
| Where SUE gets the sign wrong | 41% of beats fall, 33% of misses rise; ~1 in 3 even in the extreme quintiles. Worked example: Conagra, 20 Dec 2018 (beat, −23% in two days, −29% over 60) | `cd_surprise_errors__validation` |
| Sample sizes | primary sample: 35,508 dev / 11,375 validation / 22,145 holdout, of which 8,579 after the Llama 3.1 cutoff | `cd_sample_counts` |
| Power | minimum detectable mean IC: validation 0.028–0.052, holdout 0.021–0.039 (best case–realistic). **The 0.02 incremental-IC bar is at or below what we can detect**; contamination test MDE 0.030 at r = 0.7 | `cd_mde_*` |
| Figure for the slides | reaction and 60-day drift by SUE decile; beat / meet / miss by window | `results/figures/cd_surprise_response__validation.png` |

All numbers above use the 6 November export of part 3–4's event table. A WRDS rerun with combined
acceptance times is on `main` (8 Nov); refresh `events_final.parquet` and rerun to update them.

### Component status

| | Status |
|---|---|
| Surprise-only baseline | **done**, full sample |
| Loughran–McDonald | **done**; dictionary 1993–2025 in `resources/lm/` (git-ignored). Below the 5% coverage rule until part 5 cleans more years |
| TF-IDF | code and tests done; skips every fold until there are ≥ 500 training texts per fold (has ≤ 104) |
| LLM comparison | plumbing done and tested on synthetic scores; **waiting for part 6** |
| Regressions, 2×2, robustness | done and tested; benchmark version run on real data; text versions wait for coverage |
| Book-to-market control | **done**: Compustat `ceqq` as of the release, 94.6% coverage, median B/M 0.46; cached in `data/` (`--refresh` re-pulls WRDS, needs a Duo approval) |
| Provenance | `logs/runs.jsonl` records the git commit of the code (`-dirty` if uncommitted) |
| Rate shock | done; locked until the holdout freeze |

### Blocked on other parts

| Waiting for | Owner | Unblocks |
|---|---|---|
| Cleaned release texts 2007–2025 | part 5 | LM and TF-IDF evaluation, real text regressions and 2×2 |
| Model scores in `model_outputs.sqlite` | part 6 | the LLM rows of every table; complex vs simple models |
| Corrected release clock (55k filings) | parts 1 → 3–4 | rerun everything; may shrink the day −1 placebo |
| Holdout freeze (`splits.py --freeze`) | part 5 | the one test-set run and the 2022 rate-shock analysis |

The synthetic demo (`results/_synthetic/`) shows what every table looks like when all inputs exist.
It recovers the planted text effect and finds nothing for a pure-noise "LLM" and for the placebo.

## Data provenance and licensing

`data/` holds row-level derivatives of WRDS data (CRSP, I/B/E/S, Compustat) and stays in the team's
private Drive, like part 3-4's `data/` and `cache/`. It is git-ignored and must not be shared with
outside tools. `results/` holds summary statistics only; `outputs.save_table` refuses a table with an
`event_id`, `permno`, `accession`, `cik` or `ticker` column.

## Environment

Developed with Python 3.13 and the versions in `requirements.txt`; written to run on the course's
Python 3.12 and on Colab's pandas 2.2 as well as pandas 3.x.
