# Parts 7–8: baselines, model comparisons and financial evaluation

Owner: Max Lacroix. Part of *Priced In or Memorized?* (15.S08, Fall 2026).

This folder takes the event table of parts 3–4, the cleaned texts of part 5 and the model scores of
part 6, and answers: **does the text of an earnings release tell us anything about the stock's
abnormal return that the earnings surprise does not?** It builds the simple benchmarks a language
model has to beat (task 7) and runs the statistical tests (task 8).

* **Why it is built this way:** [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
* **What it reads from the other parts:** [`docs/INTERFACES.md`](docs/INTERFACES.md)
* **What changed and when:** [`docs/CHANGELOG.md`](docs/CHANGELOG.md)

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
    regressions.py          specification ladder, two-way clustering, Fama-MacBeth
    two_by_two.py           SUE sign x text sign, cell means, CIs, contrasts
    robustness.py           sectors, size, periods, windows, placebo, samples
    rate_shock.py           2022 rate-shock regimes and interactions
    outputs.py              saves every table as csv / md / tex and logs its provenance
    synthetic.py            fake data with a planted effect (tests and demo)
    pipeline.py             the whole thing, from the command line
    tests/                  24 tests: hand-computed answers + recovery of a planted effect
  resources/lm/             put the Loughran-McDonald master dictionary here (not redistributed)
  data/                     panel, signals, cached LM counts        WRDS-derived: never committed
  results/tables/           every table, as .csv, .md and .tex      summary statistics only
  results/regressions/      full statsmodels output of every regression
  results/_synthetic/       the same outputs on synthetic data (demo)
  logs/runs.jsonl           one line per saved result: time, sha256, git commit, upstream versions
```

## Run order

| Step | What | Reads | Writes | Time |
|---|---|---|---|---|
| `00_inputs_check` | check inputs exist and match the contract | parts 3-4, 5, 6 | nothing | < 1 min |
| `07_baselines` | build every baseline signal, compare with surprise | panel, texts | `data/signals.parquet`, `results/tables/t7_*` | 1 min today; ~15 min with all texts (TF-IDF) |
| `08_evaluation` | regressions, 2x2, robustness | panel, signals, model cache | `results/tables/t8_*`, `results/regressions/*` | 1–3 min |
| `08b_rate_shock` | rate-shock regimes | holdout, after the freeze | `results/tables/t8_rate_shock_*` | < 1 min |

Or from a terminal, at the folder root:

```bash
python code/pipeline.py --step all                 # real data, validation split
```

```bash
python code/pipeline.py --step all --synthetic     # demo on synthetic data -> results/_synthetic/
```

```bash
python code/tests/run_all.py                       # the tests (synthetic data only)
```

On Colab, mount Drive and `%cd "/content/drive/MyDrive/S08 NLP/7_8_baselines_evaluation"` first.
The paths resolve from the folder itself, so the same code runs on a laptop with the Drive folder
synced. Override with `TEAM_DIR` / `STEP78_DIR` if your layout differs.

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

## Status (6 Nov 2026)

| | Status |
|---|---|
| Surprise-only baseline | **runs on the full sample**. Validation: rank IC 0.31 with the reaction (t = 22), long-short 7.5% per event |
| Loughran–McDonald | code and tests done; waiting for the dictionary file in `resources/lm/` |
| TF-IDF | code and tests done; waiting for part 5 to clean more than 2006 (needs ≥ 500 training texts per fold) |
| LLM comparison | plumbing done and tested on synthetic scores; waiting for part 6 to fill the model cache |
| Regressions, 2×2, robustness | done and tested; run on every available text signal |
| Rate shock | done; locked until the holdout freeze |

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
