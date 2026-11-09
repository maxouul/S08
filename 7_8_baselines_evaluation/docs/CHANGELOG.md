# Changelog: parts 7–8

Add one line per change that can move a number: what, why, who. Newest first.

## 2026-11-08: v1.3, error cases, power, sample counts, figure (Max)

* `current_data.py` adds four outputs. They need no text.
  * How often the surprise sign disagrees with the reaction (`cd_surprise_errors__validation`): 41% of
    beats fall. The worked example is Conagra, 20 Dec 2018. The list naming firms stays in `data/`.
  * Events per split and after the Llama cutoff, read from the event table without outcomes
    (`cd_sample_counts`).
  * Minimum detectable effects (`cd_mde_*`). The incremental IC is clustered by quarter, and the
    contamination arms use the checkpoint's paired formula (it reproduces the checkpoint's 0.061).
  * The slide figure `cd_surprise_response__validation.png`.
* Finding to discuss: the pre-registered 0.02 incremental-IC bar is at or below the holdout's
  minimum detectable effect (0.021 at best, 0.039 realistic).
* Merged `main` into the parts 7–8 branch. Paths now come from `scripts/project_paths.py`, so run
  with `STEP78_DIR` set to this folder when `TEAM_DIR` points at the Drive copy.
* 29 tests (new: sample counts and MDE, checked against the checkpoint's example).

## 2026-11-06: v1.2, results on today's inputs, status report (Max)

* New `code/current_data.py`: the results that need no text coverage. It produces the benchmark
  regression (SUE + controls incl. B/M, two-way clustered, Fama–MacBeth), beat/meet/miss means
  with season-clustered CIs, SUE stability by sector, size, year and session, a breakdown of the
  day −1 placebo, and an exploratory LM pilot on the 115 texts. Output: `results/tables/cd_*`.
  Nothing in `pipeline.py` changes.
* The day −1 placebo is diffuse across sessions, sizes and periods (dev IC 0.052, validation
  0.036), which reads as pre-announcement drift rather than a clock error. Flagged to parts 1 and 3–4.
* 28 tests (new: beat/miss means). Status report and README rewritten for the current state.

## 2026-11-06: v1.1, LM dictionary, book-to-market, provenance fix (Max)

* Loughran–McDonald master dictionary downloaded from SRAF:
  `Loughran-McDonald_MasterDictionary_1993-2025.csv` (March 2026 update, 86,553 words, sha256
  `e2d13286…`). Git-ignored (`resources/lm/*.csv`), not redistributed. The LM baseline now runs: 115
  texts counted (105 dev, 10 validation), too few to enter the validation comparison (5% coverage rule).
* Book-to-market built from WRDS: 380,423 Compustat quarters, B/M for 205,700 of 217,549 events
  (94.6%; 11,343 with non-positive book equity), median 0.46. Fixed two pandas-3 / psycopg issues
  (mixed datetime resolutions in `merge_asof`, `ceqq` returned as `Decimal`).
* Book-to-market control: new `code/book_to_market.py` (Compustat `ceqq` of the last quarter public
  strictly before the release, `rdq` or `datadate + 90`, ≤ 365 days old, divided by `mktcap_t2` with
  the $M/$000 unit fix). `log_bm` and `bm_missing` added to `regressions.CONTROLS` and to the panel
  (fill with season median + flag, as for dispersion). Moves spec (4), (5) and Fama–MacBeth numbers
  once the B/M file exists; until then both columns are constant and dropped. The WRDS pull could
  not run non-interactively (connection waits for Duo approval): run `python code/book_to_market.py`
  once from a terminal.
* Provenance: `outputs._git_commit` looked up git from `STEP78_DIR`, which is not a git checkout on
  Drive or when overridden, so every log line had `"git": null`. It now uses the folder of the code
  itself (then `STEP78_DIR`, then `$GIT_COMMIT`), and appends `-dirty` when parts 7–8's code has
  uncommitted changes.
* Synthetic demo regenerated (specs (4), (5) and Fama–MacBeth now include log B/M; the planted
  effect is still recovered, the noise "LLM" and placebo still find nothing).
* Real-data validation run redone: surprise baseline unchanged (rank IC 0.31, t = 22).
* 27 tests pass (3 new: git lookup, B/M timing and units, panel fill-and-flag).

## 2026-11-06: v1, first version (Max)

* Repository created: panel, three baselines (surprise, Loughran–McDonald, TF-IDF), chronological
  cross-fitting with a 100-day embargo, comparison harness, regression ladder with two-way clustering
  and Fama–MacBeth, 2×2 table, robustness grid, rate-shock module, automatic saving with provenance.
* Plugged into part 3-4's `events_final.parquet` (`sample_v1`, `surprise_v2_wire_first`,
  `returns_v3_placebo_day_m1`), part 5's `clean_*.parquet` (`v2`) and model cache, and part 3-4's
  `common.py` / `returns.py` (reused, not copied).
* Small-sample p-values (Student t, seasons − 1 df) for Clark–West after the synthetic size test
  showed one noise draw at t = 3.0 with 12 seasons.
* Constant regressors are dropped with a message (`disp_missing` is constant in `in_primary`).
* 24 tests pass (`python code/tests/run_all.py`).
