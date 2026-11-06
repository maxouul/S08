# Changelog: parts 7–8

Add one line per change that can move a number: what, why, who. Newest first.

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
