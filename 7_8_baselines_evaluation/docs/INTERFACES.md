# Interfaces: what parts 7–8 read from the other parts

This is the contract between parts 7–8 and the rest of the pipeline. The readers in
`code/io_inputs.py` check it on load and stop with a clear message if a column is missing. **If you
rename or remove a column below in your part, tell Max and update this file and the reader in the
same change.**

Paths are relative to the shared `S08 NLP` folder (`TEAM_DIR`).

## 1. Events: `3_4_surprise_returns/data/events_final.parquet` (Beatriz, parts 3–4)

One row per I/B/E/S quarterly announcement. Keys: `event_id` (unique) and `accession` (the 8-K,
unique where present: checked).

| Group | Columns used | Notes |
|---|---|---|
| keys | `event_id`, `accession`, `permno`, `ticker` | `accession` joins the texts and the model scores |
| timing | `R_et`, `reaction_date`, `session_type`, `clock_status`, `R_source` | `reaction_date` defines the season and the split |
| surprise | `actual`, `consensus_median`, `surprise`, `sue_price`, `sue_disp`, `n_analysts`, `dispersion`, `price_t2` | `sue_price` is the benchmark |
| returns | `car_0_1_ff4`, `post_car_1_ff4`, `post_car_60_ff4`, `car_1_60_ff4`, `ar_m1`, `*_mkt`, `ret_prior_1m` | never recomputed here |
| firm / flags | `mktcap_t2`, `f_size`, `basis_suspect`, `timing_suspect`, `two_source`, `after_llama_cutoff`, `in_primary`, `in_text_sample`, `split` | flags, not deletions |
| versions | `surprise_version`, `returns_version`, `sample_version` | written into `logs/runs.jsonl` with every result |

Also read: `3_4_surprise_returns/cache/crsp_names.parquet` (`permno`, `namedt`, `nameenddt`,
`siccd`) for point-in-time FF12 sectors; `code/common.py` (`SPLITS`, `split_of`, `guard_holdout`)
and `code/returns.py` (`newey_west_se`, `rank_ic_by_season`), loaded by file path and not copied.

## 2. Texts: `5_data cleaning/data/clean/clean_{year}.parquet` (part 5)

| Column | Use |
|---|---|
| `accession` | key |
| `text_body` | the text the baselines read (boiler-plate sections removed by part 5) |
| `sha256_text` | part of the LM cache key: a changed text is recounted |
| `clean_version` | recorded with every result |
| `n_words_body` | length control, error analysis |
| `error` | non-empty → document dropped and counted |

Fallback while part 5 has not cleaned a year: `2_EDGAR text/results/release_texts.csv`
(`accession`, `text`), marked `text_source = fallback_release_texts`. For development only.

## 3. Model scores: `5_data cleaning/cache/model_outputs.sqlite` (part 6 writes, via part 5's `ModelCache`)

Table `outputs`, read-only from here. Columns used: `accession`, `clean_version`, `variant`,
`model_id`, `model_revision`, `prompt_id`, `score_json`, `created_at_utc`.

**The score contract (agree with part 6):**

* `score_json` is either a bare number (`0.42`) or a JSON object with a numeric `"score"` field
  (`{"score": 0.42, "label": "positive", ...}`). Extra fields are fine.
* **Higher = better news** for the firm. If a prompt gives the opposite orientation, flip it in part 6
  or register it in `io_inputs.SCORE_KEYS` with a note.
* A prompt whose number lives under another key is registered once in `io_inputs.SCORE_KEYS`
  (e.g. `{"themes_v1": "net_tone"}`).
* Unparseable scores become NaN and are counted in the provenance table.
* `variant` ∈ {`raw`, `mask`}: both are read; `placebo` is ignored here (it belongs to part 9).

Each (model, prompt, variant) becomes one signal column `llm__<model>__<prompt>__<variant>`, e.g.
`llm__llama_31_8b_instruct__tone_v1__mask`. If a document was scored twice (e.g. after a cleaning
version bump), the latest record wins; pin a version with `load_model_scores(clean_version="v2")`.
`results/tables/llm_signal_provenance.*` lists the model revisions and cleaning versions behind every
column.

## 4. Frozen holdout: `5_data cleaning/data/holdout_ids.txt` (part 5, `splits.py --freeze`)

Sorted holdout accessions. When it exists, the panel checks that its holdout events are exactly these.

## 5. What parts 7–8 produce

| Path | Content | Shareable? |
|---|---|---|
| `data/analysis_panel.parquet` | the panel, one row per event | **no** (WRDS-derived) |
| `data/signals.parquet` | every baseline signal, keyed by `event_id` | **no** |
| `data/text_features/lm_counts.parquet` | LM counts per (accession, text hash, dictionary) | no (keyed on filings) |
| `results/tables/*.{csv,md,tex}` | summary tables | yes |
| `results/regressions/*.txt` | full regression output | yes |
| `logs/runs.jsonl` | provenance of every saved result | yes |

**For part 9 (memorisation)** and **part 10 (error analysis)**: join `data/signals.parquet` to
`events_final` on `event_id`; LLM signals are re-read from the model cache with
`io_inputs.load_model_scores()`. `after_llama_cutoff` and `period` are in the panel for grouping.
