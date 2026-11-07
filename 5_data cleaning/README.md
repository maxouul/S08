# Part 5: data cleaning, caching and dataset preparation

Part of *Priced In or Memorized?* (15.S08, Fall 2026).

This folder turns the 8-K documents chosen in parts 1–2 into the inputs every later part reads: one cleaned text per filing, the pieces each model reads, a cache of model outputs, duplicate flags, and the chronological splits with the frozen test set. Everything is keyed by `accession` and covers every filing. Samples are filters applied downstream, never deletions here.

## Layout

```
5_data cleaning/
  00_setup_check.ipynb    environment record, input check, unit tests of the cleaning
  01_raw_cache.ipynb      the only notebook that talks to EDGAR: raw bytes of each document, once
  02_clean_prepare.ipynb  clean, entities, tokens and pieces, duplicates, model cache, splits, checks
  code/                   the logic; notebooks only call it
    settings.py           every path, and where the inputs of parts 1-4 are found
    cache_raw.py          5.1 raw bytes of the chosen document -> cache/raw/raw_YYYY.zip
    clean_release.py      5.2 clean(raw_bytes, filename), versioned by CLEAN_VERSION
    test_clean_release.py unit tests of clean() on built documents (no data, no network)
    entities.py           5.3 names and tickers to mask, per filing (WRDS-derived)
    token_stats.py        5.4 tokens per model and the pieces cut to fit each window
    dedup.py              5.5 exact, near-duplicate and close-event flags
    model_cache.py        5.6 SQLite cache of model outputs (ModelCache)
    splits.py             5.7 split draft and the one-time freeze of the holdout
    manifest.py           5.8 manifest.json: every output, rows, sha256, script
  cache/                  raw EDGAR bytes, model outputs                 private, git-ignored
  data/                   cleaned texts, pieces, entities, flags, splits  private, git-ignored
  results/                summary tables only
  logs/                   environment records
```

## Run order

| Step | Reads | Writes | Time |
|---|---|---|---|
| `00_setup_check` | the inputs below | `logs/env_*.json` | 1 min |
| `01_raw_cache` (200-event test, then `RUN_FULL = True` once) | `files csv/extraction_report.csv`, EDGAR | `cache/raw/raw_YYYY.zip`, `cache/raw_manifest.parquet` | 3 h 44 for the 116,983 documents (6–7 November); reruns fetch only what is missing |
| `02_clean_prepare` | `cache/raw/`, parts 3–4 | everything in `data/`, `cache/model_outputs.sqlite`, `results/token_quantiles.csv`, `manifest.json` | cleaning: 17 min for 2006 on a 4-core laptop (3 workers); later years are faster per document, about 6 h for all years (estimate); the other steps not yet timed |
| `python code/splits.py --freeze` | `data/splits_draft.parquet`, part 4's `events_final.parquet` | `data/holdout_ids.txt`, `.sha256`, `holdout_freeze.json` | once, at the end |

Every step can be stopped and rerun. `cache_raw.py` commits its archives every 500 documents and retries failed downloads on the next run. `clean_release.py` skips a year already cleaned with the current `CLEAN_VERSION` and covering every document of its archive. `token_stats.py` skips a year whose pieces match the texts and the window settings. `ModelCache` never pays for the same call twice.

Inputs (read, never changed):

- `files csv/` (or `1_2nov/nov2_data` when present): `extraction_report.csv` (accession, cik, filename of the document `fetch_releases.py` chose) and `event_table.csv` (company names).
- `scripts/fetch_releases.py`: download and choice rules, reused.
- `3_4_surprise_returns/`: `data/events_final.parquet` (event keys, reaction dates, splits, sample flags), `data/clock_provisional.csv` (permno and dates of every filing), `data/trading_calendar.csv`, `cache/crsp_names.parquet`, `cache/link_ibes_crsp.parquet`, and `SPLITS` in `code/common.py`.

Set `TEAM_DIR` when the private data live outside the checkout. The Llama 3.1 tokenizer is gated: set `HF_TOKEN`, put `HF_TOKEN=...` in the repository's `.env` (git-ignored), or run `huggingface-cli login` once.

## What the next parts read

| Output | Content | Read by |
|---|---|---|
| `data/clean/clean_YYYY.parquet` | one row per filing (year of the filing date): `text_full`, `text_body` (without safe harbor, About, contacts, conference call), `sentences` with their section, `fin_tables` (financial tables kept aside), word counts, removal counts, `earnings_like`, `sha256_text`, `clean_version`, `error` | parts 6–8 (`accession`, `text_body`, `sha256_text`, `clean_version`, `n_words_body`, `error`) |
| `data/chunks/<model>/chunks_YYYY.parquet` | the pieces each model reads: sentence ranges of `text_body`, with what they depend on; `token_stats.pieces()` rebuilds their text | part 6 |
| `cache/model_outputs.sqlite` | table `outputs`, one row per model call; table `configs`, the prompt configurations | written by part 6 through `ModelCache`; read by parts 7–9 |
| `data/entities.parquet` | EDGAR, CRSP and I/B/E/S names and tickers per filing, for identity masking | masking (parts 6 and 9) |
| `data/dedup_flags.parquet` | `accession`, `dup_group`, `flag_type` (`exact`, `near`, `close_events`) | sample filters, part 10 |
| `data/splits_draft.parquet` | split of every filing, date used and its source, part 4's sample flags | parts 6–10 |
| `data/holdout_ids.txt` | the frozen test set (sorted accessions) | parts 7–8 check their holdout against it |

`data/` and `cache/` hold WRDS-derived data (entities, part 4's flags) and stay in the team's private storage.

## Decisions

**Cleaning (`CLEAN_VERSION = "v2"`).**
- The charset is read from the document, and leftover replacement characters are counted.
- Hidden iXBRL is removed.
- Paragraphs are kept.
- Financial tables are set aside in `fin_tables`, not deleted. A table counts as financial on the 30% digit rule or the shape of a statement. Layout tables stay text.
- Page furniture and repeated content are removed and counted.
- Standard sections are labelled, not deleted. `text_body` is what the scorers read.
- Sentences are split by punkt with financial abbreviations added.
- `earnings_like` applies part 2's check (earnings terms against monthly sales or traffic terms). Part 2's hand check found about 5% of its sample was not an earnings release; those filings are flagged, not removed.
- Any change to the text rules must bump `CLEAN_VERSION`, since the model cache is keyed on it.

**Long releases and context windows.** Each model counts tokens with its own tokenizer. `text_body` is cut between sentences, with a one-sentence overlap. A single sentence longer than the window becomes its own piece and is flagged `truncated`. The windows leave room for the prompt and the answer:

| Model | Tokens of text per window |
|---|---|
| FinBERT | 510 |
| ChronoGPT | 1,500 |
| Llama 3.1 8B Instruct | 8,000 |

On the 2006 test (200 releases, cleaning v2), the median release is about 1,750 tokens with FinBERT's tokenizer and 1,800 with ChronoGPT's, and nine in ten are under 6,700: a single 8,000-token window covers about nine releases in ten, while FinBERT needs a median of four pieces. Llama's own count is not measured yet (its tokenizer needs `HF_TOKEN`). Encoders score the pieces and pool them. The generative model reads the whole `text_body`.

**Caching.**
- **Raw bytes:** cached once and never refetched.
- **Pieces:** keyed by text hash, cleaning version, tokenizer, window and overlap.
- **Model outputs:** keyed by `accession | sha256_text | clean_version | variant | model_id | model_revision | prompt_id | sha256(prompt_config)`. `variant` is `raw`, `mask` or `placebo`. `prompt_config` holds the template, the decoding parameters and the masking protocol version. Changing any of them gives a new key.
- **Batch jobs and restarts:** `ModelCache.missing(jobs)` returns what is left to run.
- **Score contract with parts 7–8:** `score_json` is a number or an object with a numeric `score`, and higher means better news.

**Duplicates and overlapping events.** All three kinds are flags, never deletions:

| Flag | Rule |
|---|---|
| `exact` | same `sha256_text` |
| `near` | Jaccard similarity of word 5-grams ≥ 0.9 (MinHash LSH) |
| `close_events` | same PERMNO with two filings at most 5 trading sessions apart |

**Splits.**
- The rule comes from `common.py` (dev 2006–2016, validation 2017–2019, 2020 excluded, holdout 2021–2025, by reaction date), so it is never restated here.
- The date is the reaction date of part 4's `events_final.parquet`, the table parts 7–8 evaluate on. Filings that no event is linked to use the release clock's date instead.
- `splits.py` checks its split against part 4's.
- The freeze writes the holdout accessions of part 4's text sample (`in_text_sample`; the primary sample is a subset of it) with their sha256. It is refused without `events_final`, if the splits disagree, or if a freeze already exists.

## Status (7 November 2026)

| | Status |
|---|---|
| Code and unit tests | done; end-to-end run on synthetic inputs, outputs read by part 7–8's own readers |
| Raw cache | complete: 116,983 documents, 2006–2025, 2.3 GB, none missing |
| Cleaning | 2006 complete with v2 (5,534 documents, 0 errors); the other years to run |
| Entities, split draft | built for all 115,976 filings from part 3–4's files of 6 November (Drive export); to rebuild after part 4's rerun |
| Tokens, duplicates | 2006 test only; Llama not counted yet (needs `HF_TOKEN`) |
| Model cache | created, empty; waiting for part 6 |
| Holdout freeze | not done: waits for the corrected clock and part 4's rerun |
