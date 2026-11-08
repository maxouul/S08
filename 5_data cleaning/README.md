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
| `02_clean_prepare` | `cache/raw/`, parts 3–4 | everything in `data/`, `cache/model_outputs.sqlite`, `results/token_quantiles.csv`, `manifest.json` | 12 h 30 for a full run on a 4-core laptop (3 workers, 7–8 November): cleaning 10 h 45 (recent documents are about three times heavier than 2006's), entities 3 min, tokens 1 h 25 to 1 h 50, duplicates 15 min, the rest under a minute; reruns skip what is up to date |
| `python code/splits.py --freeze` | `data/splits_draft.parquet`, part 4's `events_final.parquet` | `data/holdout_ids.txt`, `.sha256`, `holdout_freeze.json` | once, at the end |

Every step can be stopped and rerun. `cache_raw.py` commits its archives every 500 documents and retries failed downloads on the next run. `clean_release.py` skips a year already cleaned with the current `CLEAN_VERSION` and covering every document of its archive. `token_stats.py` skips a year whose pieces match the texts (their hash and cleaning version) and the window settings. `ModelCache` never pays for the same call twice.

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

**Cleaning (`CLEAN_VERSION = "v3"`).**
- The charset is read from the document, and leftover replacement characters are counted.
- A document that opens with a closing tag (`</head>` with no `<head>`, as some word processors export) is read in full. lxml returned it empty: 39 releases had no text in v2.
- A release filed as a PDF is read with pypdf (6 documents), not decoded as text.
- Hidden iXBRL is removed.
- Paragraphs are kept.
- Financial tables are set aside in `fin_tables`, not deleted. A table counts as financial on the 30% digit rule or the shape of a statement. Layout tables stay text.
- Page furniture and repeated content are removed and counted.
- Standard sections are labelled, not deleted. `text_body` is what the scorers read.
- A heading opens or closes a section. In the safe harbor, a factor listed on its own line ("• the loss of key suppliers; and") stays in the safe harbor even when it is short enough to pass for a heading; in v2 it closed it, and the rest of the risk factors went to the body.
- A safe harbor at the top of a transcript or a slide deck is followed by no heading. When the safe harbor holds more than half the words, the text is labelled again with a stricter rule: the safe harbor ends at the first paragraph that is not short (40 words), not a list item, and not in its vocabulary. In v2, 20 such texts had an empty body and about 600 kept under a fifth of their words.
- Sentences are split by punkt with financial abbreviations added.
- `earnings_like` applies part 2's check (earnings terms against monthly sales or traffic terms). Part 2's hand check found about 5% of its sample was not an earnings release; those filings are flagged, not removed.
- Any change to the text rules must bump `CLEAN_VERSION`, since the model cache is keyed on it.

**Long releases and context windows.** Each model counts tokens with its own tokenizer. `text_body` is cut between sentences, with a one-sentence overlap. A single sentence longer than the window becomes its own piece and is flagged `truncated`. The windows leave room for the prompt and the answer:

| Model | Tokens of text per window |
|---|---|
| FinBERT | 510 |
| ChronoGPT | 1,500 |
| Llama 3.1 8B Instruct | 8,000 |

Over the 116,983 releases (cleaning v3), the median `text_body` is 2,445 tokens with FinBERT's tokenizer and 2,521 with ChronoGPT's, and nine in ten are under 5,300. The median grows from about 2,100 in 2006 to 2,800 in 2025. With these tokenizers 97% of releases fit in 8,000 tokens, so a single Llama window should cover almost all of them. FinBERT needs a median of six pieces, ChronoGPT two (one piece for 22% of releases). Sentences longer than the window: 3,432 for FinBERT, 23 for ChronoGPT. Llama's own count is not measured yet (its tokenizer needs `HF_TOKEN`). Encoders score the pieces and pool them. The generative model reads the whole `text_body`.

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

## Status (8 November 2026)

| | Status |
|---|---|
| Code and unit tests | done (36 tests); end-to-end run on synthetic inputs, outputs read by part 7–8's own readers |
| Raw cache | complete: 116,983 documents, 2006–2025, 2.3 GB, none missing |
| Cleaning | complete with v3: 116,983 texts, 0 errors, no replacement characters. The v2 output was brought to v3 by cleaning again the 63 documents the new reading rules touch and labelling the others again; a full v3 `clean()` gives identical rows on 695 documents checked (500 at random, 45 read again, 150 relabelled) |
| Checks of `02_clean_prepare` | 22 agree, 2 do not, both expected. One text has an empty body: Hydrofarm, 0001628280-22-000647, two page-long paragraphs that both hold the safe harbor. 419 texts keep under 80% of the words of part 2's extraction, mostly repeated content removed on purpose (50 of a hand-checked sample of 60); they are listed in `results/clean_shorter_than_old.csv` |
| Entities, split draft | built for all 115,976 filings from part 3–4's files of 6 November (Drive export); to rebuild after part 4's rerun |
| Tokens, duplicates | complete for FinBERT and ChronoGPT; 142 exact duplicates, 160 near duplicates, 1,361 filings in close events; Llama not counted yet (needs `HF_TOKEN`) |
| Model cache | created, empty; waiting for part 6 |
| Holdout freeze | not done: waits for the corrected clock and part 4's rerun |
