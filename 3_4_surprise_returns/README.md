# Parts 3–4: earnings surprise and returns

Owner: Beatriz Fischberg Blank. Part of *Priced In or Memorized?* (15.S08, Fall 2026).

This folder builds the event-level financial data the rest of the project is measured against:

- **the event list:** quarterly earnings announcements, 2006–2025, linked to CRSP;
- **the release time** of each event (wire first, I/B/E/S as the check);
- **the earnings surprise (SUE):** the benchmark the text has to beat;
- **abnormal returns:** the outcome every signal is scored against.

## Layout

```
3_4_surprise_returns/
  00_setup_access.ipynb   access check, trading calendar, provisional clock
  01_download.ipynb       every WRDS pull (the only notebook that connects to WRDS)
  02_surprise.ipynb       events, wire-first release times, consensus, SUE   (part 3)
  03_returns.ipynb        abnormal returns, post-release returns, checks     (part 4)
  04_sample.ipynb         filing links, sample flags, final funnel, hand-off (part 4)
  code/                   the logic; notebooks only call it
    common.py             splits, holdout guard, trading calendar
    clock.py              release-clock schema and the provisional clock
    download.py           WRDS queries, cached per table and year
    surprise.py           part 3
    returns.py            part 4
    sample.py             part 4: links to filings, flags, funnel
    tests/                synthetic tests with hand-computed answers
  cache/                  raw WRDS pulls (parquet)            WRDS data: never submitted
  data/                   built tables keyed by event_id      WRDS-derived: never submitted
  results/                funnel, counts and summary tables (no row-level data)
  logs/                   access report with package versions
```

## Run order

| Step | Reads | Writes | Time |
|---|---|---|---|
| `00_setup_access` | WRDS; Mahaut's `1_2nov/nov2_data` | `logs/access_report_*.json`, `data/trading_calendar.csv`, `data/clock_provisional.csv` | 5 min |
| `01_download` | WRDS | `cache/` (about 840 MB), `cache/manifest.csv` | 20 min first time; reruns skip finished pulls |
| `02_surprise` | `cache/`, calendar, clock | `data/surprise.parquet`, `data/estimates_used.parquet`, `data/handcheck_sample_50.csv`, `results/funnel_part3.csv`, `results/consensus_counts.csv` | 4 min |
| `03_returns` | `data/surprise.parquet`, `cache/` | `data/returns.parquet`, `results/returns_info.csv`, `results/sue_deciles_dev.csv` | 3 min |
| `04_sample` | `data/surprise.parquet`, `data/returns.parquet`, `data/clock_provisional.csv`, `cache/` | `data/events_final.parquet`, `data/text_sample_keys.csv`, `results/funnel_final.csv`, `results/sample_by_split.csv`, `results/gaap_basis_check.csv`, `results/robustness_ic_dev.csv` | 2 min |

Each notebook finds the repository, adds its `code/` folder to the path and checks its inputs exist. Private inputs and outputs default to the checkout; set `TEAM_DIR` to use a separate data folder. WRDS logins use your own account (username, password, Duo push). Never save the password; answer `n` to the `.pgpass` question.

**For teammates:** `data/events_final.parquet` has one row per event, with surprise, returns and every flag. Filter with `in_primary` (the pre-registered sample) or `in_text_sample` (all events with text). `data/text_sample_keys.csv` lists the filings to score, by accession number. Join model scores back on `event_id` or `accession`.

When the corrected release clock (wire / EDGAR header times, part 1) is ready, put it in `data/` with the columns listed in `clock.CLOCK_COLUMNS` and rerun 02, 03 and 04. Nothing else changes.

## Definitions (to be registered before the holdout is touched)

**Event.** One I/B/E/S quarterly EPS announcement (I/B/E/S ticker, fiscal quarter end), announced 2006–2025, actual in US dollars, linked to a CRSP permno through `wrdsapps.ibcrsphist` (score 1–3, valid on the announcement date), a US common share in CRSP with a price two trading days before the reaction.

**Release time R.** The Dow Jones wire time where one exists for the same stock within four days, otherwise the I/B/E/S announcement time (`00:00:00` treated as missing). `clock_status` is `wire_ibes_agree` when the two put the release in the same session (the two-source primary sample), `wire_ibes_conflict` when they disagree (the wire is used), or `ibes_only`.

**Reaction session (day 0).** The first trading session that can react: same day for releases before 09:30, same day for releases during the session, the next session for releases at or after the close (early closes respected) or on non-trading days.

**Consensus.** Each analyst's latest unadjusted EPS estimate for the same fiscal quarter, with ANNDATS and ACTDATS both before R's date, issued within 90 days, in US dollars, from a broker that had not stopped covering the firm before R. Estimates are put on the actual's share basis with CRSP's cumulative share factor. Consensus is the median.

**SUE.** (actual − median consensus) / price at t−2 (primary); (actual − median) / dispersion when 3+ analysts (alternative). Not winsorised at construction.

**Returns.** Expected return from Fama-French three factors plus momentum, betas over trading days −280 to −31 (at least 120 returns), intercept not used.
- `car_0_1`: abnormal return on days 0 and +1, the reaction (descriptive).
- `post_car_1`, `post_car_60`: from the first CRSP price after R (day 0's open, or its close for intraday releases) to the close of day +1 / +60, the "not yet priced" outcome.
- `ret_prior_1m`: raw return over days −22 to −2, a baseline control.
- `ar_m1`: abnormal return on day −1, a placebo.
- Market-adjusted versions as robustness.
- Delisting returns are compounded with that day's return. A 60-day window past CRSP's last date is left missing.

**Sample (04).** Rules are flags, not deletions. `in_primary` = 3+ analysts, price above $5 at t−2, linked to an 8-K in the filing sample (filed between one day before and four days after the release), wire and I/B/E/S agreeing on the session, and a reaction return. `f_size` (NYSE 20th percentile at the latest month-end) is a flag for robustness, not a cut. `timing_suspect` marks I/B/E/S-only intraday times. `basis_suspect` marks events whose I/B/E/S actual equals Compustat GAAP EPS while the consensus sits much closer to operating EPS (a probable basis mismatch).

**Splits.** By reaction date: development 2006–2016, validation 2017–2019, 2020 excluded, holdout 2021–2025. `common.guard_holdout` drops holdout rows from any analysis unless `UNLOCK_HOLDOUT=1` is set, which should happen once, after the freeze.

## Data provenance and licensing

All market and analyst data come from WRDS under MIT's subscription: CRSP CIZ daily (`crsp.dsf_v2`, `crsp.stkdelists`, `crsp.stocknames_v2`, `crsp.ccmxpf_lnkhist`), I/B/E/S unadjusted actuals, detail, stopped estimates, surprise summary and guidance, Compustat quarterly, the WRDS I/B/E/S–CRSP and SEC–Compustat links, and Fama-French factors. Wire times come from RavenPack's Dow Jones feed, pulled by Mahaut (part 1) through WRDS.

`cache/` and `data/` contain licensed data or row-level derivatives of it. They stay in the team's private Drive and are not part of the submitted package, and they are not shared with outside AI tools. `results/` holds counts and summary statistics only.

## Tests

```
python 3_4_surprise_returns/code/tests/run_all.py
```

The tests use synthetic data with hand-computed answers. They cover filing links, the size flag, the GAAP-basis flag, the funnel, look-ahead, stale and stopped estimates, currency, stock splits, duplicate announcements, wire-first timing, factor betas, the open-to-close leg, delisting returns, data-end truncation, the day −1 placebo and Newey-West standard errors.

## Environment

Tested on Google Colab, Python 3.13.18 (4 Nov 2026), with the versions in `requirements.txt`. The course targets Python 3.12, and nothing here needs 3.13.
