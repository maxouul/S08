# Parts 3–4: earnings surprise and returns

**Status brief for the team meeting, 6 November 2026** Owner: Beatriz Fischberg Blank · Project: *Priced In or Memorized?* (15.S08) Mid-project checkpoint due Monday 9 November, 4 PM ET (rubric).

---

## 1\. Bottom line

- **A usable, checked event-level dataset exists (version 1).** It holds 217,611 quarterly earnings announcements (2006–2025) with surprise, returns and sample flags. The **primary sample is 73,142 events**, and **97,787 filings** have text available.  
- **The full folder was checked on 6 November.**  
  - The code matches the latest versions.  
  - All five notebooks ran without errors.  
  - All tests pass in Colab.  
  - Every saved number matches its notebook.  
  - The hand-off file has exactly one row per filing.  
- **It is provisional in one respect.** 22,825 text-sample events still have only an I/B/E/S release time. Mahaut's corrected EDGAR times are validated but cover only 44% of our filings so far. A targeted list of the 55,213 missing filings is ready, about 1.5–2 hours of fetching. When it completes, we rerun steps 02–04 (about 10 minutes).  
- **The headline result so far** (development years, holdout untouched): in our sample, the earnings surprise is fully priced by the close of the reaction day. Any "not yet priced" content in the text has to show up before then.

---

## 2\. Why parts 3–4 matter

The project asks whether earnings-release text carries information about returns beyond the earnings surprise. Parts 3–4 supply everything in that test that is not text:

| Element | What it is | Why the project needs it |
| :---- | :---- | :---- |
| SUE (earnings surprise) | (actual EPS − analyst consensus) / price | The benchmark: the text must raise the IC over SUE (and tone) by at least 0.02, the pre-registered bar |
| Abnormal returns | Returns net of a Fama-French 3-factor \+ momentum benchmark | The outcome every signal is scored against |
| Release clock | When each release became public, and which trading session first reacts | Defines "before" and "after"; a wrong clock creates look-ahead bias |
| Sample and funnel | Which releases are in, and why | Decides which releases the models score, and how many fall after Llama's training cutoff (the power of the memorization test) |

### Who uses what

| Work-division item / team | What they take from parts 3–4 |
| :---- | :---- |
| 2\. EDGAR text | `text_sample_keys.csv`: which of her filings matter (97,787 of 115,976). 11,893 of her filings (10%) match no I/B/E/S announcement, a ready candidate list for non-earnings Item 2.02 filings |
| 5\. Cleaning, caching, splits | The `split` column (development / validation / 2020 excluded / holdout, by reaction date) and the holdout guard. The chronological splits are already fixed |
| 6\. Models and inference | `text_sample_keys.csv`: the accession list to score, with `in_primary` to prioritise |
| 7\. Baselines | The surprise-only baseline (SUE) is ours. The Loughran–McDonald and TF-IDF baselines are scored on our returns |
| 8\. Financial evaluation | `events_final.parquet`: SUE, returns, the prior-month return (control), all flags; the sign of SUE is one axis of the 2×2 table |
| 9\. Memorization and contamination | `after_llama_cutoff`: 8,579 primary events after December 2023, which sets the power of the leakage test |
| 10\. Error analysis | `basis_suspect`, `timing_suspect`, and the extreme surprises |
| Report and presentation | Funnel, definitions (README), key numbers |

&nbsp;

---

## 3\. The dataset

### Files

All files are in `S08 NLP/3_4_surprise_returns/data/`. They are WRDS-derived, so they stay in the team Drive (see section 12).

| File | Rows | Contents |
| :---- | :---- | :---- |
| `events_final.parquet` | 217,611 | One row per event (I/B/E/S firm-quarter): identifiers, release time, surprise, returns, every flag |
| `text_sample_keys.csv` | 97,787 | One row per filing (unique accession): `event_id, accession, cik, permno, filing_date, R_et, R_source, clock_status, reaction_date, split, after_llama_cutoff, in_primary, in_text_sample, sample_version` |
| `surprise.parquet`, `returns.parquet` | 217,611 each | Intermediate outputs of steps 02 and 03 |
| `estimates_used.parquet` | 1,427,964 | Every analyst estimate that entered a consensus (audit trail) |
| `handcheck_sample_50.csv` | 50 | Fixed random sample for checking reported EPS against the releases |

### How teammates should use it

- **Filter on `in_primary` for headline results,** and on `in_text_sample` for robustness checks.  
- **Join model scores back on `event_id` or `accession`.**  
- **Run every analysis table through `common.guard_holdout()`,** which drops 2021–2025 until the freeze.  
- **Record the version tags with any result:** `surprise_v2_wire_first`, `returns_v3_placebo_day_m1`, `sample_v1`.

### Main columns

| Group | Columns |
| :---- | :---- |
| Identity | `event_id` (I/B/E/S ticker \| fiscal quarter end), `permno`, `accession`, `cik`, `cname`, `gvkey` |
| Timing | `R_et` (release time, Eastern), `R_source` (wire / ibes\_standin), `clock_status`, `session_type`, `reaction_date` (day 0), `entry_field` (open / close), `split`, `after_llama_cutoff` |
| Surprise | `actual`, `consensus_median`, `consensus_mean`, `dispersion`, `n_analysts`, `surprise`, `sue_price` (primary), `sue_disp` (alternative), `price_t2` |
| Returns | `car_0_1_ff4`, `car_0_1_mkt`, `post_car_1_ff4`, `post_car_60_ff4`, `car_1_60_ff4`, `post_ret_1_raw`, `post_ret_60_raw`, `ret_prior_1m`, `ar_m1`, betas, `drift_complete`, `delisted_in_window` |
| GAAP check | `gaap_eps`, `operating_eps`, `actual_basis`, `basis_suspect` |
| Flags | `f_analysts3`, `f_price5`, `f_size`, `two_source`, `timing_suspect`, `has_filing`, `in_text_sample`, `in_primary` |

### Sample sizes

| Sample | Development 2006–16 | Validation 2017–19 | 2020 (excluded) | Holdout 2021–25 | After Llama cutoff (2024–25) | Total |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Text sample (`in_text_sample`) | 51,529 | 14,523 | 5,232 | 26,503 | 9,899 | 97,787 |
| **Primary sample (`in_primary`)** | **35,508** | **11,375** | **4,114** | **22,145** | **8,579** | **73,142** |
| Primary, above NYSE 20th percentile | 35,131 | 11,304 | 4,096 | 22,060 | 8,552 | 72,591 |

### Funnel 1: building the events (step 02\)

| Step | Remaining | Removed |
| :---- | :---- | :---- |
| I/B/E/S quarterly EPS actuals announced 2006–2025 | 415,483 | — |
| One announcement per firm-quarter | 415,300 | 183 |
| Actual EPS value present | 351,756 | 63,544 (15.3%) |
| Actual reported in US dollars | 343,267 | 8,489 |
| Linked to a CRSP stock on the announcement date (score ≤ 3\) | 332,818 | 10,449 |
| Announcement time known | 332,629 | 189 |
| At least one point-in-time estimate within 90 days | 269,439 | 63,190 |
| Stock is a US common share in CRSP | 221,248 | 48,191 |
| CRSP price two trading days before day 0 | **217,611** | 3,637 |

### Funnel 2: the final sample (step 04\)

| Step | Remaining | Removed |
| :---- | :---- | :---- |
| Events with SUE and returns | 217,611 | — |
| 3+ analysts | 143,068 | 74,543 (34.3%) |
| Price above \$5 at t−2 | 131,621 | 11,447 (8.0%) |
| Linked to an 8-K in the filing sample (text available) | 97,787 | 33,834 (25.7%) |
| Wire and I/B/E/S agree on the session (two sources) | 74,217 | 23,570 (24.1%) |
| Reaction return available | **73,142** | 1,075 (1.4%) |

### Caveats to state whenever the data is used

1. **WRDS-derived data.** Team Drive only; not in the submission; not into outside AI tools.  
2. **Scope is larger firms.** 99% of the primary sample is above the NYSE 20th percentile, because Mahaut's filing sample was size-filtered. Only 377 smaller firms remain in the development primary sample.  
3. **Version 1\.** Counts and ICs will shift modestly when the EDGAR times arrive. The definitions will not change.

---

## 4\. Checklist status (work-division items 3 and 4\)

### Part 3: earnings surprise construction and validation

| Task | Status | Evidence |
| :---- | :---- | :---- |
| Finalise I/B/E/S actual EPS and consensus matching | ✅ Done | 217,611 events; actual and estimates matched on the same fiscal quarter |
| Only estimates available before each announcement | ✅ Done | An estimate counts only if dated and recorded before the release day (instructors' rule); checked in code on all 1,427,964 estimates used; release day is wire-first |
| Forecast dates, fiscal periods, units, share adjustments, duplicates | ✅ Done | Apple's split ratios recovered exactly (7.000, 4.000); 3,729 estimates rescaled for splits; 8,489 non-USD actuals and 361 non-USD estimates removed; 53,073 stale estimates from brokers that stopped coverage removed; 183 duplicate announcements resolved; each analyst's latest estimate only |
| Manually compare \~50 reported EPS values with the releases | ⏳ Open | Sample ready, with real exchange tickers and names (the I/B/E/S ticker is an internal code: "TGT" is Pactiv); 0 of 50 filled |
| GAAP vs adjusted EPS | ✅ Systematic check done; hand check pending | Compared with Compustat GAAP and operating EPS: 1.16% of the text sample flagged as a probable basis mismatch (section 5.5) |
| FactSet as an independent consensus | ⏳ Started | Workstation access obtained; API test pending; plan is a \~300-event validation |
| Save the surprise per event with analysts and dates | ✅ Done (v1) | SUE (price- and dispersion-scaled), analyst count, dispersion, all dates and clock status in `events_final.parquet` |
| Revenue and guidance surprise (instructors' feedback) | ⏳ Data ready, not computed | 441,348 revenue actuals, 4,552,627 revenue estimates, 318,568 guidance records downloaded |

### Part 4: WRDS / CRSP data pipeline

| Task | Status | Evidence |
| :---- | :---- | :---- |
| Confirm access (I/B/E/S Guidance, SEC links, CRSP) | ✅ Done | 11 tables confirmed on Beatriz's account (access report, 4 Nov); CRSP runs to 31 Dec 2025 |
| Company matching (PERMNO, CUSIP, link periods) | ✅ Done | I/B/E/S to CRSP through the WRDS link (scores 1–3, valid on the date): 97% linked. Filings: 90% of Mahaut's 115,976 linked to an event; 90% filed the same day as the release; 99.5% in the same session |
| Stock, market and delisting returns | ✅ Done | CRSP CIZ daily; daily Fama-French factors and momentum; 4,099 delisting returns compounded with that day's return (CRSP's daily return excludes them: checked, 207 of 4,306 equal) |
| Abnormal returns over predefined windows | ✅ Done | Betas for 97.6% of events (median market beta 0.98); reaction, post-release \+1 and \+60, from day 0's close, prior month, day −1 placebo; market-adjusted versions |
| Returns aligned with corrected release sessions | 🟡 Partly | Wire time used for 79,810 events; 22,825 text-sample events still use I/B/E/S time until EDGAR times arrive |
| Apply sample filters and generate the final funnel | ✅ Done | All rules are flags, not deletions; funnel computed from the data (section 3\) |
| Clean event-level dataset (surprise, texts, timestamps, returns) | ✅ Done on our side | `events_final.parquet` plus `text_sample_keys.csv`; texts (part 2\) join on accession |

---

## 5\. Results so far

All results below use **development years only (2006–2016)**. The holdout has not been touched.

### 5.1 The SUE baseline: the bar the text has to clear

The table shows the rank IC (Spearman) between SUE and each outcome, computed per announcement season and averaged, for the primary sample in development years (n \= 35,508). The t-statistics use Newey-West standard errors with 2 lags.

| Window | Rank IC | t (NW) | Reading |
| :---- | :---- | :---- | :---- |
| Reaction: days 0 and \+1 (`car_0_1`) | 0.352 | 56.1 | Descriptive: SUE and the reaction are measured at the same moment, so this is the "earnings response", not a forecast |
| First price after the release → close of day \+1 | 0.086 | 10.7 | Post-release, "not yet priced" |
| First price after the release → close of day \+60 | 0.024 | 2.6 | Post-release drift |
| **Close of day 0 → close of day \+60** | **0.002** | **0.2** | The strictest "not yet priced" window: nothing left |

**Interpretation.** All the post-release predictability of SUE comes from the move between day 0's open and its close. After the close of the reaction day, SUE predicts nothing. This fits recent research finding that post-earnings drift has largely disappeared in larger stocks when measured from the first price after the release. For the text test, this means:

- **The window from day 0's close is the cleanest test of "not yet priced."** It should be registered alongside the window from the first price.  
- **A text signal that predicts returns after day 0's close would be striking,** because SUE predicts nothing there. It would also deserve extra scrutiny for leaks.

### 5.2 Return by SUE decile

This covers the 73,585 development events with 3+ analysts and a price above \$5, across all clock statuses. Deciles are formed within each season.

|  | Decile 1 (most negative) | Decile 5 | Decile 10 (most positive) |
| :---- | :---- | :---- | :---- |
| Reaction, days 0 and \+1 | −4.86% | −0.04% | \+4.57% |
| First price → day \+1 | −1.35% | \+0.03% | \+1.14% |
| First price → day \+60 | −1.61% | −0.04% | \+1.55% |

The reaction rises steadily across all ten deciles, which only happens if timing, consensus and returns line up. The average two-day abnormal return is \+0.09%, consistent with the known "earnings announcement premium."

### 5.3 Timing diagnostics (the look-ahead checks)

| Check | Result | Conclusion |
| :---- | :---- | :---- |
| Placebo: SUE vs abnormal return on day −1 | IC 0.055 overall; **0.051 where the wire confirms the timing** | Genuine pre-release movement (industry peers reporting earlier, leakage), not a clock error |
| Placebo: SUE vs raw return, days −22 to −2 | IC 0.060 | Known pre-announcement run-up; the prior-month return belongs in the baseline as a control |
| Wire vs I/B/E/S conflicts | 787 events (1.0% of wire-covered events; 1.6% in development years), now using the wire time | Before the fix, these events' "post-release" IC was **0.46**: the reaction itself, counted as drift. After the fix: 0.096, in line with the rest |
| I/B/E/S "intraday" times | 6.3% of I/B/E/S-only events are intraday, against 0.5% where the wire confirms; their post-release IC is 0.23 | I/B/E/S intraday times are mostly wrong. These events are flagged `timing_suspect` (514 in the text sample) and are outside the primary sample anyway |

These are the development-period ICs by clock status:

| Clock status | n | Day −1 | Reaction | Post \+1 | Post \+60 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Wire and I/B/E/S agree | 36,041 | 0.051 | 0.351 | 0.086 | 0.024 |
| I/B/E/S only | 36,972 | 0.057 | 0.383 | 0.140 | 0.066 |
| Conflict (wire used) | 572 | 0.012 | 0.361 | 0.096 | 0.132 |

The higher post-release ICs for I/B/E/S-only events are likely part timing error and part smaller firms. That's why the primary sample requires two sources.

### 5.4 Robustness (primary sample, development years)

| Group | n | Reaction | Post \+1 | Post \+60 | Day 0 close → \+60 |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Primary | 35,508 | 0.352 | 0.086 | 0.024 | 0.002 |
| Primary, above NYSE 20th percentile | 35,131 | 0.352 | 0.086 | 0.024 | 0.002 |
| Primary, below | 377 | 0.337 | 0.079 | 0.008 | −0.052 |
| Text sample (any timing status) | 51,529 | 0.359 | 0.091 | 0.030 | 0.007 |

The size split cannot be run meaningfully (only 377 small events), because size was effectively decided upstream by the filing sample.

### 5.5 GAAP vs street EPS

Each I/B/E/S actual was compared with Compustat's diluted GAAP EPS and diluted operating EPS (which excludes special items) for the same quarter. The table covers the 97,787 text-sample events:

| Actual matches… | Events | Share |
| :---- | :---- | :---- |
| Neither (street EPS excludes more than special items: normal) | 53,422 | 54.6% |
| Both (no special items) | 25,607 | 26.2% |
| GAAP only | 9,568 | 9.8% |
| Operating only (the usual street basis) | 9,144 | 9.4% |
| No Compustat match | 46 | 0.0% |

**`basis_suspect` covers 1,131 events (1.16%).** In these, the actual equals GAAP, GAAP differs from operating EPS by 5+ cents, and the consensus sits much closer to operating EPS. The largest are big write-down quarters:

- Teladoc 2022 Q4: actual −23.49 against a consensus of −0.26  
- Teladoc 2022 Q2  
- Charter 2016  
- Regions 2008 Q4  
- KB Home 2007  
- Etsy 2022 Q3  
- Centex 2007  
- Moderna 2023 Q3

They stay in the data, flagged, so analyses can exclude them as a robustness check. Rank-based ICs limit their influence. None of the 50 hand-check events is flagged, but Bed Bath & Beyond (−0.92 against 0.015) isn't either, so the hand check is still needed.

---

## 6\. Validation evidence: what was checked

| Area | Check | Result |
| :---- | :---- | :---- |
| Look-ahead in the consensus | Code assertion: every estimate dated and recorded before the release day | Passed on 1,427,964 estimates |
| Share adjustment | Apple's 2014 (7:1) and 2020 (4:1) splits from CRSP factors | 7.000 and 4.000 exactly |
| Our consensus vs I/B/E/S's own figures | 131,281 matched filtered events | Actual within 1¢: 99.6%; consensus within 5¢: 93.4%; same surprise sign: 94.7% |
| Stale estimates left unscaled | Estimates without a CRSP factor among kept events | 330 of 1,205,129 |
| Trading calendar | CRSP trading days vs the NYSE calendar, 2005–2025 | Identical (0 differences); 48 early closes respected |
| Factor units | Automatic detection | Decimal (median absolute market excess return 0.0052) |
| Filing links | Filing date vs release date | 90% same day, all within −1 to \+4 days; 99.5% same session |
| Timing placebo | Day −1 IC where the wire confirms the time | 0.051, the same as overall: no clock error |
| Synthetic tests | 3 files, hand-computed answers | Cover look-ahead, stale and stopped estimates, currency, splits, duplicates, wire-first timing, betas, open-to-close, delistings, data end, placebo, Newey-West, filing links, size, GAAP flag, funnel, nullable data types. All pass in Colab (Python 3.13, pandas 2.2) and locally (Python 3.12, pandas 3.0) |

---

## 7\. Release timing: where we stand

**The problem.** The EDGAR acceptance times first used (from the SEC submissions JSON, pulled 30 October) were corrupted, as the instructors' correction email warned:

- 52,364 filings (13.7%) were stamped before 06:00, when EDGAR is closed.  
- Wire-vs-EDGAR gaps spiked at exactly ±4 and ±5 hours, matching daylight-saving periods.  
- About 31% of events with a wire time were off by exactly the Eastern offset.

Mahaut's 300-filing header check confirmed it: only 33% of JSON times were right. 128 were Eastern time labelled as UTC, and 73 were converted twice.

**The rule now applied** (instructors' feedback):

- The release time is the Dow Jones wire time where one exists.  
- I/B/E/S is the check; EDGAR is the fallback once corrected.  
- The earliest of several noisy times is never taken.  
- An event enters the primary sample only if two sources put it in the same session.

| Clock status (all 332,629 events with a known time) | Events |
| :---- | :---- |
| Wire and I/B/E/S agree | 79,023 |
| Conflict (wire used): wire later than I/B/E/S | 413 |
| Conflict (wire used): wire earlier than I/B/E/S | 374 |
| I/B/E/S only | 252,819 |

**Mahaut's corrected EDGAR times (`acceptance_times.csv`, 5 November): checked 6 November.**

- **The times are correct:**  
  - 89 of 89 match her hand-verified headers;  
  - only 3 of 130,000 fall outside EDGAR's hours;  
  - EDGAR comes a median 15 minutes after the wire, as expected;  
  - there's no timezone pattern.  
- **The fetch is incomplete:** 130,000 of 383,155 filings (33.9%). That covers **42,574 of our 97,787 filings (43.5%)**, and coverage falls by year, from 46% in 2006 to 26% in 2025\.  
- **We are not rerunning on the partial file.** It would tilt EDGAR verification toward early years for no reason other than fetch order.  
- **The targeted fetch is ready.** `acceptance_to_fetch_parts34.csv` lists the 55,213 missing filings, highest value first: 12,586 I/B/E/S-only events, then 461 conflicts, then 42,166 already two-source events. Her script runs on it unchanged, about 1.5–2 hours: `python fetch_acceptance.py --events acceptance_to_fetch_parts34.csv --out "files csv/acceptance_times.csv"`

**When it completes:**

1. EDGAR is added to our clock as the fallback.  
2. Steps 02–04 are rerun (about 10 minutes).  
3. The primary sample gains up to about 4,000 holdout events.  
4. EDGAR breaks the tie in the 745 conflicts.

**Notes on Mahaut's current timing file** (not used by our pipeline):

- Its "two of three agree" votes still use the JSON-based EDGAR times, as do its 14,718 EDGAR-based sessions. They are unreliable until rerun.  
- 569 events are marked as agreeing although no second source matches the session kept. The rule should check agreement with the final session.  
- `release_texts.csv` still carries the JSON timestamps. For merged analyses, timing should come only from our event table, so the team has one version of the truth.  
- Her 50-event timing hand-check is still blank.

---

## 8\. Statistical power

The holdout primary sample has 13,566 events before Llama 3.1's cutoff and 8,579 after.

| Assumption | Standard error | Minimum detectable effect (80% power, two-sided) |
| :---- | :---- | :---- |
| Correlation between scoring arms 0.7 | 0.0107 | **0.030** |
| Correlation 0.8 | 0.0087 | 0.024 |
| Instructors' table (20,000 per side, r \= 0.7) | 0.0077 | 0.022 |

The holdout is about 22,000 events, not the 50,000 the feedback assumed. The evaluation team should restate the minimum detectable effects for the leakage test and the main test before the freeze, as the instructors asked. EDGAR times would add up to about 4,000 holdout events.

---

## 9\. Decisions and asks for this meeting

1. **Use version 1 for the checkpoint.** `in_primary` for headline numbers, `in_text_sample` for robustness, versions recorded.  
2. **Confirm the scope:** firms above the NYSE 20th percentile, as inherited from the filing sample. Otherwise, decide now to extend Mahaut's filing download to smaller firms.  
3. **Re-register before 9 November (Evaluation).** That covers the definitions in section 11, the two post-release windows (from the first price, and from day 0's close), and the restated minimum detectable effects.  
4. **Ask Mahaut to run the targeted EDGAR fetch** (about 1.5–2 hours) and fix the two-source rule in her file.  
5. **Split the 50-event EPS hand check** (about 25 minutes per person). Look releases up by `crsp_ticker` / `crsp_name`, not the I/B/E/S code.  
6. **Agree the hand-off:** Models (item 6\) score the filings in `text_sample_keys.csv`, and every result joins back on `event_id` or `accession`.  
7. **FactSet:** run the API test, then validate about 300 events. The FactSet news archive (Business Wire, PR Newswire, GlobeNewswire) can also check release times in conflicts. StreetAccount times are only an upper bound.

---

## 10\. Next steps

**Before the checkpoint (9 November):**

- Draft the parts 3–4 paragraph for the checkpoint:  
  - runnable pipeline with environment and versions;  
  - the SUE baseline as the preliminary result;  
  - the timing diagnostic;  
  - an error case (the conflict events whose "post-release" IC of 0.46 was the reaction itself; or the Teladoc basis case);  
  - the blocker (EDGAR times) with its mitigation (wire and I/B/E/S agreement already gives a two-source sample).  
- Hand check of 50 EPS values.  
- EDGAR step added to the code, ready for the rerun.

**After the checkpoint:**

- Rerun steps 02–04 on complete EDGAR times; freeze version 2\.  
- Revenue surprise and guidance surprise (the wider baseline the instructors asked for).  
- FactSet validation of actuals and consensus (\~300 events).  
- Look into the 63,544 I/B/E/S actuals with no EPS value (15.3%), low priority.

---

## 11\. Definitions (for registration)

- **Event.** One I/B/E/S quarterly EPS announcement (I/B/E/S ticker, fiscal quarter end), announced 2006–2025. The actual is in US dollars, and the event is linked to a CRSP permno (WRDS link, score 1–3, valid on the date). The stock is a US common share with a CRSP price two trading days before day 0\.  
- **Release time R.** The Dow Jones wire time for the same stock within four days, otherwise the I/B/E/S time (`00:00:00` treated as missing). EDGAR header time will become the second fallback.  
- **Day 0 (reaction session).**  
  - A release before 09:30 reacts the same day.  
  - A release during the session reacts the same day.  
  - A release at or after the close (early closes respected), or on a non-trading day, reacts the next session.  
- **Consensus.** The median of each analyst's latest unadjusted EPS estimate for the same fiscal quarter, where:  
  - ANNDATS and ACTDATS are both before R's date;  
  - the estimate was issued within 90 days;  
  - it is in US dollars;  
  - the broker had not stopped covering the firm before R;  
  - it is rescaled to the actual's share basis with CRSP's cumulative share factor.  
- **SUE.**  
  - Primary: (actual − median consensus) / price at t−2.  
  - Alternative: (actual − median consensus) / dispersion, when 3+ analysts.  
  - Not winsorised at construction.  
- **Expected return.** Fama-French three factors plus momentum, betas over trading days −280 to −31 (at least 120 returns). The intercept is not used.  
- **Outcomes.**  
  - `car_0_1`: the reaction, days 0 and \+1 (descriptive).  
  - `post_car_1` and `post_car_60`: from the first CRSP price after R (day 0's open, or its close for intraday releases) to the close of day \+1 / \+60.  
  - `car_1_60`: from day 0's close to day \+60 (the strictest window).  
  - `ret_prior_1m`: raw, days −22 to −2 (a control).  
  - `ar_m1`: day −1 (a placebo).  
  - Market-adjusted versions as robustness checks.  
  - Delisting returns are compounded. A 60-day window past CRSP's end is left missing.  
- **Primary sample.** 3+ analysts, price above \$5 at t−2, linked to an 8-K (filed one day before to four days after the release), two sources agreeing on the session, and a reaction return.  
- **Splits.** By reaction date:  
  - development 2006–2016;  
  - validation 2017–2019;  
  - 2020 excluded;  
  - holdout 2021–2025, run once after the freeze.  
- **Inference.** Rank IC per season, averaged, with Newey-West standard errors (2 lags) for windows that overlap across seasons.

---

## 12\. Reproducibility and data handling

**Run order** (`S08 NLP/3_4_surprise_returns/`). Each notebook locates the repository, adds its `code/` folder to the path and checks its inputs. `TEAM_DIR` can point to a separate private data folder.

| Step | What it does | Time |
| :---- | :---- | :---- |
| `00_setup_access` | Access check, trading calendar, provisional clock (frozen: built 4 Nov from Mahaut's 3 Nov file; do not rerun its clock cell) | 5 min |
| `01_download` | Every WRDS pull, cached per table and year; reruns skip finished pulls | \~20 min first time |
| `02_surprise` | Events, wire-first release times, point-in-time consensus, SUE | 4 min |
| `03_returns` | Abnormal returns, post-release returns, diagnostics | 3 min |
| `04_sample` | Filing links, flags, final funnel, hand-off files | 2 min |

- **Logic lives in `code/`:** `common.py`, `clock.py`, `download.py`, `surprise.py`, `returns.py`, `sample.py`. The notebooks only call it.  
- **Tests:** `python code/tests/run_all.py` runs three synthetic test files.  
- **Environment:** Google Colab, Python 3.13.18. Versions are pinned in `requirements.txt` (wrds 3.5.0, pandas 2.2.3, numpy 2.4.6, pyarrow 23.0.1, pandas\_market\_calendars 5.5.0). The course targets Python 3.12, and nothing here needs 3.13.  
- **WRDS logins** use each person's own account. Never save the password.

**Data handling.**

- `cache/` (about 840 MB of raw WRDS pulls) and `data/` (row-level derivatives) are licensed. They stay in the team's private Drive, are excluded from the submitted package, and are never shared with outside AI tools.  
- `results/` holds counts and summary statistics only.  
- The same rule applies to FactSet data.

---

## 13\. Known limitations

- **Release timing:** 22,825 text-sample events rely on I/B/E/S time until EDGAR times arrive. I/B/E/S intraday times are unreliable (514 flagged).  
- **Scope:** results apply to firms above the NYSE 20th percentile.  
- **Opening prices:** the post-release window from the first price depends on opening prices. The day-0-close window is the check that avoids them.  
- **Basis mismatches:** about 1% of actuals are probably on the wrong basis. They are flagged, but the hand check is needed to measure the true rate.  
- **Missing actuals:** 15.3% of I/B/E/S quarterly actuals have no EPS value; not investigated.  
- **Data end:** CRSP ends 31 December 2025, so 2,668 events lack a complete 60-day window (left missing).  
- **Consensus choices:** the 90-day window and the median are design choices; sensitivity checks are possible.  
- **Frozen clock input:** `clock_provisional.csv` is a frozen input, because Mahaut's timing file has since changed format.

---

## Appendix: glossary

| Term | Meaning |
| :---- | :---- |
| SUE | Standardised unexpected earnings: (actual EPS − consensus) / price |
| Consensus | Median of analysts' latest EPS forecasts available before the release |
| Rank IC | Spearman correlation between a signal and later returns, computed per season and averaged |
| AR / CAR | Abnormal return (return minus expected return) / cumulative abnormal return over a window |
| FF4 | Fama-French market, size and value factors plus momentum |
| Day 0 | The first trading session that can react to the release |
| t−2 | Two trading days before day 0 |
| Two-source | Two independent time sources (wire, EDGAR, I/B/E/S) agree on the reaction session |
| Newey-West | Standard errors that allow for correlation between neighbouring seasons |
| MDE | Minimum detectable effect at 80% power |
| Post-earnings drift | Continued price movement in the direction of the surprise after the announcement |
| Accession | EDGAR's unique ID for a filing; the key the text is stored under |
| PERMNO | CRSP's permanent stock ID |
| I/B/E/S ticker | I/B/E/S's internal firm code, not the exchange ticker ("TGT" is Pactiv, not Target) |

&nbsp;