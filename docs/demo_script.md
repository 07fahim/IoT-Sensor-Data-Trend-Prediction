# Demo Video Script — Assessment 2: IoT Sensor Data Trend Prediction

A walkthrough script for the 5–10 minute demo video. Each section lists **what to
show on screen** and **what to say** (paraphrase in your own words — don't read it
verbatim). Target total: ~8 minutes. Timings are guidance, not hard limits.

> Before recording: do a fresh **Restart & Run All** so every cell shows live
> outputs. Have `README.md`, the notebook, and `docs/explain.md` open in tabs.

---

## 0. Intro (0:00–0:30)

**Show:** README.md title + the results table.

**Say:**
> "This is my submission for Assessment 2 — an end-to-end ML pipeline that forecasts
> carbon-monoxide concentration one hour ahead from a real, field-deployed IoT air-
> quality sensor. I'll walk through the dataset, the data cleaning, feature
> engineering, the model comparison, and the final evaluation. Everything runs
> top-to-bottom in a single notebook."

---

## 1. Dataset context & attribution (0:30–1:30)

**Show:** Notebook title cell (the attribution blockquote) and Section 1 outputs.

**Say:**
> "The data is the UCI Air Quality dataset — hourly readings from a metal-oxide
> 'electronic nose' deployed at road level in an Italian city, running continuously
> for about thirteen months in 2004–2005. It's a genuine IoT scenario: autonomous,
> unattended, with sensor noise and drift accumulating over time. The original
> research is De Vito et al., 2008 — cited in the README and notebook.
> My target is `CO(GT)`, the ground-truth CO concentration, forecast one hour ahead."

**Point out:** raw shape `(9471, 17)`, the semicolon/comma-decimal format.

---

## 2. Data quality findings (1:30–3:00)

**Show:** Section 2 — the `-200` sentinel table, NMHC missing rate, the gap check,
and the two EDA plots (raw CO time series + missingness heatmap).

**Say:**
> "Before cleaning, I empirically verified every data-quality issue. The key one:
> missing values are encoded as the literal `-200`, not NaN — I found 16,701 of them.
> Left untreated, the raw mean CO comes out as minus 34, which is physically
> impossible. `NMHC(GT)` is 90% missing, so I exclude it entirely.
> I also tested the brief's hypothesis about missing timestamps — and it didn't hold:
> the series is perfectly contiguous hourly, zero gaps. I report that honestly rather
> than invent a missing-timestamp story."

**Point out (missingness heatmap):**
> "This heatmap is my favourite finding — the long blackouts hit the *whole sensor
> array at once*. The device dropped out as a unit. That's why I don't interpolate
> across long gaps."

---

## 3. Cleaning decisions (3:00–4:00)

**Show:** Section 3 — the cleaning steps and the gap-aware imputation table.

**Say:**
> "Cleaning, in order: drop the two empty artifact columns and 114 empty trailing
> rows, build a sorted hourly DatetimeIndex, convert all `-200` to NaN, and exclude
> NMHC. For the remaining gaps I use a *gap-aware* rule: the missingness is bimodal —
> mostly one-hour dropouts, but some blackouts up to seven days. I interpolate only
> short gaps of two hours or less, and deliberately leave the long ones as NaN to be
> dropped later. I never fabricate a week of CO data."

---

## 4. Feature engineering (4:00–5:00)

**Show:** Section 4 — correlation output, correlation heatmap, diurnal-cycle plot,
the feature matrix, and the leakage-check table.

**Say:**
> "Features are all leakage-free — for a row at time t, I only use data at t or
> earlier; the label is CO at t+1. I use CO's own history — the current value plus
> three lags — rolling 24-hour mean and std, and the device's correlated sensor
> channels. I gate features on correlation rather than assuming: the brief expected
> temperature to matter, but its correlation with CO is 0.03 — basically zero — so I
> dropped it. The diurnal plot shows the twin rush-hour peaks, which is why the
> hour-of-day feature and the 24-hour window carry real signal."

---

## 5. Modeling & the cross-paradigm comparison (5:00–6:45)

**Show:** Section 5 — the CV selection table, then 5.5 cross-paradigm table + bar chart.

**Say:**
> "I didn't commit to one model. First, within the trend-regression family I select by
> time-series cross-validation on the training set only — never peeking at the test
> set. That picked HistGradientBoosting. Importantly, RandomForest had the lowest raw
> test error but a 67% train-test gap — pure memorisation — and cross-validation
> correctly rejected it. That's the overfitting guard working.
>
> Then, because the brief names three model families, I ran a full cross-paradigm
> bake-off: a classical time-series model (SARIMAX), a sequential neural network
> (an LSTM), and the boosted trees — all scored on the *same* test timestamps."

**Point out (bar chart):**
> "The result: all three real models beat the naive baselines by around 29%, and they
> cluster tightly. The LSTM edges ahead at 0.531 RMSE, with HistGradientBoosting and
> SARIMAX just behind. There's a real signal ceiling for one-hour-ahead CO and every
> sound method approaches it. One honest detail — the seasonal-naive baseline is
> *worse* than persistence, because at a one-hour horizon last hour beats same-hour-
> yesterday."

---

## 6. Evaluation & visualization (6:45–8:00)

**Show:** Section 6 — the full-window actual-vs-predicted plot, the 14-day zoom, and
the train-vs-test overfitting bar chart.

**Say:**
> "Here's the forecast against ground truth over the held-out test window. The
> predicted trace tracks the real CO signal closely, including the sharp spikes — and
> notice the gaps render as breaks, not fake straight lines, because I refuse to draw
> across the sensor blackouts. The zoom shows the hour-by-hour diurnal tracking.
>
> Finally, the overfitting check: for *both* leading models, test RMSE is essentially
> equal to train RMSE — the LSTM's test is even slightly below its train. So the
> winning model genuinely generalises; it isn't memorising. For production I'd
> actually recommend HistGradientBoosting — it's within noise of the LSTM but
> deterministic, faster, and interpretable."

---

## 7. Wrap-up (8:00–8:30)

**Show:** `docs/explain.md` scrolling briefly, then the repo structure.

**Say:**
> "Every decision is written up in full in docs/explain.md — the cleaning rationale,
> the feature choices, the model selection, and the overfitting evidence. The repo has
> the notebook, the docs, pinned requirements, and a README with setup instructions.
> Thanks for watching."

---

## Checklist — make sure the video shows all of these (per the brief)

- [ ] Dataset context **and** attribution (Section 1 / README)
- [ ] Verified data-quality findings (Section 2: `-200`, NMHC, gap check)
- [ ] Cleaning decisions (Section 3, incl. gap-aware imputation)
- [ ] Feature engineering (Section 4, incl. correlation-gated `T` drop)
- [ ] Model training (Section 5: CV selection + cross-paradigm bake-off)
- [ ] Final evaluation plot — actual vs predicted (Section 6.2/6.3)
- [ ] Trend evaluation graphs walkthrough (Section 6 + overfitting bars)

## Quick reference — headline numbers to quote

| Metric | Value |
|--------|-------|
| `-200` sentinels converted | 16,701 |
| `NMHC(GT)` missing | 90.23% (excluded) |
| Timestamp gaps | 0 (fully contiguous) |
| Modeling rows | 7,192 (train 5,753 / test 1,439) |
| Best model | LSTM — test RMSE **0.531**, MAE 0.371 |
| HistGradientBoosting | test RMSE 0.561, MAE 0.377, gap +6.4% |
| SARIMAX | test RMSE 0.568, MAE 0.385 |
| Improvement over persistence | ~29% |
