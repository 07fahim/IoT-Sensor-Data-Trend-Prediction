# Assessment 2: Must Explain

This document explains every significant decision made in
`notebook/co_forecasting.ipynb`. It is written to stand on its own: a reviewer
should be able to read it without opening the notebook and understand the
industrial context, why the data was cleaned and engineered the way it was, and
how the model was chosen and guarded against overfitting. All figures quoted here
are the actual values produced by the notebook run.

---

## 1. Industrial context and target variable

The data comes from an **Air Quality Chemical Multisensor Device** — an
"electronic nose" built from an array of five metal-oxide chemical sensors —
deployed at **road level in a significantly polluted area of an Italian city**.
It recorded **hourly averaged responses continuously from March 2004 to early
April 2005**, roughly thirteen months, making it one of the longest publicly
available recordings of a field-deployed air-quality sensor. Crucially, this is
**not a laboratory instrument**: it is an autonomous IoT device operating
unattended in the field, and a co-located certified reference analyzer provided
the ground-truth pollutant concentrations alongside the device's own raw sensor
responses (De Vito et al., *Sensors and Actuators B: Chemical*, 2008).

This is a genuine IoT scenario in every respect that matters for modelling. The
device runs continuously with no human in the loop, so faults are not corrected
in real time; sensor dropouts, hardware noise, and — over a year of operation —
**concept drift and sensor drift** all accumulate in the record. The original
research explicitly documents the metal-oxide sensors' cross-sensitivities and
their drift over the deployment, which is exactly the kind of "hardware noise"
the brief asks us to handle, and it is a direct, citable justification for
choosing robust, non-parametric models over ones that assume clean linear
relationships.

The **target variable is `CO(GT)`** — the ground-truth carbon-monoxide
concentration in mg/m³ — forecast **one hour ahead**. CO was chosen for three
concrete reasons. First, it is the pollutant this device's calibration research
is built around, so there is direct academic precedent for working with this
exact column from this exact dataset. Second, it has a workable missing rate
(~18% after sentinel conversion) compared with the unusable `NMHC(GT)` (~90%).
Third, forecasting CO has tangible public-health value: short-horizon prediction
feeds directly into urban air-quality alerting. The **one-hour horizon** matches
the data's native hourly sampling interval and is genuinely actionable, while
remaining achievable from the available feature history.

---

## 2. Data-cleaning and feature-engineering justification

Every cleaning decision below was driven by an evidence-gathering phase that
measured the issue on the actual loaded file before any code acted on it.

**The `-200` sentinel.** Missing or invalid readings in this dataset are encoded
as the literal value `-200`, not as `NaN`. This is the single most important
cleaning step: left untouched, `-200` is a perfectly valid-looking float that
silently corrupts everything downstream — the raw `describe()` shows an
impossible mean `CO(GT)` of −34.2 mg/m³ purely because of these sentinels, and
any squared-error loss would be dominated by phantom errors of ~200 against
fabricated values. The notebook detected **16,701** `-200` occurrences across the
numeric columns and converted them all to `NaN` before computing any statistic or
training any model. The per-column counts are telling: `NMHC(GT)` 8,443 (90.2%),
`CO(GT)` 1,683 (18.0%), `NO2(GT)` 1,642, `NOx(GT)` 1,639, and every remaining
sensor channel exactly 366 (3.9%).

**Excluding `NMHC(GT)`.** After sentinel conversion, `NMHC(GT)` is missing in
**90.23%** of rows. Imputing a column that is 90%+ absent does not recover
information — it manufactures signal that was never measured, and any model
leaning on it would be learning the imputation rule rather than the physics. This
matches independently published analyses of the dataset. The column is therefore
**excluded entirely** — not a feature, not a target.

**CSV export artifacts.** The file ships with two fully empty trailing columns
(`Unnamed: 15`, `Unnamed: 16`) and **114 trailing rows that are `NaN` across
every column** — both are export-tool artifacts, not data. They were confirmed
empty (0 non-null) and dropped, taking the frame from a raw `(9471, 17)` to a
clean `(9357, 12)`.

**Timestamp gaps — what the evidence actually showed.** The brief flagged
possible missing timestamps as a hypothesis to verify, not a confirmed fact. On
inspection it did **not** hold: parsing `Date` + `Time` into a single index and
comparing against a strict hourly grid over the valid range
(2004-03-10 18:00 → 2005-04-04 14:00) found **9,357 timestamps present versus
9,357 expected — zero genuine gaps, zero duplicates**. The series is fully
contiguous on the time axis. Rather than inflate a "missing timestamps"
narrative, the notebook states this plainly: no reindexing or time-axis gap
filling was needed. The only missingness is *within-row* values from the
`-200`→NaN conversion.

**Gap-aware imputation.** Those within-row gaps are **bimodal**, and that shape
dictated the strategy. For `CO(GT)`, 145 of 187 gap-runs are a single hour long,
but 18 runs exceed 24 hours and the longest is **173 hours (~7 days)**; the raw
sensor channels black out in long synchronized blocks (the missingness heatmap in
the notebook shows the whole array dropping out together). Interpolating a
one-hour dropout of a smoothly varying pollutant is a faithful reconstruction;
interpolating a week-long blackout would fabricate data and corrupt both training
labels and evaluation. The notebook therefore applies **time-based linear
interpolation only to short gaps (≤ 2 hours)** and **deliberately leaves longer
gaps as `NaN`**, to be dropped at feature-construction time. For `CO(GT)` this
filled 163 hours and left 1,520 (16.2%) genuinely missing.

**Feature engineering.** The forecast is framed as supervised learning: for a row
at time *t*, the label is `CO[t+1]` and every feature uses only information
available at or before *t*. The features are:

- **Autoregressive CO history** — the current reading `CO[t]` plus lags at
  *t−1, t−2, t−3*. CO is strongly autocorrelated, so recent history is the single
  best predictor of the near future; the permutation-importance analysis confirms
  `CO_current` dominates.
- **Rolling 24-hour mean and standard deviation** of CO, with `min_periods=12`.
  The window of 24 matches one full diurnal cycle (the EDA shows the expected
  twin rush-hour peaks and overnight trough); `min_periods=12` requires at least
  half a day of data before emitting a value, trading a little coverage for a
  statistic that is not built from one or two stray points.
- **Co-located device sensor channels at *t***, gated by a correlation check
  rather than assumed. On the cleaned data the correlations with `CO(GT)` are
  `C6H6(GT)` 0.93, `PT08.S2(NMHC)` 0.92, `PT08.S1(CO)` 0.88, `PT08.S5(O3)` 0.86 —
  all strong — so the three device-sensor channels (`PT08.S1(CO)`,
  `PT08.S5(O3)`, `PT08.S2(NMHC)`) were included. The reference-analyzer columns
  were deliberately left out to keep a clean narrative: *forecast CO from the
  device's own sensor array plus CO history.*
- **Calendar features** (`hour`, `dow`) for the diurnal and weekly cycle.

One anticipated feature was **dropped on the evidence**: the brief expected
temperature `T` to be moderately predictive, but its correlation with `CO(GT)` is
**0.031** — effectively zero. Following the brief's own instruction not to force a
weak feature, `T` (along with `RH` 0.04 and `AH` 0.05) was excluded.

**Leakage prevention.** No lag or rolling window reaches forward in time. The only
forward-looking quantity is the label `CO[t+1]`, which is never used as an input.
This is set out explicitly in the notebook as a per-feature table.

---

## 3. Model architecture and overfitting guard

**Selection within the trend-regression family.** Five candidates were compared —
`LinearRegression` and `Ridge` (scaled), `RandomForest`, `GradientBoosting`, and
`HistGradientBoosting` — using **`TimeSeriesSplit` cross-validation on the
training set only**, with five expanding-window folds that respect temporal
order. Selection was made on mean CV RMSE and never on the test set, so model
choice itself cannot leak information from the held-out window. The CV ranking
was `HistGradientBoosting` 0.666, `GradientBoosting` 0.670, `RandomForest` 0.674,
`Ridge`/`LinearRegression` 0.731 — so **HistGradientBoosting** was selected.

**Why a boosted-tree model fits this problem.** The engineered feature set is
tabular, and gradient-boosted trees are the strongest general method on tabular
data of this size. They capture nonlinear interactions (for example, lag × sensor
effects), assume nothing about stationarity, and are robust to the hardware noise
and drift documented for this device — exactly the properties the De Vito et al.
context calls for. The chosen hyperparameters are deliberate regularization
choices, not defaults: **`max_depth = 3`** keeps each tree a weak learner that
captures only low-order interactions; **`max_iter = 400`** with a modest
**`learning_rate = 0.05`** follows the standard boosting trade of many small
corrective steps over few large ones; and **`l2_regularization = 1.0`** further
penalises leaf over-confidence.

**The overfitting guard, and the empirical proof.** Three independent safeguards
support generalisation. First, the **chronological split** — the first 80% of
rows train, the last 20% test, never shuffled; shuffling would leak future
timestamps into training and make test performance meaninglessly optimistic on a
time series. Second, **leakage-free features**, as set out in Section 2. Third,
**deliberately capped model complexity** with the hyperparameters above. The
proof that this works is the train-versus-test table:

| Split | RMSE | MAE |
|-------|------|-----|
| Train | 0.529 | 0.363 |
| Test  | 0.563 | 0.379 |

The test RMSE sits only **~6%** above the train RMSE — well inside the
"generalises well" band (15–20%) and nowhere near the 30% overfitting flag. For
context, the model **beats a persistence baseline by 28.9%** on test RMSE
(0.794 → 0.563), confirming it has learned genuine structure rather than echoing
the last reading.

The model-selection process supplies its own counter-example, which is what makes
the guard meaningful: `RandomForest` posted the *lowest raw test RMSE* (0.559) but
a **~67% train-to-test gap** — the textbook signature of memorisation — and
because cross-validation ranked it *below* the boosted models, principled
selection correctly avoided it. This is precisely why the model was chosen by CV
rather than by the single test number.

**Cross-paradigm comparison.** To answer empirically which *family* of method is
best for this dataset, one representative of each category named in the brief was
evaluated on the **same held-out timestamps** (1,381 common test hours):

| Paradigm | Model | Test RMSE | Test MAE |
|----------|-------|:---------:|:--------:|
| Sequential NN | LSTM (24h lookback, 4 channels) | **0.531** | 0.371 |
| Trend-regression | HistGradientBoosting | 0.561 | 0.377 |
| Time-series | SARIMAX(1,0,1)(1,0,1,24) | 0.568 | 0.385 |
| baseline | Persistence | 0.794 | 0.514 |
| baseline | Seasonal-naive | 1.131 | 0.776 |

All three engineered paradigms beat the naive baselines decisively and cluster
within ~0.04 RMSE of one another — there is a real signal ceiling for one-hour-
ahead CO, and every reasonable method approaches it. The **LSTM is marginally
best**, learning sequential dynamics directly from the raw 24-hour window;
**HistGradientBoosting is within run-to-run noise of it** while being
deterministic, roughly ten times faster, and interpretable via permutation
importance; **SARIMAX is a close third**, impressive for a univariate model that
never sees the exogenous sensors. The two naive baselines are included as the
reference floor that makes these numbers meaningful — and they carry a small
insight of their own: seasonal-naive (same hour yesterday) is *worse* than
persistence (last hour), because at a one-hour horizon the most recent reading is
more informative than the same hour a day earlier. Both front-runners also pass
the overfitting guard on their own terms: the LSTM's test RMSE (0.531) is actually
*below* its train RMSE (0.561), and HistGradientBoosting's test sits ~6% above its
train — neither memorises the training window.

**Practical conclusion.** The LSTM wins on raw accuracy, but the margin over
HistGradientBoosting is within noise, so for a production IoT setting the
boosted-tree model is the recommended choice — reproducible, cheap, and
explainable — unless that small accuracy edge is operationally critical. The
permutation-importance breakdown for the selected model (`CO_current` ≈ 0.87,
`hour` ≈ 0.18, `CO_lag1` ≈ 0.08, `PT08.S2(NMHC)` and `PT08.S1(CO)` ≈ 0.06 each)
confirms the story the EDA told: recent CO dominates, the diurnal `hour` feature
carries real signal, and the device's correlated sensor channels add the rest.

---

### Reference

S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, "On field
calibration of an electronic nose for benzene estimation in an urban pollution
monitoring scenario," *Sensors and Actuators B: Chemical*, Vol. 129, Issue 2,
2008, pp. 750–757.
