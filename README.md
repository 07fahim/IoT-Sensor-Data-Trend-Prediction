# IoT Sensor Data Trend Prediction — Carbon Monoxide (CO) Forecasting

An end-to-end machine-learning pipeline that takes raw, noisy readings from a
field-deployed air-quality IoT device and forecasts the **carbon-monoxide
concentration `CO(GT)` one hour ahead**. The project runs the full lifecycle in a
single, navigable Jupyter notebook: data sourcing → evidence-based cleaning →
leakage-free feature engineering → a cross-paradigm model comparison
(trend-regression vs. classical time-series vs. a sequential neural network) →
evaluation with RMSE/MAE and a ground-truth-vs-predicted visualization.

## Dataset and attribution

> Dataset: UCI Air Quality Dataset, collected from a field-deployed Air Quality Chemical Multisensor Device in an Italian city (March 2004 - February 2005). Original research: S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, "On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario," Sensors and Actuators B: Chemical, 2008. Source: UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets/Air+Quality

## Results at a glance

Forecasting `CO(GT)` 1 hour ahead, evaluated on a held-out chronological test
window (last 20% of the series, identical timestamps across all models):

| Paradigm | Model | Test RMSE | Test MAE |
|----------|-------|:---------:|:--------:|
| Sequential NN | LSTM (24h lookback) | **0.531** | 0.371 |
| Trend-regression | HistGradientBoosting | 0.561 | 0.377 |
| Time-series | SARIMAX(1,0,1)(1,0,1,24) | 0.568 | 0.385 |
| _baseline_ | Persistence (`CO[t+1]=CO[t]`) | 0.794 | 0.514 |
| _baseline_ | Seasonal-naive (`CO[t+1]=CO[t-23]`) | 1.131 | 0.776 |

All three engineered models beat the persistence baseline by ~29% and cluster
tightly. The LSTM is marginally best, but **HistGradientBoosting** is the
recommended production model (deterministic, ~10x faster, interpretable) with a
train→test RMSE gap of only ~6%, confirming it generalises rather than memorises.

Full reasoning for every decision is in [`docs/explain.md`](docs/explain.md).

## Repository structure

```
iot-co-forecasting/
├── notebook/
│   └── co_forecasting.ipynb     # the full pipeline (6 phases, runs top-to-bottom)
├── data/
│   └── AirQualityUCI.csv         # place the dataset here before running
├── docs/
│   └── explain.md                # the 3 "Must Explain" answers, in full prose
├── requirements.txt              # pinned package versions
└── README.md
```

## Setup

1. **Place the dataset.** Put `AirQualityUCI.csv` in the `data/` folder.
   (Download from the [UCI Air Quality page](https://archive.ics.uci.edu/ml/datasets/Air+Quality);
   the loader expects the original semicolon-delimited, comma-decimal CSV. The
   notebook also auto-detects a Kaggle `/kaggle/input/**` path if run there.)

2. **Install dependencies.**
   ```bash
   pip install -r requirements.txt
   ```
   (CPU-only PyTorch: if needed, `pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu`.)

3. **Run the notebook.** Open and run `notebook/co_forecasting.ipynb` top to
   bottom (Restart & Run All). It executes end-to-end with no manual
   intervention beyond the CSV being present.

## Notebook structure (6 phases)

1. **Data Sourcing** — load the raw CSV, inspect shape/dtypes/head/tail/describe.
2. **Data Quality Investigation** — empirically verify the `-200` sentinels,
   empty columns, trailing rows, `NMHC(GT)` missing rate, and hourly-gap
   hypothesis (with EDA plots).
3. **Data Cleaning** — drop artifacts, build a contiguous hourly `DatetimeIndex`,
   convert `-200`→NaN, exclude `NMHC(GT)`, gap-aware short-gap interpolation.
4. **Feature Engineering** — correlation-gated features: CO lags, rolling-24h
   stats, strongly-correlated device sensor channels, calendar features; explicit
   leakage check.
5. **Modeling** — CV-based model selection within the trend-regression family,
   then a cross-paradigm bake-off (time-series, sequential NN, trend-regression)
   on identical test timestamps.
6. **Evaluation and Visualization** — RMSE/MAE, actual-vs-predicted over the test
   window (with a 14-day zoom) and a train-vs-test overfitting sanity check.
