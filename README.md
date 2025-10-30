# PJME Hourly Load Forecasting (Deep Learning + Classical TS)

A complete, production-style time series forecasting project for hourly electricity demand (`PJME_MW`) using:
- LSTM (TensorFlow/Keras)
- SARIMA (statsmodels) with compact auto-tuning to reduce MAPE
- Moving Average (baseline smoothing)

All models follow a clean repository structure, produce reproducible outputs, and save artifacts into model-specific folders under `output/`.

## 1) Dataset
- File: `data/PJME_hourly.csv`
- Columns: `Datetime`, `PJME_MW`
- Frequency: Hourly (some gaps and duplicates are handled in preprocessing)

## 2) Repository Structure
```
Load-Forecasting/
├── src/                          # Model scripts
│   ├── lstm_forecast.py          # LSTM forecasting pipeline
│   ├── sarima_forecast.py        # SARIMA with auto-tuning (MAPE-oriented)
│   └── moving_average_forecast.py# MA smoothing (baseline)
│
├── data/                         # Raw dataset(s)
│   └── PJME_hourly.csv
│
├── output/                       # Auto-created at runtime
│   ├── LSTM/                     # All LSTM artifacts (plots + model)
│   ├── SARIMA/                   # All SARIMA artifacts (plots)
│   └── Moving_Average/           # All MA artifacts (plots)
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── .gitignore                    # Clean repo hygiene (ignores output/*)
```

## 3) Environment Setup
- Python 3.8+
- Install dependencies:
```bash
pip install -r requirements.txt
# for LSTM
pip install tensorflow keras
```

## 4) How to Run (from repo root)
- LSTM:
```bash
python src/lstm_forecast.py
```
- SARIMA:
```bash
python src/sarima_forecast.py
```
- Moving Average:
```bash
python src/moving_average_forecast.py
```
All artifacts are saved in `output/<ModelName>/`.

## 5) Methodology (Shared Preprocessing)
For all models, we:
- Parse `Datetime` and set it as index
- Sort by time
- Drop duplicate timestamps
- Resample to hourly with `.asfreq('h')`
- Interpolate missing values time-wise, then back/forward fill residual edges
- Ensure no `NaN` remains before modeling

This standardization guarantees consistent, clean inputs across the models.

## 6) LSTM Pipeline (src/lstm_forecast.py)
- Preprocessing: as above
- Feature engineering: add `hour`, `dayofweek`, `month`, `year`, `dayofyear` (for analysis; univariate forecast uses only `PJME_MW` as input)
- Train-test split: time-based at `2017-01-01` (train < 2017-01-01, test ≥ 2017-01-01)
- Scaling & windowing:
  - `MinMaxScaler` on target
  - Create sliding windows of 24 hours to predict the next hour (X: 24 steps, y: next step)
  - Reshape X to `(samples, timesteps, features)` for LSTM
- Model:
  - `Input(shape=(24, 1))`
  - `LSTM(64)` + `Dropout(0.2)` + `Dense(1)`
  - `optimizer='adam'`, `loss='mse'`
- Training:
  - 30 epochs, batch size 128, validation_split 0.1
  - Saves `output/LSTM/lstm_training_loss.png`
- Evaluation:
  - Predict on test windows; inverse-transform
  - Metrics: RMSE, MAE, MAPE (percentage)
  - Plots:
    - `output/LSTM/lstm_vs_actual.png` (full test)
    - `output/LSTM/lstm_vs_actual_recent.png` (last 3 months)
- Artifacts:
  - Model saved at `output/LSTM/lstm_pjme_model.keras`

## 7) SARIMA Pipeline (src/sarima_forecast.py)
Goal: Forecast next 7 days using only the last 1 year of data, while minimizing MAPE.

- Preprocessing: same as above
- Train/test split for 7-day forecast:
  - Train window: last 365 days prior to the final 7 days in the dataset
  - Test horizon: next 7 days (168 hours)
- Auto-tuning (compact, MAPE-oriented):
  - Candidate set of sensible SARIMA orders (daily seasonality `m=24`)
  - Validation on the last 7 days of the train window
  - Optional `log1p` transform is tried for variance stabilization; inverse-transformed for scoring
  - Best config chosen by lowest validation MAPE
- Fit & Forecast:
  - Fit best config on full training year
  - Forecast 168 hours
- Metrics:
  - RMSE and MAPE (percentage)
- Plots (saved to `output/SARIMA/`):
  - `sarima_forecast.png` (train + next 7 days forecast)
  - `sarima_vs_actual.png` (actual vs predicted for 7-day horizon)

## 8) Moving Average Baseline (src/moving_average_forecast.py)
- Preprocessing: same standardization
- Focus on last 30 days for visualization speed
- Rolling averages: 6h, 12h, 24h, 72h, 168h
- Plots (saved to `output/Moving_Average/`):
  - `moving_avg_raw.png` (raw last-30D)
  - `moving_avg_smoothed.png` (smoothing comparison)

---

## 9) Research Concepts and Rationale

### 9.1 Time Series Fundamentals
- **Trend**: Long-term increase/decrease in demand (e.g., growth, macro factors).
- **Seasonality**: Systematic periodic patterns. For hourly load, strong daily (`m=24`) and weekly (`m=168`) seasonality are typical.
- **Cyclicality**: Irregular long-horizon fluctuations (economy, weather regimes).
- **Stationarity**: Many statistical models assume constant mean/variance over time. SARIMA enforces differencing (d, D) to approximate stationarity.
- **Resampling & Alignment**: `asfreq('h')` inserts missing timestamps; interpolation ensures continuous signals for both SARIMA and LSTM.

### 9.2 SARIMA Theory (ARIMA + Seasonal)
- **ARIMA(p, d, q)**: Autoregression (p), differencing (d), moving average (q).
- **Seasonal ARIMA (P, D, Q, m)**: Same components applied at seasonal lag `m` (here daily: 24). Differencing orders d, D control trend/seasonal stationarity.
- **Identification**: Conventionally via ACF/PACF and unit root/seasonal tests. Here we use a compact candidate set curated for hourly electricity load to keep tuning fast and robust.
- **Estimation & Selection**: Maximum-likelihood estimation (MLE). We select by validation MAPE (forecast accuracy) rather than just AIC, which better aligns with the objective.
- **Variance Stabilization**: Optional `log1p` improves Gaussianity and reduces heteroscedasticity; we invert via `expm1` for evaluation.

### 9.3 LSTM for Sequence Forecasting
- **Why LSTM**: Captures long-range dependencies and nonlinear interactions that linear models underfit (e.g., complex load-weather-calendar effects, nonstationarities).
- **Windowed Supervision**: We form supervised samples by slicing the last 24 hours to predict the next hour—this is standard for recurrent models.
- **Scaling**: Neural nets train better on normalized targets; we use `MinMaxScaler` fit only on train to avoid leakage, then inverse-transform predictions for evaluation.
- **Capacity vs. Overfit**: 64 units + dropout (0.2) balances expressivity with regularization. Validation loss curves reveal overfit/underfit dynamics.

### 9.4 Metrics and Their Caveats
- **RMSE**: Punishes large errors quadratically; scale-dependent (MW). Useful for reliability emphasis.
- **MAE**: Linear penalty; robust to outliers relative to RMSE.
- **MAPE**: Scale-free (%) and intuitive for business users, but unstable when true values near zero (we guard with small denominators where needed). For baseload series like PJME, zeros are rare.

### 9.5 Validation Protocols
- **Temporal Splits**: Never shuffle time series. We use fixed cutoff or rolling-origin.
- **Rolling-Origin Backtesting (suggested extension)**: Walk-forward multiple folds for stable estimates; average errors across folds.
- **Leakage Prevention**: Fit scalers and model hyperparameters on training windows only; generate windows that do not peek into the future.

### 9.6 Reproducibility, Efficiency, and Ethics
- **Reproducibility**: Fixed paths, scripted pipelines, segregated outputs, and printed configs/metrics.
- **Compute Efficiency**: Compact SARIMA search and small LSTM reduce run time and energy.
- **Energy/Ethics**: Forecasts can guide grid efficiency; we minimize compute to lower energy footprint.

### 9.7 Limitations and Future Work
- **Univariate**: Current LSTM uses only `PJME_MW`. Add temperature, calendar holidays, and exogenous variables for multivariate gains.
- **Single-Step Horizon**: Extend to multi-horizon (e.g., 24-step direct or seq2seq).
- **Seasonality m**: We use daily (24). Weekly seasonality (168) may further improve classical models.
- **Model Selection**: Swap in `pmdarima`’s `auto_arima` for broader, automated search.
- **Hyperparameter Tuning**: Optuna/KerasTuner for LSTM; MLflow for experiment tracking.

---

## 10) Outputs and Conventions
- All outputs are saved under `output/` in model-specific subfolders:
  - `output/LSTM/`
  - `output/SARIMA/`
  - `output/Moving_Average/`
- `.gitignore` excludes `output/*` to keep git history clean

## 11) Reproducibility & Tips
- Always run from the repository root (paths in scripts assume this)
- Ensure the dataset sits at `data/PJME_hourly.csv`
- If you edit the dataset or file paths, reflect changes in the scripts

## 12) Troubleshooting
- FileNotFoundError for CSV: confirm path is `data/PJME_hourly.csv` and you run from repo root
- NaN losses (LSTM): ensure interpolation is in place (already implemented)
- Long SARIMA tuning: compact MAPE-oriented search used; expand/shrink candidates as needed
- TensorFlow logs noisy: suppressed via `TF_CPP_MIN_LOG_LEVEL=2`

## 13) Roadmap / Extensions
- Multi-step forecasting (e.g., predict 24 hours ahead at once)
- Multivariate LSTM (include calendar/exogenous features)
- Auto-ARIMA (pmdarima) comparison
- Hyperparameter search (Optuna/KerasTuner)
- Model tracking (MLflow) and CI pipelines

---

Maintainer: [Kalesha Shaik](https://www.linkedin.com/in/kalesha-shaik-a27a302b6/)
