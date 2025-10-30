# ⚡️ Electric Load Forecasting — LSTM, SARIMA, and Baselines

> “All models are wrong, but some are useful.” — A time‑series exploration with deep learning and classical methods, presented with academic rigor and practical clarity.

## 🧭 Project Overview
This repository forecasts hourly electricity demand (PJME_MW) using:
- LSTM (deep learning for nonlinear temporal patterns)
- SARIMA (seasonal ARIMA, classical statistical baseline)
- Moving Average (smooth baseline)

It includes clean preprocessing, leakage‑safe splits, visualizations, metrics (RMSE/MAE/MAPE), and a professional research‑grade documentation.

## 🗂 Dataset Description
- Source file: `data/PJME_hourly.csv`
- Columns: `Datetime`, `PJME_MW`
- Frequency: Hourly (gaps/duplicates handled via resampling and interpolation)
- Period: Multi‑year PJM electricity demand series

## 🧪 Methodology
- Preprocessing: sort by time, drop duplicate timestamps, align to hourly (`asfreq('h')`), interpolate missing values.
- Models:
  - ARIMA/SARIMA: Seasonal ARIMA with compact auto‑tuning oriented to reduce MAPE (daily seasonality `m=24`).
  - LSTM: Univariate windowing (last 24 hours → next hour), `MinMaxScaler`, `LSTM(64) + Dropout(0.2) + Dense(1)`.
  - Moving Average: Rolling means (6h, 12h, 24h, 72h, 168h) for smoothing.
- Evaluation: RMSE, MAE, MAPE; holdout testing with no leakage.

## 📈 Results & Discussion (Example)
- LSTM (24→1):
  - RMSE ≈ 356 MW, MAE ≈ 259 MW, MAPE ≈ 0.84%
- SARIMA (1‑year train → 7‑day forecast):
  - Metrics depend on auto‑tuned configuration; typically performant for short horizons.
- Interpretation:
  - LSTM captures nonlinear and diurnal structure well.
  - SARIMA offers strong short‑term interpretability.
  - Both benefit from careful preprocessing and evaluation protocols.

## 🖼 Visualizations
- Predicted vs Actual (full test window)
- Zoomed (recent period)
- Training/validation loss curves (LSTM)
- SARIMA 7‑day horizon: actual vs forecast

Plots are saved under `output/<ModelName>/` (e.g., `output/LSTM/lstm_vs_actual.png`, `output/SARIMA/sarima_vs_actual.png`).

## ▶️ How to Run
Create a Python 3.8+ environment and install dependencies:
```bash
pip install -r requirements.txt
# Optional for LSTM (if not included):
pip install tensorflow keras
```
Run from the repository root:
```bash
python src/lstm_forecast.py
python src/sarima_forecast.py
python src/moving_average_forecast.py
```
Artifacts (plots/models) will appear under `output/<ModelName>/`.

## 📁 Folder Structure
```
├── data/
├── models/
├── plots/
├── src/
│   ├── lstm_forecast.py
│   ├── sarima_forecast.py
│   └── moving_average_forecast.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 👤 Author
Kalesha Shaik (Tobi) — EEE undergrad passionate about Robotics and AI‑driven systems.
- LinkedIn: https://www.linkedin.com/in/kalesha-shaik-a27a302b6/

## 🔑 License
MIT License. See `LICENSE` for details.
