
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from ..config import ensure_output_dir, PROCESSED_DATA_DIR
from ..data_loader import load_and_preprocess

def run_peak_day_analysis(yearly_path=None, peak_day_path=None):
    print("\n--- Running Peak Day Analysis ---")
    
    # Paths
    if yearly_path is None:
        yearly_path = os.path.join(PROCESSED_DATA_DIR, 'yearly_demand_National.csv')
    if peak_day_path is None:
        peak_day_path = os.path.join(PROCESSED_DATA_DIR, 'peak_day_National.csv')
        
    output_dir = ensure_output_dir('Peak_Day')
    PLOT_PATH = os.path.join(output_dir, 'peak_day_forecast.png')
    METRICS_PATH = os.path.join(output_dir, 'metrics.csv')
    
    # Load Data
    print(f"Loading Yearly Data: {yearly_path}")
    if not os.path.exists(yearly_path):
        print("Yearly data not found. Attempting to load using data loader...")
        load_and_preprocess()
    
    yearly_df = pd.read_csv(yearly_path, index_col=0, parse_dates=True)
    
    print(f"Loading Peak Day Data: {peak_day_path}")
    if not os.path.exists(peak_day_path):
        # We might need to run data loader to generate it if missing
        print("Peak day data not found.")
        return

    peak_df = pd.read_csv(peak_day_path, index_col=0, parse_dates=True)
    
    if peak_df.empty:
        print("Peak Day Data is empty.")
        return

    # Identify Peak Day Date
    peak_date = peak_df.index[0].date()
    print(f"Peak Day identified as: {peak_date}")
    
    # Identify the column name (robustness)
    col_name = peak_df.columns[0]
    
    # Prepare Training Data (All data before Peak Day)
    # Ensure yearly_df index is datetime
    train_df = yearly_df[yearly_df.index.date < peak_date]
    
    if train_df.empty:
        print("Warning: No training data found before peak day. Using all yearly data except peak day (if overlap).")
        # For demo purposes, if yearly data STARTS after peak day (unlikely but possible if files messed up), we can't forecast properly.
        # But assuming yearly data covers the year.
        return

    # Use log transform if helpful, but for short term 24h, raw is often fine.
    # Let's use simple SARIMA (1,1,1)(1,1,1,24) as requested/standard.
    y_train = train_df.iloc[:, 0] # First column
    
    # Limit training history for speed (last 4 weeks is enough for short term pattern)
    history_window = 24 * 28
    if len(y_train) > history_window:
        y_train = y_train.iloc[-history_window:]
    
    print("Training SARIMA model (Short Horizon)...")
    model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,24),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=50, method='lbfgs')
    
    # Forecast horizon matches peak_df length (usually 24h)
    steps = len(peak_df)
    print(f"Forecasting {steps} hours...")
    forecast = fit.forecast(steps=steps)
    forecast.index = peak_df.index # Align index
    
    # Metrics
    actual_peak_val = peak_df[col_name].max()
    forecast_peak_val = forecast.max()
    
    peak_error = forecast_peak_val - actual_peak_val
    peak_error_pct = (peak_error / actual_peak_val) * 100
    
    print(f"Actual Peak: {actual_peak_val:.2f} MW")
    print(f"Forecast Peak: {forecast_peak_val:.2f} MW")
    print(f"Peak Error: {peak_error:.2f} MW ({peak_error_pct:.2f}%)")
    
    # Save Metrics
    metrics_df = pd.DataFrame([{
        'Metric': 'Peak Forecast Error',
        'Actual Peak': actual_peak_val,
        'Forecast Peak': forecast_peak_val,
        'Error MW': peak_error,
        'Error %': peak_error_pct
    }])
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"Metrics saved to {METRICS_PATH}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(peak_df.index, peak_df[col_name], label='Actual Peak Day', marker='o')
    plt.plot(forecast.index, forecast, label='Forecasted Peak Day', linestyle='--', marker='x')
    plt.title(f"Peak Day Forecast vs Actual ({peak_date})\nError: {peak_error_pct:.2f}%")
    plt.xlabel("Hour")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")

if __name__ == "__main__":
    run_peak_day_analysis()
