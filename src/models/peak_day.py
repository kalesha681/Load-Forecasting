
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from ..config import ensure_output_dir, PROCESSED_DATA_DIR
from ..utils import validate_path



from pathlib import Path

def run_peak_day_pipeline(yearly_path, peak_path, output_dir):
    """Run Peak Day Analysis Pipeline."""
    output_dir = Path(output_dir) / 'Peak_Day'
    output_dir.mkdir(parents=True, exist_ok=True)
    PLOT_PATH = output_dir / 'peak_day_forecast.png'
    METRICS_PATH = output_dir / 'metrics.csv'

    METRICS_PATH = output_dir / 'metrics.csv'
    
    # Security Check
    yearly_path = validate_path(yearly_path)
    peak_path = validate_path(peak_path)
    output_dir = validate_path(output_dir)

    print(f"Loading yearly: {yearly_path}")
    yearly_df = pd.read_csv(yearly_path, index_col=0, parse_dates=True)
    
    print(f"Loading peak: {peak_path}")
    peak_df = pd.read_csv(peak_path, index_col=0, parse_dates=True)
    
    if peak_df.empty:
        print("Peak data empty.")
        return

    peak_date = peak_df.index[0].date()
    col = peak_df.columns[0]
    
    # Train on data before peak date
    train_df = yearly_df[yearly_df.index.date < peak_date]
    if train_df.empty:
        print("No training data before peak day.")
        return

    y_train = train_df.iloc[:, 0]
    # Limit history
    y_train = y_train.iloc[-24*28:]
    
    # SARIMA
    model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,24),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=50)
    
    forecast = fit.forecast(steps=len(peak_df))
    forecast.index = peak_df.index
    
    actual_peak = peak_df[col].max()
    pred_peak = forecast.max()
    error_pct = ((pred_peak - actual_peak) / actual_peak) * 100
    
    print(f"Peak Error: {error_pct:.2f}% (Pred: {pred_peak:.0f}, Actual: {actual_peak:.0f})")
    
    # Save
    pd.DataFrame([{
        'Actual': actual_peak, 'Forecast': pred_peak, 'Error %': error_pct
    }]).to_csv(METRICS_PATH, index=False)
    
    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(peak_df.index, peak_df[col], label='Actual', marker='o')
    plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
    plt.title(f"Peak Day Forecast (Err={error_pct:.2f}%)")
    plt.legend()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"Saved artifacts to {output_dir}")

def run_peak_day_analysis(): pass

