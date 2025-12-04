import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from joblib import Parallel, delayed

from ..config import ensure_output_dir
from ..data_loader import load_and_preprocess
from ..metrics import evaluate_mape

def evaluate_candidate(order, seasonal_order, train_sub, val_sub, log_flag, seasonal_period):
    try:
        y_train = np.log1p(train_sub) if log_flag else train_sub
        y_val = np.log1p(val_sub) if log_flag else val_sub
        
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=80, method='lbfgs')
        pred = fit.forecast(steps=len(val_sub))
        
        if log_flag:
            pred_inv = np.expm1(pred)
            y_val_inv = np.expm1(y_val)
            mape = evaluate_mape(y_val_inv.values, pred_inv.values)
        else:
            mape = evaluate_mape(y_val.values, pred.values)
            
        if np.isfinite(mape):
            return (order, seasonal_order, mape, log_flag)
    except Exception:
        pass
    return None

def tune_sarima_by_mape(train_series, seasonal_period=24, use_log=True):
    """Fast candidate search using validation MAPE on last 7 days (Parallelized)."""
    horizon = 24 * 7
    if len(train_series) <= horizon + 24:
        return (1, 1, 1), (1, 1, 1, seasonal_period), np.inf, False

    candidates = [
        ((1,1,1), (1,1,1,seasonal_period)),
        ((2,1,2), (1,1,1,seasonal_period)),
        ((1,1,2), (1,1,1,seasonal_period)),
        ((2,1,1), (1,1,1,seasonal_period)),
        ((0,1,1), (0,1,1,seasonal_period)),
        ((1,1,0), (0,1,1,seasonal_period)),
    ]

    train_sub = train_series.iloc[:-horizon]
    val_sub = train_series.iloc[-horizon:]
    
    tasks = []
    log_options = ([True] if use_log else [False]) + [False]
    for log_flag in set(log_options):
        for order, seasonal_order in candidates:
            tasks.append((order, seasonal_order, log_flag))
            
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_candidate)(order, seasonal_order, train_sub, val_sub, log_flag, seasonal_period)
        for order, seasonal_order, log_flag in tasks
    )
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return ((1,1,1), (1,1,1,seasonal_period), np.inf, False)
        
    valid_results.sort(key=lambda x: x[2])
    return valid_results[0]

def run_sarima():
    # Output setup
    output_dir = ensure_output_dir('SARIMA')
    PLOT1_PATH = os.path.join(output_dir, "sarima_forecast.png")
    PLOT2_PATH = os.path.join(output_dir, "sarima_vs_actual.png")

    # Data Loading
    df = load_and_preprocess()

    # Train/test split (1 year for training, next 7 days for testing)
    last_train_date = df.index.max() - pd.Timedelta(days=7)
    train_start_date = last_train_date - pd.Timedelta(days=365)
    train = df.loc[train_start_date:last_train_date, 'PJME_MW']
    test = df.loc[last_train_date+pd.Timedelta(hours=1):(last_train_date+pd.Timedelta(days=7)), 'PJME_MW']

    print(f"Training window: {train.index.min()} -- {train.index.max()} ({len(train)} hrs)")
    print(f"Test window (prediction): {test.index.min()} -- {test.index.max()} ({len(test)} hrs)")

    # Auto-tune
    print("\nTuning SARIMA hyperparameters (validation MAPE search)...")
    best_order, best_seasonal_order, val_mape, use_log = tune_sarima_by_mape(train, seasonal_period=24, use_log=True)
    print(f"Best order: {best_order}, Best seasonal_order: {best_seasonal_order}, Validation MAPE: {val_mape*100:.2f}% (log={use_log})")

    # Fit best model
    y_train_full = np.log1p(train) if use_log else train
    model = SARIMAX(y_train_full, order=best_order, seasonal_order=best_seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=100, method='lbfgs')
    pred = fit.forecast(steps=7*24)
    forecast = np.expm1(pred) if use_log else pred
    forecast = pd.Series(forecast, index=test.index)

    # Plot 1
    plt.figure(figsize=(15,6))
    plt.plot(train.index, train, label='Train Data', color='blue')
    plt.plot(test.index, test, label='Actual Load (Next 7 Days)', color='black')
    plt.plot(test.index, forecast, label='SARIMA Forecast', color='red', linestyle='--')
    plt.title("SARIMA Forecast: Next 7 Days (Trained on Last Year, Auto-tuned by MAPE)")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT1_PATH)
    plt.show()

    # Evaluation
    rmse = float(np.sqrt(np.mean((test.values - forecast.values)**2)))
    mape = evaluate_mape(test.values, forecast.values)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape*100:.2f}% (Mean Absolute Percentage Error)")

    # Plot 2
    plt.figure(figsize=(12,5))
    plt.plot(test.index, test, label='Actual Load', linewidth=2)
    plt.plot(test.index, forecast, label='Predicted Load', linestyle='--', color='red')
    plt.title('Actual vs Predicted Load (SARIMA, Next 7 Days, Auto-tuned by MAPE)')
    plt.xlabel('Time')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT2_PATH)
    plt.show()

if __name__ == "__main__":
    run_sarima()
