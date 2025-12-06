import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from joblib import Parallel, delayed

from ..config import ensure_output_dir, COL_DEMAND_YEARLY, SPLIT_DATE
from ..data_loader import train_test_split
from ..metrics import evaluate_mape, evaluate_rmse

def evaluate_candidate(order, seasonal_order, train_sub, val_sub, log_flag, seasonal_period):
    try:
        y_train = np.log1p(train_sub) if log_flag else train_sub
        y_val = np.log1p(val_sub) if log_flag else val_sub
        
        # Fast fit
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False, maxiter=50, method='lbfgs')
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
    """Fast candidate search using validation MAPE on recent history (Parallelized)."""
    # Validation horizon: 7 days
    horizon = 24 * 7
    if len(train_series) <= horizon + 24:
        # Not enough data
        return (1, 1, 1), (1, 1, 1, seasonal_period), np.inf, False

    # Candidates (simplified list for speed, can be expanded)
    candidates = [
        ((1,1,1), (1,1,1,seasonal_period)),
        ((1,1,1), (0,1,1,seasonal_period)),
        ((0,1,1), (0,1,1,seasonal_period)),
        ((1,0,1), (0,1,1,seasonal_period)),
    ]

    # Use last 3 months for tuning to speed up
    tuning_window = 24 * 90 
    train_for_tuning = train_series.iloc[-tuning_window:] if len(train_series) > tuning_window else train_series
    
    # Split validation set
    train_sub = train_for_tuning.iloc[:-horizon]
    val_sub = train_for_tuning.iloc[-horizon:]
    
    tasks = []
    log_options = ([True] if use_log else [False]) + [False]
    # Unique combinations
    seen = set()
    for log_flag in set(log_options):
        for order, seasonal_order in candidates:
            if (order, seasonal_order, log_flag) not in seen:
                tasks.append((order, seasonal_order, log_flag))
                seen.add((order, seasonal_order, log_flag))
            
    print(f"Parallel tuning on {len(tasks)} candidates...")
    results = Parallel(n_jobs=-1)(
        delayed(evaluate_candidate)(order, seasonal_order, train_sub, val_sub, log_flag, seasonal_period)
        for order, seasonal_order, log_flag in tasks
    )
    
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        # Fallback
        return ((1,1,1), (1,1,1,seasonal_period), np.inf, False)
        
    valid_results.sort(key=lambda x: x[2]) # Sort by MAPE
    best = valid_results[0]
    return best


def run_sarima_pipeline(data_path, output_dir):
    """
    Run SARIMA pipeline on specified data.
    """
    output_dir = Path(output_dir) / 'SARIMA'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    PLOT1_PATH = output_dir / "sarima_forecast.png"
    PLOT2_PATH = output_dir / "sarima_vs_actual.png"
    METRICS_PATH = output_dir / "metrics.csv"
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Identify column
    if COL_DEMAND_YEARLY in df.columns:
        col = COL_DEMAND_YEARLY
    elif "Demand_MW" in df.columns:
        col = "Demand_MW"
    else:
        col = df.columns[0]

    # Train/Test Split
    train, test = train_test_split(df, test_days=7)
    train_series = train[col]
    test_series = test[col]

    print(f"Training window: {train.index.min()} -- {train.index.max()} ({len(train)} hrs)")

    # Auto-tune
    print("Tuning SARIMA...")
    best_order, best_seasonal_order, val_mape, use_log = tune_sarima_by_mape(
        train_series, seasonal_period=24, use_log=True
    )
    print(f"Best: Order={best_order}, Seasonal={best_seasonal_order}, Log={use_log}")

    # Fit
    y_train = np.log1p(train_series) if use_log else train_series
    model = SARIMAX(y_train, order=best_order, seasonal_order=best_seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=50, method='lbfgs')
    
    # Forecast
    pred_trans = fit.forecast(steps=len(test))
    forecast = np.expm1(pred_trans) if use_log else pred_trans
    forecast.index = test.index
    
    # Evaluate
    rmse = evaluate_rmse(test_series.values, forecast.values)
    mape = evaluate_mape(test_series.values, forecast.values)
    
    print(f"SARIMA Results: RMSE={rmse:.2f}, MAPE={mape*100:.2f}%")
    
    # Save
    pd.DataFrame([{
        'Model': 'SARIMA', 'RMSE': rmse, 'MAPE': mape,
        'Order': str(best_order), 'Seasonal': str(best_seasonal_order)
    }]).to_csv(METRICS_PATH, index=False)
    
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(train_series.index[-24*7:], train_series.iloc[-24*7:], label='Train (Last Week)')
    plt.plot(test_series.index, test_series, label='Actual')
    plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
    plt.title(f"SARIMA Forecast (MAPE={mape*100:.2f}%)")
    plt.legend()
    plt.savefig(PLOT1_PATH)
    plt.close()
    print(f"Saved artifacts to {output_dir}")

# Alias for legacy or direct calling if needed
def run_sarima(data_path=None, test_days=7):
    # Backward compat stub if strictly needed, but we prefer pipeline
    pass

