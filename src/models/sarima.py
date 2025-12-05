import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from joblib import Parallel, delayed

from ..config import ensure_output_dir, COL_DEMAND_YEARLY, SPLIT_DATE
from ..data_loader import load_and_preprocess, train_test_split
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

def run_sarima(data_path=None, test_days=7):
    # Output setup
    output_dir = ensure_output_dir('SARIMA')
    PLOT1_PATH = os.path.join(output_dir, "sarima_forecast.png")
    PLOT2_PATH = os.path.join(output_dir, "sarima_vs_actual.png")
    METRICS_PATH = os.path.join(output_dir, "metrics.csv")

    # Data Loading
    if data_path:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        df = load_and_preprocess()
    
    # Identify column
    col = COL_DEMAND_YEARLY if COL_DEMAND_YEARLY in df.columns else df.columns[0]
    
    # Train/Test Split
    # We use the specified test_days (default 7) for evaluation
    train, test = train_test_split(df, test_days=test_days)
    
    train_series = train[col]
    test_series = test[col]

    print(f"Training window: {train.index.min()} -- {train.index.max()} ({len(train)} hrs)")
    print(f"Test window: {test.index.min()} -- {test.index.max()} ({len(test)} hrs)")

    # Auto-tune
    print("\nTuning SARIMA hyperparameters...")
    best_order, best_seasonal_order, val_mape, use_log = tune_sarima_by_mape(train_series, seasonal_period=24, use_log=True)
    print(f"Best Configuration: Order={best_order}, Seasonal={best_seasonal_order}, Log={use_log}")
    print(f"Validation MAPE during tuning: {val_mape*100:.2f}%")

    # Fit best model on FULL training data
    y_train = np.log1p(train_series) if use_log else train_series
    
    model = SARIMAX(y_train, order=best_order, seasonal_order=best_seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    print("Fitting final model...")
    fit = model.fit(disp=False, maxiter=100, method='lbfgs')
    
    # Forecast
    steps = len(test)
    pred_trans = fit.forecast(steps=steps)
    forecast = np.expm1(pred_trans) if use_log else pred_trans
    forecast.index = test.index # Align index
    
    # Evaluation
    rmse = evaluate_rmse(test_series.values, forecast.values)
    mape = evaluate_mape(test_series.values, forecast.values)
    
    print(f"\nFinal Results on Test Set:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape*100:.2f}%")

    # Save Metrics
    metrics_df = pd.DataFrame([{
        'Model': 'SARIMA',
        'RMSE': rmse,
        'MAPE': mape,
        'Order': str(best_order),
        'Seasonal': str(best_seasonal_order),
        'Log': use_log
    }])
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"Metrics saved to {METRICS_PATH}")

    # Plot
    plt.figure(figsize=(15,6))
    plt.plot(train_series.index[-24*14:], train_series.iloc[-24*14:], label='Train (Last 2 Weeks)')
    plt.plot(test_series.index, test_series, label='Actual Test')
    plt.plot(forecast.index, forecast, label='SARIMA Forecast', linestyle='--')
    plt.title(f"SARIMA Forecast (MAPE={mape*100:.2f}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT1_PATH)
    
    plt.figure(figsize=(10,5))
    plt.plot(test_series.index, test_series, label='Actual')
    plt.plot(forecast.index, forecast, label='Forecast', linestyle='--')
    plt.title("Detailed View: Actual vs Forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT2_PATH)
    print("Plots saved.")

if __name__ == "__main__":
    run_sarima()
