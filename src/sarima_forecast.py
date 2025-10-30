import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Try sklearn metrics but fall back to numpy implementations if unavailable
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    SKLEARN_METRICS = True
except Exception:
    SKLEARN_METRICS = False

def mean_absolute_percentage_error_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, np.nan, y_true)
    with np.errstate(invalid='ignore', divide='ignore'):
        mape = np.nanmean(np.abs((y_true - y_pred) / denom))
    return mape

def ensure_output_dir():
    out_dir = '../output' if os.path.basename(os.getcwd()) == 'src' else 'output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def evaluate_mape(y_true, y_pred):
    if SKLEARN_METRICS:
        return float(mean_absolute_percentage_error(y_true, y_pred))
    return float(mean_absolute_percentage_error_np(y_true, y_pred))

def tune_sarima_by_mape(train_series, seasonal_period=24, use_log=True):
    """Fast candidate search using validation MAPE on last 7 days.
    Returns (best_order, best_seasonal_order, best_val_mape, use_log).
    """
    horizon = 24 * 7
    if len(train_series) <= horizon + 24:
        return (1, 1, 1), (1, 1, 1, seasonal_period), np.inf, False

    # Candidate configs (compact but effective)
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

    best = (None, None, np.inf, False)

    for log_flag in ([True] if use_log else [False]) + [False]:
        y_train = np.log1p(train_sub) if log_flag else train_sub
        y_val = np.log1p(val_sub) if log_flag else val_sub
        for order, seasonal_order in candidates:
            try:
                model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                fit = model.fit(disp=False, maxiter=80, method='lbfgs')
                pred = fit.forecast(steps=horizon)
                # Inverse transform if log
                if log_flag:
                    pred_inv = np.expm1(pred)
                    y_val_inv = np.expm1(y_val)
                    mape = evaluate_mape(y_val_inv.values, pred_inv.values)
                else:
                    mape = evaluate_mape(y_val.values, pred.values)
                if np.isfinite(mape) and mape < best[2]:
                    best = (order, seasonal_order, mape, log_flag)
            except Exception:
                continue
    if best[0] is None:
        best = ((1,1,1), (1,1,1,seasonal_period), np.inf, False)
    return best

if __name__ == "__main__":
    # --- Settings ---
    DATA_PATH = "data/PJME_hourly.csv" if os.path.basename(os.getcwd()) != 'src' else "../data/PJME_hourly.csv"
    OUTPUT_ROOT = ensure_output_dir()
    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'SARIMA')
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    PLOT1_PATH = os.path.join(OUTPUT_DIR, "sarima_forecast.png")
    PLOT2_PATH = os.path.join(OUTPUT_DIR, "sarima_vs_actual.png")

    # --- Step 1: Load and prepare dataset ---
    df = pd.read_csv(DATA_PATH, parse_dates=['Datetime'], index_col='Datetime')
    df = df.sort_index()
    # Drop duplicate timestamps for .asfreq to work
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('h')
    df['PJME_MW'] = df['PJME_MW'].interpolate('time').bfill().ffill()

    # --- Step 2: Train/test split (1 year for training, next 7 days for testing) ---
    last_train_date = df.index.max() - pd.Timedelta(days=7)
    train_start_date = last_train_date - pd.Timedelta(days=365)
    train = df.loc[train_start_date:last_train_date, 'PJME_MW']
    test = df.loc[last_train_date+pd.Timedelta(hours=1):(last_train_date+pd.Timedelta(days=7)), 'PJME_MW']

    print(f"Training window: {train.index.min()} -- {train.index.max()} ({len(train)} hrs)")
    print(f"Test window (prediction): {test.index.min()} -- {test.index.max()} ({len(test)} hrs)")

    # --- Step 3: Auto-tune by validation MAPE (compact candidates, optional log) ---
    print("\nTuning SARIMA hyperparameters (validation MAPE search)...")
    best_order, best_seasonal_order, val_mape, use_log = tune_sarima_by_mape(train, seasonal_period=24, use_log=True)
    print(f"Best order: {best_order}, Best seasonal_order: {best_seasonal_order}, Validation MAPE: {val_mape*100:.2f}% (log={use_log})")

    # --- Step 4: Fit best model on full train and forecast next 7 days ---
    y_train_full = np.log1p(train) if use_log else train
    model = SARIMAX(y_train_full, order=best_order, seasonal_order=best_seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=100, method='lbfgs')
    pred = fit.forecast(steps=7*24)
    forecast = np.expm1(pred) if use_log else pred
    forecast = pd.Series(forecast, index=test.index)

    # --- Step 5: Plot historical, actual next 7 days, and forecast ---
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

    # --- Step 6: Evaluation ---
    rmse = float(np.sqrt(np.mean((test.values - forecast.values)**2)))
    mape = evaluate_mape(test.values, forecast.values)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape*100:.2f}% (Mean Absolute Percentage Error)")

    # --- Step 7: Comparison Plot Only (Zoom on Pred Window) ---
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
