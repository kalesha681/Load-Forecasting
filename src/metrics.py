import numpy as np

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    SKLEARN_METRICS = True
except ImportError:
    SKLEARN_METRICS = False

def mean_absolute_percentage_error_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(y_true == 0, np.nan, y_true)
    with np.errstate(invalid='ignore', divide='ignore'):
        mape = np.nanmean(np.abs((y_true - y_pred) / denom))
    return mape

def evaluate_mape(y_true, y_pred):
    if SKLEARN_METRICS:
        return float(mean_absolute_percentage_error(y_true, y_pred))
    return float(mean_absolute_percentage_error_np(y_true, y_pred))

def evaluate_rmse(y_true, y_pred):
    if SKLEARN_METRICS:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))
