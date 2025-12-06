import os

# Paths
# Assuming this file is in src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw Data
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'Raw')
YEARLY_DATA_PATH = os.path.join(RAW_DATA_DIR, 'Yearly Demand Profile.xlsx')
DAILY_DATA_PATH = os.path.join(RAW_DATA_DIR, 'Daily Demand Profile.xlsx')
LDC_DATA_PATH = os.path.join(RAW_DATA_DIR, 'Load Duration (in % duration of year).xlsx')

# Processed Data
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'Processed')
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

# Output
OUTPUT_ROOT = os.path.join(BASE_DIR, 'plots')

# Data Settings
SPLIT_DATE = '2024-10-01' # Approximate split for 1 year data
SEED = 42

# Data Configuration
DATA_CONFIG = {
    'test_days': 7,
    'split_ratio': 0.8,  # Fallback for small data
    'interpolation_method': 'time'
}

# Model Configuration
MODEL_CONFIG = {
    'lstm': {
        'n_input': 48,
        'epochs': 20,
        'batch_size': 32,
        'units': 64,       # First LSTM layer units
        'dropout': 0.2,
        'optimizer': 'adam',
        'loss': 'mse'
    },
    'sarima': {
        'seasonal_period': 24,
        'tuning_window_days': 90,
        'max_iter': 50,
        'method': 'lbfgs',
        'horizon_days': 7
    }
}

# Column Constants
# Yearly
COL_YEAR = 'Year'
COL_DATE_YEARLY = 'Date'
COL_DEMAND_YEARLY = 'Hourly Demand Met (in MW)'

# Daily
COL_REGION_DAILY = 'Region'
COL_DATE_DAILY = 'Date'
COL_HOUR_DAILY = 'Hour'
COL_DEMAND_DAILY = 'Hourly Demand Met (in MW)'
COL_TYPE_DAILY = 'Demand Type'

# LDC
COL_REGION_LDC = 'Region'
COL_PEAK_PCT_LDC = 'Peak Demand (in %)'
COL_TIME_PCT_LDC = '% of time of year'

def ensure_output_dir(model_name):
    path = os.path.join(OUTPUT_ROOT, model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
