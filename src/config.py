import os

# Paths
# Assuming this file is in src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'PJME_hourly.csv')
OUTPUT_ROOT = os.path.join(BASE_DIR, 'plots')

# Data
SPLIT_DATE = '2017-01-01'

# Model Settings
SEED = 42

def ensure_output_dir(model_name):
    path = os.path.join(OUTPUT_ROOT, model_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
