import pandas as pd
import numpy as np
from .config import DATA_PATH

def load_and_preprocess(file_path=DATA_PATH):
    """Load data, parse datetime, set index, handle missing/duplicates, resample."""
    df = pd.read_csv(file_path, parse_dates=['Datetime'])
    df = df.drop_duplicates(subset='Datetime')
    df = df.set_index('Datetime')
    df = df.sort_index()
    
    # Drop duplicate index if any remain
    df = df[~df.index.duplicated(keep='first')]
    
    df = df.asfreq('h')  # Ensure hourly frequency
    
    # Interpolate missing values
    df['PJME_MW'] = df['PJME_MW'].interpolate('time')
    df['PJME_MW'] = df['PJME_MW'].bfill().ffill()
    
    return df

def train_test_split(df, split_date):
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date]
    return train, test
