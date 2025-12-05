
import os
import pandas as pd
import numpy as np
from .config import (
    YEARLY_DATA_PATH, DAILY_DATA_PATH, LDC_DATA_PATH, PROCESSED_DATA_DIR,
    COL_YEAR, COL_DATE_YEARLY, COL_DEMAND_YEARLY,
    COL_REGION_DAILY, COL_DATE_DAILY, COL_HOUR_DAILY, COL_DEMAND_DAILY, COL_TYPE_DAILY,
    COL_REGION_LDC, COL_PEAK_PCT_LDC, COL_TIME_PCT_LDC
)

def validate_schema(df, required_columns, filename):
    """Strict schema validation."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Schema Mismatch in {filename}: Missing columns {missing}. Found {df.columns.tolist()}")

def parse_yearly_datetime(year_col, date_col):
    """
    Parse datetime from Year and Date (e.g., '01-Jan 12am').
    Handles '12am' as 00:00 and other hours appropriately.
    """
    # Combine Year and Date
    full_str = year_col.astype(str) + ' ' + date_col.astype(str).str.strip()
    
    # Clean up formatting issues if any (e.g., extra spaces, dots)
    full_str = full_str.str.replace('.', '', regex=False).str.upper()
    
    # Format: 2024 01-JAN 12AM
    # We can try to parse this using specific format string
    try:
        # %Y %d-%b %I%p -> 2024 01-JAN 12AM
        return pd.to_datetime(full_str, format='%Y %d-%b %I%p')
    except ValueError as e:
        print(f"Warning: fast parsing failed ({e}). Trying flexible parsing (slower)...")
        return pd.to_datetime(full_str, format='mixed', dayfirst=True)

def process_yearly_data(region='National'):
    """Load, validate, and process Yearly Hourly Demand."""
    print(f"Loading {YEARLY_DATA_PATH}...")
    if not os.path.exists(YEARLY_DATA_PATH):
        raise FileNotFoundError(f"File not found: {YEARLY_DATA_PATH}")

    df = pd.read_excel(YEARLY_DATA_PATH)
    
    # Schema Validation
    required = [COL_YEAR, COL_DATE_YEARLY, COL_DEMAND_YEARLY]
    validate_schema(df, required, "Yearly Demand Profile")
    
    # Drop rows with missing critical info
    df = df.dropna(subset=[COL_YEAR, COL_DATE_YEARLY])
    
    # Parse Datetime
    df['Datetime'] = parse_yearly_datetime(df[COL_YEAR], df[COL_DATE_YEARLY])
    
    # Set Index
    df = df.set_index('Datetime').sort_index()
    
    # Handle Duplicates (keep first or average? usually keep first for time series unless overlapping sources)
    if df.index.duplicated().any():
        print(f"Found {df.index.duplicated().sum()} duplicate timestamps. Keeping first.")
        df = df[~df.index.duplicated(keep='first')]
    
    # Enforce Hourly Continuity
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    if len(full_range) != len(df):
        print(f"Reindexing to enforce hourly continuity. Original: {len(df)}, New: {len(full_range)}")
        df = df.reindex(full_range)
    
    # Interpolate Missing Demand
    col_name = COL_DEMAND_YEARLY
    if df[col_name].isna().any():
        missing_count = df[col_name].isna().sum()
        print(f"Interpolating {missing_count} missing hours...")
        df[col_name] = df[col_name].interpolate(method='time').bfill().ffill()
        
    # Standardize column name for processed data
    # (We keep original name in file, but ensure it's clean)
    
    # Save
    out_path = os.path.join(PROCESSED_DATA_DIR, f'yearly_demand_{region}.csv')
    df[[col_name]].to_csv(out_path)
    print(f"Saved processed yearly data to {out_path}")
    return df[[col_name]]

def process_daily_data(region='National'):
    """Load, validate, and process Peak Day Hourly Demand."""
    print(f"Loading {DAILY_DATA_PATH}...")
    if not os.path.exists(DAILY_DATA_PATH):
        raise FileNotFoundError(f"File not found: {DAILY_DATA_PATH}")

    df = pd.read_excel(DAILY_DATA_PATH)
    
    # Schema Validation
    required = [COL_REGION_DAILY, COL_DATE_DAILY, COL_HOUR_DAILY, COL_DEMAND_DAILY]
    validate_schema(df, required, "Daily Demand Profile")
    
    if region:
        df = df[df[COL_REGION_DAILY] == region].copy()
        if df.empty:
            print(f"Warning: No data found for region '{region}' in Daily Demand Profile.")
    
    # Create Datetime
    # Date is likely "30 May 2024" (string) or datetime object
    # Hour is int 0-23
    
    # Clean Date
    df[COL_DATE_DAILY] = pd.to_datetime(df[COL_DATE_DAILY], format='mixed', dayfirst=True)
    
    # Add timedelta
    df['Datetime'] = df[COL_DATE_DAILY] + pd.to_timedelta(df[COL_HOUR_DAILY], unit='h')
    
    df = df.set_index('Datetime').sort_index()
    
    # Save
    out_path = os.path.join(PROCESSED_DATA_DIR, f'peak_day_{region}.csv')
    df[[COL_DEMAND_DAILY]].to_csv(out_path)
    print(f"Saved processed peak day data to {out_path}")
    return df[[COL_DEMAND_DAILY]]

def process_ldc_data():
    """Load and validate Load Duration Curve data."""
    print(f"Loading {LDC_DATA_PATH}...")
    if not os.path.exists(LDC_DATA_PATH):
        raise FileNotFoundError(f"File not found: {LDC_DATA_PATH}")

    df = pd.read_excel(LDC_DATA_PATH)
    
    required = [COL_REGION_LDC, COL_PEAK_PCT_LDC, COL_TIME_PCT_LDC]
    validate_schema(df, required, "Load Duration Curve")
    
    # Save raw (it's already clean usually, but good to have in processed)
    out_path = os.path.join(PROCESSED_DATA_DIR, 'ldc_data.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved processed LDC data to {out_path}")
    return df

# --- Legacy Support / Shared Utilities ---
def load_and_preprocess(file_path=None):
    """
    Wrapper to load processed yearly data. 
    If file_path is provided, loads that. 
    Else loads default National yearly demand.
    """
    if file_path:
        path = file_path
    else:
        path = os.path.join(PROCESSED_DATA_DIR, 'yearly_demand_National.csv')
        
    if not os.path.exists(path):
        # On first run, we might need to process
        print(f"Processed file not found at {path}. Processing raw data...")
        try:
            return process_yearly_data()
        except Exception as e:
            raise FileNotFoundError(f"Could not load or create processed data at {path}. Error: {e}")
            
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def train_test_split(df, split_date=None, test_days=None):
    """
    Split data into train and test sets.
    Prioritizes text_days (relative split) over split_date if provided/logic demands.
    For this dataset (2024), we likely want the last N days as test, or a specific date.
    """
    if test_days:
        split_idx = len(df) - (test_days * 24)
        if split_idx < 0: split_idx = 0
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
    elif split_date:
        train = df.loc[df.index < split_date].copy()
        test = df.loc[df.index >= split_date].copy()
    else:
        # Default 80/20
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
    return train, test

if __name__ == "__main__":
    # Test run
    process_yearly_data()
    process_daily_data()
    process_ldc_data()
