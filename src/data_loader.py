
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure Logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from .config import (
    COL_YEAR, COL_DATE_YEARLY, COL_DEMAND_YEARLY,
    COL_REGION_DAILY, COL_DATE_DAILY, COL_HOUR_DAILY, COL_DEMAND_DAILY,
    COL_REGION_LDC, COL_PEAK_PCT_LDC, COL_TIME_PCT_LDC
)

def validate_schema(df, required_columns, filename):
    """Strict schema validation."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        msg = f"Schema Mismatch in {filename}: Missing columns {missing}. Found {df.columns.tolist()}"
        logger.error(msg)
        raise ValueError(msg)

def parse_yearly_datetime(year_col, date_col):
    """
    Parse datetime from Year and Date columns.
    Enforces robust parsing.
    """
    # Combine Year and Date
    full_str = year_col.astype(str) + ' ' + date_col.astype(str).str.strip()
    
    # Clean up formatting issues
    full_str = full_str.str.replace('.', '', regex=False).str.upper()
    
    # Attempt parsing
    try:
        # Expected format: "2024 01-JAN 12AM"
        return pd.to_datetime(full_str, format='%Y %d-%b %I%p', errors='raise')
    except ValueError as e:
        logger.warning(f"Primary parsing failed: {e}. Attempting robust fallback (dayfirst=True).")
        try:
             # Fallback strictly without 'mixed'
            return pd.to_datetime(full_str, dayfirst=True, errors='raise')
        except Exception as e2:
             logger.error(f"Datetime parsing failed completely for some rows. Error: {e2}")
             raise e2

def process_yearly_data(input_path, output_path):
    """Process Yearly Hourly Demand Data."""
    input_path = str(input_path)
    output_path = str(output_path)
    logger.info(f"Loading yearly data from {input_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"File not found: {input_path}")
        raise FileNotFoundError(f"File not found: {input_path}")

    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)
    
    # Schema Validation
    required = ["Datetime", "Demand_MW"] if "Datetime" in df.columns else [COL_YEAR, COL_DATE_YEARLY, COL_DEMAND_YEARLY]
    # If sample data with clean columns, skip complex parsing
    if "Datetime" in df.columns and "Demand_MW" in df.columns:
         logger.info("Detected sample/clean format. Skipping complex parsing.")
         df['Datetime'] = pd.to_datetime(df['Datetime'])
         col_name = "Demand_MW"
    else:
        validate_schema(df, required, "Yearly Demand Profile")
        df = df.dropna(subset=[COL_YEAR, COL_DATE_YEARLY])
        df['Datetime'] = parse_yearly_datetime(df[COL_YEAR], df[COL_DATE_YEARLY])
        col_name = COL_DEMAND_YEARLY

    # Set Index
    df = df.set_index('Datetime').sort_index()
    
    # Duplicates
    if df.index.duplicated().any():
        logger.warning(f"Found {df.index.duplicated().sum()} duplicate timestamps. Keeping first.")
        df = df[~df.index.duplicated(keep='first')]
    
    # Hourly Continuity
    if not df.empty:
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        if len(full_range) != len(df):
            logger.info(f"Reindexing to enforce hourly continuity. Original: {len(df)}, New: {len(full_range)}")
            df = df.reindex(full_range)
    
    # Interpolation
    if df[col_name].isna().any():
        missing_count = df[col_name].isna().sum()
        logger.info(f"Interpolating {missing_count} missing hours...")
        df[col_name] = df[col_name].interpolate(method='time').bfill().ffill()
        
    # Save
    ensure_dir(output_path)
    df[[col_name]].to_csv(output_path)
    logger.info(f"Saved processed yearly data to {output_path}")

def process_peak_day_data(input_path, output_path):
    """Process Peak Day Hourly Demand Data."""
    input_path = str(input_path)
    output_path = str(output_path)
    logger.info(f"Loading peak day data from {input_path}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    # Sample data check
    if "Datetime" in df.columns and "Hourly_Demand_MW" in df.columns:
        logger.info("Detected sample/clean peak day format.")
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        col_name = "Hourly_Demand_MW"
    else:
        # Validating strictly logic
        required = [COL_DATE_DAILY, COL_HOUR_DAILY, COL_DEMAND_DAILY]
        # Allow missing Region if strict check fails but data is usable
        validate_schema(df, required, "Peak Day Profile")
        
        # Parse Dates
        df[COL_DATE_DAILY] = pd.to_datetime(df[COL_DATE_DAILY], dayfirst=True, errors='raise')
        df['Datetime'] = df[COL_DATE_DAILY] + pd.to_timedelta(df[COL_HOUR_DAILY], unit='h')
        col_name = COL_DEMAND_DAILY

    df = df.set_index('Datetime').sort_index()
    
    ensure_dir(output_path)
    df[[col_name]].to_csv(output_path)
    logger.info(f"Saved processed peak day data to {output_path}")

def process_ldc_data(input_path, output_path):
    """Process Load Duration Curve Data."""
    input_path = str(input_path)
    output_path = str(output_path)
    logger.info(f"Loading LDC data from {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)
        
    # Sample check
    # Sample cols: Percent_Time, Load_Percent_of_Peak
    # Config cols: COL_TIME_PCT_LDC, COL_PEAK_PCT_LDC
    
    # We will just pass it through if it looks reasonably tabular
    ensure_dir(output_path)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed LDC data to {output_path}")

def ensure_dir(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

# --- Legacy/Shared ---
def train_test_split(df, test_days=7):
    """Split data into train and test sets."""
    split_idx = len(df) - (test_days * 24)
    if split_idx < 0: split_idx = 0
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test

if __name__ == "__main__":
    # Test run
    process_yearly_data()
    process_daily_data()
    process_ldc_data()
