
import pandas as pd
import matplotlib.pyplot as plt
import os
from ..config import ensure_output_dir, PROCESSED_DATA_DIR, COL_REGION_LDC, COL_PEAK_PCT_LDC, COL_TIME_PCT_LDC
from ..utils import validate_path


from pathlib import Path
from ..logging_config import get_logger

logger = get_logger(__name__)

def run_ldc_pipeline(data_path, output_dir):
    """Run LDC Pipeline."""
    output_dir = Path(output_dir) / 'LDC'
    output_dir.mkdir(parents=True, exist_ok=True)
    PLOT_PATH = output_dir / 'ldc_curve.png'
    METRICS_PATH = output_dir / 'ldc_metrics.csv'
    
    METRICS_PATH = output_dir / 'ldc_metrics.csv'
    
    # Security Check
    data_path = validate_path(data_path)
    output_dir = validate_path(output_dir)

    logger.info("loading_data", path=str(data_path))
    df = pd.read_csv(data_path)
    
    # Assume cols: Percent_Time, Load_Percent_of_Peak (or similar)
    if "Percent_Time" in df.columns:
        x = df["Percent_Time"]
        y = df["Load_Percent_of_Peak"]
    elif COL_TIME_PCT_LDC in df.columns:
        x = df[COL_TIME_PCT_LDC]
        y = df[COL_PEAK_PCT_LDC]
    else:
        # Fallback to idx 0 and 1
        x = df.iloc[:,0]
        y = df.iloc[:,1]
        
    x = x.sort_values()
    y = y.sort_values(ascending=False)
    
    base_load = y.iloc[-1]
    peak_load = y.iloc[0]
    
    logger.info("ldc_metrics", base_load_pct=base_load, peak_load_pct=peak_load)
    
    pd.DataFrame([{'Base %': base_load, 'Peak %': peak_load}]).to_csv(METRICS_PATH, index=False)
    
    plt.figure()
    plt.plot(x, y, label='LDC')
    plt.fill_between(x, 0, y, alpha=0.3)
    plt.xlabel("Duration %")
    plt.ylabel("Load %")
    plt.title("Load Duration Curve")
    plt.savefig(PLOT_PATH)
    plt.close()
    logger.info("artifacts_saved", dir=str(output_dir))

def run_ldc_analysis(): pass

