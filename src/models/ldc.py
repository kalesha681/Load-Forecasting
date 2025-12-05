
import pandas as pd
import matplotlib.pyplot as plt
import os
from ..config import ensure_output_dir, PROCESSED_DATA_DIR, COL_REGION_LDC, COL_PEAK_PCT_LDC, COL_TIME_PCT_LDC

def run_ldc_analysis():
    print("\n--- Running Load Duration Curve (LDC) Analysis ---")
    
    path = os.path.join(PROCESSED_DATA_DIR, 'ldc_data.csv')
    if not os.path.exists(path):
        print("LDC data not found.")
        return
        
    df = pd.read_csv(path)
    
    output_dir = ensure_output_dir('LDC')
    PLOT_PATH = os.path.join(output_dir, 'ldc_curve.png')
    METRICS_PATH = os.path.join(output_dir, 'ldc_metrics.csv')
    
    # Filter for National
    region = 'National'
    subset = df[df[COL_REGION_LDC] == region].sort_values(by=COL_TIME_PCT_LDC)
    
    if subset.empty:
        print(f"No LDC data for {region}")
        return
        
    x = subset[COL_TIME_PCT_LDC] # Duration %
    y = subset[COL_PEAK_PCT_LDC] # Demand %
    
    # Compute Fractions
    # Base load: served 100% of time (max duration) -> Min demand in curve? 
    # Usually Base Load is the load that is present 100% of the time.
    # In LDC, x=100% -> y = Base Load % of Peak.
    # Peak Load: x=0% -> y = 100% (Peak).
    
    # We can approximate buckets:
    # Base Load: Load > 0 for 100% of time.
    # But usually we want the CAPACITY types.
    # Let's just output the curve points as metrics for now, or specific intercepts.
    
    # Example metrics:
    # Load at 100% duration (Base Load)
    # Load at 50% duration
    # Load at 1% duration (Peak)
    
    # Interpolate for specific points if needed, or just take nearest
    base_load_pct = subset.iloc[-1][COL_PEAK_PCT_LDC] # At max duration
    peak_load_pct = subset.iloc[0][COL_PEAK_PCT_LDC]   # At min duration
    
    print(f"Base Load Level: {base_load_pct}% of Peak")
    print(f"Peak Load Level: {peak_load_pct}% of Peak")
    
    # Save Metrics
    metrics_df = pd.DataFrame([{
        'Region': region,
        'Base Load %': base_load_pct,
        'Peak Load %': peak_load_pct
    }])
    metrics_df.to_csv(METRICS_PATH, index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'{region} LDC', linewidth=2)
    plt.fill_between(x, 0, y, alpha=0.3)
    plt.title("Load Duration Curve")
    plt.xlabel("Duration (% of Year)")
    plt.ylabel("Load (% of Peak)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Plot saved to {PLOT_PATH}")

if __name__ == "__main__":
    run_ldc_analysis()
