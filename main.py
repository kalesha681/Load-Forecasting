import sys
import os
import argparse
import pandas as pd

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.lstm import run_lstm
from src.models.sarima import run_sarima
from src.models.peak_day import run_peak_day_analysis
from src.models.ldc import run_ldc_analysis
from src.data_loader import process_yearly_data, process_daily_data, process_ldc_data
from src.config import PROCESSED_DATA_DIR, ensure_output_dir

def generate_summary_report():
    print("\n" + "="*40)
    print("      GENERATING FINAL RESEARCH REPORT")
    print("="*40)
    
    # 1. Forecasting Metrics (SARIMA vs LSTM)
    print("\n--- Comparative Forecasting Metrics (Yearly Data) ---")
    sarima_metrics_path = os.path.join(ensure_output_dir('SARIMA'), 'metrics.csv')
    lstm_metrics_path = os.path.join(ensure_output_dir('LSTM'), 'metrics.csv')
    
    metrics = []
    if os.path.exists(sarima_metrics_path):
        metrics.append(pd.read_csv(sarima_metrics_path))
    if os.path.exists(lstm_metrics_path):
        metrics.append(pd.read_csv(lstm_metrics_path))
        
    if metrics:
        combined = pd.concat(metrics, ignore_index=True)
        # Select common columns if schemas differ slightly
        try:
            print(combined[['Model', 'RMSE', 'MAPE']].to_markdown(index=False))
        except ImportError:
            print(combined[['Model', 'RMSE', 'MAPE']].to_string(index=False))
        
        # Save combined
        report_path = os.path.join(ensure_output_dir('Reports'), 'forecasting_comparison.csv')
        combined.to_csv(report_path, index=False)
        print(f"Saved to {report_path}")
    else:
        print("No forecasting metrics found.")

    # 2. Peak Day Metrics
    print("\n--- Peak Day Forecast Accuracy ---")
    peak_metrics_path = os.path.join(ensure_output_dir('Peak_Day'), 'metrics.csv')
    if os.path.exists(peak_metrics_path):
        peak_df = pd.read_csv(peak_metrics_path)
        try:
            print(peak_df.to_markdown(index=False))
        except ImportError:
            print(peak_df.to_string(index=False))
    else:
        print("No peak day metrics found.")

    # 3. LDC Metrics
    print("\n--- Load Duration Curve Statistics ---")
    ldc_metrics_path = os.path.join(ensure_output_dir('LDC'), 'ldc_metrics.csv')
    if os.path.exists(ldc_metrics_path):
        ldc_df = pd.read_csv(ldc_metrics_path)
        try:
            print(ldc_df.to_markdown(index=False))
        except ImportError:
            print(ldc_df.to_string(index=False))
    else:
        print("No LDC metrics found.")

def run_all(full_etl=True):
    if full_etl:
        print("\n[Stage 1] Data Engineering & Validation")
        process_yearly_data()
        process_daily_data()
        process_ldc_data()
    
    print("\n[Stage 2] Forecasting Pipelines")
    # Run SARIMA
    try:
        run_sarima()
    except Exception as e:
        print(f"SARIMA Failed: {e}")

    # Run LSTM
    try:
        run_lstm()
    except Exception as e:
        print(f"LSTM Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n[Stage 3] Analytics Pipelines")
    # Run Peak Day
    try:
        run_peak_day_analysis()
    except Exception as e:
        print(f"Peak Day Failed: {e}")

    # Run LDC
    try:
        run_ldc_analysis()
    except Exception as e:
        print(f"LDC Failed: {e}")

    # Generate Report
    print("\n[Stage 4] Comparative Reporting")
    generate_summary_report()

def main():
    parser = argparse.ArgumentParser(description="Load Forecasting & Grid Analytics Pipeline")
    parser.add_argument('mode', type=str, choices=['all', 'etl', 'lstm', 'sarima', 'peak_day', 'ldc', 'report'], 
                        help="Task to run")
    parser.add_argument('--no-etl', action='store_true', help="Skip ETL in 'all' mode")
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        run_all(full_etl=not args.no_etl)
    elif args.mode == 'etl':
        process_yearly_data()
        process_daily_data()
        process_ldc_data()
    elif args.mode == 'sarima':
        run_sarima()
    elif args.mode == 'lstm':
        run_lstm()
    elif args.mode == 'peak_day':
        run_peak_day_analysis()
    elif args.mode == 'ldc':
        run_ldc_analysis()
    elif args.mode == 'report':
        generate_summary_report()

if __name__ == "__main__":
    main()
