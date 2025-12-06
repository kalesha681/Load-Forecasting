import argparse
import os
import sys
import logging
from pathlib import Path

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Ensure src is on PYTHONPATH ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))

# --- Imports from project ---
from src.data_loader import (
    process_yearly_data,
    process_peak_day_data,
    process_ldc_data,
)

from src.models.sarima import run_sarima_pipeline
from src.models.lstm import run_lstm_pipeline
from src.models.peak_day import run_peak_day_pipeline
from src.models.ldc import run_ldc_pipeline


# -------------------------------------------------------------------
# Argument Parser
# -------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Electrical Load Forecasting Pipeline")
parser.add_argument("mode", nargs="?", default="all",
                    choices=["all", "sarima", "lstm", "peak", "ldc"],
                    help="Which pipeline to run")
parser.add_argument("--sample", action="store_true",
                    help="Run using tiny sample dataset for quick verification")

args = parser.parse_args()


# -------------------------------------------------------------------
# Path Resolution
# -------------------------------------------------------------------
if args.sample:
    logger.info("Running in SAMPLE mode")
    YEARLY_RAW = ROOT / "data/sample/yearly_sample.csv"
    PEAK_RAW   = ROOT / "data/sample/peak_day_sample.csv"
    LDC_RAW    = ROOT / "data/sample/ldc_sample.csv"

    YEARLY_OUT = ROOT / "data/sample/yearly_sample_processed.csv"
    PEAK_OUT   = ROOT / "data/sample/peak_sample_processed.csv"
    LDC_OUT    = ROOT / "data/sample/ldc_sample_processed.csv"

else:
    YEARLY_RAW = ROOT / "data/raw/yearly_hourly_demand_2024.xlsx"
    PEAK_RAW   = ROOT / "data/raw/peak_day_hourly_demand.xlsx"
    LDC_RAW    = ROOT / "data/raw/load_duration_curve.xlsx"

    YEARLY_OUT = ROOT / "data/processed/yearly_demand_National.csv"
    PEAK_OUT   = ROOT / "data/processed/peak_day_National.csv"
    LDC_OUT    = ROOT / "data/processed/ldc_data.csv"


PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------------------
# Stage 1 — Data Engineering
# -------------------------------------------------------------------
def run_data_engineering():
    logger.info("[Stage 1] Data Engineering & Validation")

    logger.info(f"Loading {YEARLY_RAW}...")
    process_yearly_data(YEARLY_RAW, YEARLY_OUT)

    logger.info(f"Loading {PEAK_RAW}...")
    process_peak_day_data(PEAK_RAW, PEAK_OUT)

    logger.info(f"Loading {LDC_RAW}...")
    process_ldc_data(LDC_RAW, LDC_OUT)

    logger.info("[Stage 1] Data processing completed.")


# -------------------------------------------------------------------
# Stage 2 — Forecasting Pipelines
# -------------------------------------------------------------------
def run_forecasting():
    logger.info("[Stage 2] Forecasting Pipelines")

    if args.mode in ("all", "sarima"):
        logger.info("--- Running SARIMA ---")
        run_sarima_pipeline(YEARLY_OUT, PLOTS_DIR)

    if args.mode in ("all", "lstm"):
        logger.info("--- Running LSTM ---")
        try:
            run_lstm_pipeline(YEARLY_OUT, PLOTS_DIR)
        except ImportError as e:
            if "TensorFlow" in str(e):
                logger.warning(f"Skipping LSTM pipeline: {e}")
                logger.warning("Tip: Install TensorFlow to enable this feature. Continuing with other stages...")
            else:
                raise e


# -------------------------------------------------------------------
# Stage 3 — Analytics Pipelines
# -------------------------------------------------------------------
def run_analytics():
    logger.info("[Stage 3] Analytics Pipelines")

    if args.mode in ("all", "peak"):
        logger.info("--- Running Peak Day Analysis ---")
        run_peak_day_pipeline(YEARLY_OUT, PEAK_OUT, PLOTS_DIR)

    if args.mode in ("all", "ldc"):
        logger.info("--- Running Load Duration Curve (LDC) Analysis ---")
        run_ldc_pipeline(LDC_OUT, PLOTS_DIR)


# -------------------------------------------------------------------
# Main Orchestrator
# -------------------------------------------------------------------
def main():
    try:
        run_data_engineering()
        run_forecasting()
        run_analytics()

        logger.info("========================================")
        logger.info("   PIPELINE EXECUTION COMPLETED")
        logger.info("========================================")

    except Exception as e:
        logger.critical("FATAL ERROR", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
