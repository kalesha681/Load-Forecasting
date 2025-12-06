import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.models.ldc import run_ldc_pipeline
from src.models.peak_day import run_peak_day_pipeline

class TestLDC:
    def test_run_ldc_pipeline(self, tmp_path):
        """Test LDC pipeline handles correct columns and file I/O."""
        # Create dummy csv
        csv_path = tmp_path / "ldc_data.csv"
        df = pd.DataFrame({
            "Percent_Time": np.linspace(0, 100, 10),
            "Load_Percent_of_Peak": np.linspace(100, 0, 10)
        })
        df.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "LDC_Output"
        
        run_ldc_pipeline(csv_path, output_dir)
        
        assert (output_dir / "LDC" / "ldc_metrics.csv").exists()
        assert (output_dir / "LDC" / "ldc_curve.png").exists()

class TestPeakDay:
    @patch('src.models.peak_day.SARIMAX')
    def test_run_peak_day_pipeline_flow(self, mock_sarimax, tmp_path):
        """Test Peak Day pipeline flow."""
        # Setup Mocks
        mock_fit = MagicMock()
        mock_sarimax.return_value.fit.return_value = mock_fit
        # Forecast length matches peak df length (24 hours)
        mock_fit.forecast.return_value = pd.Series(np.random.rand(24) * 100) 
        
        # Create Dummy Data
        yearly_path = tmp_path / "yearly.csv"
        # Dates before peak
        dates_train = pd.date_range("2024-01-01", "2024-01-31", freq='h')
        yearly = pd.DataFrame({'Demand': np.random.rand(len(dates_train))*100}, index=dates_train)
        yearly.to_csv(yearly_path)
        
        peak_path = tmp_path / "peak.csv"
        # Peak date after training
        dates_peak = pd.date_range("2024-02-01", periods=24, freq='h')
        peak = pd.DataFrame({'Demand': np.random.rand(len(dates_peak))*100 + 50}, index=dates_peak)
        peak.to_csv(peak_path)
        
        output_dir = tmp_path / "Peak_Output"
        
        run_peak_day_pipeline(yearly_path, peak_path, output_dir)
        
        assert (output_dir / "Peak_Day" / "metrics.csv").exists()
        assert (output_dir / "Peak_Day" / "peak_day_forecast.png").exists()
        
    def test_run_peak_day_empty_data(self, tmp_path, capsys):
        """Test handling of empty peak data."""
        yearly_path = tmp_path / "yearly.csv"
        pd.DataFrame({'Demand': []}).to_csv(yearly_path)
        
        peak_path = tmp_path / "peak.csv"
        pd.DataFrame({'Demand': []}).to_csv(peak_path) # Empty
        
        run_peak_day_pipeline(yearly_path, peak_path, tmp_path)
        
        captured = capsys.readouterr()
        assert "Peak data empty" in captured.out
