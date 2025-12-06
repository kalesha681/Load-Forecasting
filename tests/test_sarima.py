import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.models.sarima import evaluate_candidate, tune_sarima_by_mape, run_sarima_pipeline

class TestSarima:
    @pytest.fixture
    def sample_series(self):
        """Create a sample hourly time series."""
        dates = pd.date_range(start='2024-01-01', periods=500, freq='h')
        # Simple pattern: Trend + Daily Seasonality + Noise
        values = np.linspace(100, 200, 500) + 10 * np.sin(np.linspace(0, 100, 500))
        return pd.Series(values, index=dates, name='Demand_MW')

    def test_evaluate_candidate_success(self, sample_series):
        """Test evaluate_candidate returns a valid tuple on success."""
        train = sample_series.iloc[:400]
        val = sample_series.iloc[400:]
        
        # We assume SARIMAX works (integration test) or we could mock it.
        # For this test, let's allow it to run on small data or mock if too slow.
        # Given the data is small (400 pts), fast fit should resolve quickly.
        
        result = evaluate_candidate((1,0,0), (0,0,0,24), train, val, False, 24)
        
        assert result is not None
        assert len(result) == 4
        # Check MAPE is float and finite
        assert isinstance(result[2], float) 
        assert np.isfinite(result[2])

    def test_evaluate_candidate_failure_handling(self):
        """Test graceful failure when SARIMAX errors out."""
        # Create garbage data that might cause linalg error or we can mock SARIMAX to raise
        train = pd.Series([1, 1, 1]) # Too short
        val = pd.Series([1, 1])
        
        with patch('src.models.sarima.SARIMAX') as mock_sarimax:
            mock_sarimax.side_effect = Exception("Singular matrix")
            result = evaluate_candidate((1,1,1), (1,1,1,24), train, val, False, 24)
            assert result is None

    @patch('src.models.sarima.Parallel')
    def test_tune_sarima_fallback(self, mock_parallel, sample_series):
        """Test tuning returns fallback if all candidates fail."""
        # Setup mock to return list of Nones
        mock_parallel.return_value = lambda tasks: [None, None]
        
        # We need to mock delayed as well since it's used in the list comp
        with patch('src.models.sarima.delayed'):
            best_order, best_seasonal, best_mape, log_flag = tune_sarima_by_mape(sample_series)
            
            # Check for fallback values
            assert best_mape == np.inf
            assert best_order == (1,1,1)

    def test_run_sarima_pipeline_integration(self, tmp_path):
        """Integration test for the full pipeline using dummy data."""
        # Create dummy csv
        d = pd.DataFrame({
            'Datetime': pd.date_range('2024-01-01', periods=100, freq='h'),
            'Demand_MW': np.random.rand(100) * 100
        })
        csv_path = tmp_path / "test_data.csv"
        d.to_csv(csv_path)
        
        # Run pipeline
        # It might fail due to "dataset too small" logic in data_loader fallback or SARIMA convergence
        # But we just want to ensure it runs without crashing and produces *something* or handles error.
        # Given 100 points, train_test_split will give 80 train, 20 test.
        # Tuning window is large, so it will use all training.
        
        # We mock statsmodels to just return a dummy fit to avoid actual optimization time/errors on random data
        with patch('statsmodels.tsa.statespace.sarimax.SARIMAX.fit') as mock_fit:
            mock_results = MagicMock()
            mock_results.forecast.return_value = np.zeros(20) # Dummy forecast matches test length
            mock_fit.return_value = mock_results
            
            # Also verify it creates the directory structure
            run_sarima_pipeline(csv_path, tmp_path)
            
            assert (tmp_path / 'SARIMA').exists()
            assert (tmp_path / 'SARIMA' / 'metrics.csv').exists()
            assert (tmp_path / 'SARIMA' / 'sarima_forecast.png').exists()
