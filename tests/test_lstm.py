import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, ANY

# Import the module to test
# We used safe imports in lstm.py, so we need to be careful if TF is missing in the test env, 
# but for unit tests we can assume the *logic* runs if we mock correctly.
from src.models import lstm

class TestLSTM:
    
    def test_create_sequences_shapes(self):
        """Test that create_sequences produces correct shapes."""
        # 100 samples, 1 feature
        data = np.arange(100).reshape(-1, 1)
        n_input = 10
        
        X, y = lstm.create_sequences(data, n_input=n_input)
        
        # Expected: 100 - 10 = 90 sequences
        # X shape: (90, 10, 1)
        # y shape: (90, 1)
        assert X.shape == (90, 10, 1)
        assert y.shape == (90, 1)
        
        # Check content: X[0] should be 0..9, y[0] should be 10
        np.testing.assert_array_equal(X[0].flatten(), np.arange(10))
        assert y[0] == 10

    def test_create_sequences_validation(self):
        """Test input validation for create_sequences."""
        with pytest.raises(ValueError, match="cannot be None"):
            lstm.create_sequences(None)
            
        with pytest.raises(ValueError, match="cannot be empty"):
            lstm.create_sequences([])

    @patch('src.models.lstm.Input', create=True)
    @patch('src.models.lstm.Dense', create=True)
    @patch('src.models.lstm.Dropout', create=True)
    @patch('src.models.lstm.LSTM', create=True)
    @patch('src.models.lstm.Sequential', create=True)
    @patch('src.models.lstm.MinMaxScaler')
    @patch('src.models.lstm.pd.read_csv')
    def test_run_lstm_pipeline_mocked(self, mock_read_csv, mock_scaler, mock_sequential, 
                                      mock_lstm, mock_dropout, mock_dense, mock_input, tmp_path):
        """
        Test the full LSTM pipeline with mocked TensorFlow and Data.
        """
        # 1. Setup Data Mock
        # Create a dummy dataframe
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        df = pd.DataFrame({'Demand_MW': np.arange(100)}, index=dates)
        mock_read_csv.return_value = df
        
        # 2. Setup Scaler Mock
        scaler_instance = MagicMock()
        scaler_instance.fit_transform.side_effect = lambda x: x
        scaler_instance.transform.side_effect = lambda x: x
        scaler_instance.inverse_transform.side_effect = lambda x: x 
        mock_scaler.return_value = scaler_instance
        
        # 3. Setup TensorFlow Model Mock
        model_instance = MagicMock()
        mock_sequential.return_value = model_instance
        model_instance.predict.return_value = np.zeros((20, 1)) # Dummy prediction
        
        # 4. Run Pipeline
        input_csv = tmp_path / "dummy.csv"
        output_dir = tmp_path / "LSTM_Out"
        
        # We mock TF_AVAILABLE to True to ensure it runs
        with patch('src.models.lstm.TF_AVAILABLE', True):
             lstm.run_lstm_pipeline(input_csv, output_dir)
             
        # 5. Assertions
        mock_sequential.assert_called_once()
        model_instance.compile.assert_called()
        model_instance.fit.assert_called()
        model_instance.save.assert_called()
        
        # Verify layers were called (optional, but good for coverage)
        mock_input.assert_called()
        mock_lstm.assert_called()
        mock_dropout.assert_called()
        mock_dense.assert_called()
        
        assert (output_dir / "LSTM" / "metrics.csv").exists()
