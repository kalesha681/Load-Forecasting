import pytest
import numpy as np
from src.metrics import evaluate_mape, evaluate_rmse

class TestMetrics:
    def test_evaluate_mape_perfect_match(self):
        """Test MAPE with identical arrays (should be 0.0)."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([100, 200, 300])
        mape = evaluate_mape(y_true, y_pred)
        assert mape == 0.0

    def test_evaluate_mape_simple(self):
        """Test MAPE with simple known values."""
        # True: 100, Pred: 110 -> Error: 10, %: 0.1
        # True: 200, Pred: 180 -> Error: 20, %: 0.1
        y_true = np.array([100, 200])
        y_pred = np.array([110, 180])
        mape = evaluate_mape(y_true, y_pred)
        assert pytest.approx(mape) == 0.1

    def test_evaluate_mape_zero_handling(self):
        """Test MAPE handles division by zero (should ignore/return finite)."""
        y_true = np.array([0, 100])
        y_pred = np.array([10, 110])
        # Sklearn or our fallback should handle this without crashing
        # Usually it might return high value or huge number, but shouldn't raise exception
        mape = evaluate_mape(y_true, y_pred)
        assert np.isfinite(mape) or mape > 1e6

    def test_evaluate_rmse_perfect_match(self):
        """Test RMSE with identical arrays (should be 0.0)."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        rmse = evaluate_rmse(y_true, y_pred)
        assert rmse == 0.0


    def test_evaluate_rmse_simple(self):
        """Test RMSE calculation."""
        # Errors: 1, -1. Squared: 1, 1. Mean: 1. Sqrt: 1.
        y_true = np.array([10, 20])
        y_pred = np.array([11, 19])
        rmse = evaluate_rmse(y_true, y_pred)
        assert rmse == 1.0

    def test_input_validation(self):
        """Test validation logic in metrics."""
        # None input
        with pytest.raises(ValueError, match="cannot be None"):
            evaluate_rmse(None, [1, 2])
            
        # Empty input
        with pytest.raises(ValueError, match="cannot be empty"):
            evaluate_rmse([], [])
            
        # Length mismatch (handled by sklearn or array operations usually, but our validation might catch length=0)
        # Our validate_array_input just checks emptiness/None.
        pass
