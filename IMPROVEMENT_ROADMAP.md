# Repository Improvement Roadmap

**Based on Comprehensive Review (7.2/10 Overall Rating)**  
**Target: Achieve 9.0/10 Production-Ready Status**

---

## Phase 1: Critical Fixes (Week 1-2) 🔴

**Priority: P0 - Must Fix Immediately**  
**Estimated Time: 10-20 hours**

### ✅ Completed
- [x] Fix requirements.txt UTF-16 encoding issue → ASCII
- [x] Update TensorFlow 2.15.0 → >=2.16.0 for Python 3.12 compatibility
- [x] Create requirements-dev.txt for development dependencies

### 🔄 In Progress / Todo

#### 1.1 Add Input Validation (4-6 hours)
**Impact: Security + Stability**

```python
# File: src/validators.py (NEW)
from pathlib import Path
import numpy as np
import pandas as pd

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: list,
                      min_rows: int = 1) -> None:
    """Validate DataFrame structure."""
    if df.empty:
        raise ValidationError("DataFrame is empty")
    
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValidationError(f"Missing columns: {missing}")
    
    if len(df) < min_rows:
        raise ValidationError(f"DataFrame has only {len(df)} rows, need {min_rows}")

def validate_path(path: Path, must_exist: bool = True) -> Path:
    """Validate and sanitize file path."""
    path = Path(path).resolve()
    
    if must_exist and not path.exists():
        raise ValidationError(f"Path does not exist: {path}")
    
    # Prevent path traversal
    if '..' in str(path):
        raise ValidationError("Path traversal detected")
    
    return path

def validate_array(arr: np.ndarray, 
                  min_length: int = 1,
                  allow_nan: bool = False) -> None:
    """Validate NumPy array."""
    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"Expected np.ndarray, got {type(arr)}")
    
    if len(arr) < min_length:
        raise ValidationError(f"Array too short: {len(arr)} < {min_length}")
    
    if not allow_nan and np.any(np.isnan(arr)):
        raise ValidationError("Array contains NaN values")
```

**Files to Update:**
- src/data_loader.py (add validation calls)
- src/models/lstm.py (validate inputs)
- src/models/sarima.py (validate inputs)
- src/metrics.py (validate arrays)

#### 1.2 Implement Proper Error Handling (4-6 hours)
**Impact: Reliability**

```python
# File: src/exceptions.py (NEW)
class LoadForecastingError(Exception):
    """Base exception for all forecasting errors."""
    pass

class DataLoadError(LoadForecastingError):
    """Error loading or processing data."""
    pass

class ModelTrainingError(LoadForecastingError):
    """Error during model training."""
    pass

class ModelPredictionError(LoadForecastingError):
    """Error during model prediction."""
    pass

class ConfigurationError(LoadForecastingError):
    """Error in configuration."""
    pass
```

**Update all try/except blocks:**
```python
# Before (sarima.py line 35)
try:
    # ... SARIMA fitting
except Exception:
    pass  # ❌ Silent failure

# After
try:
    # ... SARIMA fitting
except (ValueError, LinAlgError) as e:
    logger.debug(f"SARIMA candidate {order} failed: {str(e)}")
    return None
except Exception as e:
    logger.error(f"Unexpected error in SARIMA tuning: {str(e)}")
    raise ModelTrainingError(f"SARIMA tuning failed: {str(e)}") from e
```

#### 1.3 Set Random Seeds Properly (1-2 hours)
**Impact: Reproducibility**

```python
# File: src/config.py (UPDATE)
import os
import random
import numpy as np

SEED = 42

def set_all_seeds(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Enable determinism
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except ImportError:
        pass

# Call at startup
set_all_seeds()
```

---

## Phase 2: Testing Infrastructure (Week 3-4) 🟠

**Priority: P0 - Critical for Quality**  
**Estimated Time: 40-80 hours**  
**Target: 80%+ code coverage**

### 2.1 Test Structure (2 hours)
```
tests/
├── __init__.py
├── conftest.py                  # Pytest fixtures
├── test_data_loader.py          # ✅ Exists (expand)
├── test_sarima_model.py         # ❌ NEW
├── test_lstm_model.py           # ❌ NEW
├── test_peak_day_analysis.py   # ❌ NEW
├── test_ldc_analysis.py         # ❌ NEW
├── test_metrics.py              # ❌ NEW
├── test_visualization.py        # ❌ NEW
├── test_config.py               # ❌ NEW
├── test_validators.py           # ❌ NEW
├── test_integration.py          # ❌ NEW
└── fixtures/
    ├── __init__.py
    ├── sample_data.py
    └── mock_models.py
```

### 2.2 Essential Test Examples

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_timeseries():
    """Create sample time series data."""
    dates = pd.date_range('2024-01-01', periods=200, freq='h')
    demand = 180000 + 20000 * np.sin(np.arange(200) * 2 * np.pi / 24)
    return pd.DataFrame({'Demand_MW': demand}, index=dates)

@pytest.fixture
def tmp_data_path(tmp_path):
    """Create temporary data file."""
    data_file = tmp_path / "test_data.csv"
    return data_file

# tests/test_metrics.py
import numpy as np
from src.metrics import evaluate_mape, evaluate_rmse

def test_mape_perfect_prediction():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200, 300])
    assert evaluate_mape(y_true, y_pred) == 0.0

def test_mape_calculation():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    mape = evaluate_mape(y_true, y_pred)
    expected = np.mean([0.1, 0.05, 0.0333333])
    assert abs(mape - expected) < 0.01

def test_mape_with_zeros_handled():
    y_true = np.array([0, 100, 200])
    y_pred = np.array([10, 110, 190])
    mape = evaluate_mape(y_true, y_pred)
    assert np.isfinite(mape)

def test_rmse_calculation():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    rmse = evaluate_rmse(y_true, y_pred)
    expected = np.sqrt(np.mean([100, 100, 100]))
    assert abs(rmse - expected) < 0.01

# tests/test_sarima_model.py
def test_sarima_pipeline_completes(sample_timeseries, tmp_data_path, tmp_path):
    """Test SARIMA pipeline runs without errors."""
    sample_timeseries.to_csv(tmp_data_path)
    
    from src.models.sarima import run_sarima_pipeline
    run_sarima_pipeline(tmp_data_path, tmp_path)
    
    # Check outputs
    assert (tmp_path / "SARIMA" / "metrics.csv").exists()
    assert (tmp_path / "SARIMA" / "sarima_forecast.png").exists()

# tests/test_lstm_model.py
def test_create_sequences_shape():
    """Test sequence creation produces correct shapes."""
    from src.models.lstm import create_sequences
    
    data = np.arange(100).reshape(-1, 1)
    X, y = create_sequences(data, n_input=10)
    
    assert X.shape == (90, 10, 1)
    assert y.shape == (90, 1)

def test_create_sequences_values():
    """Test sequence creation produces correct values."""
    from src.models.lstm import create_sequences
    
    data = np.arange(5).reshape(-1, 1)
    X, y = create_sequences(data, n_input=2)
    
    assert X.shape == (3, 2, 1)
    assert np.array_equal(X[0], [[0], [1]])
    assert np.array_equal(y[0], [2])

# tests/test_integration.py
def test_full_pipeline_sample_mode(tmp_path):
    """Test complete pipeline in sample mode."""
    import subprocess
    result = subprocess.run(
        ["python", "main.py", "--sample"],
        capture_output=True,
        text=True,
        timeout=60
    )
    assert result.returncode == 0
    assert "PIPELINE EXECUTION COMPLETED" in result.stdout
```

### 2.3 Coverage Setup
```bash
# Install coverage tools
pip install pytest-cov pytest-xdist

# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# Generate coverage report
coverage html
```

---

## Phase 3: Configuration Management (Week 5) 🟡

**Priority: P1 - High**  
**Estimated Time: 10-20 hours**

### 3.1 Create Configuration System

```python
# File: src/config.py (REPLACE)
from pathlib import Path
from typing import Dict, Any
import yaml
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Model hyperparameters."""
    lstm: Dict[str, Any] = field(default_factory=lambda: {
        'n_input': 48,
        'epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout': 0.2,
        'units': [64],
    })
    sarima: Dict[str, Any] = field(default_factory=lambda: {
        'seasonal_period': 24,
        'tuning_window_days': 90,
        'max_iter': 50,
        'method': 'lbfgs',
    })

@dataclass
class DataConfig:
    """Data processing parameters."""
    test_days: int = 7
    split_ratio: float = 0.8
    interpolation_method: str = 'time'
    min_data_points: int = 24

@dataclass
class Config:
    """Main configuration."""
    seed: int = 42
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    models: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        self.data_dir = self.base_dir / 'data'
        self.output_dir = self.base_dir / 'plots'
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

# Global config instance
config = Config()
```

```yaml
# config.yaml (NEW)
seed: 42

models:
  lstm:
    n_input: 48
    epochs: 20
    batch_size: 32
    learning_rate: 0.001
    dropout: 0.2
    units: [128, 64]
    
  sarima:
    seasonal_period: 24
    tuning_window_days: 90
    max_iter: 50
    method: 'lbfgs'

data:
  test_days: 7
  split_ratio: 0.8
  interpolation_method: 'time'
  min_data_points: 24
```

### 3.2 Environment Variables for Secrets
```python
# File: src/config.py (ADD)
import os
from dotenv import load_dotenv

load_dotenv()

# For future API keys, database credentials, etc.
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
```

---

## Phase 4: Documentation & Type Hints (Week 6) 🟡

**Priority: P1 - High**  
**Estimated Time: 10-20 hours**

### 4.1 Add Type Hints to All Functions

```python
# Before
def create_sequences(data, n_input=24):
    # ...

# After
from typing import Tuple
import numpy as np

def create_sequences(
    data: np.ndarray, 
    n_input: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time series forecasting.
    
    Args:
        data: Time series array of shape (n_samples,) or (n_samples, n_features)
        n_input: Lookback window size (number of time steps)
    
    Returns:
        X: Input sequences of shape (n_sequences, n_input, n_features)
        y: Target values of shape (n_sequences, n_features)
    
    Raises:
        ValueError: If n_input is larger than data length
        TypeError: If data is not a numpy array
    
    Example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> X, y = create_sequences(data, n_input=2)
        >>> X.shape
        (3, 2, 1)
    """
    # ... implementation
```

### 4.2 Generate API Documentation
```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

# Initialize
cd docs
sphinx-quickstart

# Configure autodoc in docs/conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

# Build docs
make html
```

---

## Phase 5: CI/CD Enhancement (Week 7) 🟡

**Priority: P1 - High**  
**Estimated Time: 10-20 hours**

### 5.1 Enhanced GitHub Actions Workflow

See detailed workflow in REPOSITORY_REVIEW.md Section 11.

**Key additions:**
- Linting (black, flake8, pylint)
- Security scanning (bandit, safety)
- Code coverage (codecov)
- Multi-OS testing (Linux, Windows, macOS)
- Multi-Python version (3.8-3.12)
- Artifact uploading
- Caching

### 5.2 Pre-commit Hooks
```yaml
# .pre-commit-config.yaml (NEW)
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
  
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
```

---

## Phase 6: Monitoring & Logging (Week 8) 🟡

**Priority: P1 - High**  
**Estimated Time: 10-20 hours**

### 6.1 Structured Logging

```python
# File: src/logger.py (NEW)
import logging
import structlog
from pathlib import Path

def setup_logging(log_level: str = 'INFO', log_file: Path = None):
    """Configure structured logging."""
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ]
    
    structlog.configure(
        processors=processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )

logger = structlog.get_logger()
```

---

## Phase 7: Advanced Features (Month 2-3) 💡

**Priority: P2 - Nice to Have**  
**Estimated Time: 40-80 hours**

### 7.1 Model Versioning with MLflow
### 7.2 Data Versioning with DVC
### 7.3 Model Serving API with FastAPI
### 7.4 Ensemble Methods
### 7.5 Probabilistic Forecasting
### 7.6 Monitoring Dashboards
### 7.7 Distributed Training

---

## Progress Tracking

### Week-by-Week Checklist

#### Week 1: Critical Fixes
- [x] Fix requirements.txt encoding
- [x] Update TensorFlow version
- [ ] Add input validation
- [ ] Add proper error handling
- [ ] Set random seeds properly

#### Week 2: Testing Foundation
- [ ] Create test structure
- [ ] Write 20+ unit tests
- [ ] Achieve 30% coverage

#### Week 3: More Testing
- [ ] Write 40+ more tests
- [ ] Achieve 60% coverage
- [ ] Add integration tests

#### Week 4: Complete Testing
- [ ] Write remaining tests
- [ ] Achieve 80%+ coverage
- [ ] Add CI coverage reporting

#### Week 5: Configuration
- [ ] Extract all hardcoded values
- [ ] Create config.yaml
- [ ] Add environment variables
- [ ] Add config validation

#### Week 6: Documentation
- [ ] Add type hints (100% coverage)
- [ ] Add docstrings (100% coverage)
- [ ] Generate API docs
- [ ] Create architecture diagrams

#### Week 7: CI/CD
- [ ] Add linting to CI
- [ ] Add security scanning
- [ ] Multi-OS testing
- [ ] Artifact uploading

#### Week 8: Monitoring
- [ ] Structured logging
- [ ] Metrics collection
- [ ] Health checks
- [ ] Basic dashboard

---

## Success Metrics

### Target Ratings After Completion

| Section | Current | Target | Status |
|---------|---------|--------|--------|
| Testing | 3.0/10 | 9.0/10 | 🔴 |
| Error Handling | 4.0/10 | 8.5/10 | 🔴 |
| Configuration | 4.0/10 | 8.5/10 | 🔴 |
| Dependencies | 5.0/10 | 9.0/10 | 🟡 |
| Monitoring | 2.0/10 | 8.0/10 | 🔴 |
| Code Quality | 6.5/10 | 9.0/10 | 🟡 |
| Documentation | 7.0/10 | 9.0/10 | 🟡 |

**Overall Target: 7.2/10 → 9.0/10**

---

## Resources & References

### Testing
- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices - Google](https://testing.googleblog.com/)

### Type Hints
- [PEP 484 - Type Hints](https://www.python.org/dev/peps/pep-0484/)
- [mypy Documentation](http://mypy-lang.org/)

### Configuration
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

### MLOps
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)

### Monitoring
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [structlog Documentation](https://www.structlog.org/)

---

**Last Updated**: December 6, 2025  
**Status**: Phase 1 (Critical Fixes) - In Progress
