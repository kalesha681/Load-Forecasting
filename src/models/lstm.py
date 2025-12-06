
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Safe Import for TensorFlow/Keras ---
from typing import Tuple, Union, Optional
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    TF_AVAILABLE = True
except ImportError:
    try:
        # Fallback for some environments
        import keras
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout, Input
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
        print("Warning: TensorFlow/Keras not found. LSTM will fail if run.")


from ..config import ensure_output_dir, COL_DEMAND_YEARLY, PROCESSED_DATA_DIR, MODEL_CONFIG, DATA_CONFIG
from ..data_loader import train_test_split
from ..metrics import evaluate_mape, evaluate_rmse
from ..visualization import plot_time_series, plot_train_test_split, plot_forecast_vs_actual

def add_time_features(df):
    """Add hour, dayofweek, month, etc."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

def create_sequences(data: np.ndarray, n_input: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences for time series forecasting.

    Uses NumPy stride tricks for efficient vectorized sequence generation.

    Args:
        data (np.ndarray): 1D or 2D array [samples, features].
        n_input (int, optional): Lookback window size. Defaults to 24.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            X: Input sequences of shape (n_sequences, n_input, n_features).
            y: Target values of shape (n_sequences, n_features).
    """
    # ensure 2D [samples, features]
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        
    n_samples, n_features = data.shape
    n_sequences = n_samples - n_input
    
    if n_sequences <= 0:
        return np.array([]), np.array([])
        
    # X: [n_sequences, n_input, n_features]
    # We want a sliding window view
    # Stride tricks for 2D array to get 3D view
    
    # Calculate strides
    # data.strides is (bytes_per_row, bytes_per_col)
    s0, s1 = data.strides
    
    # Shape of result: (n_sequences, n_input, n_features)
    # Strides: (step to next sequence, step to next time step, step to next feature)
    # step to next sequence = s0 (move down one row in original)
    # step to next time step = s0 (move down one row in original)
    # step to next feature = s1
    
    # Safety check: Ensure valid memory layout before striding
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
        s0, s1 = data.strides

    X = np.lib.stride_tricks.as_strided(
        data,
        shape=(n_sequences, n_input, n_features),
        strides=(s0, s0, s1)
    )
    
    # y is the next step after the window
    # y corresponding to X[i] is data[i + n_input]
    y = data[n_input:]
    
    return X, y


from pathlib import Path

def run_lstm_pipeline(data_path: Union[str, Path], output_dir: Union[str, Path]) -> None:
    """
    Execute the full LSTM forecasting pipeline.

    Steps:
    1. Load and preprocess data.
    2. Split into train/test sets.
    3. Scale data using MinMaxScaler.
    4. Create sliding window sequences.
    5. Build and train LSTM model.
    6. Generate predictions and evaluate performance (RMSE, MAPE).
    7. Save metrics, plots, and the trained model.

    Args:
        data_path (Union[str, Path]): Path to the input CSV file.
        output_dir (Union[str, Path]): Directory to save artifacts.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    if not TF_AVAILABLE:
        print("ERROR: TensorFlow/Keras is not installed or failed to import.")
        print("Please ensure tensorflow is installed correctly.")
        raise ImportError("TensorFlow not available")

    output_dir = Path(output_dir) / 'LSTM'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    PLOT_OUT = output_dir / 'lstm_forecast.png'
    METRICS_PATH = output_dir / 'metrics.csv'
    MODEL_OUT = output_dir / 'lstm_model.keras'
    
    lstm_config = MODEL_CONFIG['lstm']
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Identify column
    if COL_DEMAND_YEARLY in df.columns:
        col = COL_DEMAND_YEARLY
    elif "Demand_MW" in df.columns:
        col = "Demand_MW"
    else:
        col = df.columns[0]
        
    train, test = train_test_split(df, test_days=DATA_CONFIG['test_days'])
    
    # Scaling
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[[col]])
    test_scaled = scaler.transform(test[[col]])
    
    N_INPUT = lstm_config['n_input']
    X_train, y_train = create_sequences(train_scaled, n_input=N_INPUT)
    
    # For test, we need context
    concat_test = np.vstack([train_scaled[-N_INPUT:], test_scaled])
    X_test, y_test = create_sequences(concat_test, n_input=N_INPUT)
    
    if len(X_train) == 0:
        print("Not enough data for LSTM training.")
        return

    # Model
    model = Sequential([
        Input(shape=(N_INPUT, 1)),
        LSTM(lstm_config['units']),
        Dropout(lstm_config['dropout']),
        Dense(1)
    ])
    model.compile(optimizer=lstm_config['optimizer'], loss=lstm_config['loss'])
    
    # Train
    model.fit(X_train, y_train, epochs=lstm_config['epochs'], batch_size=lstm_config['batch_size'], verbose=0)
    
    # Predict
    pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(pred_scaled)
    y_true = scaler.inverse_transform(y_test)
    
    # Evaluate
    mape = evaluate_mape(y_true, y_pred)
    rmse = evaluate_rmse(y_true, y_pred)
    
    print(f"LSTM Results: RMSE={rmse:.2f}, MAPE={mape*100:.2f}%")
    
    # Save
    pd.DataFrame([{'Model': 'LSTM', 'RMSE': rmse, 'MAPE': mape}]).to_csv(METRICS_PATH, index=False)
    
    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(test.index, y_true, label='Actual')
    plt.plot(test.index, y_pred, label='LSTM', linestyle='--')
    plt.title(f"LSTM Forecast (MAPE={mape*100:.2f}%)")
    plt.legend()
    plt.savefig(PLOT_OUT)
    plt.close()
    
    model.save(MODEL_OUT)
    print(f"Saved artifacts to {output_dir}")

def run_lstm(): pass

