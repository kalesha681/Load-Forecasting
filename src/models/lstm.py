
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error

from ..config import ensure_output_dir, COL_DEMAND_YEARLY, PROCESSED_DATA_DIR
from ..data_loader import load_and_preprocess, train_test_split
from ..metrics import evaluate_mape, evaluate_rmse
from ..visualization import plot_time_series, plot_train_test_split, plot_forecast_vs_actual

def add_time_features(df):
    """Add hour, dayofweek, month, etc."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

def create_sequences(data, n_input=24):
    """
    Vectorized sequence creation using stride_tricks.
    data: 1D or 2D array [samples, features]
    n_input: lookback window size
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
    
    X = np.lib.stride_tricks.as_strided(
        data,
        shape=(n_sequences, n_input, n_features),
        strides=(s0, s0, s1)
    )
    
    # y is the next step after the window
    # y corresponding to X[i] is data[i + n_input]
    y = data[n_input:]
    
    return X, y

def run_lstm(data_path=None, test_days=7):
    # Config
    N_INPUT = 24 * 2 # Lookback 48 hours
    EPOCHS = 20 # Keep it reasonable
    BATCH_SIZE = 64
    
    output_dir = ensure_output_dir('LSTM')
    PLOT_OUT = os.path.join(output_dir, 'lstm_forecast.png')
    METRICS_PATH = os.path.join(output_dir, 'metrics.csv')
    MODEL_OUT = os.path.join(output_dir, 'lstm_model.keras')

    # Load Data
    if data_path:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        df = load_and_preprocess()

    col = COL_DEMAND_YEARLY if COL_DEMAND_YEARLY in df.columns else df.columns[0]
    
    # Train/Test Split
    train_df, test_df = train_test_split(df, test_days=test_days)
    
    train_data = train_df[[col]].values
    test_data = test_df[[col]].values
    
    # Scale
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    # Use same scaler
    test_scaled = scaler.transform(test_data)
    
    # Create Sequences
    # For testing, we need history from train to predict first elements of test if strictly sequential
    # But usually simple approach: create sequences from train, and from test (loosing first N_INPUT of test)
    # Better: concat last N_INPUT of train to test
    
    X_train, y_train = create_sequences(train_scaled, n_input=N_INPUT)
    
    # Prepare test input
    # We need last N_INPUT from train to start predicting test
    concat_test = np.vstack([train_scaled[-N_INPUT:], test_scaled])
    X_test, y_test = create_sequences(concat_test, n_input=N_INPUT)
    
    print(f"Train X shape: {X_train.shape}")
    print(f"Test X shape: {X_test.shape}")
    
    # Build Model
    model = Sequential([
        Input(shape=(N_INPUT, 1)),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    print("Training LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )
    
    # Predict
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    
    # y_test is scaled, inverse it for metrics
    y_true = scaler.inverse_transform(y_test)
    
    # Align dates for plotting/saving
    # y_true corresponds to test_df (since we padded carefully)
    # Check lengths
    if len(y_pred) != len(test_df):
        print(f"Warning: Length mismatch. Pred: {len(y_pred)}, Test DF: {len(test_df)}")
        # If we did it right, they should match exactly because of the padding
    
    # Evaluation
    rmse = evaluate_rmse(y_true, y_pred)
    mape = evaluate_mape(y_true, y_pred)
    
    print(f"\nFinal Results on Test Set:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape*100:.2f}%")
    
    # Save Metrics
    metrics_df = pd.DataFrame([{
        'Model': 'LSTM',
        'RMSE': rmse,
        'MAPE': mape,
        'Lookback': N_INPUT,
        'Epochs': EPOCHS
    }])
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"Metrics saved to {METRICS_PATH}")
    
    # Plot
    plt.figure(figsize=(15,6))
    plt.plot(test_df.index, y_true, label='Actual')
    plt.plot(test_df.index, y_pred, label='LSTM Forecast', linestyle='--')
    plt.title(f"LSTM Forecast (MAPE={mape*100:.2f}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_OUT)
    
    # Save Model
    model.save(MODEL_OUT)
    print("Model saved.")

if __name__ == "__main__":
    run_lstm()
