import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import shared modules
from ..config import SPLIT_DATE, ensure_output_dir
from ..data_loader import load_and_preprocess, train_test_split
from ..metrics import evaluate_mape
from ..visualization import plot_time_series, plot_train_test_split, plot_forecast_vs_actual

# --- Feature Engineering ---
def add_time_features(df):
    """Add hour, dayofweek, month, year, dayofyear as features."""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

# --- Scaling & Windowing ---
def scale_data(train, test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled, scaler

def create_sequences(data, n_input=24):
    # Vectorized sequence creation
    if data.ndim > 1:
        data = data.flatten()
        
    n_samples = len(data) - n_input
    if n_samples <= 0:
        return np.array([]), np.array([])
        
    strides = data.strides[0]
    X = np.lib.stride_tricks.as_strided(
        data, 
        shape=(n_samples, n_input), 
        strides=(strides, strides)
    )
    y = data[n_input:]
    return X, y

# --- Model Building ---
def build_lstm_model(n_timesteps, n_features):
    model = Sequential()
    model.add(Input(shape=(n_timesteps, n_features)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Main Pipeline ---
def run_lstm():
    # Config
    N_INPUT = 24
    EPOCHS = 30
    BATCH_SIZE = 128
    
    # Output setup
    output_dir = ensure_output_dir('LSTM')
    PLOT_OUT = os.path.join(output_dir, 'lstm_raw_timeseries.png')
    SPLIT_PLOT_OUT = os.path.join(output_dir, 'lstm_train_test_split.png')
    LOSS_PLOT_OUT = os.path.join(output_dir, 'lstm_training_loss.png')
    VS_PLOT_OUT = os.path.join(output_dir, 'lstm_vs_actual.png')
    ZOOM_PLOT_OUT = os.path.join(output_dir, 'lstm_vs_actual_recent.png')
    MODEL_OUT = os.path.join(output_dir, 'lstm_pjme_model.keras')

    # Data Loading
    df = load_and_preprocess()
    plot_time_series(df, 'PJME Hourly Power Demand', outpath=PLOT_OUT)
    
    df = add_time_features(df)
    train_df, test_df = train_test_split(df, split_date=SPLIT_DATE)
    
    print(f"Train samples: {len(train_df)}  | Dates: {train_df.index.min()} — {train_df.index.max()}")
    print(f"Test samples:  {len(test_df)}  | Dates: {test_df.index.min()} — {test_df.index.max()}")
    plot_train_test_split(train_df, test_df, outpath=SPLIT_PLOT_OUT)

    # Prepare data for LSTM
    train, test = pd.DataFrame(train_df['PJME_MW']), pd.DataFrame(test_df['PJME_MW'])
    train_scaled, test_scaled, scaler = scale_data(train, test)
    X_train, y_train = create_sequences(train_scaled, n_input=N_INPUT)
    X_test, y_test = create_sequences(np.vstack([train_scaled[-N_INPUT:], test_scaled]), n_input=N_INPUT)

    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Model Building
    print("\n--- Model Building ---")
    model = build_lstm_model(N_INPUT, 1)
    model.summary()

    # Training
    print("\n--- Training Model ---")
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=2)

    # Plot Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss over Epochs')
    plt.savefig(LOSS_PLOT_OUT)
    plt.show()

    # Evaluation
    print("\n--- Evaluation on Test Data ---")
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0]
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]
    
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = evaluate_mape(y_test_inv, y_pred_inv)
    
    print(f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}')
    print(f'Mean Absolute Percentage Error: {mape*100:.2f}%')

    # Plots
    plot_forecast_vs_actual(test_df.index[-len(y_test_inv):], y_test_inv, y_pred_inv, 
                          'LSTM: Actual vs Forecast (Full Test Period)', outpath=VS_PLOT_OUT)
                          
    recent_n = 24*30*3
    plot_forecast_vs_actual(test_df.index[-recent_n:], y_test_inv[-recent_n:], y_pred_inv[-recent_n:], 
                          'LSTM: Actual vs Forecast (Zoomed: Last 3 Months)', outpath=ZOOM_PLOT_OUT)

    # Save Model
    print(f"\nSaving trained model to {MODEL_OUT} ...")
    model.save(MODEL_OUT)
    print("Model saved!")

if __name__ == "__main__":
    run_lstm()
