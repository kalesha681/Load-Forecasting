import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
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

# --- 1. Data Loading & Preprocessing ---
def load_and_preprocess(file_path):
    """Load data, parse datetime, set index, handle missing/duplicates, resample."""
    df = pd.read_csv(file_path, parse_dates=['Datetime'])
    df = df.drop_duplicates(subset='Datetime')
    df = df.set_index('Datetime')
    df = df.sort_index()
    df = df.asfreq('h')  # Use 'h' per pandas warning
    # Interpolate missing values after resample
    df['PJME_MW'] = df['PJME_MW'].interpolate('time')
    df['PJME_MW'] = df['PJME_MW'].bfill().ffill()  # fill any remaining leading/trailing nans
    # Debug print to confirm no NaNs remain
    print('Missing values after preprocessing:', df.isna().sum().to_dict())
    return df

# --- 2. Visualization: Raw Time Series ---
def plot_time_series(df, title, outpath=None):
    plt.figure(figsize=(16, 4))
    plt.plot(df.index, df['PJME_MW'])
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('PJME_MW')
    if outpath:
        plt.savefig(outpath)
    plt.show()

# --- 3. Feature Engineering ---
def add_time_features(df):
    """Add hour, dayofweek, month, year, dayofyear as features."""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

# --- 4. Train-Test Split ---
def train_test_split(df, split_date='2017-01-01'):
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date]
    return train, test

def plot_train_test_split(train_df, test_df, outpath=None):
    plt.figure(figsize=(16, 4))
    plt.plot(train_df.index, train_df['PJME_MW'], label='Train')
    plt.plot(test_df.index, test_df['PJME_MW'], label='Test')
    plt.title('Train/Test Split')
    plt.xlabel('Datetime')
    plt.ylabel('PJME_MW')
    plt.legend()
    if outpath:
        plt.savefig(outpath)
    plt.show()

# --- 5. Scaling & Windowing ---
def scale_data(train, test):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    return train_scaled, test_scaled, scaler

def create_sequences(data, n_input=24):
    X, y = [], []
    for i in range(n_input, len(data)):
        X.append(data[i - n_input:i, 0])
        y.append(data[i, 0])
    X = np.array(X)
    return X, np.array(y)

# --- 6. Model Building ---
def build_lstm_model(n_timesteps, n_features):
    model = Sequential()
    model.add(Input(shape=(n_timesteps, n_features)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 7. Main Pipeline ---
def main():
    # Config
    CSV_PATH = 'data/PJME_hourly.csv'
    SPLIT_DATE = '2017-01-01'
    N_INPUT = 24
    EPOCHS = 30
    BATCH_SIZE = 128
    PLOT_OUT = 'lstm_raw_timeseries.png'
    SPLIT_PLOT_OUT = 'lstm_train_test_split.png'

    # Data Loading and Preprocessing
    df = load_and_preprocess(CSV_PATH)
    # Ensure model-specific output directory exists
    output_root = 'output'
    output_dir = os.path.join(output_root, 'LSTM')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    PLOT_OUT = os.path.join(output_dir, 'lstm_raw_timeseries.png')
    SPLIT_PLOT_OUT = os.path.join(output_dir, 'lstm_train_test_split.png')
    LOSS_PLOT_OUT = os.path.join(output_dir, 'lstm_training_loss.png')
    VS_PLOT_OUT = os.path.join(output_dir, 'lstm_vs_actual.png')
    ZOOM_PLOT_OUT = os.path.join(output_dir, 'lstm_vs_actual_recent.png')
    MODEL_OUT = os.path.join(output_dir, 'lstm_pjme_model.keras')

    plot_time_series(df, 'PJME Hourly Power Demand', outpath=PLOT_OUT)
    df = add_time_features(df)
    train_df, test_df = train_test_split(df, split_date=SPLIT_DATE)
    
    # Print split info
    print(f"Train samples: {len(train_df)}  | Dates: {train_df.index.min()} — {train_df.index.max()}")
    print(f"Test samples:  {len(test_df)}  | Dates: {test_df.index.min()} — {test_df.index.max()}")
    plot_train_test_split(train_df, test_df, outpath=SPLIT_PLOT_OUT)

    # We forecast PJME_MW only for now
    train, test = pd.DataFrame(train_df['PJME_MW']), pd.DataFrame(test_df['PJME_MW'])
    train_scaled, test_scaled, scaler = scale_data(train, test)
    X_train, y_train = create_sequences(train_scaled, n_input=N_INPUT)
    X_test, y_test = create_sequences(np.vstack([train_scaled[-N_INPUT:], test_scaled]), n_input=N_INPUT)

    # Enhanced: print data shapes and a sample inverse transformed window.
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    example_idx = 0
    example_input = X_train[example_idx]
    example_target = y_train[example_idx]
    # Inverse scale for display
    example_input_inv = scaler.inverse_transform(example_input.reshape(-1, 1)).flatten()
    example_target_inv = scaler.inverse_transform(np.array([[example_target]])).flatten()[0]
    print("Example input sequence (original values):", example_input_inv)
    print("Example target (original value, next hour):", example_target_inv)

    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # --- Model Building ---
    print("\n--- Model Building ---")
    model = build_lstm_model(N_INPUT, 1)
    # Show summary after model creation
    print("\n--- Model Summary ---")
    model.summary()

    # --- Model Training ---
    print("\n--- Training Model ---")
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=2)

    # --- Plot Loss Curves: Train & Validation Loss ---
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss over Epochs')
    plt.savefig(LOSS_PLOT_OUT)
    plt.show()

    # --- Evaluation on Test Data ---
    print("\n--- Evaluation on Test Data ---")
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0]
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0]
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    print(f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}')

    # Mean Absolute Percentage Error (MAPE)
    try:
        from sklearn.metrics import mean_absolute_percentage_error
        mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)
    except ImportError:
        # Fallback manually if sklearn too old
        y_t = np.array(y_test_inv)
        y_p = np.array(y_pred_inv)
        mape = np.mean(np.abs((y_t - y_p) / np.clip(np.abs(y_t), 1e-10, None)))
    print(f'Mean Absolute Percentage Error: {mape*100:.2f}%')

    # --- Plot Actual vs Predicted (Full Test Set) ---
    plt.figure(figsize=(16,6))
    plt.plot(test_df.index[-len(y_test_inv):], y_test_inv, label='Actual')
    plt.plot(test_df.index[-len(y_pred_inv):], y_pred_inv, label='Forecast')
    plt.title('LSTM: Actual vs Forecast (Full Test Period)')
    plt.ylabel('PJME_MW')
    plt.xlabel('Datetime')
    plt.legend()
    plt.savefig(VS_PLOT_OUT)
    plt.show()

    # --- Plot Actual vs Predicted (Zoom: Last 3 Months) ---
    recent_n = 24*30*3
    plt.figure(figsize=(16,6))
    plt.plot(test_df.index[-recent_n:], y_test_inv[-recent_n:], label='Actual')
    plt.plot(test_df.index[-recent_n:], y_pred_inv[-recent_n:], label='Forecast')
    plt.title('LSTM: Actual vs Forecast (Zoomed: Last 3 Months)')
    plt.ylabel('PJME_MW')
    plt.xlabel('Datetime')
    plt.legend()
    plt.savefig(ZOOM_PLOT_OUT)
    plt.show()

    # --- Save Model ---
    print(f"\nSaving trained model to {MODEL_OUT} ...")
    model.save(MODEL_OUT)
    print("Model saved!")

if __name__ == "__main__":
    main()
