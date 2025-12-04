import os
import pandas as pd
import matplotlib.pyplot as plt

from ..config import ensure_output_dir
from ..data_loader import load_and_preprocess

def run_ma():
    # Output setup
    output_dir = ensure_output_dir('Moving_Average')

    # Data Loading
    df = load_and_preprocess()

    # Keep only the last 30 days for faster plotting
    df = df.last("30D")

    # Plot raw
    plt.figure(figsize=(12, 4))
    plt.plot(df['PJME_MW'], label='Raw Load Data', alpha=0.6)
    plt.title("Raw Electricity Load (Last 30 Days)")
    plt.xlabel("Date")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'moving_avg_raw.png'))
    plt.show()

    # Apply moving averages
    df['MA_6H'] = df['PJME_MW'].rolling(window=6).mean()
    df['MA_24H'] = df['PJME_MW'].rolling(window=24).mean()
    df['MA_72H'] = df['PJME_MW'].rolling(window=72).mean()
    df['MA_12H'] = df['PJME_MW'].rolling(window=12).mean()
    df['MA_168H'] = df['PJME_MW'].rolling(window=168).mean()

    # Plot smoothed
    plt.figure(figsize=(12, 5))
    plt.plot(df['PJME_MW'], label='Raw Load', color='gray', alpha=0.5)
    plt.plot(df['MA_6H'], label='6-Hour MA', color='blue')
    plt.plot(df['MA_24H'], label='24-Hour MA', color='red')
    plt.plot(df['MA_72H'], label='72-Hour MA', color='green')
    plt.plot(df['MA_12H'], label='12-Hour MA', color='purple')
    plt.plot(df['MA_168H'], label='168-Hour MA', color='orange')
    plt.title("Electric Load Smoothing using Moving Averages")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'moving_avg_smoothed.png'))
    plt.show()

if __name__ == "__main__":
    run_ma()
