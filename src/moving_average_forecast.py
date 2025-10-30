import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure output subfolder
output_root = 'output'
output_dir = os.path.join(output_root, 'Moving_Average')
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# --- Step 1: Load dataset ---
df = pd.read_csv("data/PJME_hourly.csv", parse_dates=['Datetime'], index_col='Datetime')
df = df.sort_index()
# Handle duplicates and ensure hourly frequency
df = df[~df.index.duplicated(keep='first')]
df = df.asfreq('h')
df['PJME_MW'] = df['PJME_MW'].interpolate('time').bfill().ffill()

# Keep only the last 30 days for faster plotting
df = df.last("30D")

# --- Step 2: Plot raw load data ---
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

# --- Step 3: Apply moving averages ---
df['MA_6H'] = df['PJME_MW'].rolling(window=6).mean()   # 6-hour average
df['MA_24H'] = df['PJME_MW'].rolling(window=24).mean() # 1-day average
df['MA_72H'] = df['PJME_MW'].rolling(window=72).mean() # 3-day average
df['MA_12H'] = df['PJME_MW'].rolling(window=12).mean()
df['MA_168H'] = df['PJME_MW'].rolling(window=168).mean()  # 7 days

# --- Step 4: Plot all together ---
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
