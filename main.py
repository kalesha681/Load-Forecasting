import sys
import os

USAGE = """
Load-Forecasting entry point

Usage:
  python main.py [lstm|sarima|ma]

Examples:
  python main.py lstm
  python main.py sarima
  python main.py ma

Note:
  - Ensure dependencies: pip install -r requirements.txt
  - For LSTM, you may also need: pip install tensorflow keras
"""

def run_lstm():
	os.system("python src/lstm_forecast.py")

def run_sarima():
	os.system("python src/sarima_forecast.py")

def run_ma():
	os.system("python src/moving_average_forecast.py")

def main():
	if len(sys.argv) < 2:
		print(USAGE)
		return
	cmd = sys.argv[1].lower()
	if cmd == 'lstm':
		run_lstm()
	elif cmd == 'sarima':
		run_sarima()
	elif cmd in ('ma', 'moving_average', 'moving-average'):
		run_ma()
	else:
		print(USAGE)

if __name__ == "__main__":
	main()
