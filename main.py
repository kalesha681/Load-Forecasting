import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.lstm import run_lstm
from src.models.sarima import run_sarima
from src.models.moving_average import run_ma

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
