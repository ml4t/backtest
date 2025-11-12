"""
Debug Test: Print signal values to understand phantom trades
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from common import load_real_crypto_data, BacktestConfig

# Load data
ohlcv = load_real_crypto_data(symbol="BTC", data_type="spot", n_bars=200)

# Create explicit signals
entries = pd.Series([False] * 200, index=ohlcv.index)
exits = pd.Series([False] * 200, index=ohlcv.index)

# Only these bars should have signals
entries.iloc[10] = True
exits.iloc[20] = True
entries.iloc[60] = True
exits.iloc[70] = True
entries.iloc[110] = True
exits.iloc[120] = True

print("Entry signals:")
for idx, val in entries[entries].items():
    print(f"  Bar {entries.index.get_loc(idx)}: {idx} → {val}")

print("\nExit signals:")
for idx, val in exits[exits].items():
    print(f"  Bar {exits.index.get_loc(idx)}: {idx} → {val}")

# Check specific bars that cause phantom trades
problem_bars = [20, 21, 70, 71, 120, 121]
print("\nSignals at problem bars:")
for bar_idx in problem_bars:
    entry = entries.iloc[bar_idx]
    exit = exits.iloc[bar_idx]
    print(f"  Bar {bar_idx}: entry={entry}, exit={exit}")
