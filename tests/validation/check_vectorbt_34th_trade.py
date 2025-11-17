"""Check what VectorBT's 34th trade is."""

import pickle
from pathlib import Path
from frameworks.vectorbt_adapter import VectorBTAdapter
from frameworks.base import FrameworkConfig

# Load signals
signal_file = Path(__file__).parent / "signals" / "sp500_top10_sma_crossover.pkl"
with open(signal_file, 'rb') as f:
    signal_set = pickle.load(f)

asset_data = signal_set['assets']['AAPL']
data = asset_data['data']
signals = asset_data['signals']

# Run VectorBT
config = FrameworkConfig.for_matching()
adapter = VectorBTAdapter()
result = adapter.run_with_signals(data, signals, config)

print(f"VectorBT Results:")
print(f"  Total trades: {result.num_trades}")
print(f"  Extracted trades: {len(result.trades)}")
print()

# Show last 5 trades
print("Last 5 trades:")
for i in range(max(0, len(result.trades)-5), len(result.trades)):
    t = result.trades[i]
    print(f"  Trade #{i+1}: {t.timestamp.strftime('%Y-%m-%d')} {t.action} {t.quantity:.2f} @ ${t.price:.2f}")

# Check signal stats
print()
print(f"Signal stats:")
print(f"  ENTRY signals: {signals['entry'].sum()}")
print(f"  EXIT signals: {signals['exit'].sum()}")

# Find last signal
last_entry_idx = None
last_exit_idx = None
for i in range(len(signals)-1, -1, -1):
    if last_entry_idx is None and signals['entry'].iloc[i]:
        last_entry_idx = i
    if last_exit_idx is None and signals['exit'].iloc[i]:
        last_exit_idx = i
    if last_entry_idx and last_exit_idx:
        break

if last_entry_idx is not None:
    print(f"  Last ENTRY: iloc[{last_entry_idx}] = {signals.index[last_entry_idx].strftime('%Y-%m-%d')}")
if last_exit_idx is not None:
    print(f"  Last EXIT:  iloc[{last_exit_idx}] = {signals.index[last_exit_idx].strftime('%Y-%m-%d')}")
