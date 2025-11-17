"""
Debug why ml4t.backtest isn't executing the final signal.
"""

import pickle
from pathlib import Path

from frameworks.backtrader_adapter import BacktraderAdapter
from frameworks.qengine_adapter import BacktestAdapter
from frameworks.vectorbt_adapter import VectorBTAdapter
from frameworks.base import FrameworkConfig


def main():
    # Load signals
    signal_file = Path(__file__).parent / "signals" / "sp500_top10_sma_crossover.pkl"
    with open(signal_file, 'rb') as f:
        signal_set = pickle.load(f)

    asset_data = signal_set['assets']['AAPL']
    data = asset_data['data']
    signals = asset_data['signals']

    print("="*80)
    print("SIGNAL DATASET ANALYSIS")
    print("="*80)
    print(f"Data length: {len(data)}")
    print(f"Signals length: {len(signals)}")
    print(f"Data index range: {data.index[0]} to {data.index[-1]}")
    print(f"Signals index range: {signals.index[0]} to {signals.index[-1]}")
    print()

    # Find last few signals
    print("Last 10 signals:")
    last_signals = []
    for i in range(len(signals) - 1, max(-1, len(signals) - 50), -1):
        if signals['entry'].iloc[i] or signals['exit'].iloc[i]:
            sig_type = "ENTRY" if signals['entry'].iloc[i] else "EXIT"
            last_signals.append((i, signals.index[i], sig_type))
            if len(last_signals) >= 10:
                break

    for idx, date, sig_type in reversed(last_signals):
        print(f"  iloc[{idx}] = {date.strftime('%Y-%m-%d')} ({sig_type})")
    print()

    # Check if data and signals are aligned
    print("Alignment check:")
    print(f"  Data last date:   {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Signal last date: {signals.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Indices match: {(data.index == signals.index).all()}")
    print()

    # Check what happens with last signal
    last_signal_idx = last_signals[0][0]
    last_signal_date = last_signals[0][1]
    last_signal_type = last_signals[0][2]

    print(f"Last signal: iloc[{last_signal_idx}] = {last_signal_date.strftime('%Y-%m-%d')} ({last_signal_type})")
    print(f"Data has row for this date: {last_signal_date in data.index}")

    if last_signal_date in data.index:
        data_row = data.loc[last_signal_date]
        print(f"  OHLC: O={data_row['open']:.2f} H={data_row['high']:.2f} L={data_row['low']:.2f} C={data_row['close']:.2f}")


if __name__ == "__main__":
    main()
