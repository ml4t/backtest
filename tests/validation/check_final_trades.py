"""
Check final trades to understand why ml4t.backtest has 68 vs 67 for others.
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

    config = FrameworkConfig.for_matching()

    frameworks = {
        'ml4t.backtest': BacktestAdapter(),
        'Backtrader': BacktraderAdapter(),
        'VectorBT': VectorBTAdapter(),
    }

    results = {}
    for name, adapter in frameworks.items():
        result = adapter.run_with_signals(data, signals, config)
        results[name] = result

    # Show last 5 trades from each
    print("="*80)
    print("LAST 5 TRADES FROM EACH FRAMEWORK")
    print("="*80)

    for name in ['ml4t.backtest', 'Backtrader', 'VectorBT']:
        trades = results[name].trades
        print(f"\n{name} (total: {len(trades)} trades):")
        for t in trades[-5:]:
            print(f"  {t.timestamp.strftime('%Y-%m-%d')} {t.action:<4} {t.quantity:>8.2f} @ ${t.price:.2f}")

    # Check signal totals
    print("\n" + "="*80)
    print("SIGNAL VERIFICATION")
    print("="*80)
    print(f"Total ENTRY signals: {signals['entry'].sum()}")
    print(f"Total EXIT signals: {signals['exit'].sum()}")

    # Find last signal
    for i in range(len(signals) - 1, -1, -1):
        if signals['entry'].iloc[i] or signals['exit'].iloc[i]:
            sig_type = "ENTRY" if signals['entry'].iloc[i] else "EXIT"
            print(f"Last signal: iloc[{i}] = {signals.index[i].strftime('%Y-%m-%d')} ({sig_type})")
            break


if __name__ == "__main__":
    main()
