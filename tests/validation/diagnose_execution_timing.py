"""
Diagnose why frameworks execute on different signals.

Simple test: Load first 10 signals, show exactly when each framework
decides to trade or not trade.
"""

import pickle
from pathlib import Path
import pandas as pd

from frameworks.backtrader_adapter import BacktraderAdapter
from frameworks.qengine_adapter import BacktestAdapter
from frameworks.vectorbt_adapter import VectorBTAdapter
from frameworks.base import FrameworkConfig


SIGNAL_DIR = Path(__file__).parent / "signals"


def diagnose_signal_execution():
    """Show signal-by-signal execution decisions for all frameworks."""

    # Load SMA crossover signals
    signal_file = SIGNAL_DIR / "sp500_top10_sma_crossover.pkl"
    with open(signal_file, 'rb') as f:
        signal_set = pickle.load(f)

    # Use AAPL
    asset_data = signal_set['assets']['AAPL']
    data = asset_data['data']
    signals = asset_data['signals']

    print("="*80)
    print("EXECUTION TIMING DIAGNOSTIC")
    print("="*80)
    print(f"Asset: AAPL")
    print(f"Total bars: {len(data)}")
    print(f"Entry signals: {signals['entry'].sum()}")
    print(f"Exit signals: {signals['exit'].sum()}")
    print()

    # Show first 10 signals with prices
    print("Signal Timeline (first 10 signals):")
    print("="*80)

    signal_count = 0
    for i, (timestamp, row) in enumerate(data.iterrows()):
        entry_signal = signals['entry'].iloc[i] if i < len(signals) else False
        exit_signal = signals['exit'].iloc[i] if i < len(signals) else False

        if entry_signal or exit_signal:
            signal_count += 1
            signal_type = "ENTRY" if entry_signal else "EXIT"

            print(f"\nSignal #{signal_count} @ {timestamp.strftime('%Y-%m-%d')} ({signal_type})")
            print(f"  Bar #{i}")
            print(f"  Close: ${row['close']:.2f}")
            next_open = data['open'].iloc[i+1] if i+1 < len(data) else None
            if next_open is not None:
                print(f"  Next Open: ${next_open:.2f}")
            else:
                print(f"  Next Open: N/A")

            if signal_count >= 10:
                break

    print("\n" + "="*80)
    print("FRAMEWORK EXECUTION")
    print("="*80)

    # Run each framework
    config = FrameworkConfig.realistic()

    frameworks = {
        'ml4t.backtest': BacktestAdapter(),
        'Backtrader': BacktraderAdapter(),
        'VectorBT': VectorBTAdapter(),
    }

    results = {}
    for name, adapter in frameworks.items():
        print(f"\nRunning {name}...")
        result = adapter.run_with_signals(data, signals, config)
        results[name] = result

        print(f"  Final value: ${result.final_value:,.2f}")
        print(f"  Total return: {result.total_return:.2f}%")
        print(f"  Trades: {result.num_trades}")
        print(f"  Trade records extracted: {len(result.trades)}")

    # Compare execution sequences
    print("\n" + "="*80)
    print("TRADE EXECUTION COMPARISON")
    print("="*80)

    # Show first few trades from each
    for name in ['ml4t.backtest', 'Backtrader', 'VectorBT']:
        result = results[name]
        print(f"\n{name} (first 5 trades):")
        if len(result.trades) == 0:
            print("  [NO TRADES EXTRACTED]")
        else:
            for i, trade in enumerate(result.trades[:5]):
                print(f"  {i+1}. {trade.timestamp.strftime('%Y-%m-%d')} {trade.action} "
                      f"{trade.quantity:.2f} @ ${trade.price:.2f}")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    qengine_count = len(results['ml4t.backtest'].trades)
    backtrader_count = len(results['Backtrader'].trades)
    vectorbt_count = len(results['VectorBT'].trades)

    print(f"\nTrade count comparison:")
    print(f"  ml4t.backtest:  {qengine_count} trades")
    print(f"  Backtrader:     {backtrader_count} trades")
    print(f"  VectorBT:       {vectorbt_count} trades")

    if qengine_count != backtrader_count:
        print(f"\n⚠️  ml4t.backtest and Backtrader executed different number of trades!")
        print(f"    Difference: {abs(qengine_count - backtrader_count)} trades")

        if qengine_count > backtrader_count:
            print(f"    ml4t.backtest executed MORE trades")
            print(f"    Hypothesis: Backtrader may be rejecting orders due to cash constraints")
        else:
            print(f"    Backtrader executed MORE trades")
            print(f"    Hypothesis: ml4t.backtest may be rejecting orders or has different state logic")

    if vectorbt_count == 0:
        print(f"\n⚠️  VectorBT trade extraction failed!")
        print(f"    Reported {results['VectorBT'].num_trades} trades but extracted 0")
        print(f"    This is a bug in vectorbt_adapter.py - trade extraction not implemented in run_with_signals()")

    print("\n✅ Diagnostic complete")
    print("\nNext steps:")
    print("  1. Fix VectorBT trade extraction in run_with_signals()")
    print("  2. Add detailed logging to see WHY frameworks skip certain signals")
    print("  3. Check position state before each signal (are they tracking positions correctly?)")


if __name__ == "__main__":
    diagnose_signal_execution()
