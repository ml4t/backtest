"""
Test 100% alignment across all 4 frameworks using for_zipline_matching() config.

This test uses next-day execution (fill_timing="next_open") to match Zipline's
inherent T+1 execution model.
"""

import pickle
from pathlib import Path

from frameworks.backtrader_adapter import BacktraderAdapter
from frameworks.qengine_adapter import BacktestAdapter
from frameworks.vectorbt_adapter import VectorBTAdapter
from frameworks.zipline_adapter import ZiplineAdapter
from frameworks.base import FrameworkConfig


def main():
    # Load test signals
    signal_file = Path(__file__).parent / "signals" / "sp500_top10_sma_crossover.pkl"
    print(f"Loading signals from: {signal_file}")
    print()

    with open(signal_file, 'rb') as f:
        signal_set = pickle.load(f)

    # Get AAPL data
    asset_data = signal_set['assets']['AAPL']
    data = asset_data['data']
    signals = asset_data['signals']

    print(f"Signal verification:")
    print(f"  Total ENTRY signals: {signals['entry'].sum()}")
    print(f"  Total EXIT signals: {signals['exit'].sum()}")
    print()

    # Use for_zipline_matching() config (next-day fills for ALL frameworks)
    config = FrameworkConfig.for_zipline_matching()
    print("Configuration: FrameworkConfig.for_zipline_matching()")
    print(f"  fill_timing: {config.fill_timing}")
    print(f"  commission: {config.commission_pct*100}%")
    print(f"  slippage: {config.slippage_pct*100}%")
    print(f"  Expected: All frameworks execute signals 1 day later (T+1)")
    print()

    frameworks = {
        'ml4t.backtest': BacktestAdapter(),
        'Backtrader': BacktraderAdapter(),
        'VectorBT': VectorBTAdapter(),
        'Zipline': ZiplineAdapter(),
    }

    results = {}
    for name, adapter in frameworks.items():
        print(f"Running {name}...")
        result = adapter.run_with_signals(data, signals, config)
        results[name] = result
        print(f"  Final: ${result.final_value:,.2f} ({result.total_return:.2f}%)")
        print(f"  Trades: {result.num_trades} ({len(result.trades)} extracted)")
        if len(result.trades) > 0:
            print(f"  First trade: {result.trades[0].timestamp.strftime('%Y-%m-%d')} {result.trades[0].action}")
        print()

    # Compare first 10 trades
    print("="*80)
    print("FIRST 10 TRADES COMPARISON")
    print("="*80)

    for i in range(10):
        print(f"\nTrade #{i+1}:")
        for name in ['ml4t.backtest', 'Backtrader', 'VectorBT', 'Zipline']:
            trades = results[name].trades
            if i < len(trades):
                t = trades[i]
                print(f"  {name:<15}: {t.timestamp.strftime('%Y-%m-%d')} {t.action} {t.quantity:.2f} @ ${t.price:.2f}")
            else:
                print(f"  {name:<15}: [NO TRADE]")

    # Matching analysis
    print("\n" + "="*80)
    print("MATCHING ANALYSIS")
    print("="*80)

    # Check first trade dates
    print("\nFirst trade dates:")
    first_dates = {}
    for name in ['ml4t.backtest', 'Backtrader', 'VectorBT', 'Zipline']:
        if len(results[name].trades) > 0:
            first_dates[name] = results[name].trades[0].timestamp
            print(f"  {name:<15}: {first_dates[name].strftime('%Y-%m-%d')} {results[name].trades[0].action}")

    # Check if all match
    if len(set(d.date() for d in first_dates.values())) == 1:
        print("\n✓ All first trades match by date!")
    else:
        print("\n✗ First trades differ")

    # Trade counts
    print("\nTrade counts:")
    trade_counts = {}
    for name in ['ml4t.backtest', 'Backtrader', 'VectorBT', 'Zipline']:
        count = len(results[name].trades)
        trade_counts[name] = count
        print(f"  {name:<15}: {count}")

    if len(set(trade_counts.values())) == 1:
        print("\n✓ All frameworks have same trade count!")
    else:
        print("\n✗ Trade counts differ")

    # Final values
    print("\nFinal values:")
    final_values = {}
    for name in ['ml4t.backtest', 'Backtrader', 'VectorBT', 'Zipline']:
        value = results[name].final_value
        final_values[name] = value
        print(f"  {name:<15}: ${value:,.2f}")

    # Calculate variance
    import numpy as np
    values = list(final_values.values())
    avg_value = np.mean(values)
    max_variance = max(abs(v - avg_value) / avg_value * 100 for v in values)

    print(f"\nMaximum variance from average: {max_variance:.4f}%")

    if max_variance < 0.01:
        print("✓ PERFECT ALIGNMENT (<0.01% variance)")
    elif max_variance < 0.1:
        print("✓ Excellent alignment (<0.1% variance)")
    elif max_variance < 1.0:
        print("⚠ Good alignment (<1.0% variance)")
    else:
        print(f"✗ High variance ({max_variance:.2f}%) - further investigation needed")


if __name__ == "__main__":
    main()
