"""
Test aligned execution timing across frameworks.

This tests the for_matching() preset to see if all frameworks produce
identical results when using same-bar close execution.
"""

import pickle
from pathlib import Path

from frameworks.backtrader_adapter import BacktraderAdapter
from frameworks.qengine_adapter import BacktestAdapter
from frameworks.vectorbt_adapter import VectorBTAdapter
from frameworks.zipline_adapter import ZiplineAdapter
from frameworks.base import FrameworkConfig


def main():
    # Load signals
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
    print(f"  Total bars: {len(signals)}")
    print(f"  Total ENTRY signals: {signals['entry'].sum()}")
    print(f"  Total EXIT signals: {signals['exit'].sum()}")

    # Find first signal
    for i in range(len(signals)):
        if signals['entry'].iloc[i] or signals['exit'].iloc[i]:
            sig_type = "ENTRY" if signals['entry'].iloc[i] else "EXIT"
            print(f"  First signal: iloc[{i}] = {signals.index[i].strftime('%Y-%m-%d')} ({sig_type})")
            break
    print()

    # Test with aligned execution timing
    print("="*80)
    print("TESTING: FrameworkConfig.for_matching()")
    print("="*80)
    print("Configuration:")
    print("  - Same-bar close fills (all frameworks)")
    print("  - Backtrader COC enabled")
    print("  - No fees (commission=0, slippage=0)")
    print("  - Fractional shares enabled")
    print()

    config = FrameworkConfig.for_matching()

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
            first = result.trades[0]
            last = result.trades[-1]
            print(f"  First trade: {first.timestamp.strftime('%Y-%m-%d')} {first.action}")
            print(f"  Last trade:  {last.timestamp.strftime('%Y-%m-%d')} {last.action}")
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
                print(f"  {name:<15}: {t.timestamp.strftime('%Y-%m-%d')} {t.action:<4} {t.quantity:>8.2f} @ ${t.price:.2f}")
            else:
                print(f"  {name:<15}: [NO TRADE]")

    # Matching analysis
    print("\n" + "="*80)
    print("MATCHING ANALYSIS")
    print("="*80)

    all_frameworks = ['ml4t.backtest', 'Backtrader', 'VectorBT', 'Zipline']

    # Check first trades
    if all(len(results[fw].trades) > 0 for fw in all_frameworks):
        print(f"\nFirst trade dates:")
        for name in all_frameworks:
            first = results[name].trades[0]
            print(f"  {name:<15}: {first.timestamp.strftime('%Y-%m-%d')} {first.action}")

        # Check alignment
        first_trades = [results[fw].trades[0] for fw in all_frameworks]
        dates_match = all(t.timestamp == first_trades[0].timestamp for t in first_trades)
        actions_match = all(t.action == first_trades[0].action for t in first_trades)

        if dates_match and actions_match:
            print("\n✓ All frameworks match on first trade!")
        else:
            print("\n✗ First trades differ:")
            if not dates_match:
                print("  - Dates don't match")
            if not actions_match:
                print("  - Actions don't match")

    # Check trade counts
    print(f"\nTrade counts:")
    for name in all_frameworks:
        print(f"  {name:<15}: {len(results[name].trades)}")

    trade_counts = [len(results[fw].trades) for fw in all_frameworks]
    if len(set(trade_counts)) == 1:
        print("\n✓ All frameworks have same trade count!")
    else:
        print("\n✗ Trade counts differ")

    # Check final values
    print(f"\nFinal values:")
    for name in all_frameworks:
        print(f"  {name:<15}: ${results[name].final_value:,.2f}")

    # Calculate variance
    final_values = [results[fw].final_value for fw in all_frameworks]
    avg_final = sum(final_values) / len(final_values)

    print(f"\nVariance from average:")
    variances = {}
    for name in all_frameworks:
        var = abs(results[name].final_value - avg_final) / avg_final * 100
        variances[name] = var
        print(f"  {name:<15}: {var:.4f}%")

    max_var = max(variances.values())
    if max_var < 0.01:
        print("\n✓ Excellent alignment (< 0.01% variance)")
    elif max_var < 0.1:
        print("\n✓ Good alignment (< 0.1% variance)")
    elif max_var < 1.0:
        print("\n⚠️ Moderate variance (< 1.0%)")
    else:
        print("\n✗ High variance (> 1.0%) - further investigation needed")


if __name__ == "__main__":
    main()
