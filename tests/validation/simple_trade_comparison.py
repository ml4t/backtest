"""
Simple trade comparison - just show first 10 trades from each framework.
"""

import pickle
from pathlib import Path

from frameworks.backtrader_adapter import BacktraderAdapter
from frameworks.qengine_adapter import BacktestAdapter
from frameworks.vectorbt_adapter import VectorBTAdapter
from frameworks.base import FrameworkConfig


def main():
    # Load fresh signals
    signal_file = Path(__file__).parent / "signals" / "sp500_top10_sma_crossover.pkl"
    print(f"Loading signals from: {signal_file}")
    print(f"File modified: {signal_file.stat().st_mtime}")
    print()

    with open(signal_file, 'rb') as f:
        signal_set = pickle.load(f)

    # Get AAPL data
    asset_data = signal_set['assets']['AAPL']
    data = asset_data['data']
    signals = asset_data['signals']

    print(f"Data verification:")
    print(f"  Total bars: {len(data)}")
    print(f"  Duplicate index values: {data.index.duplicated().sum()}")
    print(f"  data.index[0:5]: {list(data.index[0:5])}")
    print(f"  data.index[66]: {data.index[66]}")
    print()

    print(f"Signal verification:")
    print(f"  Total bars: {len(signals)}")
    print(f"  Duplicate index values: {signals.index.duplicated().sum()}")
    print(f"  Total TRUE entries: {signals['entry'].sum()}")
    print(f"  Total TRUE exits: {signals['exit'].sum()}")

    # Find first TRUE signal
    for i in range(len(signals)):
        if signals['entry'].iloc[i] or signals['exit'].iloc[i]:
            sig_type = "ENTRY" if signals['entry'].iloc[i] else "EXIT"
            print(f"  First signal: iloc[{i}] = {signals.index[i].strftime('%Y-%m-%d')} ({sig_type})")
            break
    print()

    # Run frameworks with aligned timing configuration
    config = FrameworkConfig.for_matching()
    print("Configuration:")
    print(f"  fill_timing: {config.fill_timing}")
    print(f"  backtrader_coc: {config.backtrader_coc}")
    print(f"  vectorbt_accumulate: {config.vectorbt_accumulate}")
    print(f"  commission: {config.commission_pct*100}%")
    print(f"  slippage: {config.slippage_pct*100}%")
    print()

    frameworks = {
        'ml4t.backtest': BacktestAdapter(),
        'Backtrader': BacktraderAdapter(),
        'VectorBT': VectorBTAdapter(),
    }

    results = {}
    for name, adapter in frameworks.items():
        print(f"Running {name}...")
        result = adapter.run_with_signals(data, signals, config)
        results[name] = result
        print(f"  Final: ${result.final_value:,.2f} ({result.total_return:.2f}%)")
        print(f"  Trades: {result.num_trades} ({len(result.trades)} extracted)")
        print()

    # Compare first 10 trades
    print("="*80)
    print("FIRST 10 TRADES COMPARISON")
    print("="*80)

    for i in range(10):
        print(f"\nTrade #{i+1}:")
        for name in ['ml4t.backtest', 'Backtrader', 'VectorBT']:
            trades = results[name].trades
            if i < len(trades):
                t = trades[i]
                print(f"  {name:<15}: {t.timestamp.strftime('%Y-%m-%d')} {t.action} {t.quantity:.2f} @ ${t.price:.2f}")
            else:
                print(f"  {name:<15}: [NO TRADE]")

    # Check if first trades match
    print("\n" + "="*80)
    print("MATCHING ANALYSIS")
    print("="*80)

    qengine_trades = results['ml4t.backtest'].trades
    backtrader_trades = results['Backtrader'].trades

    if len(qengine_trades) > 0 and len(backtrader_trades) > 0:
        qe_first = qengine_trades[0]
        bt_first = backtrader_trades[0]

        print(f"\nFirst trade dates:")
        print(f"  ml4t.backtest: {qe_first.timestamp.strftime('%Y-%m-%d')}")
        print(f"  Backtrader:    {bt_first.timestamp.strftime('%Y-%m-%d')}")

        if qe_first.timestamp == bt_first.timestamp:
            print("\n✓ First trades match by date!")
        else:
            days_diff = abs((qe_first.timestamp - bt_first.timestamp).days)
            print(f"\n✗ First trades differ by {days_diff} days")

            # Check if ml4t.backtest first trade is BEFORE first signal
            first_signal_date = None
            for i in range(len(signals)):
                if signals['entry'].iloc[i]:
                    first_signal_date = signals.index[i]
                    break

            if first_signal_date:
                if qe_first.timestamp < first_signal_date:
                    print(f"\n⚠️  ml4t.backtest trades BEFORE first signal!")
                    print(f"   First signal: {first_signal_date.strftime('%Y-%m-%d')}")
                    print(f"   First trade:  {qe_first.timestamp.strftime('%Y-%m-%d')}")
                    print(f"   This suggests a bug in BacktestWrapper signal handling")


if __name__ == "__main__":
    main()
