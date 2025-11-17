"""
Detailed trade-by-trade comparison across all 4 frameworks.

This script extracts and compares EVERY trade (all 67) to identify the exact
systematic differences causing 2.4% variance between framework groups.

Analysis includes:
- Fill prices (open vs close, which bar)
- Quantities (fractional share handling, rounding)
- Commission calculations
- Value accumulation
- Cumulative P&L differences
"""

import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from frameworks.backtrader_adapter import BacktraderAdapter
from frameworks.qengine_adapter import BacktestAdapter
from frameworks.vectorbt_adapter import VectorBTAdapter
from frameworks.zipline_adapter import ZiplineAdapter
from frameworks.base import FrameworkConfig


def compare_trades_detailed(results: dict, max_trades: int = None):
    """
    Compare trades across all frameworks with full details.

    Args:
        results: Dict of framework_name -> ValidationResult
        max_trades: If set, only compare first N trades (None = all trades)
    """
    frameworks = ['ml4t.backtest', 'Backtrader', 'VectorBT', 'Zipline']

    # Determine how many trades to compare
    min_trades = min(len(results[fw].trades) for fw in frameworks)
    num_trades = min_trades if max_trades is None else min(min_trades, max_trades)

    print("=" * 120)
    print(f"DETAILED TRADE-BY-TRADE COMPARISON (First {num_trades} trades)")
    print("=" * 120)
    print()

    # Track cumulative differences
    cumulative_diffs = {fw: 0.0 for fw in frameworks}

    for i in range(num_trades):
        print(f"\n{'=' * 120}")
        print(f"TRADE #{i+1}")
        print('=' * 120)

        # Collect this trade from each framework
        trade_data = {}
        for fw in frameworks:
            t = results[fw].trades[i]
            commission = t.commission if t.commission is not None else 0.0

            # Calculate actual value (quantity * price) consistently
            # Framework's t.value field has different meanings across frameworks!
            actual_value = t.quantity * t.price

            # Calculate net cash impact (sign matters!)
            # BUY: negative (cash out)
            # SELL: positive (cash in)
            sign = 1 if t.action == 'SELL' else -1
            net_cash = sign * actual_value - abs(commission)  # Commission always reduces cash

            trade_data[fw] = {
                'timestamp': t.timestamp,
                'action': t.action,
                'quantity': t.quantity,
                'price': t.price,
                'value': actual_value,  # Use calculated value, not framework's value
                'commission': commission,
                'net_cash_impact': net_cash  # Total cash change
            }

        # Print formatted comparison
        print(f"\n{'Framework':<15} {'Date':<12} {'Action':<6} {'Quantity':>12} {'Price':>10} {'Value':>12} {'Commission':>10} {'Net Cash':>12}")
        print('-' * 120)

        for fw in frameworks:
            td = trade_data[fw]
            print(
                f"{fw:<15} "
                f"{td['timestamp'].strftime('%Y-%m-%d'):<12} "
                f"{td['action']:<6} "
                f"{td['quantity']:>12.6f} "
                f"${td['price']:>9.4f} "
                f"${td['value']:>11.2f} "
                f"${td['commission']:>9.2f} "
                f"${td['net_cash_impact']:>11.2f}"
            )

        # Calculate differences from ml4t.backtest (baseline)
        print(f"\n{'Differences from ml4t.backtest:':<40}")
        baseline = trade_data['ml4t.backtest']

        for fw in ['Backtrader', 'VectorBT', 'Zipline']:
            td = trade_data[fw]

            # Date difference (handle timezone-aware/naive)
            try:
                baseline_ts = baseline['timestamp']
                fw_ts = td['timestamp']
                # Convert both to naive if one is aware
                if baseline_ts.tzinfo is not None:
                    baseline_ts = baseline_ts.replace(tzinfo=None)
                if fw_ts.tzinfo is not None:
                    fw_ts = fw_ts.replace(tzinfo=None)
                date_diff = (fw_ts - baseline_ts).days
                date_str = f"same" if date_diff == 0 else f"+{date_diff}d" if date_diff > 0 else f"{date_diff}d"
            except:
                date_str = "???"

            # Quantity difference
            qty_diff = td['quantity'] - baseline['quantity']
            qty_pct = (qty_diff / baseline['quantity'] * 100) if baseline['quantity'] != 0 else 0

            # Price difference
            price_diff = td['price'] - baseline['price']
            price_pct = (price_diff / baseline['price'] * 100) if baseline['price'] != 0 else 0

            # Value difference (this is what matters for P&L)
            value_diff = td['value'] - baseline['value']

            # Commission difference
            comm_diff = td['commission'] - baseline['commission']

            # Net cash impact difference (what actually affects final portfolio value)
            net_diff = td['net_cash_impact'] - baseline['net_cash_impact']
            cumulative_diffs[fw] += net_diff

            print(f"  {fw:<15}: Date={date_str:>6} | Qty={qty_diff:+.6f} ({qty_pct:+.3f}%) | "
                  f"Price=${price_diff:+.4f} ({price_pct:+.4f}%) | Value=${value_diff:+.2f} | "
                  f"Comm=${comm_diff:+.2f} | Net=${net_diff:+.2f} | Cumulative=${cumulative_diffs[fw]:+.2f}")

    # Summary of cumulative differences
    print(f"\n\n{'=' * 120}")
    print("CUMULATIVE NET CASH IMPACT DIFFERENCES (vs ml4t.backtest)")
    print('=' * 120)
    print(f"\n{'Framework':<15} {'Cumulative $':>15} {'As % of Initial':>15}")
    print('-' * 50)

    initial_capital = results['ml4t.backtest'].initial_capital
    for fw in ['Backtrader', 'VectorBT', 'Zipline']:
        cum_diff = cumulative_diffs[fw]
        pct_of_initial = (cum_diff / initial_capital * 100)
        print(f"{fw:<15} ${cum_diff:>14.2f} {pct_of_initial:>14.4f}%")

    print()


def analyze_price_selection(results: dict, data: pd.DataFrame):
    """
    Analyze which prices frameworks are using (open vs close, which bar).

    Args:
        results: Dict of framework_name -> ValidationResult
        data: OHLCV DataFrame with price data
    """
    frameworks = ['ml4t.backtest', 'Backtrader', 'VectorBT', 'Zipline']

    print("\n" + "=" * 120)
    print("PRICE SELECTION ANALYSIS (First 10 BUY trades)")
    print("=" * 120)
    print()

    # Get first 10 BUY trades from each framework
    buy_count = 0
    for i in range(100):  # Look through first 100 trades to find 10 buys
        if buy_count >= 10:
            break

        # Check if this is a BUY in all frameworks
        is_buy = all(
            i < len(results[fw].trades) and results[fw].trades[i].action == 'BUY'
            for fw in frameworks
        )

        if not is_buy:
            continue

        buy_count += 1
        print(f"\nBUY Trade #{buy_count} (overall trade #{i+1}):")
        print(f"{'Framework':<15} {'Date':<12} {'Fill Price':>12} {'Trade Day Open':>15} {'Trade Day Close':>16} {'Prev Close':>12} {'Next Open':>12}")
        print('-' * 120)

        for fw in frameworks:
            trade = results[fw].trades[i]
            trade_date = trade.timestamp

            # Get prices from data
            try:
                # Find the row for this date
                if trade_date in data.index:
                    row = data.loc[trade_date]
                    day_open = row['Open']
                    day_close = row['Close']

                    # Get previous close
                    idx_pos = data.index.get_loc(trade_date)
                    prev_close = data.iloc[idx_pos - 1]['Close'] if idx_pos > 0 else np.nan

                    # Get next open
                    next_open = data.iloc[idx_pos + 1]['Open'] if idx_pos < len(data) - 1 else np.nan
                else:
                    day_open = day_close = prev_close = next_open = np.nan

                # Determine which price was used
                fill_price = trade.price
                if abs(fill_price - day_open) < 0.01:
                    price_type = "Trade Day OPEN ✓"
                elif abs(fill_price - day_close) < 0.01:
                    price_type = "Trade Day CLOSE ✓"
                elif abs(fill_price - prev_close) < 0.01:
                    price_type = "Prev CLOSE ✓"
                elif abs(fill_price - next_open) < 0.01:
                    price_type = "Next OPEN ✓"
                else:
                    price_type = "UNKNOWN"

                print(f"{fw:<15} {trade_date.strftime('%Y-%m-%d'):<12} "
                      f"${fill_price:>11.4f} "
                      f"${day_open:>14.4f} "
                      f"${day_close:>15.4f} "
                      f"${prev_close:>11.4f} "
                      f"${next_open:>11.4f} "
                      f"  {price_type}")

            except Exception as e:
                print(f"{fw:<15} ERROR: {e}")

    print()


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

    # Use for_zipline_matching() config
    config = FrameworkConfig.for_zipline_matching()
    print("Configuration: FrameworkConfig.for_zipline_matching()")
    print(f"  fill_timing: {config.fill_timing}")
    print(f"  commission: {config.commission_pct*100}% (${config.commission_fixed} fixed)")
    print(f"  slippage: {config.slippage_pct*100}% (${config.slippage_fixed} fixed)")
    print(f"  fractional_shares: {config.fractional_shares}")
    print()

    # Run all frameworks
    frameworks_list = {
        'ml4t.backtest': BacktestAdapter(),
        'Backtrader': BacktraderAdapter(),
        'VectorBT': VectorBTAdapter(),
        'Zipline': ZiplineAdapter(),
    }

    results = {}
    print("Running frameworks...")
    for name, adapter in frameworks_list.items():
        print(f"  {name}...", end='', flush=True)
        result = adapter.run_with_signals(data, signals, config)
        results[name] = result
        print(f" ✓ (${result.final_value:,.2f}, {len(result.trades)} trades)")

    print()

    # Detailed trade comparison (all 67 trades)
    compare_trades_detailed(results, max_trades=None)

    # Price selection analysis
    analyze_price_selection(results, data)

    # Final summary
    print("\n" + "=" * 120)
    print("FINAL PORTFOLIO VALUES")
    print("=" * 120)

    final_values = {name: result.final_value for name, result in results.items()}
    avg_value = np.mean(list(final_values.values()))

    print(f"\n{'Framework':<15} {'Final Value':>15} {'Variance from Avg':>18}")
    print('-' * 50)

    for name in ['ml4t.backtest', 'Backtrader', 'VectorBT', 'Zipline']:
        value = final_values[name]
        variance = (value - avg_value) / avg_value * 100
        print(f"{name:<15} ${value:>14,.2f} {variance:>17.4f}%")

    print(f"\nAverage: ${avg_value:,.2f}")
    max_variance = max(abs(v - avg_value) / avg_value * 100 for v in final_values.values())
    print(f"Maximum variance: {max_variance:.4f}%")
    print()


if __name__ == "__main__":
    main()
