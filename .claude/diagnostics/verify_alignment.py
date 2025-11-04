#!/usr/bin/env python3
"""
Verify Zipline and VectorBT now produce aligned results after fix.
"""

from tests.validation.data_loader import UniversalDataLoader
from tests.validation.frameworks.zipline_adapter import ZiplineAdapter
from tests.validation.frameworks.vectorbtpro_adapter import VectorBTProAdapter

def main():
    print("=" * 80)
    print("CROSS-FRAMEWORK VALIDATION: Zipline vs VectorBT Pro")
    print("=" * 80)

    # Load data
    loader = UniversalDataLoader()

    # Test parameters
    strategy_params = {
        "name": "MovingAverageCrossover",
        "short_window": 10,
        "slow_window": 30,  # Using slow_window for compatibility
    }
    initial_capital = 10000.0

    print("\nStrategy: MA(10/30) Crossover")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Test Period: 2017-01-03 to 2017-12-29")

    # Run VectorBT
    print("\n" + "-" * 80)
    print("Running VectorBT Pro...")
    print("-" * 80)
    vbt_adapter = VectorBTProAdapter()
    df = loader.load_daily_equities(tickers=['AAPL'], start_date='2017-01-03', end_date='2017-12-29', source='wiki')

    # VectorBT expects DataFrame with timestamp index
    vbt_data = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    vbt_data.set_index('timestamp', inplace=True)

    vbt_result = vbt_adapter.run_backtest(vbt_data, strategy_params, initial_capital)

    print(f"\nVectorBT Results:")
    print(f"  Final Value: ${vbt_result.final_value:,.2f}")
    print(f"  Total Return: {vbt_result.total_return:.2f}%")
    print(f"  Num Trades: {vbt_result.num_trades}")
    print(f"  Sharpe Ratio: {vbt_result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {vbt_result.max_drawdown:.2f}%")

    # Run Zipline
    print("\n" + "-" * 80)
    print("Running Zipline-Reloaded...")
    print("-" * 80)
    zl_adapter = ZiplineAdapter()
    # Zipline needs data to determine date range (but loads OHLCV from bundle)
    zl_result = zl_adapter.run_backtest(vbt_data, strategy_params, initial_capital)

    print(f"\nZipline Results:")
    print(f"  Final Value: ${zl_result.final_value:,.2f}")
    print(f"  Total Return: {zl_result.total_return:.2f}%")
    print(f"  Num Trades: {zl_result.num_trades}")
    print(f"  Sharpe Ratio: {zl_result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {zl_result.max_drawdown:.2f}%")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    return_diff = abs(vbt_result.total_return - zl_result.total_return)
    trade_diff = abs(vbt_result.num_trades - zl_result.num_trades)
    sharpe_diff = abs(vbt_result.sharpe_ratio - zl_result.sharpe_ratio)
    dd_diff = abs(vbt_result.max_drawdown - zl_result.max_drawdown)

    print(f"\nReturn Difference: {return_diff:.2f}%")
    print(f"Trade Count Difference: {trade_diff}")
    print(f"Sharpe Difference: {sharpe_diff:.2f}")
    print(f"Max DD Difference: {dd_diff:.2f}%")

    # Acceptance criteria
    print("\n" + "=" * 80)
    print("VALIDATION STATUS")
    print("=" * 80)

    checks = []
    checks.append(("Return variance < 0.5%", return_diff < 0.5, return_diff))
    checks.append(("Trade count matches", trade_diff == 0, trade_diff))
    checks.append(("Sharpe variance < 10%", sharpe_diff / max(vbt_result.sharpe_ratio, 0.01) < 0.10, sharpe_diff))
    checks.append(("Max DD matches closely", dd_diff < 1.0, dd_diff))

    all_passed = True
    for check_name, passed, value in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name} (actual: {value:.4f})")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 VALIDATION SUCCESSFUL - Frameworks are aligned!")
    else:
        print("\n⚠️  VALIDATION FAILED - Further investigation needed")

    return all_passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
