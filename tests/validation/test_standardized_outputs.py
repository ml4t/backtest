"""
Test that all 3 frameworks return standardized outputs.

This validates the common output format required for apples-to-apples comparison:
- Trades (list of TradeRecord)
- Daily returns (pd.Series)
- Equity curve (pd.Series)
"""

import pickle
from pathlib import Path

import pandas as pd

from frameworks.backtrader_adapter import BacktraderAdapter
from frameworks.qengine_adapter import BacktestAdapter  # ml4t.backtest adapter
from frameworks.vectorbt_adapter import VectorBTAdapter
from frameworks.base import FrameworkConfig


def test_all_frameworks_return_standardized_outputs():
    """Verify all frameworks return daily_returns, equity_curve, and trades."""

    # Load BTC signal dataset
    signal_file = Path(__file__).parent / 'signals' / 'btc_sma_crossover_daily.pkl'
    with open(signal_file, 'rb') as f:
        signal_data = pickle.load(f)

    data = signal_data['data']
    signals = signal_data['signals']

    # Use realistic configuration (next-bar open fill)
    config = FrameworkConfig.realistic()

    # Run all 3 frameworks
    adapters = [
        BacktestAdapter(),  # ml4t.backtest
        BacktraderAdapter(),
        VectorBTAdapter(),
    ]

    results = {}
    for adapter in adapters:
        print(f"\n{'='*60}")
        print(f"Testing {adapter.framework_name}")
        print('='*60)
        result = adapter.run_with_signals(data, signals, config)
        results[adapter.framework_name] = result

        # Verify standardized outputs exist
        assert result.daily_returns is not None, f"{adapter.framework_name} must return daily_returns"
        assert isinstance(result.daily_returns, pd.Series), f"{adapter.framework_name} daily_returns must be pd.Series"

        assert result.equity_curve is not None, f"{adapter.framework_name} must return equity_curve"
        assert isinstance(result.equity_curve, pd.Series), f"{adapter.framework_name} equity_curve must be pd.Series"

        assert isinstance(result.trades, list), f"{adapter.framework_name} trades must be list"

        print(f"✓ Daily returns: {len(result.daily_returns)} periods")
        print(f"✓ Equity curve: {len(result.equity_curve)} periods")
        print(f"✓ Trades: {len(result.trades)} executed trades")
        print(f"✓ Final value: ${result.final_value:,.2f}")
        print(f"✓ Total return: {result.total_return:.2f}%")

    # Comparison summary
    print(f"\n{'='*60}")
    print("Framework Comparison Summary")
    print('='*60)
    print(f"{'Framework':<15} {'Final Value':<15} {'Return %':<12} {'Trades':<8} {'Data Points'}")
    print('-'*60)

    for name, result in results.items():
        print(f"{name:<15} ${result.final_value:>12,.2f} {result.total_return:>10.2f}% {result.num_trades:>6} {len(result.daily_returns):>11}")

    print("\n✓ All frameworks return standardized outputs!")
    print("  - daily_returns (pd.Series)")
    print("  - equity_curve (pd.Series)")
    print("  - trades (list[TradeRecord])")
    print("\n✓ Ready for comprehensive validation testing!")


if __name__ == "__main__":
    test_all_frameworks_return_standardized_outputs()
