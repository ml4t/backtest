"""Quick test of signal-based adapter implementation."""

import sys
from pathlib import Path

# Add necessary paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.validation.signals.generate import load_signal_set
from tests.validation.frameworks.qengine_adapter import BacktestAdapter
from tests.validation.frameworks.backtrader_adapter import BacktraderAdapter
from tests.validation.frameworks.vectorbt_adapter import VectorBTAdapter


def test_ml4t_backtest_adapter():
    """Test ml4t.backtest adapter with generated BTC signals."""
    print("=" * 80)
    print("Testing ml4t.backtest Signal Adapter")
    print("=" * 80)

    # Load signal set
    print("\n1. Loading BTC SMA crossover signals...")
    signal_set = load_signal_set("btc_sma_crossover_daily")

    data = signal_set["data"]
    signals = signal_set["signals"]
    metadata = signal_set["metadata"]

    print(f"   Data: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    print(f"   Signals: {signals['entry'].sum()} entries, {signals['exit'].sum()} exits")
    print(f"   Metadata: {metadata}")

    # Test adapter
    print("\n2. Running backtest with ml4t.backtest adapter...")
    adapter = BacktestAdapter()

    result = adapter.run_with_signals(
        data=data,
        signals=signals,
        initial_capital=10000.0,
        commission_rate=0.001,
    )

    # Display results
    print("\n3. Results:")
    print(f"   Framework: {result.framework}")
    print(f"   Initial Capital: ${result.initial_capital:,.2f}")
    print(f"   Final Value: ${result.final_value:,.2f}")
    print(f"   Total Return: {result.total_return:.2f}%")
    print(f"   Number of Trades: {result.num_trades}")
    print(f"   Win Rate: {result.win_rate:.2%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"   Execution Time: {result.execution_time:.3f}s")
    print(f"   Memory Usage: {result.memory_usage:.1f} MB")

    if result.has_errors:
        print(f"\n   ⚠️ ERRORS: {result.errors}")
        return False

    print("\n✓ Test PASSED - Adapter working correctly")
    return True


def test_backtrader_adapter():
    """Test Backtrader adapter with generated BTC signals."""
    print("\n" + "=" * 80)
    print("Testing Backtrader Signal Adapter")
    print("=" * 80)

    # Load signal set
    print("\n1. Loading BTC SMA crossover signals...")
    signal_set = load_signal_set("btc_sma_crossover_daily")

    data = signal_set["data"]
    signals = signal_set["signals"]

    print(f"   Data: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    print(f"   Signals: {signals['entry'].sum()} entries, {signals['exit'].sum()} exits")

    # Test adapter
    print("\n2. Running backtest with Backtrader adapter...")
    adapter = BacktraderAdapter()

    result = adapter.run_with_signals(
        data=data,
        signals=signals,
        initial_capital=10000.0,
        commission_rate=0.001,
    )

    # Display results
    print("\n3. Results:")
    print(f"   Framework: {result.framework}")
    print(f"   Final Value: ${result.final_value:,.2f}")
    print(f"   Total Return: {result.total_return:.2f}%")
    print(f"   Number of Trades: {result.num_trades}")

    if result.has_errors:
        print(f"\n   ⚠️ ERRORS: {result.errors}")
        return False

    print("\n✓ Test PASSED - Adapter working correctly")
    return True


def test_vectorbt_adapter():
    """Test VectorBT adapter with generated BTC signals."""
    print("\n" + "=" * 80)
    print("Testing VectorBT Signal Adapter")
    print("=" * 80)

    # Load signal set
    print("\n1. Loading BTC SMA crossover signals...")
    signal_set = load_signal_set("btc_sma_crossover_daily")

    data = signal_set["data"]
    signals = signal_set["signals"]

    print(f"   Data: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    print(f"   Signals: {signals['entry'].sum()} entries, {signals['exit'].sum()} exits")

    # Test adapter
    print("\n2. Running backtest with VectorBT adapter...")
    adapter = VectorBTAdapter()

    result = adapter.run_with_signals(
        data=data,
        signals=signals,
        initial_capital=10000.0,
        commission_rate=0.001,
    )

    # Display results
    print("\n3. Results:")
    print(f"   Framework: {result.framework}")
    print(f"   Final Value: ${result.final_value:,.2f}")
    print(f"   Total Return: {result.total_return:.2f}%")
    print(f"   Number of Trades: {result.num_trades}")

    if result.has_errors:
        print(f"\n   ⚠️ ERRORS: {result.errors}")
        return False

    print("\n✓ Test PASSED - Adapter working correctly")
    return True


if __name__ == "__main__":
    success1 = test_ml4t_backtest_adapter()
    success2 = test_backtrader_adapter()
    success3 = test_vectorbt_adapter()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"ml4t.backtest: {'✓ PASSED' if success1 else '✗ FAILED'}")
    print(f"Backtrader:    {'✓ PASSED' if success2 else '✗ FAILED'}")
    print(f"VectorBT:      {'✓ PASSED' if success3 else '✗ FAILED'}")
    print("=" * 80)

    sys.exit(0 if (success1 and success2 and success3) else 1)
