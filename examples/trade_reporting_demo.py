"""Demo of trade reporting functionality in ml4t.backtest.

Shows how to access trades DataFrame after a backtest and compare with VectorBT format.
"""

import pandas as pd
from datetime import datetime, timezone

from ml4t.backtest.core.assets import AssetRegistry, AssetSpec, AssetClass
from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import AssetId, OrderSide, OrderType, MarketDataType
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.execution.order import Order


def demo_trade_reporting():
    """Demonstrate trade reporting with clean output format."""

    print("=" * 80)
    print("ml4t.backtest Trade Reporting Demo")
    print("=" * 80)
    print()

    # Setup
    registry = AssetRegistry()
    registry.register(
        AssetSpec(
            asset_id="BTC-USD",
            asset_class=AssetClass.CRYPTO,
            tick_size=0.01,
            lot_size=0.001,
        )
    )

    broker = SimulationBroker(
        initial_cash=100000.0,
        asset_registry=registry,
        execution_delay=False,  # Immediate execution for demo
    )

    # Simulate market data
    prices = [50000, 51000, 52000, 51500, 53000, 52500, 54000, 55000, 54500, 56000]
    timestamps = [
        datetime(2024, 1, 1, 10 + i, 0, tzinfo=timezone.utc) for i in range(len(prices))
    ]

    print("Running backtest with 5 round-trip trades...")
    print()

    # Execute 5 trades
    for i in range(5):
        # Entry
        entry_idx = i * 2
        entry_event = MarketEvent(
            timestamp=timestamps[entry_idx],
            asset_id="BTC-USD",
            data_type=MarketDataType.BAR,
            open=prices[entry_idx],
            high=prices[entry_idx] * 1.001,
            low=prices[entry_idx] * 0.999,
            close=prices[entry_idx],
        )
        broker.on_market_event(entry_event)

        # Buy order
        entry_order = Order(
            asset_id="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        broker.submit_order(entry_order, timestamps[entry_idx])

        # Process fill
        broker.on_market_event(entry_event)

        # Exit
        exit_idx = entry_idx + 1
        exit_event = MarketEvent(
            timestamp=timestamps[exit_idx],
            asset_id="BTC-USD",
            data_type=MarketDataType.BAR,
            open=prices[exit_idx],
            high=prices[exit_idx] * 1.001,
            low=prices[exit_idx] * 0.999,
            close=prices[exit_idx],
        )
        broker.on_market_event(exit_event)

        # Sell order
        exit_order = Order(
            asset_id="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        broker.submit_order(exit_order, timestamps[exit_idx])

        # Process fill
        broker.on_market_event(exit_event)

    print("✓ Backtest complete")
    print()

    # Get trades DataFrame (clean column names!)
    trades_df = broker.trades

    print(f"Total trades: {len(trades_df)}")
    print()

    # Display trades (convert to pandas for prettier printing)
    trades_pd = trades_df.to_pandas()

    # Show key columns
    key_cols = [
        "trade_id",
        "entry_dt",
        "entry_price",
        "exit_dt",
        "exit_price",
        "pnl",
        "return_pct",
        "duration_bars",
    ]

    print("Trades Summary:")
    print("-" * 80)
    print(trades_pd[key_cols].to_string(index=False))
    print()

    # Show statistics
    total_pnl = trades_df["pnl"].sum()
    avg_pnl = trades_df["pnl"].mean()
    win_rate = (trades_df["pnl"] > 0).sum() / len(trades_df) * 100
    avg_duration = trades_df["duration_bars"].mean()

    print("Performance Metrics:")
    print("-" * 80)
    print(f"Total P&L:        ${total_pnl:,.2f}")
    print(f"Average P&L:      ${avg_pnl:,.2f}")
    print(f"Win Rate:         {win_rate:.1f}%")
    print(f"Avg Duration:     {avg_duration:.1f} bars")
    print()

    # Show comparison with VectorBT format
    print("Column Name Comparison (ml4t.backtest vs VectorBT):")
    print("-" * 80)
    comparison = [
        ("trade_id", "Exit Trade Id"),
        ("asset_id", "Column"),
        ("entry_dt", "Entry Index"),
        ("entry_price", "Avg Entry Price"),
        ("entry_commission", "Entry Fees"),
        ("exit_dt", "Exit Index"),
        ("exit_price", "Avg Exit Price"),
        ("exit_commission", "Exit Fees"),
        ("pnl", "PnL"),
        ("return_pct", "Return"),
        ("direction", "Direction"),
    ]

    for ml4t.backtest_col, vbt_col in comparison:
        print(f"  {ml4t.backtest_col:20s} → {vbt_col}")

    print()
    print("✓ Clean snake_case column names (no capitals or spaces!)")
    print("✓ Polars DataFrame for maximum performance")
    print("✓ Easy conversion to pandas: broker.trades.to_pandas()")
    print()

    # Show DataFrame info
    print("DataFrame Info:")
    print("-" * 80)
    print(f"Shape: {trades_df.shape}")
    print(f"Columns: {len(trades_df.columns)}")
    print(f"Memory efficient: Polars uses Arrow format")
    print()

    return trades_df


def benchmark_performance():
    """Benchmark trade tracking performance."""
    import time

    print("=" * 80)
    print("Performance Benchmark")
    print("=" * 80)
    print()

    registry = AssetRegistry()
    registry.register(
        AssetSpec(
            asset_id="BTC-USD",
            asset_class=AssetClass.CRYPTO,
        )
    )

    broker = SimulationBroker(initial_cash=100000.0, asset_registry=registry)

    # Simulate 1000 trades
    n_trades = 1000
    print(f"Processing {n_trades} round-trip trades...")

    start = time.time()

    for i in range(n_trades):
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

        # Entry
        event = MarketEvent(
            timestamp=timestamp,
            asset_id="BTC-USD",
            data_type=MarketDataType.BAR,
            close=50000.0 + i,
        )
        broker.on_market_event(event)

        order = Order(
            asset_id="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        broker.submit_order(order, timestamp)
        broker.on_market_event(event)

        # Exit
        broker.on_market_event(event)
        order = Order(
            asset_id="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        broker.submit_order(order, timestamp)
        broker.on_market_event(event)

    elapsed = time.time() - start

    # Get DataFrame
    df_start = time.time()
    trades_df = broker.trades
    df_elapsed = time.time() - df_start

    print()
    print("Results:")
    print(f"  Trades processed: {len(trades_df)}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Trades/sec: {n_trades / elapsed:,.0f}")
    print(f"  DataFrame creation: {df_elapsed*1000:.2f}ms")
    print(f"  Overhead per trade: {(elapsed / n_trades) * 1000:.4f}ms")
    print()
    print("✓ Minimal overhead - suitable for high-frequency backtesting")
    print()


if __name__ == "__main__":
    # Run demo
    trades = demo_trade_reporting()

    # Run benchmark
    benchmark_performance()

    print("=" * 80)
    print("Demo Complete")
    print("=" * 80)
