"""
Simple baseline evaluation runner using real data from projects/

This script runs a simplified evaluation of strategies using real market data
to verify ml4t.backtest's correctness and performance.
"""

import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project paths
ml4t.backtest_src = Path(__file__).parent.parent.parent / "src"
projects_dir = Path(__file__).parent.parent.parent.parent / "projects"
sys.path.insert(0, str(ml4t.backtest_src))

from unittest.mock import Mock

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import MarketDataType
from ml4t.backtest.strategy.adapters import PITData
from ml4t.backtest.strategy.crypto_basis_adapter import (
    CryptoBasisExternalStrategy,
)
from ml4t.backtest.strategy.spy_order_flow_adapter import (
    SPYOrderFlowExternalStrategy,
    create_spy_order_flow_strategy,
)


def load_spy_data():
    """Load SPY order flow data."""
    spy_path = projects_dir / "spy_order_flow" / "spy_features.parquet"
    if not spy_path.exists():
        print(f"ERROR: SPY data not found at {spy_path}")
        return None

    df = pd.read_parquet(spy_path)
    print(f"Loaded SPY data: {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Columns available: {', '.join(df.columns[:15])}...")
    return df


def load_crypto_data():
    """Load crypto spot and futures data."""
    spot_path = projects_dir / "crypto_futures" / "data" / "spot" / "BTC.parquet"
    futures_path = projects_dir / "crypto_futures" / "data" / "futures" / "BTC.parquet"

    if not spot_path.exists() or not futures_path.exists():
        print("ERROR: Crypto data not found")
        return None, None

    spot_df = pd.read_parquet(spot_path)
    futures_df = pd.read_parquet(futures_path)

    print(f"Loaded BTC spot: {len(spot_df)} rows")
    print(f"Loaded BTC futures: {len(futures_df)} rows")

    return spot_df, futures_df


def run_spy_strategy_test(data: pd.DataFrame, max_rows: int = 1000):
    """Test SPY order flow strategy with real data."""
    print("\n" + "=" * 60)
    print("SPY ORDER FLOW STRATEGY TEST")
    print("=" * 60)

    # Limit data for testing
    test_data = data.head(max_rows)
    print(f"Using {len(test_data)} data points for test")

    # Create standalone strategy
    strategy = SPYOrderFlowExternalStrategy(
        asset_id="SPY",
        lookback_window=50,
        momentum_window_short=5,
        momentum_window_long=20,
        imbalance_threshold=0.65,
        momentum_threshold=0.002,
        min_data_points=20,
        signal_cooldown=5,
    )
    strategy.initialize()

    # Track performance
    tracemalloc.start()
    start_time = time.time()

    signals = []
    for _idx, row in test_data.iterrows():
        # Create PITData
        pit_data = PITData(
            timestamp=pd.to_datetime(row["timestamp"]),
            asset_data={
                "SPY": {
                    "volume": row.get("volume", 1000000),
                    "buy_volume": row.get("buy_volume", row.get("volume", 1000000) * 0.5),
                    "sell_volume": row.get("sell_volume", row.get("volume", 1000000) * 0.5),
                },
            },
            market_prices={"SPY": row.get("close", 450.0)},
        )

        # Generate signal
        signal = strategy.generate_signal(pd.to_datetime(row["timestamp"]), pit_data)

        if signal:
            signals.append(
                {
                    "timestamp": signal.timestamp,
                    "position": signal.position,
                    "confidence": signal.confidence,
                    "metadata": signal.metadata,
                },
            )

    # Performance metrics
    execution_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak / 1024 / 1024

    # Results
    print("\nResults:")
    print(f"  Signals generated: {len(signals)}")
    print(f"  Execution time: {execution_time:.3f}s")
    print(f"  Memory usage: {memory_mb:.2f}MB")
    print(f"  Data points/sec: {len(test_data) / execution_time:.0f}")

    if signals:
        print("\nFirst signal:")
        first = signals[0]
        print(f"  Timestamp: {first['timestamp']}")
        print(f"  Position: {first['position']:.3f}")
        print(f"  Confidence: {first['confidence']:.3f}")
        print(f"  Factors: {first['metadata'].get('factors', [])}")

    strategy.finalize()

    return {
        "signals": len(signals),
        "execution_time": execution_time,
        "memory_mb": memory_mb,
        "data_points": len(test_data),
    }


def run_crypto_basis_test(spot_df: pd.DataFrame, futures_df: pd.DataFrame, max_rows: int = 500):
    """Test crypto basis strategy with real data."""
    print("\n" + "=" * 60)
    print("CRYPTO BASIS STRATEGY TEST")
    print("=" * 60)

    # Align and limit data
    spot_test = spot_df.head(max_rows)
    futures_test = futures_df.head(max_rows)

    print(f"Using {len(spot_test)} spot and {len(futures_test)} futures data points")

    # Create strategy
    strategy = CryptoBasisExternalStrategy(
        spot_asset_id="BTC",
        futures_asset_id="BTC_FUTURE",
        lookback_window=50,
        entry_threshold=2.0,
        exit_threshold=0.5,
        min_data_points=20,
    )
    strategy.initialize()

    # Track performance
    tracemalloc.start()
    start_time = time.time()

    signals = []

    # Process synchronized data
    for i in range(min(len(spot_test), len(futures_test))):
        spot_row = spot_test.iloc[i]
        futures_row = futures_test.iloc[i]

        # Use spot timestamp (or whichever is available)
        timestamp = pd.to_datetime(spot_row.get("timestamp", spot_row.get("date")))

        # Create PITData with both prices
        pit_data = PITData(
            timestamp=timestamp,
            asset_data={},
            market_prices={
                "BTC": spot_row.get("close", 50000),
                "BTC_FUTURE": futures_row.get("close", 50100),
            },
        )

        # Generate signal
        signal = strategy.generate_signal(timestamp, pit_data)

        if signal:
            signals.append(
                {
                    "timestamp": signal.timestamp,
                    "position": signal.position,
                    "confidence": signal.confidence,
                    "basis": signal.metadata.get("basis", 0),
                    "z_score": signal.metadata.get("z_score", 0),
                },
            )

    # Performance metrics
    execution_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak / 1024 / 1024

    # Calculate basis statistics
    basis_values = []
    for i in range(min(len(spot_test), len(futures_test))):
        spot_price = spot_test.iloc[i].get("close", 0)
        futures_price = futures_test.iloc[i].get("close", 0)
        if spot_price > 0 and futures_price > 0:
            basis_values.append(futures_price - spot_price)

    # Results
    print("\nResults:")
    print(f"  Signals generated: {len(signals)}")
    print(f"  Execution time: {execution_time:.3f}s")
    print(f"  Memory usage: {memory_mb:.2f}MB")
    print(f"  Data points/sec: {len(spot_test) / execution_time:.0f}")

    if basis_values:
        print("\nBasis Statistics:")
        print(f"  Mean basis: ${np.mean(basis_values):.2f}")
        print(f"  Std basis: ${np.std(basis_values):.2f}")
        print(f"  Min basis: ${min(basis_values):.2f}")
        print(f"  Max basis: ${max(basis_values):.2f}")

    if signals:
        print("\nFirst signal:")
        first = signals[0]
        print(f"  Timestamp: {first['timestamp']}")
        print(f"  Position: {first['position']:.3f}")
        print(f"  Basis: ${first['basis']:.2f}")
        print(f"  Z-score: {first['z_score']:.3f}")

    strategy.finalize()

    return {
        "signals": len(signals),
        "execution_time": execution_time,
        "memory_mb": memory_mb,
        "data_points": min(len(spot_test), len(futures_test)),
    }


def run_backtest_adapter_test(data: pd.DataFrame, max_rows: int = 500):
    """Test ml4t.backtest adapter integration with real data."""
    print("\n" + "=" * 60)
    print("ML4T.BACKTEST ADAPTER INTEGRATION TEST")
    print("=" * 60)

    test_data = data.head(max_rows)
    print(f"Using {len(test_data)} data points")

    # Create adapter
    adapter = create_spy_order_flow_strategy(
        asset_id="SPY",
        lookback_window=50,
        position_scaling=0.2,
    )

    # Mock broker
    adapter.broker = Mock()
    adapter.broker.submit_order = Mock(return_value="order_123")
    adapter.broker.get_cash = Mock(return_value=100000)

    adapter.on_start()

    # Track performance
    tracemalloc.start()
    start_time = time.time()

    order_count = 0

    for _idx, row in test_data.iterrows():
        # Create market event
        event = MarketEvent(
            timestamp=pd.to_datetime(row["timestamp"]),
            asset_id="SPY",
            data_type=MarketDataType.BAR,
            open=row.get("open", row.get("close", 450)),
            high=row.get("high", row.get("close", 450)),
            low=row.get("low", row.get("close", 450)),
            close=row.get("close", 450),
            volume=row.get("volume", 1000000),
            metadata={
                "buy_volume": row.get("buy_volume", row.get("volume", 1000000) * 0.5),
                "sell_volume": row.get("sell_volume", row.get("volume", 1000000) * 0.5),
            },
        )

        # Process event
        adapter.on_market_event(event)

        # Check if order was submitted
        if adapter.broker.submit_order.called:
            order_count = adapter.broker.submit_order.call_count

    # Performance metrics
    execution_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_mb = peak / 1024 / 1024

    # Get diagnostics
    diagnostics = adapter.get_strategy_diagnostics()

    # Results
    print("\nResults:")
    print(f"  Orders submitted: {order_count}")
    print(f"  Execution time: {execution_time:.3f}s")
    print(f"  Memory usage: {memory_mb:.2f}MB")
    print(f"  Data points/sec: {len(test_data) / execution_time:.0f}")

    if "order_flow_statistics" in diagnostics:
        stats = diagnostics["order_flow_statistics"]
        print("\nStrategy Statistics:")
        print(f"  Data points: {stats.get('data_points', 0)}")
        print(f"  Signal count: {stats.get('signal_count', 0)}")
        print(f"  Current position: {stats.get('current_position', 0):.3f}")

    adapter.on_end()

    return {
        "orders": order_count,
        "execution_time": execution_time,
        "memory_mb": memory_mb,
        "data_points": len(test_data),
    }


def main():
    """Run all baseline evaluations."""
    print("\n" + "#" * 60)
    print("# BASELINE EVALUATION WITH REAL DATA")
    print("#" * 60)
    print(f"Timestamp: {datetime.now()}")

    results = {}

    # Test 1: SPY Order Flow Strategy
    spy_data = load_spy_data()
    if spy_data is not None:
        results["spy_standalone"] = run_spy_strategy_test(spy_data, max_rows=1000)
        results["spy_adapter"] = run_ml4t.backtest_adapter_test(spy_data, max_rows=1000)

    # Test 2: Crypto Basis Strategy
    spot_data, futures_data = load_crypto_data()
    if spot_data is not None and futures_data is not None:
        results["crypto_basis"] = run_crypto_basis_test(spot_data, futures_data, max_rows=500)

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for test_name, test_results in results.items():
        print(f"\n{test_name}:")
        print(f"  Data points: {test_results['data_points']}")
        print(f"  Execution time: {test_results['execution_time']:.3f}s")
        print(f"  Memory: {test_results['memory_mb']:.2f}MB")
        print(
            f"  Throughput: {test_results['data_points'] / test_results['execution_time']:.0f} points/sec",
        )

    # Performance verdict
    print("\n" + "-" * 60)
    print("PERFORMANCE VERDICT:")

    total_time = sum(r["execution_time"] for r in results.values())
    total_memory = max(r["memory_mb"] for r in results.values())
    total_points = sum(r["data_points"] for r in results.values())

    print(f"  Total data processed: {total_points:,} points")
    print(f"  Total execution time: {total_time:.2f}s")
    print(f"  Peak memory usage: {total_memory:.2f}MB")
    print(f"  Average throughput: {total_points / total_time:.0f} points/sec")

    if total_time < 10 and total_memory < 500:
        print("\n✓ ml4t.backtest demonstrates excellent performance!")
        print("  - Fast execution (<10s for all tests)")
        print("  - Low memory footprint (<500MB)")
    elif total_time < 30:
        print("\n✓ ml4t.backtest shows good performance")
        print("  - Reasonable execution time")
        print("  - Consider optimization for better throughput")
    else:
        print("\n⚠ Performance optimization needed")
        print(f"  - Execution time: {total_time:.1f}s (target: <10s)")
        print(f"  - Memory usage: {total_memory:.1f}MB (target: <500MB)")

    print("\n" + "#" * 60)
    print("# EVALUATION COMPLETE")
    print("#" * 60)


if __name__ == "__main__":
    main()
