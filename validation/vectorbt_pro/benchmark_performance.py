#!/usr/bin/env python3
"""Performance benchmark: ml4t.backtest vs VectorBT Pro.

This script benchmarks runtime performance across different data sizes
and number of assets.

Run from .venv-vectorbt-pro environment:
    source .venv-vectorbt-pro/bin/activate
    python validation/vectorbt_pro/benchmark_performance.py

Metrics:
- Total runtime (seconds)
- Trades per second
- Memory usage (MB)
"""

import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def generate_benchmark_data(n_bars: int, n_assets: int, seed: int = 42) -> tuple:
    """Generate benchmark data for both frameworks."""
    np.random.seed(seed)

    dates = pd.date_range(start="2020-01-01", periods=n_bars, freq="D")

    # Generate price data for multiple assets
    asset_data = {}
    for i in range(n_assets):
        base_price = 100.0 + i * 10
        returns = np.random.randn(n_bars) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))

        asset_data[f"ASSET_{i:03d}"] = pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(n_bars) * 0.005),
                "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.01),
                "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.01),
                "close": prices,
                "volume": np.random.randint(100000, 1000000, n_bars).astype(float),
            },
            index=dates,
        )

    # Generate signals - entry every 20 bars, exit 10 bars later
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)

    i = 0
    while i < n_bars - 11:
        entries[i] = True
        exits[i + 10] = True
        i += 20

    return asset_data, entries, exits, dates


def benchmark_vectorbt_pro(asset_data: dict, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Benchmark VectorBT Pro."""
    import vectorbtpro as vbt

    # Prepare close prices DataFrame
    close_df = pd.DataFrame({name: df["close"] for name, df in asset_data.items()})

    # Warm-up run
    _ = vbt.Portfolio.from_signals(
        close=close_df.iloc[:100],
        entries=entries[:100],
        exits=exits[:100],
        init_cash=100_000.0,
        size=100,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
    )

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries,
        exits=exits,
        init_cash=100_000.0,
        size=100,
        size_type="amount",
        fees=0.0,
        slippage=0.0,
    )

    # Force computation
    _ = pf.total_return
    num_trades = len(pf.trades.records_readable)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "framework": "VectorBT Pro",
        "runtime_sec": end_time - start_time,
        "num_trades": num_trades,
        "memory_mb": peak / 1024 / 1024,
    }


def benchmark_ml4t_backtest(
    asset_data: dict, entries: np.ndarray, exits: np.ndarray, dates
) -> dict:
    """Benchmark ml4t.backtest."""
    import polars as pl

    from ml4t.backtest import DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy

    # Prepare data in polars format
    rows = []
    for asset_name, df in asset_data.items():
        for i, (ts, row) in enumerate(df.iterrows()):
            rows.append(
                {
                    "timestamp": ts.to_pydatetime(),
                    "asset": asset_name,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
            )

    prices_pl = pl.DataFrame(rows)

    # Prepare signals
    signal_rows = []
    for asset_name in asset_data:
        for i, ts in enumerate(dates):
            signal_rows.append(
                {
                    "timestamp": ts.to_pydatetime(),
                    "asset": asset_name,
                    "entry": bool(entries[i]),
                    "exit": bool(exits[i]),
                }
            )

    signals_pl = pl.DataFrame(signal_rows)

    class BenchmarkStrategy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            for asset_name, asset_data in data.items():
                signals = asset_data.get("signals", {})
                position = broker.get_position(asset_name)
                current_qty = position.quantity if position else 0

                if signals.get("exit") and current_qty > 0:
                    broker.close_position(asset_name)
                elif signals.get("entry") and current_qty == 0:
                    broker.submit_order(asset_name, 100)

    # Warm-up run with small data
    small_prices = prices_pl.head(100 * len(asset_data))
    small_signals = signals_pl.head(100 * len(asset_data))
    small_feed = DataFeed(prices_df=small_prices, signals_df=small_signals)
    small_engine = Engine(
        small_feed,
        BenchmarkStrategy(),
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
    )
    _ = small_engine.run()

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = BenchmarkStrategy()

    engine = Engine(
        feed,
        strategy,
        initial_cash=100_000.0,
        allow_short_selling=False,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=ExecutionMode.SAME_BAR,
    )

    results = engine.run()
    num_trades = results["num_trades"]

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "framework": "ml4t.backtest",
        "runtime_sec": end_time - start_time,
        "num_trades": num_trades,
        "memory_mb": peak / 1024 / 1024,
    }


def run_benchmark(n_bars: int, n_assets: int) -> dict:
    """Run benchmark for specific configuration."""
    print(f"\n  Generating data: {n_bars} bars x {n_assets} assets...")
    asset_data, entries, exits, dates = generate_benchmark_data(n_bars, n_assets)

    print("  Running VectorBT Pro...")
    vbt_results = benchmark_vectorbt_pro(asset_data, entries, exits)

    print("  Running ml4t.backtest...")
    ml4t_results = benchmark_ml4t_backtest(asset_data, entries, exits, dates)

    # Calculate speedup
    speedup = (
        vbt_results["runtime_sec"] / ml4t_results["runtime_sec"]
        if ml4t_results["runtime_sec"] > 0
        else 0
    )

    return {
        "n_bars": n_bars,
        "n_assets": n_assets,
        "vbt_runtime": vbt_results["runtime_sec"],
        "vbt_trades": vbt_results["num_trades"],
        "vbt_memory": vbt_results["memory_mb"],
        "ml4t_runtime": ml4t_results["runtime_sec"],
        "ml4t_trades": ml4t_results["num_trades"],
        "ml4t_memory": ml4t_results["memory_mb"],
        "speedup": speedup,
    }


def main():
    print("=" * 70)
    print("Performance Benchmark: ml4t.backtest vs VectorBT Pro")
    print("=" * 70)

    # Test configurations
    configs = [
        (100, 1),
        (1000, 1),
        (10000, 1),
        (1000, 10),
        (10000, 10),
        (1000, 100),
    ]

    results = []

    for n_bars, n_assets in configs:
        print(f"\nBenchmark: {n_bars} bars x {n_assets} assets")
        try:
            result = run_benchmark(n_bars, n_assets)
            results.append(result)

            print(
                f"  VBT: {result['vbt_runtime']:.3f}s, {result['vbt_trades']} trades, {result['vbt_memory']:.1f} MB"
            )
            print(
                f"  ML4T: {result['ml4t_runtime']:.3f}s, {result['ml4t_trades']} trades, {result['ml4t_memory']:.1f} MB"
            )
            print(
                f"  Speedup: {result['speedup']:.2f}x {'(VBT faster)' if result['speedup'] > 1 else '(ML4T faster)'}"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'VBT (s)':<12} {'ML4T (s)':<12} {'Speedup':<12} {'Winner'}")
    print("-" * 70)

    for r in results:
        config = f"{r['n_bars']}x{r['n_assets']}"
        winner = "VBT" if r["speedup"] > 1 else "ML4T"
        speedup_str = f"{r['speedup']:.2f}x" if r["speedup"] > 1 else f"{1/r['speedup']:.2f}x"
        print(
            f"{config:<20} {r['vbt_runtime']:<12.3f} {r['ml4t_runtime']:<12.3f} {speedup_str:<12} {winner}"
        )

    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
