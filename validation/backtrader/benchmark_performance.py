#!/usr/bin/env python3
"""Performance benchmark: ml4t.backtest vs Backtrader.

This script benchmarks runtime performance across different data sizes
and number of assets.

Run from .venv-backtrader environment:
    .venv-backtrader/bin/python3 validation/backtrader/benchmark_performance.py

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


def benchmark_backtrader(asset_data: dict, entries: np.ndarray, exits: np.ndarray) -> dict:
    """Benchmark Backtrader."""
    import backtrader as bt

    class PandasData(bt.feeds.PandasData):
        params = (
            ("datetime", None),
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("volume", "volume"),
            ("openinterest", -1),
        )

    class BenchmarkStrategy(bt.Strategy):
        params = (("entries", None), ("exits", None))

        def __init__(self):
            self.bar_count = 0
            self.trade_count = 0

        def next(self):
            idx = self.bar_count
            for i, data in enumerate(self.datas):
                if not self.getposition(data):
                    if idx < len(self.params.entries) and self.params.entries[idx]:
                        self.buy(data=data, size=100)
                else:
                    if idx < len(self.params.exits) and self.params.exits[idx]:
                        self.close(data=data)
            self.bar_count += 1

        def notify_trade(self, trade):
            if trade.isclosed:
                self.trade_count += 1

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0)

    for name, df in asset_data.items():
        data = PandasData(dataname=df)
        cerebro.adddata(data, name=name)

    cerebro.addstrategy(BenchmarkStrategy, entries=entries, exits=exits)

    results = cerebro.run()
    strategy = results[0]
    num_trades = strategy.trade_count

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "framework": "Backtrader",
        "runtime_sec": end_time - start_time,
        "num_trades": num_trades,
        "memory_mb": peak / 1024 / 1024,
    }


def benchmark_ml4t_backtest(
    asset_data: dict, entries: np.ndarray, exits: np.ndarray, dates
) -> dict:
    """Benchmark ml4t.backtest."""
    import polars as pl

    from ml4t.backtest._validation_imports import DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy

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
        execution_mode=ExecutionMode.NEXT_BAR,  # Match Backtrader
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

    print("  Running Backtrader...")
    bt_results = benchmark_backtrader(asset_data, entries, exits)

    print("  Running ml4t.backtest...")
    ml4t_results = benchmark_ml4t_backtest(asset_data, entries, exits, dates)

    # Calculate speedup (>1 means Backtrader is faster)
    speedup = (
        bt_results["runtime_sec"] / ml4t_results["runtime_sec"]
        if ml4t_results["runtime_sec"] > 0
        else 0
    )

    return {
        "n_bars": n_bars,
        "n_assets": n_assets,
        "bt_runtime": bt_results["runtime_sec"],
        "bt_trades": bt_results["num_trades"],
        "bt_memory": bt_results["memory_mb"],
        "ml4t_runtime": ml4t_results["runtime_sec"],
        "ml4t_trades": ml4t_results["num_trades"],
        "ml4t_memory": ml4t_results["memory_mb"],
        "speedup": speedup,
    }


def main():
    print("=" * 70)
    print("Performance Benchmark: ml4t.backtest vs Backtrader")
    print("=" * 70)

    # Test configurations (smaller for Backtrader - it's slower)
    configs = [
        (100, 1),
        (1000, 1),
        (10000, 1),
        (1000, 10),
        (1000, 50),
    ]

    results = []

    for n_bars, n_assets in configs:
        print(f"\nBenchmark: {n_bars} bars x {n_assets} assets")
        try:
            result = run_benchmark(n_bars, n_assets)
            results.append(result)

            print(
                f"  BT: {result['bt_runtime']:.3f}s, {result['bt_trades']} trades, {result['bt_memory']:.1f} MB"
            )
            print(
                f"  ML4T: {result['ml4t_runtime']:.3f}s, {result['ml4t_trades']} trades, {result['ml4t_memory']:.1f} MB"
            )
            print(
                f"  Speedup: {result['speedup']:.2f}x {'(BT faster)' if result['speedup'] > 1 else '(ML4T faster)'}"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'BT (s)':<12} {'ML4T (s)':<12} {'Speedup':<12} {'Winner'}")
    print("-" * 70)

    for r in results:
        config = f"{r['n_bars']}x{r['n_assets']}"
        winner = "BT" if r["speedup"] > 1 else "ML4T"
        speedup_str = f"{r['speedup']:.2f}x" if r["speedup"] > 1 else f"{1/r['speedup']:.2f}x"
        print(
            f"{config:<20} {r['bt_runtime']:<12.3f} {r['ml4t_runtime']:<12.3f} {speedup_str:<12} {winner}"
        )

    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
