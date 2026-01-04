#!/usr/bin/env python3
"""Performance benchmark: ml4t.backtest vs Zipline-Reloaded.

This script benchmarks runtime performance across different data sizes
and number of assets.

Run from .venv-zipline or .venv-validation environment:
    source .venv-zipline/bin/activate
    python validation/zipline/benchmark_performance.py

Metrics:
- Total runtime (seconds)
- Trades per second
- Memory usage (MB)
"""

import gc
import sys
import time
import traceback
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def generate_benchmark_data(n_bars: int, n_assets: int, seed: int = 42) -> tuple:
    """Generate benchmark data for both frameworks."""
    np.random.seed(seed)

    # Use NYSE trading days
    try:
        import exchange_calendars as xcals
        nyse = xcals.get_calendar("XNYS")
        start = pd.Timestamp("2013-01-02")  # No timezone for exchange_calendars
        sessions = nyse.sessions_in_range(start, start + pd.Timedelta(days=n_bars * 2))
        dates = pd.DatetimeIndex(sessions[:n_bars]).tz_localize("UTC")
    except ImportError:
        # Fallback to business days
        dates = pd.date_range(start="2013-01-02", periods=n_bars, freq="B", tz="UTC")

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


def setup_zipline_bundle(asset_data: dict, dates, bundle_name: str = "perf_benchmark"):
    """Register and ingest a custom bundle with multi-asset test data."""
    from zipline.data.bundles import ingest, register

    def make_ingest_func(asset_data_dict, dates_idx):
        def ingest_func(
            environ,
            asset_db_writer,
            minute_bar_writer,
            daily_bar_writer,
            adjustment_writer,
            calendar,
            start_session,
            end_session,
            cache,
            show_progress,
            output_dir,
        ):
            sessions = calendar.sessions_in_range(start_session, end_session)
            asset_names = list(asset_data_dict.keys())

            # Write equity metadata
            asset_db_writer.write(
                equities=pd.DataFrame(
                    {
                        "symbol": asset_names,
                        "asset_name": [f"Asset {i}" for i in range(len(asset_names))],
                        "exchange": ["NYSE"] * len(asset_names),
                    }
                )
            )

            # Write daily bars for each asset
            def gen_data():
                for sid, (asset_name, df) in enumerate(asset_data_dict.items()):
                    df_naive = df.copy()
                    if df_naive.index.tz is not None:
                        df_naive.index = df_naive.index.tz_convert(None)
                    valid_mask = df_naive.index.isin(sessions)
                    trading_df = df_naive[valid_mask].copy()
                    if len(trading_df) > 0:
                        yield sid, trading_df[["open", "high", "low", "close", "volume"]]

            daily_bar_writer.write(gen_data(), show_progress=show_progress)
            adjustment_writer.write()

        return ingest_func

    start_session = dates[0]
    end_session = dates[-1]
    if hasattr(start_session, 'tz') and start_session.tz is not None:
        start_session = start_session.tz_convert(None)
        end_session = end_session.tz_convert(None)

    register(
        bundle_name,
        make_ingest_func(asset_data, dates),
        calendar_name="XNYS",
        start_session=start_session,
        end_session=end_session,
    )

    ingest(bundle_name, show_progress=False)
    return bundle_name


def benchmark_zipline(asset_data: dict, entries: np.ndarray, exits: np.ndarray, dates) -> dict:
    """Benchmark Zipline-Reloaded using bundle-based data."""
    from zipline import run_algorithm
    from zipline.api import order, set_commission, set_slippage, sid
    from zipline.finance.commission import NoCommission
    from zipline.finance.slippage import FixedSlippage

    asset_names = list(asset_data.keys())
    trade_count = [0]

    # Set up bundle (not timed - this is data prep)
    try:
        bundle_name = setup_zipline_bundle(asset_data, dates)
    except Exception as e:
        print(f"    Zipline bundle error: {e}")
        return {"framework": "Zipline", "success": False, "error": f"Bundle setup failed: {e}"}

    def initialize(context):
        context.entries = entries
        context.exits = exits
        context.bar_idx = 0
        context.n_assets = len(asset_names)
        set_commission(NoCommission())
        set_slippage(FixedSlippage(spread=0.0))

    def handle_data(context, data):
        if context.bar_idx >= len(context.entries):
            return

        for asset_idx in range(context.n_assets):
            try:
                asset = sid(asset_idx)
            except Exception:
                continue

            position = context.portfolio.positions.get(asset)
            current_qty = position.amount if position else 0

            if context.exits[context.bar_idx] and current_qty > 0:
                order(asset, -current_qty)
                trade_count[0] += 1
            elif context.entries[context.bar_idx] and current_qty == 0:
                order(asset, 100)
                trade_count[0] += 1

        context.bar_idx += 1

    # Convert to tz-naive for Zipline (it adds tz internally)
    start_date = dates[0]
    end_date = dates[-1]
    if hasattr(start_date, 'tz') and start_date.tz is not None:
        start_date = start_date.tz_convert(None)
        end_date = end_date.tz_convert(None)

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        run_algorithm(
            start=start_date,
            end=end_date,
            initialize=initialize,
            handle_data=handle_data,
            capital_base=100_000_000.0,  # $100M for large portfolios
            bundle=bundle_name,
        )
        success = True
    except Exception as e:
        print(f"    Zipline error: {e}")
        return {"framework": "Zipline", "success": False, "error": str(e)}

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "framework": "Zipline",
        "runtime_sec": end_time - start_time,
        "num_trades": trade_count[0],
        "memory_mb": peak / 1024 / 1024,
        "success": success,
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
                    "timestamp": ts.to_pydatetime().replace(tzinfo=None),
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
                    "timestamp": ts.to_pydatetime().replace(tzinfo=None),
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
        account_type="cash",
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
        account_type="cash",
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

    print("  Running Zipline...")
    zipline_results = benchmark_zipline(asset_data, entries, exits, dates)

    print("  Running ml4t.backtest...")
    ml4t_results = benchmark_ml4t_backtest(asset_data, entries, exits, dates)

    # Calculate speedup
    if ml4t_results["runtime_sec"] > 0 and zipline_results.get("success", False):
        speedup = zipline_results["runtime_sec"] / ml4t_results["runtime_sec"]
    else:
        speedup = 0

    return {
        "n_bars": n_bars,
        "n_assets": n_assets,
        "zipline_runtime": zipline_results.get("runtime_sec", 0),
        "zipline_trades": zipline_results.get("num_trades", 0),
        "zipline_memory": zipline_results.get("memory_mb", 0),
        "zipline_success": zipline_results.get("success", False),
        "zipline_error": zipline_results.get("error", None),
        "ml4t_runtime": ml4t_results["runtime_sec"],
        "ml4t_trades": ml4t_results["num_trades"],
        "ml4t_memory": ml4t_results["memory_mb"],
        "speedup": speedup,
    }


def main():
    print("=" * 70)
    print("Performance Benchmark: ml4t.backtest vs Zipline-Reloaded")
    print("=" * 70)

    # Test configurations (reduced from VBT benchmark due to Zipline overhead)
    configs = [
        (100, 1),
        (500, 1),
        (1000, 1),
        (500, 5),
        (1000, 10),
    ]

    results = []

    for n_bars, n_assets in configs:
        print(f"\nBenchmark: {n_bars} bars x {n_assets} assets")
        try:
            result = run_benchmark(n_bars, n_assets)
            results.append(result)

            if result["zipline_success"]:
                print(
                    f"  Zipline: {result['zipline_runtime']:.3f}s, {result['zipline_trades']} trades, {result['zipline_memory']:.1f} MB"
                )
            else:
                err = result.get("zipline_error", "Unknown error")
                print(f"  Zipline: FAILED - {err}")
            print(
                f"  ML4T: {result['ml4t_runtime']:.3f}s, {result['ml4t_trades']} trades, {result['ml4t_memory']:.1f} MB"
            )
            if result["speedup"] > 0:
                print(
                    f"  Speedup: {result['speedup']:.2f}x {'(Zipline faster)' if result['speedup'] > 1 else '(ML4T faster)'}"
                )
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'Zipline (s)':<12} {'ML4T (s)':<12} {'Speedup':<12} {'Winner'}")
    print("-" * 70)

    for r in results:
        config = f"{r['n_bars']}x{r['n_assets']}"
        if r["zipline_success"] and r["speedup"] > 0:
            winner = "Zipline" if r["speedup"] > 1 else "ML4T"
            speedup_str = f"{r['speedup']:.2f}x" if r["speedup"] > 1 else f"{1/r['speedup']:.2f}x"
            print(
                f"{config:<20} {r['zipline_runtime']:<12.3f} {r['ml4t_runtime']:<12.3f} {speedup_str:<12} {winner}"
            )
        else:
            print(f"{config:<20} {'FAILED':<12} {r['ml4t_runtime']:<12.3f} {'N/A':<12} {'N/A'}")

    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
