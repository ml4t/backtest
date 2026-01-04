#!/usr/bin/env python3
"""Unified Performance Benchmark Runner.

Runs performance benchmarks across all frameworks and generates a summary report.

Usage:
    # Run all frameworks
    python validation/run_all_benchmarks.py

    # Run specific framework
    python validation/run_all_benchmarks.py --framework ml4t

    # Run specific configuration
    python validation/run_all_benchmarks.py --config "100x1" "1000x1"

Output:
    - Console summary
    - validation/BENCHMARK_RESULTS.md
"""

import argparse
import gc
import json
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Directory structure
VALIDATION_DIR = Path(__file__).parent
PROJECT_ROOT = VALIDATION_DIR.parent

# Add project root to path for ml4t imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Framework configurations
FRAMEWORKS = {
    "ml4t": {
        "venv": ".venv",
        "display_name": "ml4t.backtest",
        "function": "benchmark_ml4t",
    },
    "vectorbt_pro": {
        "venv": ".venv-vectorbt-pro",
        "display_name": "VectorBT Pro",
        "function": "benchmark_vectorbt_pro",
    },
    "vectorbt_oss": {
        "venv": ".venv",
        "display_name": "VectorBT OSS",
        "function": "benchmark_vectorbt_oss",
    },
    "backtrader": {
        "venv": ".venv",
        "display_name": "Backtrader",
        "function": "benchmark_backtrader",
    },
    "zipline": {
        "venv": ".venv",
        "display_name": "Zipline",
        "function": "benchmark_zipline",
    },
    "lean": {
        "venv": None,  # Uses Docker
        "display_name": "LEAN CLI",
        "function": "benchmark_lean",
    },
}

# Test configurations (n_bars, n_assets)
CONFIGS = {
    "100x1": (100, 1),
    "500x1": (500, 1),
    "1000x1": (1000, 1),
    "500x5": (500, 5),
    "1000x10": (1000, 10),
    "2520x500": (2520, 500),  # ~10 years daily
}


def generate_benchmark_data(n_bars: int, n_assets: int, seed: int = 42) -> tuple:
    """Generate benchmark data for testing."""
    np.random.seed(seed)

    # Generate date index (business days)
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


def benchmark_ml4t(asset_data: dict, entries: np.ndarray, exits: np.ndarray, dates) -> dict:
    """Benchmark ml4t.backtest."""
    import polars as pl

    from ml4t.backtest import DataFeed, Engine, ExecutionMode, NoCommission, NoSlippage, Strategy

    # Prepare data in polars format
    rows = []
    for asset_name, df in asset_data.items():
        for ts, row in df.iterrows():
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

    # Warm-up run
    small_prices = prices_pl.head(100 * len(asset_data))
    small_signals = signals_pl.head(100 * len(asset_data))
    small_feed = DataFeed(prices_df=small_prices, signals_df=small_signals)
    small_engine = Engine(
        small_feed,
        BenchmarkStrategy(),
        initial_cash=100_000_000.0,  # Match main run
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
    engine = Engine(
        feed,
        BenchmarkStrategy(),
        initial_cash=100_000_000.0,  # $100M to handle 500 assets
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
        "success": True,
    }


def benchmark_vectorbt_oss(
    asset_data: dict, entries: np.ndarray, exits: np.ndarray, dates
) -> dict:
    """Benchmark VectorBT OSS."""
    try:
        import vectorbt as vbt
    except ImportError:
        return {"framework": "VectorBT OSS", "success": False, "error": "Not installed"}

    # Build price panel
    close_df = pd.DataFrame({name: df["close"] for name, df in asset_data.items()})

    # Expand signals to match assets
    entries_2d = np.column_stack([entries] * len(asset_data))
    exits_2d = np.column_stack([exits] * len(asset_data))

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries_2d,
        exits=exits_2d,
        init_cash=100_000.0,
        fees=0.0,
        slippage=0.0,
        size=100,
        accumulate=False,
    )

    trades_count = pf.trades.count()
    # Handle both scalar and Series (multi-asset case)
    if hasattr(trades_count, 'sum'):
        num_trades = int(trades_count.sum())
    else:
        num_trades = int(trades_count)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "framework": "VectorBT OSS",
        "runtime_sec": end_time - start_time,
        "num_trades": num_trades,
        "memory_mb": peak / 1024 / 1024,
        "success": True,
    }


def benchmark_vectorbt_pro(
    asset_data: dict, entries: np.ndarray, exits: np.ndarray, dates
) -> dict:
    """Benchmark VectorBT Pro."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        return {"framework": "VectorBT Pro", "success": False, "error": "Not installed"}

    # Build price panel
    close_df = pd.DataFrame({name: df["close"] for name, df in asset_data.items()})

    # Expand signals to match assets
    entries_2d = np.column_stack([entries] * len(asset_data))
    exits_2d = np.column_stack([exits] * len(asset_data))

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    pf = vbt.Portfolio.from_signals(
        close=close_df,
        entries=entries_2d,
        exits=exits_2d,
        init_cash=100_000.0,
        fees=0.0,
        slippage=0.0,
        size=100,
        accumulate=False,
    )

    trades_count = pf.trades.count()
    # Handle both scalar and Series (multi-asset case)
    if hasattr(trades_count, 'sum'):
        num_trades = int(trades_count.sum())
    else:
        num_trades = int(trades_count)

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "framework": "VectorBT Pro",
        "runtime_sec": end_time - start_time,
        "num_trades": num_trades,
        "memory_mb": peak / 1024 / 1024,
        "success": True,
    }


def benchmark_backtrader(
    asset_data: dict, entries: np.ndarray, exits: np.ndarray, dates
) -> dict:
    """Benchmark Backtrader."""
    try:
        import backtrader as bt
    except ImportError:
        return {"framework": "Backtrader", "success": False, "error": "Not installed"}

    trade_count = [0]

    class BenchmarkStrategy(bt.Strategy):
        def __init__(self):
            self.bar_idx = 0

        def next(self):
            if self.bar_idx >= len(entries):
                return

            for data in self.datas:
                position = self.getposition(data)
                current_qty = position.size

                if exits[self.bar_idx] and current_qty > 0:
                    self.close(data)
                    trade_count[0] += 1
                elif entries[self.bar_idx] and current_qty == 0:
                    self.buy(data, size=100)
                    trade_count[0] += 1

            self.bar_idx += 1

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    cerebro = bt.Cerebro()
    cerebro.addstrategy(BenchmarkStrategy)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.0)

    for asset_name, df in asset_data.items():
        bt_data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )
        cerebro.adddata(bt_data, name=asset_name)

    cerebro.run()

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "framework": "Backtrader",
        "runtime_sec": end_time - start_time,
        "num_trades": trade_count[0],
        "memory_mb": peak / 1024 / 1024,
        "success": True,
    }


def benchmark_zipline(
    asset_data: dict, entries: np.ndarray, exits: np.ndarray, dates
) -> dict:
    """Benchmark Zipline-Reloaded."""
    try:
        from zipline import run_algorithm
        from zipline.api import order, set_commission, set_slippage, symbol
        from zipline.finance.commission import NoCommission
        from zipline.finance.slippage import FixedSlippage
    except ImportError:
        return {"framework": "Zipline", "success": False, "error": "Not installed"}

    # Build panel data for Zipline
    panel_data = {}
    for field in ["open", "high", "low", "close", "volume"]:
        panel_data[field] = pd.DataFrame({name: df[field] for name, df in asset_data.items()})

    asset_names = list(asset_data.keys())
    trade_count = [0]

    def initialize(context):
        context.entries = entries
        context.exits = exits
        context.bar_idx = 0
        context.asset_names = asset_names
        set_commission(NoCommission())
        set_slippage(FixedSlippage(spread=0.0))

    def handle_data(context, data):
        if context.bar_idx >= len(context.entries):
            return

        for asset_name in context.asset_names:
            try:
                asset = symbol(asset_name)
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

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        run_algorithm(
            start=dates[0],
            end=dates[-1],
            initialize=initialize,
            handle_data=handle_data,
            capital_base=100_000.0,
            data=panel_data,
        )
        success = True
    except Exception as e:
        success = False
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


def benchmark_lean(
    asset_data: dict, entries: np.ndarray, exits: np.ndarray, dates
) -> dict:
    """Benchmark LEAN CLI (requires Docker)."""
    lean_workspace = VALIDATION_DIR / "lean" / "workspace"

    if not lean_workspace.exists():
        return {"framework": "LEAN CLI", "success": False, "error": "Workspace not found"}

    # Check if lean CLI is available
    try:
        result = subprocess.run(
            ["lean", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"framework": "LEAN CLI", "success": False, "error": "CLI not available"}
    except Exception as e:
        return {"framework": "LEAN CLI", "success": False, "error": str(e)}

    # Run a simple backtest using the test_validation project
    test_project = lean_workspace / "test_validation"
    if not test_project.exists():
        return {"framework": "LEAN CLI", "success": False, "error": "Test project not found"}

    gc.collect()
    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            ["lean", "backtest", str(test_project)],
            capture_output=True,
            text=True,
            timeout=300,  # LEAN can be slow (Docker + initialization)
            cwd=str(lean_workspace),
        )
        success = result.returncode == 0

        # Parse output for performance info
        output = result.stdout + result.stderr
        if "Total Trades" in output:
            # Try to extract trade count from LEAN output
            for line in output.split("\n"):
                if "Total Trades" in line:
                    try:
                        num_trades = int(line.split()[-1])
                        break
                    except (ValueError, IndexError):
                        num_trades = 0
        else:
            num_trades = 0

    except subprocess.TimeoutExpired:
        return {"framework": "LEAN CLI", "success": False, "error": "Timeout (300s)"}
    except Exception as e:
        return {"framework": "LEAN CLI", "success": False, "error": str(e)}

    end_time = time.perf_counter()

    return {
        "framework": "LEAN CLI",
        "runtime_sec": end_time - start_time,
        "num_trades": num_trades,
        "memory_mb": 0,  # Cannot measure Docker memory easily
        "success": success,
        "note": "Docker-based execution, includes container startup overhead",
    }


def run_benchmark(config_name: str, framework: str) -> dict:
    """Run a single benchmark configuration."""
    n_bars, n_assets = CONFIGS[config_name]

    print(f"\n  Generating data: {n_bars} bars x {n_assets} assets...")
    asset_data, entries, exits, dates = generate_benchmark_data(n_bars, n_assets)

    benchmark_func = {
        "ml4t": benchmark_ml4t,
        "vectorbt_pro": benchmark_vectorbt_pro,
        "vectorbt_oss": benchmark_vectorbt_oss,
        "backtrader": benchmark_backtrader,
        "zipline": benchmark_zipline,
        "lean": benchmark_lean,
    }.get(framework)

    if not benchmark_func:
        return {"framework": framework, "success": False, "error": "Unknown framework"}

    try:
        result = benchmark_func(asset_data, entries, exits, dates)
        result["config"] = config_name
        result["n_bars"] = n_bars
        result["n_assets"] = n_assets
        return result
    except Exception as e:
        return {
            "framework": framework,
            "config": config_name,
            "success": False,
            "error": str(e),
        }


def run_all_benchmarks(
    frameworks: list = None, configs: list = None
) -> list:
    """Run all benchmarks and return results."""
    if frameworks is None:
        frameworks = list(FRAMEWORKS.keys())
    if configs is None:
        configs = list(CONFIGS.keys())

    results = []

    for framework in frameworks:
        config = FRAMEWORKS.get(framework)
        if not config:
            print(f"Unknown framework: {framework}")
            continue

        print(f"\n{'='*60}")
        print(f"Framework: {config['display_name']}")
        print(f"{'='*60}")

        for config_name in configs:
            print(f"\n  Config: {config_name}")
            result = run_benchmark(config_name, framework)
            results.append(result)

            if result.get("success"):
                print(
                    f"    Runtime: {result['runtime_sec']:.3f}s, "
                    f"Trades: {result['num_trades']}, "
                    f"Memory: {result.get('memory_mb', 0):.1f} MB"
                )
            else:
                print(f"    FAILED: {result.get('error', 'Unknown error')}")

    return results


def generate_report(results: list) -> str:
    """Generate markdown report from results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Performance Benchmark Results",
        "",
        f"**Generated**: {now}",
        "",
        "## Summary",
        "",
        "| Framework | Config | Runtime (s) | Trades | Memory (MB) | Status |",
        "|-----------|--------|-------------|--------|-------------|--------|",
    ]

    for r in results:
        framework = r.get("framework", "Unknown")
        config = r.get("config", "N/A")
        if r.get("success"):
            runtime = f"{r['runtime_sec']:.3f}"
            trades = str(r.get("num_trades", "N/A"))
            memory = f"{r.get('memory_mb', 0):.1f}"
            status = "PASS"
        else:
            runtime = "-"
            trades = "-"
            memory = "-"
            status = f"FAIL: {r.get('error', 'Unknown')[:30]}"

        lines.append(f"| {framework} | {config} | {runtime} | {trades} | {memory} | {status} |")

    # Add comparison table by config
    lines.extend([
        "",
        "## Performance Comparison by Configuration",
        "",
    ])

    # Group by config
    configs_seen = set(r.get("config") for r in results if r.get("config"))
    for config_name in sorted(configs_seen):
        config_results = [r for r in results if r.get("config") == config_name and r.get("success")]
        if not config_results:
            continue

        lines.extend([
            f"### {config_name}",
            "",
            "| Framework | Runtime (s) | vs ml4t |",
            "|-----------|-------------|---------|",
        ])

        # Find ml4t baseline
        ml4t_time = None
        for r in config_results:
            if r.get("framework") == "ml4t.backtest":
                ml4t_time = r["runtime_sec"]
                break

        for r in sorted(config_results, key=lambda x: x["runtime_sec"]):
            framework = r["framework"]
            runtime = r["runtime_sec"]
            if ml4t_time and ml4t_time > 0:
                speedup = runtime / ml4t_time
                vs_ml4t = f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x faster"
            else:
                vs_ml4t = "-"
            lines.append(f"| {framework} | {runtime:.3f} | {vs_ml4t} |")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument(
        "--framework",
        type=str,
        help="Specific framework to test (ml4t, vectorbt_pro, vectorbt_oss, backtrader, zipline, lean)",
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        help="Specific configurations to run (e.g., 100x1, 1000x1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation/BENCHMARK_RESULTS.md",
        help="Output file path",
    )

    args = parser.parse_args()

    frameworks = [args.framework] if args.framework else None
    configs = args.config if args.config else None

    print("=" * 60)
    print("Performance Benchmark Runner")
    print("=" * 60)

    results = run_all_benchmarks(frameworks=frameworks, configs=configs)

    # Generate and save report
    report = generate_report(results)
    output_path = PROJECT_ROOT / args.output
    output_path.write_text(report)

    print(f"\n{'='*60}")
    print(f"Report saved to: {output_path}")
    print("=" * 60)

    # Summary
    passed = sum(1 for r in results if r.get("success"))
    failed = len(results) - passed
    print(f"\nTotal: {len(results)} benchmarks ({passed} passed, {failed} failed)")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
