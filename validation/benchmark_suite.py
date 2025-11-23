#!/usr/bin/env python3
"""Comprehensive Performance Benchmark Suite.

This script benchmarks ml4t.backtest against VectorBT Pro, VectorBT OSS,
Backtrader, and Zipline across realistic trading scenarios.

Key scenarios:
1. Long/short top-N/bottom-N with daily rebalancing
2. Stop-loss and take-profit orders
3. Commission and slippage models
4. Scale from small (100 assets × 1 month) to large (500 assets × 1 year)

Usage:
    # Run with VectorBT Pro
    source .venv-vectorbt-pro/bin/activate
    python validation/benchmark_suite.py --framework vbt-pro

    # Run with Backtrader
    source .venv-backtrader/bin/activate
    python validation/benchmark_suite.py --framework backtrader

    # Run ml4t only (any venv)
    python validation/benchmark_suite.py --framework ml4t

    # Run all scenarios for specific framework
    python validation/benchmark_suite.py --framework vbt-pro --all

    # Run specific scenario
    python validation/benchmark_suite.py --framework ml4t --scenario baseline
"""

import argparse
import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    n_bars: int
    n_assets: int
    frequency: str  # "D" for daily, "1min" for minute
    top_n: int  # Number of assets to go long
    bottom_n: int  # Number of assets to go short (0 = long only)
    rebalance_freq: int  # Bars between rebalancing
    stop_loss: float | None = None  # e.g., 0.02 = 2%
    take_profit: float | None = None  # e.g., 0.05 = 5%
    commission_pct: float = 0.0
    slippage_pct: float = 0.0


# Benchmark scenarios - progressive scaling
SCENARIOS = {
    # === Baseline: Quick sanity check ===
    "baseline": BenchmarkConfig(
        name="Baseline (100×1mo minute)",
        n_bars=8_580,  # ~1 month of minute data (390 min/day × 22 days)
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,  # Daily rebalance
    ),
    # === Scale tests ===
    "scale_1": BenchmarkConfig(
        name="Scale 1 (100×3mo minute)",
        n_bars=25_740,  # ~3 months
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,
    ),
    "scale_2": BenchmarkConfig(
        name="Scale 2 (250×6mo minute)",
        n_bars=51_480,  # ~6 months
        n_assets=250,
        frequency="1min",
        top_n=25,
        bottom_n=25,
        rebalance_freq=390,
    ),
    "scale_3": BenchmarkConfig(
        name="Scale 3 (500×1yr minute)",
        n_bars=97_500,  # ~1 year (390 × 250 trading days)
        n_assets=500,
        frequency="1min",
        top_n=25,
        bottom_n=25,
        rebalance_freq=390,
    ),
    # === Daily data (faster, 10 years) ===
    "daily_baseline": BenchmarkConfig(
        name="Daily (500×10yr daily)",
        n_bars=2_520,  # 10 years of daily data
        n_assets=500,
        frequency="D",
        top_n=25,
        bottom_n=25,
        rebalance_freq=1,  # Daily rebalance
    ),
    # === Feature tests ===
    "stop_loss": BenchmarkConfig(
        name="Stop-loss (100×1mo)",
        n_bars=8_580,
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,
        stop_loss=0.02,  # 2% stop-loss
    ),
    "take_profit": BenchmarkConfig(
        name="Take-profit (100×1mo)",
        n_bars=8_580,
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,
        take_profit=0.05,  # 5% take-profit
    ),
    "stop_and_profit": BenchmarkConfig(
        name="Stop+Take (100×1mo)",
        n_bars=8_580,
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,
        stop_loss=0.02,
        take_profit=0.05,
    ),
    "with_costs": BenchmarkConfig(
        name="With costs (100×1mo)",
        n_bars=8_580,
        n_assets=100,
        frequency="1min",
        top_n=10,
        bottom_n=10,
        rebalance_freq=390,
        commission_pct=0.001,  # 10 bps
        slippage_pct=0.0005,  # 5 bps
    ),
    # === Long-only variant ===
    "long_only": BenchmarkConfig(
        name="Long-only (100×1mo)",
        n_bars=8_580,
        n_assets=100,
        frequency="1min",
        top_n=25,
        bottom_n=0,  # No shorts
        rebalance_freq=390,
    ),
}


def generate_benchmark_data(config: BenchmarkConfig, seed: int = 42) -> tuple:
    """Generate synthetic market data and signals for benchmarking.

    Returns:
        Tuple of (price_data, signals, dates) where:
        - price_data: dict of asset_name -> DataFrame with OHLCV
        - signals: DataFrame with timestamp, asset, score columns
        - dates: DatetimeIndex
    """
    np.random.seed(seed)

    # Generate dates
    if config.frequency == "1min":
        # Generate minute bars (market hours only: 9:30-16:00 = 390 mins/day)
        start = datetime(2023, 1, 3, 9, 30)  # First trading day
        dates = []
        current = start
        bars_generated = 0
        while bars_generated < config.n_bars:
            # Add minute
            dates.append(current)
            bars_generated += 1
            current += timedelta(minutes=1)
            # Skip to next day at 16:00
            if current.hour == 16 and current.minute == 0:
                current = current.replace(hour=9, minute=30) + timedelta(days=1)
                # Skip weekends
                while current.weekday() >= 5:
                    current += timedelta(days=1)
        dates = pd.DatetimeIndex(dates)
    else:
        # Use NYSE calendar for actual trading days (not just business days)
        # This ensures data matches Zipline's calendar and respects US holidays
        try:
            import exchange_calendars as xcals
            nyse = xcals.get_calendar("XNYS")
            # Get NYSE sessions starting from 2013-01-02
            all_sessions = nyse.sessions_in_range("2013-01-02", "2025-12-31")
            dates = all_sessions[:config.n_bars]
        except ImportError:
            # Fallback to business days if exchange_calendars not available
            dates = pd.date_range(start="2013-01-02", periods=config.n_bars, freq="B")

    # Generate price data for each asset
    price_data = {}
    for i in range(config.n_assets):
        asset_name = f"ASSET_{i:03d}"
        base_price = 50.0 + np.random.rand() * 150  # $50-200 starting price

        # Generate realistic returns (different vol/drift per asset)
        daily_vol = 0.01 + np.random.rand() * 0.03  # 1-4% daily vol
        drift = -0.0001 + np.random.rand() * 0.0002  # Small drift

        if config.frequency == "1min":
            # Scale vol for minute bars
            bar_vol = daily_vol / np.sqrt(390)
            bar_drift = drift / 390
        else:
            bar_vol = daily_vol
            bar_drift = drift

        returns = np.random.randn(config.n_bars) * bar_vol + bar_drift
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV with realistic intraday patterns
        high_mult = 1 + np.abs(np.random.randn(config.n_bars)) * bar_vol
        low_mult = 1 - np.abs(np.random.randn(config.n_bars)) * bar_vol
        open_offset = np.random.randn(config.n_bars) * bar_vol * 0.3

        price_data[asset_name] = pd.DataFrame(
            {
                "open": prices * (1 + open_offset),
                "high": prices * high_mult,
                "low": prices * low_mult,
                "close": prices,
                "volume": np.random.randint(10000, 1000000, config.n_bars).astype(float),
            },
            index=dates,
        )

    # Generate signals (scores for ranking)
    # Score = momentum + noise, changes each rebalance period
    signal_rows = []
    rebalance_bars = list(range(0, config.n_bars, config.rebalance_freq))

    for bar_idx in rebalance_bars:
        ts = dates[bar_idx]
        # Generate random scores for all assets
        scores = np.random.randn(config.n_assets)
        for i, score in enumerate(scores):
            signal_rows.append(
                {
                    "timestamp": ts,
                    "asset": f"ASSET_{i:03d}",
                    "score": score,
                }
            )

    signals = pd.DataFrame(signal_rows)

    return price_data, signals, dates


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""

    framework: str
    scenario: str
    runtime_sec: float
    num_trades: int
    final_value: float
    memory_mb: float
    error: str | None = None
    trades_df: pd.DataFrame | None = None  # Trade log for validation
    positions_df: pd.DataFrame | None = None  # PyFolio positions (Backtrader/Zipline)
    transactions_df: pd.DataFrame | None = None  # PyFolio transactions (Backtrader/Zipline)


def save_trades(result: BenchmarkResult, output_dir: Path):
    """Save trade log to CSV for validation."""
    if result.trades_df is not None and len(result.trades_df) > 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{result.framework.replace(' ', '_').lower()}_{result.scenario.replace(' ', '_').replace('×', 'x').lower()}.csv"
        filepath = output_dir / filename
        result.trades_df.to_csv(filepath, index=False)
        print(f"  Saved trades to: {filepath}")


def compare_trades(results: list[BenchmarkResult]) -> dict:
    """Compare trades between frameworks to detect compounding errors."""
    if len(results) < 2:
        return {}

    comparisons = {}
    baseline = None
    for r in results:
        if r.framework == "ml4t.backtest" and r.trades_df is not None:
            baseline = r
            break

    if baseline is None:
        return {}

    for r in results:
        if r == baseline or r.trades_df is None:
            continue

        # Compare trade counts
        count_diff = abs(r.num_trades - baseline.num_trades)
        count_pct = count_diff / baseline.num_trades * 100 if baseline.num_trades > 0 else 0

        # Compare final values
        value_diff = abs(r.final_value - baseline.final_value)
        value_pct = value_diff / baseline.final_value * 100 if baseline.final_value > 0 else 0

        comparisons[r.framework] = {
            "trade_count_diff": count_diff,
            "trade_count_pct": count_pct,
            "final_value_diff": value_diff,
            "final_value_pct": value_pct,
        }

    return comparisons


def benchmark_ml4t(
    config: BenchmarkConfig,
    price_data: dict,
    signals: pd.DataFrame,
    dates,
    execution_mode: str = "same_bar",
) -> BenchmarkResult:
    """Benchmark ml4t.backtest with given configuration.

    Args:
        execution_mode: "same_bar" (default, matches VectorBT) or "next_bar" (matches Backtrader)
    """
    import polars as pl

    from ml4t.backtest import (
        DataFeed,
        Engine,
        ExecutionMode,
        NoCommission,
        NoSlippage,
        PercentageCommission,
        PercentageSlippage,
        Strategy,
    )

    # Select execution mode
    exec_mode = ExecutionMode.NEXT_BAR if execution_mode == "next_bar" else ExecutionMode.SAME_BAR
    framework_name = "ml4t.backtest" if execution_mode == "same_bar" else "ml4t.backtest (backtrader-mode)"

    # Convert price data to Polars format
    price_rows = []
    for asset_name, df in price_data.items():
        for ts, row in df.iterrows():
            price_rows.append(
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
    prices_pl = pl.DataFrame(price_rows)

    # Convert signals to Polars
    signals_pl = pl.DataFrame(
        {
            "timestamp": signals["timestamp"].tolist(),
            "asset": signals["asset"].tolist(),
            "score": signals["score"].tolist(),
        }
    )

    class TopBottomStrategy(Strategy):
        """Long top N, short bottom N based on score signals."""

        def __init__(self, top_n: int, bottom_n: int, stop_loss: float | None, take_profit: float | None):
            self.top_n = top_n
            self.bottom_n = bottom_n
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            self.current_signals: dict[str, float] = {}

        def on_start(self, broker):
            """Set up position rules for stop-loss and take-profit."""
            from ml4t.backtest.risk import StopLoss, TakeProfit, RuleChain

            rules = []
            if self.stop_loss is not None:
                rules.append(StopLoss(pct=self.stop_loss))
            if self.take_profit is not None:
                rules.append(TakeProfit(pct=self.take_profit))

            if rules:
                broker.set_position_rules(RuleChain(rules))

        def on_data(self, timestamp, data, context, broker):
            # Update signals from context (if available)
            for asset_name, asset_data in data.items():
                if "signals" in asset_data and "score" in asset_data["signals"]:
                    self.current_signals[asset_name] = asset_data["signals"]["score"]

            if not self.current_signals:
                return

            # Rank assets by score
            sorted_assets = sorted(self.current_signals.items(), key=lambda x: x[1], reverse=True)
            top_assets = set(a for a, _ in sorted_assets[: self.top_n])
            bottom_assets = set(a for a, _ in sorted_assets[-self.bottom_n :]) if self.bottom_n > 0 else set()

            # Rebalance: close positions not in target, open new positions
            for asset_name in data.keys():
                position = broker.get_position(asset_name)
                current_qty = position.quantity if position else 0

                # Target position
                if asset_name in top_assets:
                    target_qty = 100  # Long 100 shares
                elif asset_name in bottom_assets:
                    target_qty = -100  # Short 100 shares
                else:
                    target_qty = 0

                # Adjust position
                if current_qty != target_qty:
                    if current_qty != 0:
                        broker.close_position(asset_name)
                    if target_qty != 0:
                        broker.submit_order(asset_name, target_qty)

    # Set up commission/slippage
    commission = (
        PercentageCommission(config.commission_pct) if config.commission_pct > 0 else NoCommission()
    )
    slippage = PercentageSlippage(config.slippage_pct) if config.slippage_pct > 0 else NoSlippage()

    # Warm-up run (smaller data)
    n_warmup = min(1000, config.n_bars // 10)
    warmup_prices = prices_pl.head(n_warmup * config.n_assets)
    warmup_signals = signals_pl.filter(pl.col("timestamp") <= dates[n_warmup])
    warmup_feed = DataFeed(prices_df=warmup_prices, signals_df=warmup_signals)
    warmup_engine = Engine(
        warmup_feed,
        TopBottomStrategy(config.top_n, config.bottom_n, config.stop_loss, config.take_profit),
        initial_cash=1e15,  # Unlimited cash to eliminate margin rejections
        account_type="margin",
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        execution_mode=exec_mode,
    )
    _ = warmup_engine.run()

    # Actual benchmark
    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    strategy = TopBottomStrategy(config.top_n, config.bottom_n, config.stop_loss, config.take_profit)

    engine = Engine(
        feed,
        strategy,
        initial_cash=1e15,  # Unlimited cash to eliminate margin rejections
        account_type="margin",
        commission_model=commission,
        slippage_model=slippage,
        execution_mode=exec_mode,
    )

    results = engine.run()

    end_time = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract trade log for validation
    trades_df = None
    if results.get("trades"):
        trade_records = []
        for t in results["trades"]:
            trade_records.append({
                "timestamp": t.entry_time,
                "exit_time": t.exit_time,
                "asset": t.asset,
                "side": "long" if t.quantity > 0 else "short",
                "quantity": abs(t.quantity),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
            })
        trades_df = pd.DataFrame(trade_records)

    return BenchmarkResult(
        framework=framework_name,
        scenario=config.name,
        runtime_sec=end_time - start_time,
        num_trades=results["num_trades"],
        final_value=results["final_value"],
        memory_mb=peak / 1024 / 1024,
        trades_df=trades_df,
    )


def benchmark_vectorbt_pro(
    config: BenchmarkConfig, price_data: dict, signals: pd.DataFrame, dates
) -> BenchmarkResult:
    """Benchmark VectorBT Pro with given configuration."""
    try:
        import vectorbtpro as vbt
    except ImportError:
        return BenchmarkResult(
            framework="VectorBT Pro",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error="VectorBT Pro not installed",
        )

    # Prepare close prices DataFrame
    close_df = pd.DataFrame({name: df["close"] for name, df in price_data.items()})

    # For VectorBT, we need to compute entries/exits from signals
    # This is a limitation - VectorBT is signal-based, not order-based
    # We'll use the Portfolio.from_orders API instead

    # Create ranking-based entries
    signal_pivot = signals.pivot(index="timestamp", columns="asset", values="score")
    signal_pivot = signal_pivot.reindex(dates).ffill()

    # Rank each row
    ranks = signal_pivot.rank(axis=1, ascending=False)

    # Long top N, short bottom N
    long_mask = ranks <= config.top_n
    short_mask = ranks > (config.n_assets - config.bottom_n) if config.bottom_n > 0 else pd.DataFrame(False, index=ranks.index, columns=ranks.columns)

    # Convert to target shares: +100 for long, -100 for short, 0 otherwise
    target_shares = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    target_shares[long_mask.reindex(target_shares.index).ffill().fillna(False)] = 100
    if config.bottom_n > 0:
        target_shares[short_mask.reindex(target_shares.index).ffill().fillna(False)] = -100

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    # Use from_orders with target shares
    # cash_sharing=True ensures single cash pool across all assets (like real portfolio)
    pf = vbt.Portfolio.from_orders(
        close=close_df,
        size=target_shares,
        size_type="targetamount",
        init_cash=1_000_000.0,
        cash_sharing=True,  # Critical: single cash pool, not per-column
        fees=config.commission_pct,
        slippage=config.slippage_pct,
        # Note: VectorBT Pro doesn't have native stop-loss in from_orders
        # Would need from_signals with sl_stop parameter
    )

    # Force computation
    final_value = pf.value.iloc[-1].sum() if hasattr(pf.value.iloc[-1], "sum") else pf.value.iloc[-1]
    trades_readable = pf.trades.records_readable
    num_trades = len(trades_readable)

    end_time = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract trade log for validation
    trades_df = None
    if num_trades > 0:
        # Sort by entry date for proper comparison (VBT returns sorted by column/asset)
        trades_readable = trades_readable.sort_values("Entry Index")
        trade_records = []
        for _, row in trades_readable.iterrows():
            # VectorBT Pro's Entry Index is already a Timestamp
            entry_ts = row.get("Entry Index")
            trade_records.append({
                "timestamp": entry_ts,
                "asset": row.get("Column", "unknown"),
                "side": "long" if str(row.get("Direction", "Long")).lower() == "long" else "short",
                "quantity": abs(row.get("Size", 0)),
                "entry_price": row.get("Avg Entry Price", 0),
                "exit_price": row.get("Avg Exit Price", 0),
                "pnl": row.get("PnL", 0),
            })
        trades_df = pd.DataFrame(trade_records)

    return BenchmarkResult(
        framework="VectorBT Pro",
        scenario=config.name,
        runtime_sec=end_time - start_time,
        num_trades=num_trades,
        final_value=float(final_value),
        memory_mb=peak / 1024 / 1024,
        trades_df=trades_df,
    )


def benchmark_vectorbt_oss(
    config: BenchmarkConfig, price_data: dict, signals: pd.DataFrame, dates
) -> BenchmarkResult:
    """Benchmark VectorBT OSS with given configuration."""
    try:
        import vectorbt as vbt
    except ImportError:
        return BenchmarkResult(
            framework="VectorBT OSS",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error="VectorBT OSS not installed",
        )

    # Prepare close prices DataFrame
    close_df = pd.DataFrame({name: df["close"] for name, df in price_data.items()})

    # Create ranking-based entries
    signal_pivot = signals.pivot(index="timestamp", columns="asset", values="score")
    signal_pivot = signal_pivot.reindex(dates).ffill()

    # Rank each row
    ranks = signal_pivot.rank(axis=1, ascending=False)

    # Long top N, short bottom N
    long_mask = ranks <= config.top_n
    short_mask = ranks > (config.n_assets - config.bottom_n) if config.bottom_n > 0 else pd.DataFrame(False, index=ranks.index, columns=ranks.columns)

    # Convert to target shares: +100 for long, -100 for short, 0 otherwise
    target_shares = pd.DataFrame(0, index=close_df.index, columns=close_df.columns)
    target_shares[long_mask.reindex(target_shares.index).ffill().fillna(False)] = 100
    if config.bottom_n > 0:
        target_shares[short_mask.reindex(target_shares.index).ffill().fillna(False)] = -100

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    # VectorBT OSS uses from_orders API
    # cash_sharing=True ensures single cash pool across all assets (like real portfolio)
    # lock_cash=True enforces cash constraints on short selling (default is False in OSS!)
    pf = vbt.Portfolio.from_orders(
        close=close_df,
        size=target_shares,
        size_type="targetamount",
        init_cash=1_000_000.0,
        cash_sharing=True,  # Critical: single cash pool, not per-column
        lock_cash=True,  # Critical: enforce cash constraints (VBT OSS default is False!)
        fees=config.commission_pct,
        slippage=config.slippage_pct,
    )

    # Force computation
    final_value = pf.value().iloc[-1].sum() if hasattr(pf.value().iloc[-1], "sum") else pf.value().iloc[-1]
    trades_readable = pf.trades.records_readable
    num_trades = len(trades_readable)

    end_time = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract trade log for validation
    trades_df = None
    if num_trades > 0:
        # Sort by entry date for proper comparison (VBT returns sorted by column/asset)
        entry_col = "Entry Timestamp" if "Entry Timestamp" in trades_readable.columns else "Entry Index"
        trades_readable = trades_readable.sort_values(entry_col)
        trade_records = []
        for _, row in trades_readable.iterrows():
            trade_records.append({
                "timestamp": row.get("Entry Timestamp", row.get("Entry Index")),
                "asset": row.get("Column", "unknown"),
                "side": "long" if row.get("Direction", "Long") == "Long" else "short",
                "quantity": abs(row.get("Size", 0)),
                "entry_price": row.get("Avg Entry Price", row.get("Entry Price", 0)),
                "exit_price": row.get("Avg Exit Price", row.get("Exit Price", 0)),
                "pnl": row.get("PnL", 0),
            })
        trades_df = pd.DataFrame(trade_records)

    return BenchmarkResult(
        framework="VectorBT OSS",
        scenario=config.name,
        runtime_sec=end_time - start_time,
        num_trades=num_trades,
        final_value=float(final_value),
        memory_mb=peak / 1024 / 1024,
        trades_df=trades_df,
    )


def benchmark_zipline(
    config: BenchmarkConfig, price_data: dict, signals: pd.DataFrame, dates
) -> BenchmarkResult:
    """Benchmark Zipline with given configuration.

    Creates a multi-asset bundle with all test data and runs a proper
    top-N/bottom-N ranking strategy.
    """
    try:
        from zipline import run_algorithm
        from zipline.api import order_target, symbol, symbols, set_slippage, get_datetime, sid
        from zipline.data.bundles import register, ingest
        from zipline.finance.slippage import SlippageModel
        import exchange_calendars as xcals
    except ImportError as e:
        return BenchmarkResult(
            framework="Zipline",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error=f"Zipline not installed: {e}",
        )

    # Zipline only supports daily data in bundles
    if config.frequency == "1min":
        return BenchmarkResult(
            framework="Zipline",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error="Zipline bundles only support daily data",
        )

    # Get NYSE calendar sessions for proper date alignment
    nyse = xcals.get_calendar("XNYS")

    # Prepare all assets for the bundle
    asset_names = list(price_data.keys())
    n_assets = len(asset_names)

    # Convert price data to have NYSE-aligned dates
    # Filter to only NYSE trading days
    first_df = price_data[asset_names[0]]
    start_date = first_df.index[0]
    end_date = first_df.index[-1]

    # Get actual NYSE sessions
    nyse_sessions = nyse.sessions_in_range(
        pd.Timestamp(start_date).tz_localize(None) if pd.Timestamp(start_date).tz else start_date,
        pd.Timestamp(end_date).tz_localize(None) if pd.Timestamp(end_date).tz else end_date
    )

    # Custom slippage model for open-price fills (matching ml4t NEXT_BAR mode)
    class OpenPriceSlippage(SlippageModel):
        @staticmethod
        def process_order(data, order):
            return (data.current(order.asset, "open"), order.amount)

    # Build signal lookup: timestamp -> {asset: score}
    signal_lookup = {}
    for _, row in signals.iterrows():
        ts = row["timestamp"]
        # Normalize to tz-naive for comparison
        if hasattr(ts, 'tz') and ts.tz is not None:
            ts = ts.tz_convert(None)
        if ts not in signal_lookup:
            signal_lookup[ts] = {}
        signal_lookup[ts][row["asset"]] = row["score"]

    # Bundle ingest function for multi-asset
    def make_multi_asset_ingest(price_data_dict, asset_list):
        def ingest_func(environ, asset_db_writer, minute_bar_writer, daily_bar_writer,
                       adjustment_writer, calendar, start_session, end_session, cache,
                       show_progress, output_dir):
            sessions = calendar.sessions_in_range(start_session, end_session)

            # Write equity metadata for all assets
            equities_df = pd.DataFrame({
                'symbol': asset_list,
                'asset_name': [f'Asset {name}' for name in asset_list],
                'exchange': ['NYSE'] * len(asset_list),
            })
            asset_db_writer.write(equities=equities_df)

            # Write daily bars for each asset
            bar_data = []
            for sid, asset_name in enumerate(asset_list):
                df = price_data_dict[asset_name].copy()
                # Convert to tz-naive
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                # Filter to valid sessions
                valid_mask = df.index.isin(sessions)
                trading_df = df[valid_mask][['open', 'high', 'low', 'close', 'volume']]
                if len(trading_df) > 0:
                    bar_data.append((sid, trading_df))

            daily_bar_writer.write(bar_data, show_progress=show_progress)
            adjustment_writer.write()

        return ingest_func

    # Register and ingest bundle (cached - only created once per config)
    bundle_name = f"bench_multi_{n_assets}_{config.n_bars}"
    start_session = nyse_sessions[0]
    end_session = nyse_sessions[-1] if len(nyse_sessions) <= config.n_bars else nyse_sessions[config.n_bars - 1]

    # Check if bundle already exists ON DISK to avoid re-ingestion
    # The in-memory `bundles` registry resets each process - check filesystem instead
    import os
    from pathlib import Path
    zipline_root = Path(os.environ.get('ZIPLINE_ROOT', Path.home() / '.zipline'))
    bundle_dir = zipline_root / 'data' / bundle_name  # Zipline stores in data/, not bundles/
    bundle_exists = bundle_dir.exists() and any(bundle_dir.iterdir()) if bundle_dir.exists() else False

    bundle_time = 0.0
    if not bundle_exists:
        bundle_start = time.perf_counter()
        try:
            register(
                bundle_name,
                make_multi_asset_ingest(price_data, asset_names),
                calendar_name='XNYS',
                start_session=start_session,
                end_session=end_session,
            )
            ingest(bundle_name, show_progress=False)
            bundle_time = time.perf_counter() - bundle_start
            print(f"  Bundle creation: {bundle_time:.1f}s (one-time setup)")
        except Exception as e:
            return BenchmarkResult(
                framework="Zipline",
                scenario=config.name,
                runtime_sec=0,
                num_trades=0,
                final_value=0,
                memory_mb=0,
                error=f"Bundle setup failed: {e}",
            )
    else:
        # Bundle exists on disk - just re-register (required for run_algorithm in this process)
        print(f"  Using cached bundle: {bundle_name}")
        register(
            bundle_name,
            make_multi_asset_ingest(price_data, asset_names),
            calendar_name='XNYS',
            start_session=start_session,
            end_session=end_session,
        )

    # Algorithm state
    algo_state = {
        "signal_lookup": signal_lookup,
        "asset_names": asset_names,
        "top_n": config.top_n,
        "bottom_n": config.bottom_n,
        "current_signals": {},
    }

    def initialize(context):
        context.state = algo_state
        # Get all asset objects by sid
        context.assets = [sid(i) for i in range(len(context.state["asset_names"]))]
        context.asset_map = {name: context.assets[i] for i, name in enumerate(context.state["asset_names"])}
        set_slippage(OpenPriceSlippage())

    def handle_data(context, data):
        dt = get_datetime()
        # Normalize datetime for lookup
        dt_naive = dt.tz_convert(None) if dt.tz else dt
        dt_normalized = dt_naive.normalize()

        # Update signals if available for this date
        if dt_normalized in context.state["signal_lookup"]:
            context.state["current_signals"] = context.state["signal_lookup"][dt_normalized]

        if not context.state["current_signals"]:
            return

        # Rank assets by score
        scores = context.state["current_signals"]
        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        top_n = context.state["top_n"]
        bottom_n = context.state["bottom_n"]
        top_assets = set(a for a, _ in sorted_assets[:top_n])
        bottom_assets = set(a for a, _ in sorted_assets[-bottom_n:]) if bottom_n > 0 else set()

        # Rebalance positions
        for asset_name, asset in context.asset_map.items():
            if not data.can_trade(asset):
                continue

            current_pos = context.portfolio.positions[asset].amount

            if asset_name in top_assets:
                target = 100  # Long
            elif asset_name in bottom_assets:
                target = -100  # Short
            else:
                target = 0  # Flat

            if current_pos != target:
                order_target(asset, target)

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        results = run_algorithm(
            start=start_session,
            end=end_session,
            initialize=initialize,
            handle_data=handle_data,
            capital_base=1_000_000.0,
            bundle=bundle_name,
            data_frequency='daily',
        )

        final_value = results['portfolio_value'].iloc[-1]

        end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Extract positions and transactions using pyfolio utilities
        import pyfolio as pf
        returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(results)

        # Convert transactions to completed trades by tracking position changes
        # transactions format: index=datetime, columns=[amount, price, sid/symbol, value]
        # A trade completes when position goes to 0 or flips sign
        trades_df = None
        trade_records = []

        # Get symbol column name (may be 'symbol' or 'sid')
        symbol_col = "symbol" if "symbol" in transactions.columns else "sid" if "sid" in transactions.columns else None

        if len(transactions) > 0 and symbol_col:
            for symbol in transactions[symbol_col].unique():
                symbol_txns = transactions[transactions[symbol_col] == symbol].sort_index()

                running_pos = 0
                entry_time = None
                entry_price = None
                entry_size = 0

                for dt, row in symbol_txns.iterrows():
                    amount = row["amount"]
                    price = row["price"]
                    prev_pos = running_pos
                    running_pos += amount

                    # Check if position just opened
                    if prev_pos == 0 and running_pos != 0:
                        entry_time = dt
                        entry_price = price
                        entry_size = amount

                    # Check if position just closed (or flipped)
                    elif prev_pos != 0 and running_pos == 0:
                        # Position closed completely
                        exit_price = price
                        pnl = (exit_price - entry_price) * entry_size
                        trade_records.append({
                            "entry_date": entry_time,
                            "exit_date": dt,
                            "asset": str(symbol),
                            "side": "long" if entry_size > 0 else "short",
                            "quantity": abs(entry_size),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                        })
                        entry_time = None

                    # Handle position flip (e.g., long -> short or short -> long)
                    elif prev_pos != 0 and running_pos != 0 and (prev_pos > 0) != (running_pos > 0):
                        # Close old position first
                        exit_price = price
                        pnl = (exit_price - entry_price) * entry_size
                        trade_records.append({
                            "entry_date": entry_time,
                            "exit_date": dt,
                            "asset": str(symbol),
                            "side": "long" if entry_size > 0 else "short",
                            "quantity": abs(entry_size),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                        })
                        # Open new position
                        entry_time = dt
                        entry_price = price
                        entry_size = running_pos

        num_trades = len(trade_records)
        if trade_records:
            trades_df = pd.DataFrame(trade_records)
            trades_df = trades_df.sort_values("entry_date")

        return BenchmarkResult(
            framework="Zipline",
            scenario=config.name,
            runtime_sec=end_time - start_time,
            num_trades=num_trades,
            final_value=float(final_value),
            memory_mb=peak / 1024 / 1024,
            trades_df=trades_df,
            positions_df=positions,
            transactions_df=transactions,
        )
    except Exception as e:
        tracemalloc.stop()
        import traceback
        return BenchmarkResult(
            framework="Zipline",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error=f"{e}\n{traceback.format_exc()}",
        )


def benchmark_backtrader(
    config: BenchmarkConfig, price_data: dict, signals: pd.DataFrame, dates
) -> BenchmarkResult:
    """Benchmark Backtrader with given configuration."""
    try:
        import backtrader as bt
    except ImportError:
        return BenchmarkResult(
            framework="Backtrader",
            scenario=config.name,
            runtime_sec=0,
            num_trades=0,
            final_value=0,
            memory_mb=0,
            error="Backtrader not installed",
        )

    # Convert signals to dict for lookup - normalize timestamps for comparison
    signal_lookup = {}
    for _, row in signals.iterrows():
        ts = row["timestamp"]
        # Convert to date string for reliable comparison (Backtrader uses datetime objects)
        ts_key = pd.Timestamp(ts).strftime("%Y-%m-%d")
        if ts_key not in signal_lookup:
            signal_lookup[ts_key] = {}
        signal_lookup[ts_key][row["asset"]] = row["score"]

    class TopBottomBTStrategy(bt.Strategy):
        params = (
            ("top_n", config.top_n),
            ("bottom_n", config.bottom_n),
        )

        def __init__(self):
            self.signal_lookup = signal_lookup
            self.current_signals = {}

        def next(self):
            dt = self.datas[0].datetime.datetime(0)
            dt_key = dt.strftime("%Y-%m-%d")  # Match signal_lookup key format

            # Update signals if available
            if dt_key in self.signal_lookup:
                self.current_signals = self.signal_lookup[dt_key]

            if not self.current_signals:
                return

            # Rank and rebalance
            sorted_assets = sorted(self.current_signals.items(), key=lambda x: x[1], reverse=True)
            top_assets = set(a for a, _ in sorted_assets[: self.p.top_n])
            bottom_assets = (
                set(a for a, _ in sorted_assets[-self.p.bottom_n :])
                if self.p.bottom_n > 0
                else set()
            )

            for i, data in enumerate(self.datas):
                asset_name = data._name
                pos = self.getposition(data)

                if asset_name in top_assets and pos.size <= 0:
                    if pos.size < 0:
                        self.close(data)
                    self.buy(data, size=100)
                elif asset_name in bottom_assets and pos.size >= 0:
                    if pos.size > 0:
                        self.close(data)
                    self.sell(data, size=100)
                elif asset_name not in top_assets and asset_name not in bottom_assets:
                    if pos.size != 0:
                        self.close(data)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(TopBottomBTStrategy)

    # Add data feeds
    for asset_name, df in price_data.items():
        data = bt.feeds.PandasData(dataname=df, name=asset_name)
        cerebro.adddata(data)

    # Use unlimited cash to eliminate margin rejections for exact comparison
    cerebro.broker.setcash(1e15)

    if config.commission_pct > 0:
        cerebro.broker.setcommission(commission=config.commission_pct)

    # Add PyFolio analyzer to get standardized positions/transactions output
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    results = cerebro.run()
    final_value = cerebro.broker.getvalue()

    end_time = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract positions and transactions via PyFolio analyzer
    strat = results[0]
    pyfolio_analyzer = strat.analyzers.getbyname("pyfolio")
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()

    # Convert transactions to completed trades by tracking position changes
    # transactions format: index=datetime, columns=[amount, price, sid, symbol, value]
    # A trade completes when position goes to 0 or flips sign
    trades_df = None
    trade_records = []

    # Group transactions by symbol and process sequentially
    if len(transactions) > 0 and "symbol" in transactions.columns:
        for symbol in transactions["symbol"].unique():
            symbol_txns = transactions[transactions["symbol"] == symbol].sort_index()

            running_pos = 0
            entry_time = None
            entry_price = None
            entry_size = 0

            for dt, row in symbol_txns.iterrows():
                amount = row["amount"]
                price = row["price"]
                prev_pos = running_pos
                running_pos += amount

                # Check if position just opened
                if prev_pos == 0 and running_pos != 0:
                    entry_time = dt
                    entry_price = price
                    entry_size = amount

                # Check if position just closed (or flipped)
                elif prev_pos != 0 and running_pos == 0:
                    # Position closed completely
                    exit_price = price
                    pnl = (exit_price - entry_price) * entry_size
                    trade_records.append({
                        "entry_date": entry_time,
                        "exit_date": dt,
                        "asset": str(symbol),
                        "side": "long" if entry_size > 0 else "short",
                        "quantity": abs(entry_size),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                    })
                    entry_time = None

                # Handle position flip (e.g., long -> short or short -> long)
                elif prev_pos != 0 and running_pos != 0 and (prev_pos > 0) != (running_pos > 0):
                    # Close old position first
                    exit_price = price
                    pnl = (exit_price - entry_price) * entry_size
                    trade_records.append({
                        "entry_date": entry_time,
                        "exit_date": dt,
                        "asset": str(symbol),
                        "side": "long" if entry_size > 0 else "short",
                        "quantity": abs(entry_size),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                    })
                    # Open new position
                    entry_time = dt
                    entry_price = price
                    entry_size = running_pos

    num_trades = len(trade_records)
    if trade_records:
        trades_df = pd.DataFrame(trade_records)
        trades_df = trades_df.sort_values("entry_date")

    return BenchmarkResult(
        framework="Backtrader",
        scenario=config.name,
        runtime_sec=end_time - start_time,
        num_trades=num_trades,
        final_value=final_value,
        memory_mb=peak / 1024 / 1024,
        trades_df=trades_df,
        positions_df=positions,
        transactions_df=transactions,
    )


def run_scenario(scenario_name: str, frameworks: list[str]) -> list[BenchmarkResult]:
    """Run a benchmark scenario across specified frameworks."""
    config = SCENARIOS[scenario_name]
    print(f"\n{'='*70}")
    print(f"Scenario: {config.name}")
    print(f"  Bars: {config.n_bars:,} | Assets: {config.n_assets} | Freq: {config.frequency}")
    print(f"  Long top {config.top_n}, Short bottom {config.bottom_n}")
    if config.stop_loss:
        print(f"  Stop-loss: {config.stop_loss*100:.1f}%")
    if config.take_profit:
        print(f"  Take-profit: {config.take_profit*100:.1f}%")
    if config.commission_pct > 0:
        print(f"  Commission: {config.commission_pct*100:.2f}%")
    if config.slippage_pct > 0:
        print(f"  Slippage: {config.slippage_pct*100:.2f}%")
    print(f"{'='*70}")

    print("\nGenerating data...")
    price_data, signals, dates = generate_benchmark_data(config)
    print(f"  Generated {len(price_data)} assets with {config.n_bars:,} bars each")

    results = []

    for framework in frameworks:
        print(f"\nRunning {framework}...")
        try:
            if framework == "ml4t":
                result = benchmark_ml4t(config, price_data, signals, dates, execution_mode="same_bar")
            elif framework == "ml4t-backtrader":
                # ML4T with Backtrader-compatible settings (next-bar execution)
                result = benchmark_ml4t(config, price_data, signals, dates, execution_mode="next_bar")
            elif framework == "vbt-pro":
                result = benchmark_vectorbt_pro(config, price_data, signals, dates)
            elif framework == "vbt-oss":
                result = benchmark_vectorbt_oss(config, price_data, signals, dates)
            elif framework == "zipline":
                result = benchmark_zipline(config, price_data, signals, dates)
            elif framework == "backtrader":
                result = benchmark_backtrader(config, price_data, signals, dates)
            else:
                result = BenchmarkResult(
                    framework=framework,
                    scenario=config.name,
                    runtime_sec=0,
                    num_trades=0,
                    final_value=0,
                    memory_mb=0,
                    error=f"Unknown framework: {framework}",
                )

            results.append(result)

            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(f"  Runtime: {result.runtime_sec:.3f}s")
                print(f"  Trades: {result.num_trades:,}")
                print(f"  Final value: ${result.final_value:,.2f}")
                print(f"  Memory: {result.memory_mb:.1f} MB")

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                BenchmarkResult(
                    framework=framework,
                    scenario=config.name,
                    runtime_sec=0,
                    num_trades=0,
                    final_value=0,
                    memory_mb=0,
                    error=str(e),
                )
            )

    return results


def print_summary(all_results: list[BenchmarkResult]):
    """Print summary comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Group by scenario
    scenarios = {}
    for r in all_results:
        if r.scenario not in scenarios:
            scenarios[r.scenario] = {}
        scenarios[r.scenario][r.framework] = r

    print(f"{'Scenario':<30} {'Framework':<15} {'Runtime':<12} {'Trades':<10} {'Memory':<10}")
    print("-" * 80)

    for scenario, frameworks in scenarios.items():
        first = True
        for framework, result in frameworks.items():
            scenario_col = scenario if first else ""
            first = False

            if result.error:
                print(f"{scenario_col:<30} {framework:<15} {'ERROR':<12} {'-':<10} {'-':<10}")
            else:
                runtime_str = f"{result.runtime_sec:.3f}s"
                if result.runtime_sec > 60:
                    runtime_str = f"{result.runtime_sec/60:.1f}m"
                print(
                    f"{scenario_col:<30} {framework:<15} {runtime_str:<12} "
                    f"{result.num_trades:<10} {result.memory_mb:.0f} MB"
                )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark suite for backtesting frameworks")
    parser.add_argument(
        "--framework",
        choices=["ml4t", "ml4t-backtrader", "vbt-pro", "vbt-oss", "backtrader", "zipline", "all"],
        default="ml4t",
        help="Framework to benchmark (ml4t-backtrader uses next-bar execution to match Backtrader)",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="baseline",
        help="Scenario to run",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scenarios",
    )
    parser.add_argument(
        "--save-trades",
        action="store_true",
        help="Save trade logs to CSV for validation",
    )
    args = parser.parse_args()

    # Determine frameworks
    if args.framework == "all":
        frameworks = ["ml4t", "vbt-pro", "backtrader"]
    else:
        frameworks = [args.framework]

    # Determine scenarios
    if args.all or args.scenario == "all":
        scenario_names = list(SCENARIOS.keys())
    else:
        scenario_names = [args.scenario]

    print("=" * 70)
    print("Backtesting Framework Benchmark Suite")
    print("=" * 70)
    print(f"Frameworks: {', '.join(frameworks)}")
    print(f"Scenarios: {', '.join(scenario_names)}")

    all_results = []
    output_dir = PROJECT_ROOT / "validation" / "trade_logs"

    for scenario_name in scenario_names:
        results = run_scenario(scenario_name, frameworks)
        all_results.extend(results)

        # Save trade logs if requested
        if args.save_trades:
            for r in results:
                save_trades(r, output_dir)

    print_summary(all_results)

    # Print trade comparison if multiple frameworks
    if len(frameworks) > 1:
        comparisons = compare_trades(all_results)
        if comparisons:
            print("\nTRADE COMPARISON (vs ml4t.backtest baseline)")
            print("-" * 60)
            for framework, comp in comparisons.items():
                print(f"{framework}:")
                print(f"  Trade count diff: {comp['trade_count_diff']:,} ({comp['trade_count_pct']:.2f}%)")
                print(f"  Final value diff: ${comp['final_value_diff']:,.2f} ({comp['final_value_pct']:.2f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
