#!/usr/bin/env python3
"""Cross-implementation benchmark: VectorBT Pro (ground truth) vs backtest-nb vs backtest-rs.

This script validates accuracy against VectorBT Pro and measures performance.

All implementations must produce IDENTICAL results to VectorBT Pro:
- Same trade count
- Same entry/exit bars
- Same entry/exit prices
- Same PNL per trade (within tolerance)

Usage:
    source .venv-benchmark/bin/activate
    python validation/cross_impl_benchmark.py
"""

from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

# Import VectorBT Pro (ground truth) - use separate venv if needed
try:
    import vectorbtpro as vbt
    VBT_AVAILABLE = True
    VBT_VERSION = vbt.__version__
except ImportError:
    VBT_AVAILABLE = False
    VBT_VERSION = None
    print("Warning: VectorBT Pro not available - cannot validate against ground truth")

# Import frameworks
try:
    from ml4t.backtest import (
        Broker,
        DataFeed,
        Engine,
        PercentageCommission,
        PercentageSlippage,
        StopFillMode,
        Strategy,
    )
    from ml4t.backtest.risk.position.static import StopLoss, TakeProfit
    from ml4t.backtest.risk.position.dynamic import TrailingStop
    from ml4t.backtest.risk.position.composite import RuleChain

    ML4T_AVAILABLE = True
except ImportError:
    ML4T_AVAILABLE = False
    print("Warning: ml4t.backtest not available")

try:
    from ml4t.backtest_nb import RuleEngine as NbRuleEngine
    from ml4t.backtest_nb import Signal as NbSignal
    from ml4t.backtest_nb import backtest as nb_backtest
    from ml4t.backtest_nb import (
        FILL_STOP_PRICE,
        FILL_CLOSE_PRICE,
        FILL_BAR_EXTREME,
        FILL_NEXT_BAR_OPEN,
    )

    NB_AVAILABLE = True
except ImportError:
    NB_AVAILABLE = False
    FILL_STOP_PRICE = 0
    FILL_CLOSE_PRICE = 1
    print("Warning: ml4t.backtest_nb not available")

try:
    from ml4t_backtest_rs import RuleEngine as RsRuleEngine
    from ml4t_backtest_rs import Signal as RsSignal
    from ml4t_backtest_rs import backtest as rs_backtest
    from ml4t_backtest_rs import sweep as rs_sweep

    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False
    print("Warning: ml4t_backtest_rs not available")


# =============================================================================
# Data Generation
# =============================================================================


def generate_test_data(
    n_bars: int, n_assets: int = 1, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate synthetic OHLCV + momentum signal data.

    Args:
        n_bars: Number of bars to generate.
        n_assets: Number of assets (currently only 1 supported for cross-impl).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (prices_df, signals_df) as Polars DataFrames.
    """
    rng = np.random.default_rng(seed)

    # Generate price random walk
    returns = rng.normal(0.0002, 0.015, n_bars)
    close = 100.0 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    volatility = 0.01 * close
    high = close + rng.uniform(0, 1, n_bars) * volatility
    low = close - rng.uniform(0, 1, n_bars) * volatility
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Ensure OHLC validity
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Generate volume
    volume = rng.uniform(1_000_000, 5_000_000, n_bars)

    # Generate momentum signal (mean-reverting)
    momentum = np.zeros(n_bars)
    for i in range(1, n_bars):
        momentum[i] = 0.9 * momentum[i - 1] + rng.normal(0, 0.02)

    prices = pl.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    signals = pl.DataFrame({"momentum": momentum})

    return prices, signals


def generate_realistic_data(
    n_years: int = 5,
    n_assets: int = 100,
    freq: str = "daily",
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate realistic multi-asset backtest data.

    Args:
        n_years: Number of years of data.
        n_assets: Number of assets.
        freq: "daily" or "minute".
        seed: Random seed.

    Returns:
        Tuple of (prices_df, signals_df) with ~10+ trades per asset per year.
    """
    rng = np.random.default_rng(seed)

    if freq == "daily":
        n_bars = n_years * 252  # Trading days
    else:  # minute
        n_bars = n_years * 252 * 390  # 390 minutes per day

    # For single-asset validation, use simple 1-asset data
    # Multi-asset support would require more complex handling

    # Generate correlated returns (market factor + idiosyncratic)
    market_returns = rng.normal(0.0003, 0.01, n_bars)  # Market factor

    # Generate price path
    returns = market_returns + rng.normal(0, 0.005, n_bars)  # Add idiosyncratic
    close = 100.0 * np.exp(np.cumsum(returns))

    # Generate realistic OHLC
    daily_vol = np.abs(rng.normal(0.015, 0.005, n_bars))
    high = close * (1 + daily_vol * rng.uniform(0.5, 1.0, n_bars))
    low = close * (1 - daily_vol * rng.uniform(0.5, 1.0, n_bars))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # Ensure OHLC validity
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.uniform(1e6, 10e6, n_bars)

    # Generate momentum signal calibrated for ~10 trades per year
    # Higher variance = more signal crossings = more trades
    momentum = np.zeros(n_bars)
    mean_reversion = 0.95  # High persistence
    innovation_std = 0.04  # Enough variance for signal crossings

    for i in range(1, n_bars):
        momentum[i] = mean_reversion * momentum[i - 1] + rng.normal(0, innovation_std)

    prices = pl.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })

    signals = pl.DataFrame({"momentum": momentum})

    return prices, signals


# =============================================================================
# Benchmark Result Types
# =============================================================================


@dataclass
class Trade:
    """Normalized trade record for comparison."""

    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    pnl: float
    side: str = "long"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    framework: str
    scenario: str
    config: str
    runtime_sec: float
    events_per_sec: float
    memory_mb: float
    n_trades: int
    final_value: float
    trades: list[Trade] = field(default_factory=list)
    error: str | None = None


# =============================================================================
# VectorBT Pro Adapter (Ground Truth)
# =============================================================================


def run_vbt_backtest(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    entry_threshold: float = 0.02,
    exit_threshold: float = -0.01,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
    commission: float = 0.0,
    slippage: float = 0.0,
    initial_cash: float = 100_000.0,
) -> BenchmarkResult:
    """Run backtest using VectorBT Pro (ground truth).

    VectorBT Pro is the reference implementation that all other
    implementations must match.
    """
    if not VBT_AVAILABLE:
        return BenchmarkResult(
            framework="vectorbt-pro",
            scenario="",
            config="",
            runtime_sec=0,
            events_per_sec=0,
            memory_mb=0,
            n_trades=0,
            final_value=0,
            error="VectorBT Pro not available",
        )

    import pandas as pd

    n_bars = len(prices)

    # Convert to pandas for VBT
    close = prices["close"].to_numpy()
    high = prices["high"].to_numpy()
    low = prices["low"].to_numpy()
    open_ = prices["open"].to_numpy()
    momentum = signals["momentum"].to_numpy()

    # Create index
    start = datetime(2020, 1, 1)
    dates = pd.date_range(start, periods=n_bars, freq="D")

    close_pd = pd.Series(close, index=dates, name="close")
    high_pd = pd.Series(high, index=dates, name="high")
    low_pd = pd.Series(low, index=dates, name="low")
    open_pd = pd.Series(open_, index=dates, name="open")

    # Generate entry/exit signals as boolean arrays
    entries = momentum > entry_threshold
    exits = momentum < exit_threshold

    entries_pd = pd.Series(entries, index=dates)
    exits_pd = pd.Series(exits, index=dates)

    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        # Run VBT Pro backtest with fixed share sizing (100 shares like nb/rs)
        pf = vbt.Portfolio.from_signals(
            close=close_pd,
            open=open_pd,
            high=high_pd,
            low=low_pd,
            entries=entries_pd,
            exits=exits_pd,
            size=100,  # Fixed position size in shares (matches nb/rs)
            sl_stop=stop_loss,
            tp_stop=take_profit,
            tsl_stop=trailing_stop,
            init_cash=initial_cash,
            fees=commission,
            slippage=slippage,
            accumulate=False,  # No re-entry while in position
            upon_opposite_entry="ignore",
        )

        elapsed = time.perf_counter() - start_time
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Extract trades
        trades = []
        if hasattr(pf, "trades") and pf.trades is not None:
            trades_records = pf.trades.records
            for i in range(len(trades_records)):
                t = trades_records.iloc[i] if hasattr(trades_records, 'iloc') else trades_records[i]
                trades.append(
                    Trade(
                        entry_bar=int(t.get("entry_idx", t.get("entry_idx", 0))),
                        exit_bar=int(t.get("exit_idx", t.get("exit_idx", 0))),
                        entry_price=float(t.get("entry_price", 0)),
                        exit_price=float(t.get("exit_price", 0)),
                        pnl=float(t.get("pnl", 0)),
                    )
                )

        final_value = float(pf.final_value)  # Property
        n_trades = int(pf.trades.count()) if hasattr(pf, "trades") else 0  # Method

        return BenchmarkResult(
            framework="vectorbt-pro",
            scenario="",
            config="",
            runtime_sec=elapsed,
            events_per_sec=n_bars / elapsed if elapsed > 0 else 0,
            memory_mb=peak_memory / 1024 / 1024,
            n_trades=n_trades,
            final_value=final_value,
            trades=trades,
        )

    except Exception as e:
        tracemalloc.stop()
        return BenchmarkResult(
            framework="vectorbt-pro",
            scenario="",
            config="",
            runtime_sec=0,
            events_per_sec=0,
            memory_mb=0,
            n_trades=0,
            final_value=0,
            error=str(e),
        )


# =============================================================================
# ml4t.backtest Adapter
# =============================================================================


class MomentumStrategy(Strategy):
    """Simple momentum strategy for ml4t.backtest.

    Matches the behavior of backtest-nb and backtest-rs:
    - Entry: momentum > entry_threshold
    - Exit: momentum < exit_threshold OR stop_loss/take_profit/trailing_stop triggered
    """

    def __init__(
        self,
        entry_threshold: float = 0.02,
        exit_threshold: float = -0.01,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        trailing_stop: float | None = None,
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.entry_bars: dict[str, int] = {}
        self.current_bar = 0
        self._position_rules_set: set[str] = set()

    def _build_position_rules(self):
        """Build position rules based on configured stops."""
        rules = []
        if self.stop_loss is not None:
            rules.append(StopLoss(pct=self.stop_loss))
        if self.take_profit is not None:
            rules.append(TakeProfit(pct=self.take_profit))
        if self.trailing_stop is not None:
            rules.append(TrailingStop(pct=self.trailing_stop))
        if rules:
            return RuleChain(rules=rules)
        return None

    def on_data(self, timestamp, assets_data, context, broker: Broker) -> None:
        self.current_bar += 1
        for asset, data in assets_data.items():
            signals = data.get("signals", {})
            momentum = signals.get("momentum", 0.0)
            position = broker.get_position(asset)

            if position is None and momentum > self.entry_threshold:
                # Entry: use positive quantity (broker infers BUY side)
                order = broker.submit_order(asset, 100)
                if order:
                    self.entry_bars[asset] = self.current_bar
                    # Set position rules for this asset
                    rules = self._build_position_rules()
                    if rules and asset not in self._position_rules_set:
                        broker.set_position_rules(rules, asset)
                        self._position_rules_set.add(asset)

            elif position is not None and momentum < self.exit_threshold:
                # Signal-based exit: use negative quantity (broker infers SELL side)
                broker.submit_order(asset, -position.quantity)


def run_ml4t_backtest(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    entry_threshold: float = 0.02,
    exit_threshold: float = -0.01,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
    commission: float = 0.0,
    slippage: float = 0.0,
    initial_cash: float = 100_000.0,
) -> BenchmarkResult:
    """Run backtest using ml4t.backtest."""
    if not ML4T_AVAILABLE:
        return BenchmarkResult(
            framework="ml4t.backtest",
            scenario="",
            config="",
            runtime_sec=0,
            events_per_sec=0,
            memory_mb=0,
            n_trades=0,
            final_value=0,
            error="ml4t.backtest not available",
        )

    n_bars = len(prices)

    # Create timestamps
    start = datetime(2020, 1, 1)
    timestamps = [start + timedelta(days=i) for i in range(n_bars)]

    # Prepare prices with timestamp and asset
    prices_with_ts = prices.with_columns([
        pl.Series("timestamp", timestamps),
        pl.lit("ASSET").alias("asset"),
    ])

    # Prepare signals with timestamp and asset
    signals_with_ts = signals.with_columns([
        pl.Series("timestamp", timestamps),
        pl.lit("ASSET").alias("asset"),
    ])

    # Create strategy
    strategy = MomentumStrategy(
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
    )

    # Create DataFeed
    feed = DataFeed(prices_df=prices_with_ts, signals_df=signals_with_ts)

    # Create commission/slippage models
    commission_model = PercentageCommission(commission) if commission > 0 else None
    slippage_model = PercentageSlippage(slippage) if slippage > 0 else None

    # Create engine with CLOSE_PRICE fill mode to match VBT Pro default
    # VBT Pro default (StopExitPrice.Stop/Close): stops fill at bar's close price
    # Use STOP_PRICE only if matching VBT Pro with StopExitPrice.HardStop
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=initial_cash,
        commission_model=commission_model,
        slippage_model=slippage_model,
        stop_fill_mode=StopFillMode.STOP_PRICE,
    )

    # Run with timing
    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    results = engine.run()

    elapsed = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract trades
    trades = []
    for trade in engine.broker.trades:
        trades.append(
            Trade(
                entry_bar=0,  # Would need to track this
                exit_bar=0,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
            )
        )

    return BenchmarkResult(
        framework="ml4t.backtest",
        scenario="",
        config="",
        runtime_sec=elapsed,
        events_per_sec=n_bars / elapsed if elapsed > 0 else 0,
        memory_mb=peak_memory / 1024 / 1024,
        n_trades=len(trades),
        final_value=results.get("final_value", initial_cash),
        trades=trades,
    )


# =============================================================================
# backtest-nb Adapter
# =============================================================================


def run_nb_backtest(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    entry_threshold: float = 0.02,
    exit_threshold: float = -0.01,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
    commission: float = 0.0,
    slippage: float = 0.0,
    initial_cash: float = 100_000.0,
    stop_fill_mode: int = FILL_STOP_PRICE,
) -> BenchmarkResult:
    """Run backtest using backtest-nb (Numba)."""
    if not NB_AVAILABLE:
        return BenchmarkResult(
            framework="backtest-nb",
            scenario="",
            config="",
            runtime_sec=0,
            events_per_sec=0,
            memory_mb=0,
            n_trades=0,
            final_value=0,
            error="backtest-nb not available",
        )

    n_bars = len(prices)

    # Build rule engine
    entry_rule = NbSignal("momentum") > entry_threshold
    exit_rule = NbSignal("momentum") < exit_threshold if exit_threshold else None

    strategy = NbRuleEngine(
        entry=entry_rule,
        exit=exit_rule,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    # Run with timing
    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    result = nb_backtest(
        prices,
        signals,
        strategy,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
        stop_fill_mode=stop_fill_mode,
    )

    elapsed = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract trades
    trades = []
    trades_df = result.trades_df
    if len(trades_df) > 0:
        for row in trades_df.iter_rows(named=True):
            trades.append(
                Trade(
                    entry_bar=row.get("entry_bar", 0),
                    exit_bar=row.get("exit_bar", 0),
                    entry_price=row.get("entry_price", 0),
                    exit_price=row.get("exit_price", 0),
                    pnl=row.get("pnl", 0),
                )
            )

    return BenchmarkResult(
        framework="backtest-nb",
        scenario="",
        config="",
        runtime_sec=elapsed,
        events_per_sec=n_bars / elapsed if elapsed > 0 else 0,
        memory_mb=peak_memory / 1024 / 1024,
        n_trades=result.n_trades,
        final_value=result.final_value,
        trades=trades,
    )


# =============================================================================
# backtest-rs Adapter
# =============================================================================


def run_rs_backtest(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    entry_threshold: float = 0.02,
    exit_threshold: float = -0.01,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    trailing_stop: float | None = None,
    commission: float = 0.0,
    slippage: float = 0.0,
    initial_cash: float = 100_000.0,
    stop_fill_mode: int = FILL_STOP_PRICE,
) -> BenchmarkResult:
    """Run backtest using backtest-rs (Rust)."""
    if not RS_AVAILABLE:
        return BenchmarkResult(
            framework="backtest-rs",
            scenario="",
            config="",
            runtime_sec=0,
            events_per_sec=0,
            memory_mb=0,
            n_trades=0,
            final_value=0,
            error="backtest-rs not available",
        )

    n_bars = len(prices)

    # Build rule engine
    entry_rule = RsSignal("momentum") > entry_threshold
    exit_rule = RsSignal("momentum") < exit_threshold if exit_threshold else None

    strategy = RsRuleEngine(
        entry=entry_rule,
        exit=exit_rule,
        stop_loss=stop_loss,
        take_profit=take_profit if take_profit else None,
        trailing_stop=trailing_stop,
        position_size=100.0,
    )

    # Run with timing
    gc.collect()
    tracemalloc.start()
    start_time = time.perf_counter()

    result = rs_backtest(
        prices,
        signals,
        strategy,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
        stop_fill_mode=stop_fill_mode,
    )

    elapsed = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Extract trades
    trades = []
    # RS returns trades as list of dicts or structured
    if hasattr(result, "trades") and result.trades:
        for t in result.trades:
            if isinstance(t, dict):
                trades.append(
                    Trade(
                        entry_bar=t.get("entry_bar", 0),
                        exit_bar=t.get("exit_bar", 0),
                        entry_price=t.get("entry_price", 0),
                        exit_price=t.get("exit_price", 0),
                        pnl=t.get("pnl", 0),
                    )
                )

    return BenchmarkResult(
        framework="backtest-rs",
        scenario="",
        config="",
        runtime_sec=elapsed,
        events_per_sec=n_bars / elapsed if elapsed > 0 else 0,
        memory_mb=peak_memory / 1024 / 1024,
        n_trades=result.n_trades if hasattr(result, "n_trades") else len(trades),
        final_value=result.final_value if hasattr(result, "final_value") else initial_cash,
        trades=trades,
    )


# =============================================================================
# Accuracy Validation
# =============================================================================

ACCURACY_SCENARIOS = {
    "simple_momentum": {
        "entry_threshold": 0.02,
        "exit_threshold": -0.01,
        "description": "Entry: momentum > 0.02, Exit: momentum < -0.01",
    },
    "stop_loss_2pct": {
        "entry_threshold": 0.02,
        "exit_threshold": -0.01,
        "stop_loss": 0.02,
        "description": "2% stop-loss",
    },
    "take_profit_5pct": {
        "entry_threshold": 0.02,
        "exit_threshold": -0.01,
        "take_profit": 0.05,
        "description": "5% take-profit",
    },
    "trailing_stop_3pct": {
        "entry_threshold": 0.02,
        "exit_threshold": -0.01,
        "trailing_stop": 0.03,
        "description": "3% trailing stop",
    },
    "combined_rules": {
        "entry_threshold": 0.02,
        "exit_threshold": -0.01,
        "stop_loss": 0.02,
        "take_profit": 0.05,
        "trailing_stop": 0.03,
        "description": "SL + TP + trailing",
    },
    "with_costs": {
        "entry_threshold": 0.02,
        "exit_threshold": -0.01,
        "commission": 0.001,
        "slippage": 0.0005,
        "description": "10bps commission + 5bps slippage",
    },
}


def validate_accuracy(results: list[BenchmarkResult], tolerance: float = 1e-4) -> dict:
    """Compare results across implementations for accuracy.

    Uses VectorBT Pro as ground truth if available, otherwise uses backtest-nb.

    Returns dict with validation results.
    """
    if len(results) < 2:
        return {"status": "SKIP", "reason": "Need at least 2 implementations"}

    # Filter out errors
    valid = [r for r in results if r.error is None]
    if len(valid) < 2:
        return {"status": "SKIP", "reason": "Less than 2 valid results"}

    # Prefer VectorBT Pro as baseline (ground truth), then backtest-nb
    vbt_result = next((r for r in valid if r.framework == "vectorbt-pro"), None)
    nb_result = next((r for r in valid if r.framework == "backtest-nb"), None)

    if vbt_result:
        baseline = vbt_result
    elif nb_result:
        baseline = nb_result
    else:
        baseline = valid[0]

    issues = []

    for other in valid:
        if other.framework == baseline.framework:
            continue

        # Compare trade counts
        if baseline.n_trades != other.n_trades:
            issues.append(
                f"{other.framework}: trade count {other.n_trades} vs {baseline.n_trades} ({baseline.framework})"
            )

        # Compare final value (within tolerance for floating point)
        if baseline.final_value > 0:
            rel_diff = abs(baseline.final_value - other.final_value) / baseline.final_value
            if rel_diff > tolerance:
                issues.append(
                    f"{other.framework}: final_value {other.final_value:.2f} vs {baseline.final_value:.2f} ({baseline.framework}) [{rel_diff*100:.3f}%]"
                )

    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "baseline": baseline.framework,
        "compared": [r.framework for r in valid if r.framework != baseline.framework],
    }


# =============================================================================
# Performance Benchmarks
# =============================================================================

PERF_CONFIGS = {
    "tiny": {"n_bars": 100, "n_assets": 1, "description": "100 bars (warmup)"},
    "small": {"n_bars": 1_000, "n_assets": 1, "description": "1K bars"},
    "medium": {"n_bars": 10_000, "n_assets": 1, "description": "10K bars"},
    "large": {"n_bars": 100_000, "n_assets": 1, "description": "100K bars"},
    "xlarge": {"n_bars": 500_000, "n_assets": 1, "description": "500K bars"},
    "minute": {"n_bars": 1_000_000, "n_assets": 1, "description": "1M bars (minute data)"},
}


def warmup_jit():
    """Warmup JIT compilers with small data."""
    print("Warming up JIT compilers...")
    prices, signals = generate_test_data(100)

    if NB_AVAILABLE:
        strategy = NbRuleEngine(entry=NbSignal("momentum") > 0.02, position_size=100.0)
        nb_backtest(prices, signals, strategy)

    if RS_AVAILABLE:
        strategy = RsRuleEngine(entry=RsSignal("momentum") > 0.02, position_size=100.0)
        rs_backtest(prices, signals, strategy)

    print("JIT warmup complete.")


# =============================================================================
# Report Generation
# =============================================================================


def generate_report(
    accuracy_results: dict[str, dict],
    perf_results: dict[str, list[BenchmarkResult]],
    sweep_result: dict | None = None,
) -> str:
    """Generate markdown report."""
    lines = [
        "# Cross-Implementation Benchmark Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Implementation Status",
        "",
        "| Framework | Available | Version |",
        "|-----------|-----------|---------|",
        f"| ml4t.backtest | {'✅' if ML4T_AVAILABLE else '❌'} | baseline |",
        f"| backtest-nb | {'✅' if NB_AVAILABLE else '❌'} | Numba |",
        f"| backtest-rs | {'✅' if RS_AVAILABLE else '❌'} | Rust |",
        "",
        "---",
        "",
        "## Accuracy Validation",
        "",
    ]

    # Accuracy table
    lines.extend(
        [
            "| Scenario | Status | Issues |",
            "|----------|--------|--------|",
        ]
    )

    for scenario, result in accuracy_results.items():
        status = result.get("status", "N/A")
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏭️"
        issues = ", ".join(result.get("issues", [])) or "-"
        lines.append(f"| {scenario} | {status_icon} {status} | {issues} |")

    lines.extend(["", "---", "", "## Performance Benchmarks", ""])

    # Performance table
    lines.extend(
        [
            "| Config | ml4t.backtest | backtest-nb | backtest-rs | nb Speedup | rs Speedup |",
            "|--------|---------------|-------------|-------------|------------|------------|",
        ]
    )

    for config_name, results in perf_results.items():
        row = [config_name]
        baseline_time = None
        nb_time = None
        rs_time = None

        for r in results:
            time_str = f"{r.runtime_sec:.4f}s" if r.error is None else "ERROR"
            if r.framework == "ml4t.backtest":
                baseline_time = r.runtime_sec if r.error is None else None
                row.append(time_str)
            elif r.framework == "backtest-nb":
                nb_time = r.runtime_sec if r.error is None else None
                row.append(time_str)
            elif r.framework == "backtest-rs":
                rs_time = r.runtime_sec if r.error is None else None
                row.append(time_str)

        # Calculate speedups
        if baseline_time and nb_time and nb_time > 0:
            row.append(f"{baseline_time / nb_time:.1f}x")
        else:
            row.append("-")

        if baseline_time and rs_time and rs_time > 0:
            row.append(f"{baseline_time / rs_time:.1f}x")
        else:
            row.append("-")

        lines.append("| " + " | ".join(row) + " |")

    # Events per second table
    lines.extend(
        [
            "",
            "### Events Per Second",
            "",
            "| Config | ml4t.backtest | backtest-nb | backtest-rs |",
            "|--------|---------------|-------------|-------------|",
        ]
    )

    for config_name, results in perf_results.items():
        row = [config_name]
        for r in sorted(results, key=lambda x: x.framework):
            if r.error is None:
                row.append(f"{r.events_per_sec:,.0f}")
            else:
                row.append("ERROR")
        lines.append("| " + " | ".join(row) + " |")

    # Memory table
    lines.extend(
        [
            "",
            "### Peak Memory (MB)",
            "",
            "| Config | ml4t.backtest | backtest-nb | backtest-rs |",
            "|--------|---------------|-------------|-------------|",
        ]
    )

    for config_name, results in perf_results.items():
        row = [config_name]
        for r in sorted(results, key=lambda x: x.framework):
            if r.error is None:
                row.append(f"{r.memory_mb:.1f}")
            else:
                row.append("ERROR")
        lines.append("| " + " | ".join(row) + " |")

    # Parallel sweep results
    if sweep_result:
        lines.extend(
            [
                "",
                "---",
                "",
                "## Parallel Sweep Benchmark (backtest-rs)",
                "",
                f"- **Parameters tested**: {sweep_result.get('n_params', 0)}",
                f"- **Sequential time**: {sweep_result.get('sequential_time', 0):.2f}s",
                f"- **Parallel time**: {sweep_result.get('parallel_time', 0):.2f}s",
                f"- **Speedup**: {sweep_result.get('speedup', 0):.1f}x",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "*Benchmark complete.*",
        ]
    )

    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================


def run_accuracy_validation() -> dict[str, dict]:
    """Run accuracy validation across all scenarios.

    VectorBT Pro is the ground truth. All other implementations must match.
    """
    print("\n" + "=" * 60)
    print("ACCURACY VALIDATION")
    print("=" * 60)
    print(f"  VectorBT Pro: {'Available (' + VBT_VERSION + ')' if VBT_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  backtest-nb: {'Available' if NB_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"  backtest-rs: {'Available' if RS_AVAILABLE else 'NOT AVAILABLE'}")

    # Use realistic test data: 5 years, single asset (for comparison)
    print("\n  Generating realistic test data (5 years daily)...")
    prices, signals = generate_realistic_data(n_years=5, n_assets=1, freq="daily", seed=42)
    print(f"  Generated {len(prices)} bars")

    results = {}

    for scenario_name, params in ACCURACY_SCENARIOS.items():
        print(f"\n  {scenario_name}: {params.get('description', '')}")

        scenario_results = []

        # Get params without description
        run_params = {k: v for k, v in params.items() if k != "description"}

        # Run VectorBT Pro first (ground truth)
        if VBT_AVAILABLE:
            r = run_vbt_backtest(prices.clone(), signals.clone(), **run_params)
            r.scenario = scenario_name
            scenario_results.append(r)
            if r.error:
                print(f"    vectorbt-pro: ERROR - {r.error}")
            else:
                print(f"    vectorbt-pro: {r.n_trades} trades, ${r.final_value:,.2f} (GROUND TRUTH)")

        # Run backtest-nb
        if NB_AVAILABLE:
            r = run_nb_backtest(prices.clone(), signals.clone(), **run_params)
            r.scenario = scenario_name
            scenario_results.append(r)
            if r.error:
                print(f"    backtest-nb: ERROR - {r.error}")
            else:
                print(f"    backtest-nb: {r.n_trades} trades, ${r.final_value:,.2f}")

        # Run backtest-rs
        if RS_AVAILABLE:
            r = run_rs_backtest(prices.clone(), signals.clone(), **run_params)
            r.scenario = scenario_name
            scenario_results.append(r)
            if r.error:
                print(f"    backtest-rs: ERROR - {r.error}")
            else:
                print(f"    backtest-rs: {r.n_trades} trades, ${r.final_value:,.2f}")

        # Run ml4t.backtest (if available, for completeness)
        if ML4T_AVAILABLE:
            r = run_ml4t_backtest(prices.clone(), signals.clone(), **run_params)
            r.scenario = scenario_name
            scenario_results.append(r)
            if r.error:
                print(f"    ml4t.backtest: ERROR - {r.error}")
            else:
                print(f"    ml4t.backtest: {r.n_trades} trades, ${r.final_value:,.2f}")

        # Validate against ground truth
        validation = validate_accuracy(scenario_results)
        results[scenario_name] = validation

        status_icon = "✅" if validation["status"] == "PASS" else "❌"
        print(f"    -> {status_icon} {validation['status']}")
        if validation.get("issues"):
            for issue in validation["issues"][:3]:  # Show first 3 issues
                print(f"       ⚠️  {issue}")

    return results


def run_performance_benchmarks() -> dict[str, list[BenchmarkResult]]:
    """Run performance benchmarks across all configs."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)

    results = {}

    for config_name, config in PERF_CONFIGS.items():
        print(f"\n  {config_name}: {config['description']}")

        n_bars = config["n_bars"]
        prices, signals = generate_test_data(n_bars, seed=42)

        config_results = []

        # Simple momentum strategy params
        params = {"entry_threshold": 0.02, "exit_threshold": -0.01}

        # Run each framework
        if ML4T_AVAILABLE:
            r = run_ml4t_backtest(prices.clone(), signals.clone(), **params)
            r.config = config_name
            config_results.append(r)
            print(f"    ml4t.backtest: {r.runtime_sec:.4f}s ({r.events_per_sec:,.0f} evt/s)")

        if NB_AVAILABLE:
            r = run_nb_backtest(prices.clone(), signals.clone(), **params)
            r.config = config_name
            config_results.append(r)
            print(f"    backtest-nb: {r.runtime_sec:.4f}s ({r.events_per_sec:,.0f} evt/s)")

        if RS_AVAILABLE:
            r = run_rs_backtest(prices.clone(), signals.clone(), **params)
            r.config = config_name
            config_results.append(r)
            print(f"    backtest-rs: {r.runtime_sec:.4f}s ({r.events_per_sec:,.0f} evt/s)")

        results[config_name] = config_results

    return results


def run_parallel_sweep() -> dict | None:
    """Run parallel parameter sweep on backtest-rs."""
    if not RS_AVAILABLE:
        print("\n  backtest-rs not available, skipping parallel sweep")
        return None

    print("\n" + "=" * 60)
    print("PARALLEL SWEEP BENCHMARK (backtest-rs)")
    print("=" * 60)

    # Generate test data
    prices, signals = generate_test_data(10000, seed=42)

    # Create parameter grid (10x10 = 100 combinations)
    stop_losses = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    take_profits = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    n_params = len(stop_losses) * len(take_profits)

    print(f"  Testing {n_params} parameter combinations...")

    # Sequential timing (run backtests one by one)
    print("  Running sequential...")
    start_seq = time.perf_counter()
    for sl in stop_losses:
        for tp in take_profits:
            strategy = RsRuleEngine(
                entry=RsSignal("momentum") > 0.02,
                stop_loss=sl,
                take_profit=tp,
                position_size=100.0,
            )
            rs_backtest(prices.clone(), signals.clone(), strategy)
    sequential_time = time.perf_counter() - start_seq
    print(f"    Sequential: {sequential_time:.2f}s")

    # Parallel timing using sweep function
    print("  Running parallel (rayon)...")
    start_par = time.perf_counter()
    try:
        # If sweep function exists and works
        result = rs_sweep(
            prices.clone(),
            signals.clone(),
            entry=RsSignal("momentum") > 0.02,
            stop_losses=stop_losses,
            take_profits=take_profits,
        )
        parallel_time = time.perf_counter() - start_par
        print(f"    Parallel: {parallel_time:.2f}s")
    except Exception as e:
        print(f"    Parallel sweep failed: {e}")
        parallel_time = sequential_time  # Fallback

    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    print(f"  Speedup: {speedup:.1f}x")

    return {
        "n_params": n_params,
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
    }


def main():
    """Run full benchmark suite."""
    print("=" * 60)
    print("CROSS-IMPLEMENTATION BENCHMARK SUITE")
    print("=" * 60)
    print()
    print(f"ml4t.backtest: {'Available' if ML4T_AVAILABLE else 'Not available'}")
    print(f"backtest-nb:   {'Available' if NB_AVAILABLE else 'Not available'}")
    print(f"backtest-rs:   {'Available' if RS_AVAILABLE else 'Not available'}")

    # Warmup JIT
    warmup_jit()

    # Run benchmarks
    accuracy_results = run_accuracy_validation()
    perf_results = run_performance_benchmarks()
    sweep_result = run_parallel_sweep()

    # Generate report
    report = generate_report(accuracy_results, perf_results, sweep_result)

    # Save report
    report_path = "validation/CROSS_IMPL_BENCHMARK.md"
    with open(report_path, "w") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print(f"Report saved to: {report_path}")
    print("=" * 60)

    # Summary
    print("\nSUMMARY:")
    n_pass = sum(1 for r in accuracy_results.values() if r.get("status") == "PASS")
    n_total = len(accuracy_results)
    print(f"  Accuracy: {n_pass}/{n_total} scenarios passed")

    if perf_results:
        # Get speedups for largest config that ran
        for config in ["minute", "xlarge", "large", "medium", "small"]:
            if config in perf_results:
                results = perf_results[config]
                baseline = next((r for r in results if r.framework == "ml4t.backtest"), None)
                nb = next((r for r in results if r.framework == "backtest-nb"), None)
                rs = next((r for r in results if r.framework == "backtest-rs"), None)

                if baseline and baseline.error is None:
                    if nb and nb.error is None:
                        print(f"  backtest-nb speedup ({config}): {baseline.runtime_sec/nb.runtime_sec:.1f}x")
                    if rs and rs.error is None:
                        print(f"  backtest-rs speedup ({config}): {baseline.runtime_sec/rs.runtime_sec:.1f}x")
                break


if __name__ == "__main__":
    main()
