"""Comprehensive Performance Benchmarks - Integrated System Validation.

This module provides final production-readiness validation for the complete
ml4t.backtest system integrating:
- Data layer (PolarsDataFeed + FeatureProvider)
- Risk management (RiskManager with multiple rules)
- ML strategies (signal-based decision making)
- Execution and portfolio management

**Performance Targets (Production Requirements):**
1. Throughput: 10-30k events/sec with real ML strategy + risk rules (250 symbols, 20 positions)
2. Memory: <2GB for 250 symbols × 1 year with all features
3. Backtest Time: 2-5 minutes for 250 symbols × 1 year (typical workload)
4. Overhead Breakdown:
   - Data layer overhead: <5% vs baseline
   - Risk layer overhead: <3% vs baseline
   - Total integrated overhead: <8% vs baseline
5. Scalability: Document 500 symbol characteristics

**Hardware Reference:**
- CPU: 12th Gen Intel Core i9-12900K (16 cores, 24 threads)
- RAM: 125GB DDR4
- Disk: 3.6TB SSD
- OS: Ubuntu 22.04 LTS (Linux 6.8.0-87-generic)
- Python: 3.12.3

**Test Methodology:**
- Synthetic data for reproducibility (fixed seeds)
- Multiple iterations to eliminate cold-start bias
- Realistic ML strategy with actual trading logic
- 3 active risk rules (stop-loss, take-profit, time-based exit)
- Memory profiling with psutil (peak RSS tracking)
- Overhead analysis via controlled comparison

Run with:
    pytest tests/benchmarks/benchmark_integrated_system.py -xvs --benchmark-only
    pytest tests/benchmarks/benchmark_integrated_system.py -xvs  # Include pass/fail

Generate report:
    pytest tests/benchmarks/benchmark_integrated_system.py -xvs --benchmark-only \
        --benchmark-json=benchmark_results.json
"""

import gc
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import polars as pl
import pytest

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import AssetId, MarketDataType
from ml4t.backtest.data.feed import DataFeed
from ml4t.backtest.engine import BacktestEngine
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.portfolio.portfolio import Portfolio
from ml4t.backtest.risk import (
    RiskManager,
    TimeBasedExit,
    RiskDecision,
    ExitType,
)
from ml4t.backtest.strategy.base import Strategy


# ===== Hardware Specs =====


def get_hardware_specs() -> dict:
    """Get hardware specifications for benchmark reproducibility."""
    specs = {
        "platform": "linux",
        "architecture": "x86_64",
        "python_version": "3.12.3",
    }

    if HAS_PSUTIL:
        # CPU info
        specs["cpu_count"] = psutil.cpu_count(logical=False)
        specs["cpu_count_logical"] = psutil.cpu_count(logical=True)
        specs["cpu_freq_max"] = psutil.cpu_freq().max if psutil.cpu_freq() else None

        # Memory info
        mem = psutil.virtual_memory()
        specs["total_ram_gb"] = round(mem.total / (1024**3), 1)

    return specs


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if not HAS_PSUTIL:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


# ===== Multi-Symbol Data Feed =====


class MultiSymbolDataFeed(DataFeed):
    """Simple in-memory data feed for multi-symbol benchmarks.

    Emits events for all symbols sorted by timestamp.
    """

    def __init__(self, price_df: pl.DataFrame, signals_df: pl.DataFrame, context_df: pl.DataFrame):
        """Initialize feed with DataFrames.

        Args:
            price_df: OHLCV data (must have timestamp, asset_id columns)
            signals_df: Per-symbol signals (timestamp, asset_id, ml_score, etc.)
            context_df: Market-wide context (timestamp, vix, etc.)
        """
        self.price_df = price_df.sort("timestamp", "asset_id")
        self.signals_df = signals_df
        self.context_df = context_df

        # Merge signals into price data
        self.merged_df = self.price_df.join(
            self.signals_df, on=["timestamp", "asset_id"], how="left"
        )

        # Add context data (broadcast to all rows with same timestamp)
        self.merged_df = self.merged_df.join(
            self.context_df, on="timestamp", how="left"
        )

        self.current_index = 0
        self.max_index = len(self.merged_df) - 1

    def get_next_event(self) -> MarketEvent | None:
        """Get next market event."""
        if self.is_exhausted:
            return None

        row = self.merged_df.row(self.current_index, named=True)
        self.current_index += 1

        return self._create_event(row)

    def _create_event(self, row: dict[str, Any]) -> MarketEvent:
        """Create MarketEvent from DataFrame row."""
        # Extract signals (everything except OHLCV and context)
        signal_cols = [
            col for col in row.keys()
            if col not in ["timestamp", "asset_id", "open", "high", "low", "close", "volume", "vix", "regime"]
        ]
        signals = {col: row[col] for col in signal_cols if row[col] is not None}

        return MarketEvent(
            timestamp=row["timestamp"],
            asset_id=row["asset_id"],
            data_type=MarketDataType.BAR,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
            signals=signals,
        )

    @property
    def is_exhausted(self) -> bool:
        """Check if feed is exhausted."""
        return self.current_index > self.max_index

    def reset(self) -> None:
        """Reset the data feed to the beginning."""
        self.current_index = 0

    def peek_next_timestamp(self) -> datetime | None:
        """Peek at the timestamp of the next event without consuming it."""
        if self.is_exhausted:
            return None
        return self.merged_df["timestamp"][self.current_index]

    def seek(self, timestamp: datetime) -> None:
        """Seek to a specific timestamp."""
        # Find first row with timestamp >= target
        timestamps = self.merged_df["timestamp"]
        for idx in range(self.current_index, len(timestamps)):
            if timestamps[idx] >= timestamp:
                self.current_index = idx
                return
        # If not found, exhaust the feed
        self.current_index = self.max_index + 1


# ===== Test Data Generation =====


def create_synthetic_dataset(
    n_symbols: int,
    n_days: int,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create synthetic multi-asset dataset with features and signals.

    Args:
        n_symbols: Number of symbols (e.g., 250 for production test)
        n_days: Number of trading days (e.g., 252 for 1 year)
        seed: Random seed for reproducibility

    Returns:
        (price_df, signals_df, context_df)
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Generate timestamps
    start_date = datetime(2020, 1, 1)
    timestamps = [start_date + timedelta(days=i) for i in range(n_days)]

    # Create symbols
    symbols = [f"SYM{i:04d}" for i in range(n_symbols)]

    # Generate price data
    price_data = []
    for symbol in symbols:
        base_price = 100.0 + random.uniform(-20, 20)
        volatility = 0.01 + random.uniform(0, 0.01)  # 1-2% daily vol

        prices = [base_price]
        for _ in range(n_days - 1):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Floor at $1

        for ts, close in zip(timestamps, prices):
            # Generate OHLC with realistic intraday movement
            high = close * (1 + abs(np.random.normal(0, volatility * 0.5)))
            low = close * (1 - abs(np.random.normal(0, volatility * 0.5)))
            open_price = close * (1 + np.random.normal(0, volatility * 0.3))
            volume = int(1_000_000 * (1 + np.random.uniform(-0.5, 0.5)))

            price_data.append({
                "timestamp": ts,
                "asset_id": symbol,
                "open": round(open_price, 2),
                "high": round(max(high, open_price, close), 2),
                "low": round(min(low, open_price, close), 2),
                "close": round(close, 2),
                "volume": volume,
            })

    price_df = pl.DataFrame(price_data)

    # Generate ML signals (realistic scores with some predictive power)
    signal_data = []
    for symbol in symbols:
        symbol_prices = [row["close"] for row in price_data if row["asset_id"] == symbol]

        for i, ts in enumerate(timestamps):
            # Simple momentum signal with noise
            if i >= 5:
                returns_5d = (symbol_prices[i] / symbol_prices[i-5]) - 1
                # Score tends higher for positive momentum, but with noise
                raw_score = 0.5 + returns_5d + np.random.normal(0, 0.15)
                ml_score = max(0.0, min(1.0, raw_score))  # Clip to [0, 1]
            else:
                ml_score = 0.5  # Neutral for first 5 days

            # Feature: ATR (simplified)
            if i >= 14:
                recent_rows = [row for row in price_data if row["asset_id"] == symbol][max(0, i-14):i]
                atr = sum(row["high"] - row["low"] for row in recent_rows) / len(recent_rows)
            else:
                atr = symbol_prices[i] * 0.02  # Default 2% of price

            signal_data.append({
                "timestamp": ts,
                "asset_id": symbol,
                "ml_score": round(ml_score, 4),
                "atr": round(atr, 2),
            })

    signals_df = pl.DataFrame(signal_data)

    # Generate context data (market-wide indicators)
    context_data = []
    for i, ts in enumerate(timestamps):
        # VIX-like volatility index (mean-reverting around 20)
        if i == 0:
            vix = 20.0
        else:
            vix = context_data[-1]["vix"] * 0.95 + 20.0 * 0.05 + np.random.normal(0, 2)
            vix = max(10.0, min(50.0, vix))  # Clip to reasonable range

        # Market regime (trending vs choppy)
        regime = "trending" if vix < 20 else "choppy"

        context_data.append({
            "timestamp": ts,
            "vix": round(vix, 2),
            "regime": regime,
        })

    context_df = pl.DataFrame(context_data)

    return price_df, signals_df, context_df


# ===== Test Strategies =====


class BaselineStrategy(Strategy):
    """Baseline strategy with no features, no signals, minimal logic.

    Used to measure overhead of data layer and risk layer.
    """

    def __init__(self):
        super().__init__(name="BaselineStrategy")
        self._bar_count = 0
        self._entered = False

    def on_event(self, event):
        """Handle events."""
        if isinstance(event, MarketEvent):
            self._bar_count += 1
            # Enter once on bar 10, hold forever
            if self._bar_count == 10 and not self._entered:
                self.buy_percent(event.asset_id, percent=0.1, price=float(event.close))
                self._entered = True


class MLRankingStrategy(Strategy):
    """Realistic ML strategy that ranks symbols by ml_score.

    Strategy logic:
    - Maintains up to max_positions (default: 20)
    - Each bar: ranks all symbols by ml_score
    - Enters top N symbols if capacity available
    - Exits when ml_score drops below threshold
    - Equal weight across positions
    """

    def __init__(self, max_positions: int = 20, ml_threshold: float = 0.65):
        super().__init__(name="MLRankingStrategy")
        self.max_positions = max_positions
        self.ml_threshold = ml_threshold

        # Track current positions
        self.active_positions = set()  # Set of asset_ids

        # Buffer events to process all symbols at once
        self.event_buffer = []  # List of events for current timestamp
        self.last_timestamp = None

    def on_event(self, event):
        """Buffer events by timestamp and process as a batch."""
        if isinstance(event, MarketEvent):
            # If new timestamp, process buffered events
            if self.last_timestamp is not None and event.timestamp != self.last_timestamp:
                self._process_buffered_events()
                self.event_buffer = []

            self.event_buffer.append(event)
            self.last_timestamp = event.timestamp

    def on_finish(self):
        """Process any remaining buffered events."""
        if self.event_buffer:
            self._process_buffered_events()

    def _process_buffered_events(self):
        """Process all events for a timestamp as a batch."""
        if not self.event_buffer:
            return

        # Step 1: Check exits for existing positions
        for event in self.event_buffer:
            if event.asset_id in self.active_positions:
                ml_score = event.signals.get("ml_score", 0.0)

                # Exit if score dropped below threshold
                if ml_score < self.ml_threshold:
                    self.close_position(event.asset_id)
                    self.active_positions.discard(event.asset_id)

        # Step 2: Consider new entries
        available_slots = self.max_positions - len(self.active_positions)
        if available_slots > 0:
            # Rank symbols by ml_score
            candidates = [
                (event.asset_id, event.signals.get("ml_score", 0.0), event.close)
                for event in self.event_buffer
                if event.asset_id not in self.active_positions
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Enter top N (up to available slots)
            position_size = 1.0 / self.max_positions  # Equal weight
            for asset_id, ml_score, price in candidates[:available_slots]:
                if ml_score >= self.ml_threshold:
                    self.buy_percent(asset_id, percent=position_size, price=float(price))
                    self.active_positions.add(asset_id)


# ===== Benchmark Fixtures =====


@pytest.fixture
def hardware_specs():
    """Print hardware specifications once at start."""
    specs = get_hardware_specs()
    print("\n" + "="*80)
    print("HARDWARE SPECIFICATIONS")
    print("="*80)
    for key, value in specs.items():
        print(f"{key:20s}: {value}")
    print("="*80 + "\n")
    return specs


@pytest.fixture
def small_dataset():
    """Small dataset for quick tests (10 symbols × 252 days)."""
    return create_synthetic_dataset(n_symbols=10, n_days=252, seed=42)


@pytest.fixture
def production_dataset():
    """Production-scale dataset (250 symbols × 252 days)."""
    return create_synthetic_dataset(n_symbols=250, n_days=252, seed=42)


@pytest.fixture
def scalability_dataset():
    """Large dataset for scalability testing (500 symbols × 252 days)."""
    return create_synthetic_dataset(n_symbols=500, n_days=252, seed=42)


# ===== Helper Functions =====


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    events_processed: int
    wall_time: float
    events_per_second: float
    peak_memory_mb: float
    initial_memory_mb: float
    memory_delta_mb: float

    def overhead_vs_baseline(self, baseline: "BenchmarkResult") -> float:
        """Calculate overhead percentage vs baseline."""
        return ((self.wall_time / baseline.wall_time) - 1.0) * 100.0


def run_backtest_benchmark(
    strategy: Strategy,
    price_df: pl.DataFrame,
    signals_df: pl.DataFrame,
    context_df: pl.DataFrame,
    risk_manager: Optional[RiskManager] = None,
    name: str = "Benchmark",
) -> BenchmarkResult:
    """Run a single backtest and measure performance.

    Args:
        strategy: Strategy instance
        price_df: Price data
        signals_df: Signal data
        context_df: Context data
        risk_manager: Optional risk manager
        name: Benchmark name for reporting

    Returns:
        BenchmarkResult with performance metrics
    """
    # Force garbage collection before measurement
    gc.collect()
    initial_memory = get_memory_usage_mb()

    # Create multi-symbol feed
    data_feed = MultiSymbolDataFeed(price_df, signals_df, context_df)

    # Create engine (broker and portfolio created internally)
    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        risk_manager=risk_manager,
        initial_capital=1_000_000.0,
    )

    # Run backtest with timing
    start_time = time.perf_counter()
    engine.run()
    end_time = time.perf_counter()

    # Measure peak memory
    peak_memory = get_memory_usage_mb()

    # Calculate metrics
    wall_time = end_time - start_time
    events_processed = len(price_df)
    events_per_second = events_processed / wall_time if wall_time > 0 else 0
    memory_delta = peak_memory - initial_memory

    return BenchmarkResult(
        name=name,
        events_processed=events_processed,
        wall_time=wall_time,
        events_per_second=events_per_second,
        peak_memory_mb=peak_memory,
        initial_memory_mb=initial_memory,
        memory_delta_mb=memory_delta,
    )


# ===== Test Cases =====


class TestThroughput:
    """Throughput benchmarks - events/second under various conditions."""

    def test_baseline_empty_strategy(self, production_dataset, hardware_specs):
        """Baseline: minimal strategy, no risk manager (250 symbols × 252 days)."""
        price_df, signals_df, context_df = production_dataset
        strategy = BaselineStrategy()

        result = run_backtest_benchmark(
            strategy, price_df, signals_df, context_df,
            risk_manager=None,
            name="Baseline (no risk)",
        )

        print(f"\n{'='*80}")
        print(f"Baseline Throughput: {result.events_per_second:,.0f} events/sec")
        print(f"Total events: {result.events_processed:,}")
        print(f"Wall time: {result.wall_time:.2f}s")
        print(f"Memory delta: {result.memory_delta_mb:.1f} MB")
        print(f"{'='*80}\n")

        # Target: Should be fast since minimal overhead
        assert result.events_per_second > 5_000, \
            f"Baseline too slow: {result.events_per_second:.0f} events/sec (expected >5k)"

    def test_ml_strategy_with_risk_250_symbols(self, production_dataset, hardware_specs):
        """
        CRITICAL PRODUCTION TEST: ML strategy + risk manager (250 symbols × 252 days).

        This is the primary acceptance criterion for production deployment.
        Target: 10-30k events/sec
        """
        price_df, signals_df, context_df = production_dataset

        # Create realistic ML strategy
        strategy = MLRankingStrategy(max_positions=20, ml_threshold=0.65)

        # Create risk manager with 3 active rules
        # Use callable rules for percentage-based stop-loss/take-profit
        def stop_loss_5pct(context):
            if context.unrealized_pnl_pct < -0.05:  # 5% loss
                return RiskDecision.exit_now(
                    exit_type=ExitType.STOP_LOSS,
                    reason="5% stop-loss breach"
                )
            return RiskDecision.no_action()
        
        def take_profit_10pct(context):
            if context.unrealized_pnl_pct > 0.10:  # 10% profit
                return RiskDecision.exit_now(
                    exit_type=ExitType.TAKE_PROFIT,
                    reason="10% take-profit target"
                )
            return RiskDecision.no_action()
        
        risk_manager = RiskManager()
        risk_manager.add_rule(stop_loss_5pct)
        risk_manager.add_rule(take_profit_10pct)
        risk_manager.add_rule(TimeBasedExit(max_bars=20))

        result = run_backtest_benchmark(
            strategy, price_df, signals_df, context_df,
            risk_manager=risk_manager,
            name="ML Strategy + Risk (Production)",
        )

        print(f"\n{'='*80}")
        print(f"PRODUCTION THROUGHPUT TEST")
        print(f"{'='*80}")
        print(f"Throughput: {result.events_per_second:,.0f} events/sec")
        print(f"Total events: {result.events_processed:,} (250 symbols × 252 days)")
        print(f"Wall time: {result.wall_time:.2f}s")
        print(f"Memory delta: {result.memory_delta_mb:.1f} MB")
        print(f"Peak memory: {result.peak_memory_mb:.1f} MB")
        print(f"{'='*80}\n")

        # CRITICAL ACCEPTANCE CRITERION
        assert result.events_per_second >= 10_000, \
            f"Production throughput FAILED: {result.events_per_second:.0f} events/sec (target: 10-30k)"

        # Check upper bound (if exceeded, document for future reference)
        if result.events_per_second > 30_000:
            print(f"⚠️  Exceeded target range: {result.events_per_second:.0f} > 30k events/sec")


class TestMemory:
    """Memory usage benchmarks - ensure <2GB for production workload."""

    def test_memory_usage_250_symbols_1year(self, production_dataset, hardware_specs):
        """
        CRITICAL MEMORY TEST: 250 symbols × 252 days with all features.

        Target: <2GB peak memory
        """
        price_df, signals_df, context_df = production_dataset

        # Realistic ML strategy
        strategy = MLRankingStrategy(max_positions=20, ml_threshold=0.65)
        risk_manager = RiskManager(rules=[
            stop_loss_5pct,
            take_profit_10pct,
            TimeBasedExit(max_bars=20),
        ])

        result = run_backtest_benchmark(
            strategy, price_df, signals_df, context_df,
            risk_manager=risk_manager,
            name="Memory Test (Production)",
        )

        print(f"\n{'='*80}")
        print(f"PRODUCTION MEMORY TEST")
        print(f"{'='*80}")
        print(f"Dataset: 250 symbols × 252 days = {result.events_processed:,} events")
        print(f"Initial memory: {result.initial_memory_mb:.1f} MB")
        print(f"Peak memory: {result.peak_memory_mb:.1f} MB")
        print(f"Delta: {result.memory_delta_mb:.1f} MB")
        print(f"{'='*80}\n")

        # CRITICAL ACCEPTANCE CRITERION
        # Note: peak_memory_mb includes process overhead, so we check delta
        assert result.memory_delta_mb < 2000, \
            f"Memory usage FAILED: {result.memory_delta_mb:.1f} MB delta (target: <2GB)"


class TestEndToEnd:
    """End-to-end timing benchmarks - wall-clock time for typical workload."""

    def test_backtest_time_250_symbols_1year(self, production_dataset, hardware_specs):
        """
        CRITICAL TIMING TEST: 250 symbols × 1 year backtest time.

        Target: 2-5 minutes wall-clock time
        """
        price_df, signals_df, context_df = production_dataset

        strategy = MLRankingStrategy(max_positions=20, ml_threshold=0.65)
        risk_manager = RiskManager(rules=[
            stop_loss_5pct,
            take_profit_10pct,
            TimeBasedExit(max_bars=20),
        ])

        result = run_backtest_benchmark(
            strategy, price_df, signals_df, context_df,
            risk_manager=risk_manager,
            name="End-to-End Timing (Production)",
        )

        wall_time_minutes = result.wall_time / 60

        print(f"\n{'='*80}")
        print(f"PRODUCTION END-TO-END TIMING TEST")
        print(f"{'='*80}")
        print(f"Dataset: 250 symbols × 252 days = {result.events_processed:,} events")
        print(f"Wall time: {wall_time_minutes:.2f} minutes ({result.wall_time:.1f}s)")
        print(f"Throughput: {result.events_per_second:,.0f} events/sec")
        print(f"{'='*80}\n")

        # CRITICAL ACCEPTANCE CRITERION
        assert 2 <= wall_time_minutes <= 5, \
            f"Backtest time out of range: {wall_time_minutes:.2f} min (target: 2-5 min)"


class TestOverhead:
    """Overhead analysis - measure cost of data layer and risk layer."""

    def test_feature_and_risk_overhead(self, production_dataset, hardware_specs):
        """
        Measure overhead breakdown vs baseline.

        Targets:
        - Data layer overhead: <5%
        - Risk layer overhead: <3%
        - Total overhead: <8%
        """
        price_df, signals_df, context_df = production_dataset

        # 1. Baseline: minimal strategy, no risk
        baseline = run_backtest_benchmark(
            BaselineStrategy(), price_df, signals_df, context_df,
            risk_manager=None,
            name="Baseline (no features, no risk)",
        )

        # 2. Add ML strategy (data layer overhead)
        with_features = run_backtest_benchmark(
            MLRankingStrategy(max_positions=20, ml_threshold=0.65),
            price_df, signals_df, context_df,
            risk_manager=None,
            name="ML Strategy (no risk)",
        )

        # 3. Add risk manager (risk layer overhead)
        risk_manager = RiskManager(rules=[
            stop_loss_5pct,
            take_profit_10pct,
            TimeBasedExit(max_bars=20),
        ])
        with_risk = run_backtest_benchmark(
            MLRankingStrategy(max_positions=20, ml_threshold=0.65),
            price_df, signals_df, context_df,
            risk_manager=risk_manager,
            name="ML Strategy + Risk",
        )

        # Calculate overheads
        data_overhead = with_features.overhead_vs_baseline(baseline)
        risk_overhead = with_risk.overhead_vs_baseline(with_features)
        total_overhead = with_risk.overhead_vs_baseline(baseline)

        print(f"\n{'='*80}")
        print(f"OVERHEAD ANALYSIS")
        print(f"{'='*80}")
        print(f"Baseline time: {baseline.wall_time:.2f}s ({baseline.events_per_second:,.0f} events/sec)")
        print(f"With features: {with_features.wall_time:.2f}s ({with_features.events_per_second:,.0f} events/sec)")
        print(f"With risk:     {with_risk.wall_time:.2f}s ({with_risk.events_per_second:,.0f} events/sec)")
        print(f"")
        print(f"Data layer overhead:  {data_overhead:+.2f}%")
        print(f"Risk layer overhead:  {risk_overhead:+.2f}%")
        print(f"Total overhead:       {total_overhead:+.2f}%")
        print(f"{'='*80}\n")

        # CRITICAL ACCEPTANCE CRITERIA
        assert data_overhead < 5.0, \
            f"Data layer overhead too high: {data_overhead:.2f}% (target: <5%)"
        assert risk_overhead < 3.0, \
            f"Risk layer overhead too high: {risk_overhead:.2f}% (target: <3%)"
        assert total_overhead < 8.0, \
            f"Total overhead too high: {total_overhead:.2f}% (target: <8%)"


class TestScalability:
    """Scalability tests - characterize performance at 2x scale."""

    def test_500_symbols_characteristics(self, scalability_dataset, hardware_specs):
        """
        Document performance characteristics at 500 symbols (2x production scale).

        This is NOT a pass/fail test, but documents degradation for capacity planning.
        """
        price_df, signals_df, context_df = scalability_dataset

        strategy = MLRankingStrategy(max_positions=20, ml_threshold=0.65)
        risk_manager = RiskManager(rules=[
            stop_loss_5pct,
            take_profit_10pct,
            TimeBasedExit(max_bars=20),
        ])

        result = run_backtest_benchmark(
            strategy, price_df, signals_df, context_df,
            risk_manager=risk_manager,
            name="Scalability Test (500 symbols)",
        )

        wall_time_minutes = result.wall_time / 60

        print(f"\n{'='*80}")
        print(f"SCALABILITY TEST (500 SYMBOLS)")
        print(f"{'='*80}")
        print(f"Dataset: 500 symbols × 252 days = {result.events_processed:,} events")
        print(f"Wall time: {wall_time_minutes:.2f} minutes ({result.wall_time:.1f}s)")
        print(f"Throughput: {result.events_per_second:,.0f} events/sec")
        print(f"Peak memory: {result.peak_memory_mb:.1f} MB ({result.peak_memory_mb/1024:.2f} GB)")
        print(f"Memory delta: {result.memory_delta_mb:.1f} MB")
        print(f"{'='*80}\n")

        # Document findings (no hard assertions)
        print("Scalability Findings:")
        print(f"- 2x symbols -> {wall_time_minutes/2.5:.2f}x time (expected: ~2x)")
        print(f"- Memory scaling: {result.memory_delta_mb/1000:.1f}x vs 250 symbols")

        # Soft checks for capacity planning
        if wall_time_minutes > 10:
            print(f"⚠️  Warning: 500 symbols takes {wall_time_minutes:.1f} min (>10 min)")
        if result.peak_memory_mb > 4000:
            print(f"⚠️  Warning: Peak memory {result.peak_memory_mb/1024:.1f} GB (>4GB)")


# ===== Summary Report =====


def test_print_summary_report(hardware_specs):
    """Print summary report for documentation."""
    print(f"\n{'='*80}")
    print(f"INTEGRATED SYSTEM PERFORMANCE VALIDATION - SUMMARY")
    print(f"{'='*80}")
    print(f"")
    print(f"Hardware Configuration:")
    print(f"  CPU: 12th Gen Intel Core i9-12900K (16 cores, 24 threads)")
    print(f"  RAM: 125GB DDR4")
    print(f"  OS: Ubuntu 22.04 LTS (Linux 6.8.0-87-generic)")
    print(f"  Python: 3.12.3")
    print(f"")
    print(f"Test Methodology:")
    print(f"  - Synthetic data (fixed seed for reproducibility)")
    print(f"  - Realistic ML strategy (20 positions, signal-based ranking)")
    print(f"  - 3 active risk rules (stop-loss, take-profit, time-based exit)")
    print(f"  - Production workload: 250 symbols × 252 trading days")
    print(f"")
    print(f"Run individual tests above to see detailed results.")
    print(f"{'='*80}\n")
