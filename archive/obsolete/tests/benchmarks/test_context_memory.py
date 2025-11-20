"""Memory and performance benchmarks for ContextCache vs embedded context.

This benchmark validates the claim that ContextCache provides ~50x memory savings
compared to embedding context data in every MarketEvent.

Scenarios tested:
- Small scale: 10 assets × 252 days = 2,520 events
- Medium scale: 100 assets × 252 days = 25,200 events
- Large scale: 500 assets × 252 days = 126,000 events

Measurements:
- Peak memory usage (MB)
- Memory per event (bytes)
- Execution time (seconds)
- Memory efficiency ratio (ContextCache vs Embedded)
"""

import gc
import sys
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest

from ml4t.backtest.core.context import Context, ContextCache
from ml4t.backtest.core.event import MarketEvent
from ml4t.backtest.core.types import MarketDataType


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    scenario: str
    approach: str
    num_assets: int
    num_days: int
    total_events: int
    peak_memory_mb: float
    memory_per_event_bytes: float
    execution_time_sec: float
    events_per_second: float


class ContextCacheApproach:
    """Approach A: Use ContextCache (one Context per timestamp)."""

    def __init__(self):
        self.cache = ContextCache()
        self.events = []

    def create_events(
        self, num_assets: int, num_days: int, context_data: dict
    ) -> list[MarketEvent]:
        """Create market events using ContextCache."""
        base_date = datetime(2024, 1, 1, 9, 30)
        events = []

        for day in range(num_days):
            timestamp = base_date + timedelta(days=day)
            # Create one context per timestamp (cached and shared)
            context = self.cache.get_or_create(timestamp=timestamp, data=context_data)

            for asset_idx in range(num_assets):
                asset_id = f"ASSET_{asset_idx:03d}"
                event = MarketEvent(
                    timestamp=timestamp,
                    asset_id=asset_id,
                    data_type=MarketDataType.BAR,
                    open=100.0,
                    high=105.0,
                    low=95.0,
                    close=102.0,
                    volume=1_000_000,
                )
                events.append(event)

        self.events = events
        return events


class EmbeddedContextApproach:
    """Approach B: Embed context in every MarketEvent."""

    def __init__(self):
        self.events = []

    def create_events(
        self, num_assets: int, num_days: int, context_data: dict
    ) -> list[MarketEvent]:
        """Create market events with embedded context."""
        base_date = datetime(2024, 1, 1, 9, 30)
        events = []

        for day in range(num_days):
            timestamp = base_date + timedelta(days=day)

            for asset_idx in range(num_assets):
                asset_id = f"ASSET_{asset_idx:03d}"
                # Embed context in metadata (separate copy per event)
                event = MarketEvent(
                    timestamp=timestamp,
                    asset_id=asset_id,
                    data_type=MarketDataType.BAR,
                    open=100.0,
                    high=105.0,
                    low=95.0,
                    close=102.0,
                    volume=1_000_000,
                    metadata={"context": context_data.copy()},
                )
                events.append(event)

        self.events = events
        return events


def run_benchmark(
    scenario: str, num_assets: int, num_days: int, approach_class, context_data: dict
) -> BenchmarkResult:
    """Run a single benchmark scenario."""
    # Force garbage collection before measurement
    gc.collect()

    # Start memory tracking
    tracemalloc.start()
    start_time = time.perf_counter()

    # Create events
    approach = approach_class()
    events = approach.create_events(num_assets, num_days, context_data)

    # End timing
    end_time = time.perf_counter()

    # Get peak memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate metrics
    total_events = len(events)
    peak_memory_mb = peak / (1024 * 1024)  # Convert to MB
    memory_per_event_bytes = peak / total_events if total_events > 0 else 0
    execution_time_sec = end_time - start_time
    events_per_second = total_events / execution_time_sec if execution_time_sec > 0 else 0

    # Clean up
    del approach
    del events
    gc.collect()

    return BenchmarkResult(
        scenario=scenario,
        approach=approach_class.__name__.replace("Approach", ""),
        num_assets=num_assets,
        num_days=num_days,
        total_events=total_events,
        peak_memory_mb=peak_memory_mb,
        memory_per_event_bytes=memory_per_event_bytes,
        execution_time_sec=execution_time_sec,
        events_per_second=events_per_second,
    )


@pytest.fixture
def context_data():
    """Realistic context data for benchmarks."""
    return {
        "VIX": 18.5,
        "SPY": 485.0,
        "QQQ": 410.0,
        "regime": "bull",
        "market_hours": "regular",
        "volume_profile": "normal",
        "trend_strength": 0.85,
        "volatility_regime": "low",
    }


@pytest.fixture
def large_context_data():
    """Large context data with many market indicators (realistic ML scenario)."""
    context = {
        # Major indices
        "SPY": 485.0,
        "QQQ": 410.0,
        "IWM": 195.0,
        "DIA": 380.0,
        # Volatility indicators
        "VIX": 18.5,
        "VIX9D": 17.2,
        "VVIX": 85.0,
        # Sector ETFs (11 sectors)
        "XLK": 175.0,  # Technology
        "XLF": 38.0,   # Financials
        "XLV": 145.0,  # Healthcare
        "XLE": 89.0,   # Energy
        "XLI": 120.0,  # Industrials
        "XLC": 82.0,   # Communications
        "XLY": 180.0,  # Consumer Discretionary
        "XLP": 78.0,   # Consumer Staples
        "XLB": 88.0,   # Materials
        "XLRE": 42.0,  # Real Estate
        "XLU": 68.0,   # Utilities
        # Market breadth indicators
        "NYSE_ADV": 2500,
        "NYSE_DEC": 1200,
        "NYSE_ADV_VOL": 1.2e9,
        "NYSE_DEC_VOL": 0.5e9,
        "NYSE_NEW_HIGHS": 150,
        "NYSE_NEW_LOWS": 25,
        # Technical indicators
        "SPY_RSI_14": 55.0,
        "SPY_MACD": 1.2,
        "SPY_BB_UPPER": 490.0,
        "SPY_BB_LOWER": 480.0,
        "SPY_ATR_14": 5.5,
        # Regime indicators
        "regime": "bull",
        "trend_strength": 0.85,
        "volatility_regime": "low",
        "correlation_regime": "normal",
        "liquidity_regime": "high",
        # Economic indicators
        "US_10Y_YIELD": 4.25,
        "US_2Y_YIELD": 4.75,
        "YIELD_CURVE": -0.50,
        "DXY": 103.5,  # Dollar index
        "GOLD": 2050.0,
        "OIL_WTI": 78.5,
        # Time features
        "market_hours": "regular",
        "day_of_week": 3,
        "week_of_year": 42,
        "month": 10,
        "quarter": 4,
        "is_month_end": False,
        "is_quarter_end": False,
        "days_to_opex": 15,
    }
    return context


class TestContextMemoryBenchmarks:
    """Memory and performance benchmarks for ContextCache."""

    def test_small_scale_benchmark_minimal_context(self, context_data):
        """Benchmark: 10 assets × 252 days = 2,520 events (baseline)."""
        scenario = "small"
        num_assets = 10
        num_days = 252

        # Run both approaches
        result_cache = run_benchmark(
            scenario, num_assets, num_days, ContextCacheApproach, context_data
        )
        result_embedded = run_benchmark(
            scenario, num_assets, num_days, EmbeddedContextApproach, context_data
        )

        # Calculate efficiency
        memory_ratio = result_embedded.peak_memory_mb / result_cache.peak_memory_mb
        speed_ratio = result_cache.events_per_second / result_embedded.events_per_second

        # Print results
        print("\n" + "=" * 80)
        print(f"SMALL SCALE: {num_assets} assets × {num_days} days = {result_cache.total_events:,} events")
        print("=" * 80)
        print(f"\nContextCache:")
        print(f"  Peak Memory:     {result_cache.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_cache.memory_per_event_bytes:.0f} bytes")
        print(f"  Execution Time:  {result_cache.execution_time_sec:.3f} sec")
        print(f"  Events/Second:   {result_cache.events_per_second:,.0f}")

        print(f"\nEmbedded Context:")
        print(f"  Peak Memory:     {result_embedded.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_embedded.memory_per_event_bytes:.0f} bytes")
        print(f"  Execution Time:  {result_embedded.execution_time_sec:.3f} sec")
        print(f"  Events/Second:   {result_embedded.events_per_second:,.0f}")

        print(f"\nEfficiency Ratio:")
        print(f"  Memory Savings:  {memory_ratio:.1f}x (ContextCache uses {100/memory_ratio:.1f}% of embedded)")
        print(f"  Speed Ratio:     {speed_ratio:.2f}x")
        print("=" * 80 + "\n")

        # Assertions
        assert result_cache.total_events == num_assets * num_days
        assert result_embedded.total_events == num_assets * num_days
        assert result_cache.peak_memory_mb > 0
        assert result_embedded.peak_memory_mb > 0

    def test_medium_scale_benchmark_minimal_context(self, context_data):
        """Benchmark: 100 assets × 252 days = 25,200 events (typical retail)."""
        scenario = "medium"
        num_assets = 100
        num_days = 252

        # Run both approaches
        result_cache = run_benchmark(
            scenario, num_assets, num_days, ContextCacheApproach, context_data
        )
        result_embedded = run_benchmark(
            scenario, num_assets, num_days, EmbeddedContextApproach, context_data
        )

        # Calculate efficiency
        memory_ratio = result_embedded.peak_memory_mb / result_cache.peak_memory_mb
        speed_ratio = result_cache.events_per_second / result_embedded.events_per_second

        # Print results
        print("\n" + "=" * 80)
        print(f"MEDIUM SCALE: {num_assets} assets × {num_days} days = {result_cache.total_events:,} events")
        print("=" * 80)
        print(f"\nContextCache:")
        print(f"  Peak Memory:     {result_cache.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_cache.memory_per_event_bytes:.0f} bytes")
        print(f"  Execution Time:  {result_cache.execution_time_sec:.3f} sec")
        print(f"  Events/Second:   {result_cache.events_per_second:,.0f}")

        print(f"\nEmbedded Context:")
        print(f"  Peak Memory:     {result_embedded.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_embedded.memory_per_event_bytes:.0f} bytes")
        print(f"  Execution Time:  {result_embedded.execution_time_sec:.3f} sec")
        print(f"  Events/Second:   {result_embedded.events_per_second:,.0f}")

        print(f"\nEfficiency Ratio:")
        print(f"  Memory Savings:  {memory_ratio:.1f}x (ContextCache uses {100/memory_ratio:.1f}% of embedded)")
        print(f"  Speed Ratio:     {speed_ratio:.2f}x")
        print("=" * 80 + "\n")

        # Assertions
        assert result_cache.total_events == num_assets * num_days
        assert result_embedded.total_events == num_assets * num_days
        assert memory_ratio > 1.0  # ContextCache should use less memory

    def test_large_scale_benchmark_minimal_context(self, context_data):
        """Benchmark: 500 assets × 252 days = 126,000 events (institutional)."""
        scenario = "large"
        num_assets = 500
        num_days = 252

        # Run both approaches
        result_cache = run_benchmark(
            scenario, num_assets, num_days, ContextCacheApproach, context_data
        )
        result_embedded = run_benchmark(
            scenario, num_assets, num_days, EmbeddedContextApproach, context_data
        )

        # Calculate efficiency
        memory_ratio = result_embedded.peak_memory_mb / result_cache.peak_memory_mb
        speed_ratio = result_cache.events_per_second / result_embedded.events_per_second

        # Print results
        print("\n" + "=" * 80)
        print(f"LARGE SCALE: {num_assets} assets × {num_days} days = {result_cache.total_events:,} events")
        print("=" * 80)
        print(f"\nContextCache:")
        print(f"  Peak Memory:     {result_cache.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_cache.memory_per_event_bytes:.0f} bytes")
        print(f"  Execution Time:  {result_cache.execution_time_sec:.3f} sec")
        print(f"  Events/Second:   {result_cache.events_per_second:,.0f}")

        print(f"\nEmbedded Context:")
        print(f"  Peak Memory:     {result_embedded.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_embedded.memory_per_event_bytes:.0f} bytes")
        print(f"  Execution Time:  {result_embedded.execution_time_sec:.3f} sec")
        print(f"  Events/Second:   {result_embedded.events_per_second:,.0f}")

        print(f"\nEfficiency Ratio:")
        print(f"  Memory Savings:  {memory_ratio:.1f}x (ContextCache uses {100/memory_ratio:.1f}% of embedded)")
        print(f"  Speed Ratio:     {speed_ratio:.2f}x")
        print("=" * 80 + "\n")

        # Assertions
        assert result_cache.total_events == num_assets * num_days
        assert result_embedded.total_events == num_assets * num_days
        assert memory_ratio > 1.5  # Should see savings even with minimal context

    def test_small_scale_benchmark_large_context(self, large_context_data):
        """Benchmark: 10 assets × 252 days with LARGE context (50+ indicators)."""
        scenario = "small_large_ctx"
        num_assets = 10
        num_days = 252

        # Run both approaches
        result_cache = run_benchmark(
            scenario, num_assets, num_days, ContextCacheApproach, large_context_data
        )
        result_embedded = run_benchmark(
            scenario, num_assets, num_days, EmbeddedContextApproach, large_context_data
        )

        # Calculate efficiency
        memory_ratio = result_embedded.peak_memory_mb / result_cache.peak_memory_mb

        # Print results
        print("\n" + "=" * 80)
        print(f"SMALL SCALE (LARGE CONTEXT): {num_assets} assets × {num_days} days = {result_cache.total_events:,} events")
        print(f"Context size: {len(large_context_data)} indicators")
        print("=" * 80)
        print(f"\nContextCache:")
        print(f"  Peak Memory:     {result_cache.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_cache.memory_per_event_bytes:.0f} bytes")

        print(f"\nEmbedded Context:")
        print(f"  Peak Memory:     {result_embedded.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_embedded.memory_per_event_bytes:.0f} bytes")

        print(f"\nEfficiency Ratio:")
        print(f"  Memory Savings:  {memory_ratio:.1f}x")
        print("=" * 80 + "\n")

        assert memory_ratio > 1.0

    def test_medium_scale_benchmark_large_context(self, large_context_data):
        """Benchmark: 100 assets × 252 days with LARGE context (50+ indicators)."""
        scenario = "medium_large_ctx"
        num_assets = 100
        num_days = 252

        result_cache = run_benchmark(
            scenario, num_assets, num_days, ContextCacheApproach, large_context_data
        )
        result_embedded = run_benchmark(
            scenario, num_assets, num_days, EmbeddedContextApproach, large_context_data
        )

        memory_ratio = result_embedded.peak_memory_mb / result_cache.peak_memory_mb

        print("\n" + "=" * 80)
        print(f"MEDIUM SCALE (LARGE CONTEXT): {num_assets} assets × {num_days} days = {result_cache.total_events:,} events")
        print(f"Context size: {len(large_context_data)} indicators")
        print("=" * 80)
        print(f"\nContextCache:")
        print(f"  Peak Memory:     {result_cache.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_cache.memory_per_event_bytes:.0f} bytes")

        print(f"\nEmbedded Context:")
        print(f"  Peak Memory:     {result_embedded.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_embedded.memory_per_event_bytes:.0f} bytes")

        print(f"\nEfficiency Ratio:")
        print(f"  Memory Savings:  {memory_ratio:.1f}x")
        print("=" * 80 + "\n")

        assert memory_ratio > 3.0  # Should see good savings with large context

    def test_large_scale_benchmark_large_context(self, large_context_data):
        """Benchmark: 500 assets × 252 days with LARGE context (50+ indicators)."""
        scenario = "large_large_ctx"
        num_assets = 500
        num_days = 252

        result_cache = run_benchmark(
            scenario, num_assets, num_days, ContextCacheApproach, large_context_data
        )
        result_embedded = run_benchmark(
            scenario, num_assets, num_days, EmbeddedContextApproach, large_context_data
        )

        memory_ratio = result_embedded.peak_memory_mb / result_cache.peak_memory_mb

        print("\n" + "=" * 80)
        print(f"LARGE SCALE (LARGE CONTEXT): {num_assets} assets × {num_days} days = {result_cache.total_events:,} events")
        print(f"Context size: {len(large_context_data)} indicators")
        print("=" * 80)
        print(f"\nContextCache:")
        print(f"  Peak Memory:     {result_cache.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_cache.memory_per_event_bytes:.0f} bytes")

        print(f"\nEmbedded Context:")
        print(f"  Peak Memory:     {result_embedded.peak_memory_mb:.2f} MB")
        print(f"  Memory/Event:    {result_embedded.memory_per_event_bytes:.0f} bytes")

        print(f"\nEfficiency Ratio:")
        print(f"  Memory Savings:  {memory_ratio:.1f}x")
        print("=" * 80 + "\n")

        assert memory_ratio > 4.0  # Should see significant savings with large context

    def test_context_cache_memory_sharing(self, context_data):
        """Verify that ContextCache actually shares Context instances."""
        cache = ContextCache()
        base_date = datetime(2024, 1, 1, 9, 30)

        # Create multiple contexts for same timestamp
        contexts = []
        for _ in range(100):
            ctx = cache.get_or_create(timestamp=base_date, data=context_data)
            contexts.append(ctx)

        # All should be the same object instance
        assert all(ctx is contexts[0] for ctx in contexts)

        # Cache should have only one entry
        assert cache.size() == 1

        print("\n" + "=" * 80)
        print("CONTEXT SHARING VERIFICATION")
        print("=" * 80)
        print(f"Created 100 context requests for same timestamp")
        print(f"Unique objects:  {len(set(id(ctx) for ctx in contexts))}")
        print(f"Cache size:      {cache.size()}")
        print(f"Memory sharing:  {'✓ VERIFIED' if cache.size() == 1 else '✗ FAILED'}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    """Run benchmarks standalone."""
    minimal_context = {
        "VIX": 18.5,
        "SPY": 485.0,
        "QQQ": 410.0,
        "regime": "bull",
        "market_hours": "regular",
        "volume_profile": "normal",
        "trend_strength": 0.85,
        "volatility_regime": "low",
    }

    large_context = {
        # Major indices
        "SPY": 485.0,
        "QQQ": 410.0,
        "IWM": 195.0,
        "DIA": 380.0,
        # Volatility indicators
        "VIX": 18.5,
        "VIX9D": 17.2,
        "VVIX": 85.0,
        # Sector ETFs (11 sectors)
        "XLK": 175.0, "XLF": 38.0, "XLV": 145.0, "XLE": 89.0,
        "XLI": 120.0, "XLC": 82.0, "XLY": 180.0, "XLP": 78.0,
        "XLB": 88.0, "XLRE": 42.0, "XLU": 68.0,
        # Market breadth
        "NYSE_ADV": 2500, "NYSE_DEC": 1200,
        "NYSE_ADV_VOL": 1.2e9, "NYSE_DEC_VOL": 0.5e9,
        "NYSE_NEW_HIGHS": 150, "NYSE_NEW_LOWS": 25,
        # Technical indicators
        "SPY_RSI_14": 55.0, "SPY_MACD": 1.2,
        "SPY_BB_UPPER": 490.0, "SPY_BB_LOWER": 480.0, "SPY_ATR_14": 5.5,
        # Regime indicators
        "regime": "bull", "trend_strength": 0.85,
        "volatility_regime": "low", "correlation_regime": "normal",
        "liquidity_regime": "high",
        # Economic indicators
        "US_10Y_YIELD": 4.25, "US_2Y_YIELD": 4.75, "YIELD_CURVE": -0.50,
        "DXY": 103.5, "GOLD": 2050.0, "OIL_WTI": 78.5,
        # Time features
        "market_hours": "regular", "day_of_week": 3, "week_of_year": 42,
        "month": 10, "quarter": 4,
        "is_month_end": False, "is_quarter_end": False, "days_to_opex": 15,
    }

    print("\n" + "=" * 80)
    print("CONTEXT MEMORY BENCHMARK SUITE")
    print("Validating ContextCache vs Embedded Context Memory Usage")
    print("=" * 80)

    # Part 1: Minimal Context (8 indicators)
    print("\n" + "=" * 80)
    print("PART 1: MINIMAL CONTEXT (8 indicators)")
    print("=" * 80)

    print("\nRunning Small Scale (10 assets × 252 days)...")
    result_cache_s_min = run_benchmark("small_min", 10, 252, ContextCacheApproach, minimal_context)
    result_embed_s_min = run_benchmark("small_min", 10, 252, EmbeddedContextApproach, minimal_context)

    print("\nRunning Medium Scale (100 assets × 252 days)...")
    result_cache_m_min = run_benchmark("medium_min", 100, 252, ContextCacheApproach, minimal_context)
    result_embed_m_min = run_benchmark("medium_min", 100, 252, EmbeddedContextApproach, minimal_context)

    print("\nRunning Large Scale (500 assets × 252 days)...")
    result_cache_l_min = run_benchmark("large_min", 500, 252, ContextCacheApproach, minimal_context)
    result_embed_l_min = run_benchmark("large_min", 500, 252, EmbeddedContextApproach, minimal_context)

    # Part 2: Large Context (50+ indicators - realistic ML scenario)
    print("\n" + "=" * 80)
    print("PART 2: LARGE CONTEXT (50+ indicators - realistic ML scenario)")
    print("=" * 80)

    print("\nRunning Small Scale (10 assets × 252 days)...")
    result_cache_s_large = run_benchmark("small_large", 10, 252, ContextCacheApproach, large_context)
    result_embed_s_large = run_benchmark("small_large", 10, 252, EmbeddedContextApproach, large_context)

    print("\nRunning Medium Scale (100 assets × 252 days)...")
    result_cache_m_large = run_benchmark("medium_large", 100, 252, ContextCacheApproach, large_context)
    result_embed_m_large = run_benchmark("medium_large", 100, 252, EmbeddedContextApproach, large_context)

    print("\nRunning Large Scale (500 assets × 252 days)...")
    result_cache_l_large = run_benchmark("large_large", 500, 252, ContextCacheApproach, large_context)
    result_embed_l_large = run_benchmark("large_large", 500, 252, EmbeddedContextApproach, large_context)

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print("\nMINIMAL CONTEXT (8 indicators):")
    print(f"{'Scenario':<25} {'Events':>10} {'Cache MB':>12} {'Embed MB':>12} {'Ratio':>8}")
    print("-" * 80)
    min_scenarios = [
        ("Small (10 assets)", result_cache_s_min, result_embed_s_min),
        ("Medium (100 assets)", result_cache_m_min, result_embed_m_min),
        ("Large (500 assets)", result_cache_l_min, result_embed_l_min),
    ]
    for name, cache_res, embed_res in min_scenarios:
        ratio = embed_res.peak_memory_mb / cache_res.peak_memory_mb
        print(
            f"{name:<25} {cache_res.total_events:>10,} "
            f"{cache_res.peak_memory_mb:>12.2f} {embed_res.peak_memory_mb:>12.2f} "
            f"{ratio:>7.1f}x"
        )

    print("\nLARGE CONTEXT (50+ indicators):")
    print(f"{'Scenario':<25} {'Events':>10} {'Cache MB':>12} {'Embed MB':>12} {'Ratio':>8}")
    print("-" * 80)
    large_scenarios = [
        ("Small (10 assets)", result_cache_s_large, result_embed_s_large),
        ("Medium (100 assets)", result_cache_m_large, result_embed_m_large),
        ("Large (500 assets)", result_cache_l_large, result_embed_l_large),
    ]
    for name, cache_res, embed_res in large_scenarios:
        ratio = embed_res.peak_memory_mb / cache_res.peak_memory_mb
        print(
            f"{name:<25} {cache_res.total_events:>10,} "
            f"{cache_res.peak_memory_mb:>12.2f} {embed_res.peak_memory_mb:>12.2f} "
            f"{ratio:>7.1f}x"
        )

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    min_avg_ratio = sum(
        embed.peak_memory_mb / cache.peak_memory_mb
        for _, cache, embed in min_scenarios
    ) / len(min_scenarios)

    large_avg_ratio = sum(
        embed.peak_memory_mb / cache.peak_memory_mb
        for _, cache, embed in large_scenarios
    ) / len(large_scenarios)

    print(f"\nMinimal Context (8 indicators):")
    print(f"  Average Memory Savings: {min_avg_ratio:.1f}x")
    print(f"  Status: {'✓ EFFICIENT' if min_avg_ratio > 2 else '✗ INEFFICIENT'}")

    print(f"\nLarge Context (50+ indicators):")
    print(f"  Average Memory Savings: {large_avg_ratio:.1f}x")
    print(f"  Status: {'✓ EFFICIENT' if large_avg_ratio > 5 else '✗ INEFFICIENT'}")

    print(f"\nKey Finding:")
    if large_avg_ratio > 5:
        print(f"  ✓ ContextCache provides {large_avg_ratio:.0f}x memory savings with realistic ML context")
        print(f"  ✓ Validates architectural decision for multi-asset strategies")
    else:
        print(f"  ⚠ Memory savings ({large_avg_ratio:.1f}x) lower than expected")
        print(f"  ⚠ May need to re-evaluate design for different use cases")

    print("=" * 80 + "\n")
