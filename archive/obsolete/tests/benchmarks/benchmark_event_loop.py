"""
Benchmark suite for event loop performance.

Tests core event generation throughput with different data sizes and
iteration strategies.

Run with: pytest tests/benchmarks/benchmark_event_loop.py -v --benchmark-only
"""

import polars as pl
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_data_10k():
    """Generate 10k rows of sample data (10 symbols, 1000 bars each)."""
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=i) for i in range(1000)]
    symbols = [f"SYM{i:02d}" for i in range(10)]

    data = []
    for symbol in symbols:
        for ts in timestamps:
            data.append({
                "timestamp": ts,
                "symbol": symbol,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10000.0,
            })

    return pl.DataFrame(data).sort(["timestamp", "symbol"])


@pytest.fixture
def sample_data_100k():
    """Generate 100k rows of sample data (50 symbols, 2000 bars each)."""
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=i) for i in range(2000)]
    symbols = [f"SYM{i:03d}" for i in range(50)]

    data = []
    for symbol in symbols:
        for ts in timestamps:
            data.append({
                "timestamp": ts,
                "symbol": symbol,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10000.0,
            })

    return pl.DataFrame(data).sort(["timestamp", "symbol"])


@pytest.fixture
def sample_data_1m():
    """Generate 1M rows of sample data (250 symbols, 4000 bars each)."""
    timestamps = [datetime(2023, 1, 1) + timedelta(minutes=i) for i in range(4000)]
    symbols = [f"SYM{i:03d}" for i in range(250)]

    data = []
    for symbol in symbols:
        for ts in timestamps:
            data.append({
                "timestamp": ts,
                "symbol": symbol,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10000.0,
            })

    return pl.DataFrame(data).sort(["timestamp", "symbol"])


def iterate_filter_naive(df: pl.DataFrame):
    """
    Naive iteration: filter entire DataFrame for each timestamp.

    This is the SLOW approach that was in the original proposal.
    O(T × N) complexity where T = unique timestamps, N = total rows.
    """
    timestamps = df["timestamp"].unique().sort()
    event_count = 0

    for ts in timestamps:
        # BAD: Filters entire DataFrame for each timestamp
        rows = df.filter(pl.col("timestamp") == ts)
        for row in rows.iter_rows(named=True):
            event_count += 1

    return event_count


def iterate_group_by_optimized(df: pl.DataFrame):
    """
    Optimized iteration: group by timestamp once, iterate groups.

    This is the FAST approach from external review fix.
    O(N) complexity - single pass through data.
    """
    event_count = 0

    # GOOD: Group once, iterate groups
    for (ts_tuple, group) in df.group_by("timestamp", maintain_order=True):
        # group is already filtered, no extra work needed
        for row in group.iter_rows(named=True):
            event_count += 1

    return event_count


def iterate_partition_by_optimized(df: pl.DataFrame):
    """
    Alternative optimized iteration: partition_by timestamp.

    partition_by creates an iterator over groups without materializing.
    Similar performance to group_by but lower memory.
    """
    event_count = 0

    # Also GOOD: Partitions without materializing groups upfront
    for group_df in df.partition_by("timestamp", as_dict=False, maintain_order=True):
        for row in group_df.iter_rows(named=True):
            event_count += 1

    return event_count


# Benchmarks for 10k rows
def test_benchmark_filter_10k(benchmark, sample_data_10k):
    """Benchmark naive filter approach with 10k rows."""
    result = benchmark(iterate_filter_naive, sample_data_10k)
    assert result == 10000


def test_benchmark_group_by_10k(benchmark, sample_data_10k):
    """Benchmark optimized group_by approach with 10k rows."""
    result = benchmark(iterate_group_by_optimized, sample_data_10k)
    assert result == 10000


def test_benchmark_partition_by_10k(benchmark, sample_data_10k):
    """Benchmark partition_by approach with 10k rows."""
    result = benchmark(iterate_partition_by_optimized, sample_data_10k)
    assert result == 10000


# Benchmarks for 100k rows
def test_benchmark_filter_100k(benchmark, sample_data_100k):
    """Benchmark naive filter approach with 100k rows."""
    result = benchmark(iterate_filter_naive, sample_data_100k)
    assert result == 100000


def test_benchmark_group_by_100k(benchmark, sample_data_100k):
    """Benchmark optimized group_by approach with 100k rows."""
    result = benchmark(iterate_group_by_optimized, sample_data_100k)
    assert result == 100000


def test_benchmark_partition_by_100k(benchmark, sample_data_100k):
    """Benchmark partition_by approach with 100k rows."""
    result = benchmark(iterate_partition_by_optimized, sample_data_100k)
    assert result == 100000


# Benchmarks for 1M rows
@pytest.mark.slow
def test_benchmark_filter_1m(benchmark, sample_data_1m):
    """Benchmark naive filter approach with 1M rows."""
    # This will be VERY slow - may take minutes
    result = benchmark(iterate_filter_naive, sample_data_1m)
    assert result == 1000000


@pytest.mark.slow
def test_benchmark_group_by_1m(benchmark, sample_data_1m):
    """Benchmark optimized group_by approach with 1M rows."""
    result = benchmark(iterate_group_by_optimized, sample_data_1m)
    assert result == 1000000


@pytest.mark.slow
def test_benchmark_partition_by_1m(benchmark, sample_data_1m):
    """Benchmark partition_by approach with 1M rows."""
    result = benchmark(iterate_partition_by_optimized, sample_data_1m)
    assert result == 1000000


# Direct comparison test
def test_speedup_comparison(sample_data_100k):
    """
    Measure speedup of optimized vs naive approach.

    Expected: 10-50x speedup for optimized approach.
    """
    import time

    # Naive approach
    start = time.perf_counter()
    naive_count = iterate_filter_naive(sample_data_100k)
    naive_time = time.perf_counter() - start

    # Optimized approach
    start = time.perf_counter()
    optimized_count = iterate_group_by_optimized(sample_data_100k)
    optimized_time = time.perf_counter() - start

    assert naive_count == optimized_count == 100000

    speedup = naive_time / optimized_time
    print(f"\nSpeedup: {speedup:.1f}x")
    print(f"Naive: {naive_time:.3f}s ({naive_count/naive_time:.0f} events/sec)")
    print(f"Optimized: {optimized_time:.3f}s ({optimized_count/optimized_time:.0f} events/sec)")

    # Expect at least 10x speedup
    assert speedup >= 10.0, f"Speedup {speedup:.1f}x is less than expected 10x minimum"


if __name__ == "__main__":
    # Quick manual run
    print("Generating test data...")
    data = pl.DataFrame([
        {
            "timestamp": datetime(2023, 1, 1) + timedelta(minutes=i),
            "symbol": f"SYM{j:02d}",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10000.0,
        }
        for i in range(100)
        for j in range(10)
    ]).sort(["timestamp", "symbol"])

    print(f"Data: {len(data)} rows")

    import time

    # Naive
    start = time.perf_counter()
    count1 = iterate_filter_naive(data)
    time1 = time.perf_counter() - start
    print(f"Naive: {count1} events in {time1:.3f}s ({count1/time1:.0f} events/sec)")

    # Optimized
    start = time.perf_counter()
    count2 = iterate_group_by_optimized(data)
    time2 = time.perf_counter() - start
    print(f"Optimized: {count2} events in {time2:.3f}s ({count2/time2:.0f} events/sec)")

    print(f"Speedup: {time1/time2:.1f}x")
