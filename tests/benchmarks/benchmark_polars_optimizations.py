"""Benchmarks for Polars optimization features.

This module benchmarks:
1. Compression: zstd vs snappy vs gzip vs lz4 vs uncompressed
2. Categorical encoding: memory savings for 500+ symbols
3. Partitioning: query performance for selective loading
4. Lazy evaluation: memory efficiency

Target: <2GB memory for 250 symbols × 1 year × daily bars
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import psutil

from ml4t.backtest.data.polars_feed import (
    PolarsDataFeed,
    create_partitioned_dataset,
    load_partitioned_dataset,
    write_optimized_parquet,
)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_large_dataset(
    n_symbols: int,
    n_days: int,
    bars_per_day: int = 1,
) -> pl.DataFrame:
    """Create large OHLCV dataset for benchmarking.

    Args:
        n_symbols: Number of symbols (e.g., 250 for SP500 subset)
        n_days: Number of trading days (e.g., 252 for 1 year)
        bars_per_day: Bars per day (1 for daily, 390 for minute bars)

    Returns:
        DataFrame with OHLCV data
    """
    print(f"Creating dataset: {n_symbols} symbols × {n_days} days × {bars_per_day} bars/day")

    base_time = datetime(2025, 1, 1, 9, 30)
    symbols = [f"SYM{i:04d}" for i in range(n_symbols)]

    rows = []
    for symbol in symbols:
        for day in range(n_days):
            for bar in range(bars_per_day):
                timestamp = base_time + timedelta(days=day, minutes=bar)
                price = 100.0 + day * 0.5 + (bar % 10) * 0.1
                rows.append(
                    {
                        "timestamp": timestamp,
                        "asset_id": symbol,
                        "open": price,
                        "high": price + 1.0,
                        "low": price - 1.0,
                        "close": price + 0.5,
                        "volume": 1000000 + day * 1000,
                    }
                )

    df = pl.DataFrame(rows)
    print(f"  Total rows: {len(df):,}")
    return df


def benchmark_compression():
    """Benchmark compression codecs: size and write/read speed."""
    print("\n" + "=" * 80)
    print("BENCHMARK 1: Compression Codecs")
    print("=" * 80)

    # Create test dataset (250 symbols × 252 days)
    df = create_large_dataset(n_symbols=250, n_days=252, bars_per_day=1)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        results = {}

        # Test each compression codec
        for compression in [None, "snappy", "gzip", "lz4", "zstd"]:
            print(f"\nTesting compression: {compression or 'uncompressed'}")

            # Write benchmark
            path = tmpdir / f"data_{compression or 'none'}.parquet"
            start = time.perf_counter()
            write_optimized_parquet(df, path, compression=compression)
            write_time = time.perf_counter() - start

            # Read benchmark
            start = time.perf_counter()
            df_read = pl.read_parquet(path)
            read_time = time.perf_counter() - start

            # Size
            size_mb = path.stat().st_size / 1024 / 1024

            # Verify correctness
            assert len(df_read) == len(df)

            results[compression or "none"] = {
                "size_mb": size_mb,
                "write_time": write_time,
                "read_time": read_time,
            }

            print(f"  Size: {size_mb:.2f} MB")
            print(f"  Write time: {write_time:.3f}s")
            print(f"  Read time: {read_time:.3f}s")

        # Summary
        print("\n" + "-" * 80)
        print("Compression Summary:")
        print("-" * 80)
        baseline_size = results["none"]["size_mb"]
        for name, metrics in results.items():
            reduction = (1 - metrics["size_mb"] / baseline_size) * 100
            print(
                f"{name:12s}: {metrics['size_mb']:7.2f} MB "
                f"({reduction:+5.1f}%), "
                f"write={metrics['write_time']:.3f}s, "
                f"read={metrics['read_time']:.3f}s"
            )

    return results


def benchmark_categorical_encoding():
    """Benchmark memory savings from categorical encoding."""
    print("\n" + "=" * 80)
    print("BENCHMARK 2: Categorical Encoding")
    print("=" * 80)

    # Create dataset with many symbols (500 for realistic test)
    df = create_large_dataset(n_symbols=500, n_days=252, bars_per_day=1)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Baseline: no categorical
        print("\n1. Without categorical encoding:")
        path_baseline = tmpdir / "baseline.parquet"
        df.write_parquet(path_baseline)
        size_baseline = path_baseline.stat().st_size / 1024 / 1024

        # Load and measure memory
        mem_before = get_memory_usage_mb()
        df_baseline = pl.read_parquet(path_baseline)
        mem_after = get_memory_usage_mb()
        mem_baseline = mem_after - mem_before

        print(f"  File size: {size_baseline:.2f} MB")
        print(f"  Memory delta: {mem_baseline:.2f} MB")

        # With categorical
        print("\n2. With categorical encoding:")
        path_categorical = tmpdir / "categorical.parquet"
        write_optimized_parquet(
            df, path_categorical, use_categorical=True, categorical_columns=["asset_id"]
        )
        size_categorical = path_categorical.stat().st_size / 1024 / 1024

        # Load and measure memory
        mem_before = get_memory_usage_mb()
        df_categorical = pl.read_parquet(path_categorical)
        mem_after = get_memory_usage_mb()
        mem_categorical = mem_after - mem_before

        print(f"  File size: {size_categorical:.2f} MB")
        print(f"  Memory delta: {mem_categorical:.2f} MB")

        # Verify correctness
        assert len(df_categorical) == len(df_baseline)

        # Summary
        print("\n" + "-" * 80)
        print("Categorical Encoding Summary:")
        print("-" * 80)
        size_reduction = (1 - size_categorical / size_baseline) * 100
        mem_reduction = (1 - mem_categorical / mem_baseline) * 100
        print(f"File size reduction: {size_reduction:.1f}%")
        print(f"Memory reduction: {mem_reduction:.1f}%")

        return {
            "baseline": {"size_mb": size_baseline, "memory_mb": mem_baseline},
            "categorical": {"size_mb": size_categorical, "memory_mb": mem_categorical},
            "size_reduction_pct": size_reduction,
            "memory_reduction_pct": mem_reduction,
        }


def benchmark_partitioning():
    """Benchmark query performance with partitioning."""
    print("\n" + "=" * 80)
    print("BENCHMARK 3: Partitioning")
    print("=" * 80)

    # Create 1 year dataset
    df = create_large_dataset(n_symbols=250, n_days=252, bars_per_day=1)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 1. Single file (baseline)
        print("\n1. Single file (baseline):")
        single_file = tmpdir / "single.parquet"
        df.write_parquet(single_file, compression="zstd")

        # Query for 1 month of data (should scan entire file)
        start = time.perf_counter()
        df_single = pl.scan_parquet(single_file).filter(
            (pl.col("timestamp") >= datetime(2025, 1, 1))
            & (pl.col("timestamp") < datetime(2025, 2, 1))
        ).collect()
        time_single = time.perf_counter() - start
        print(f"  Query time (1 month): {time_single:.3f}s")
        print(f"  Rows returned: {len(df_single):,}")

        # 2. Partitioned by month
        print("\n2. Partitioned by month:")
        partitioned_dir = tmpdir / "partitioned"
        partitions = create_partitioned_dataset(
            df,
            partitioned_dir,
            partition_by="month",
            compression="zstd",
        )
        print(f"  Created {len(partitions)} partitions")

        # Query for 1 month (should only load 1 partition)
        start = time.perf_counter()
        df_partitioned = load_partitioned_dataset(
            partitioned_dir,
            partitions=["2025-01"],  # Load only January
            lazy=False,
        )
        time_partitioned = time.perf_counter() - start
        print(f"  Query time (1 month): {time_partitioned:.3f}s")
        print(f"  Rows returned: {len(df_partitioned):,}")

        # Summary
        print("\n" + "-" * 80)
        print("Partitioning Summary:")
        print("-" * 80)
        speedup = time_single / time_partitioned
        print(f"Single file: {time_single:.3f}s")
        print(f"Partitioned: {time_partitioned:.3f}s")
        print(f"Speedup: {speedup:.2f}x")

        return {
            "single_file_time": time_single,
            "partitioned_time": time_partitioned,
            "speedup": speedup,
        }


def benchmark_memory_target():
    """Verify memory target: <2GB for 250 symbols × 1 year."""
    print("\n" + "=" * 80)
    print("BENCHMARK 4: Memory Target Verification")
    print("=" * 80)

    # Create target dataset
    df = create_large_dataset(n_symbols=250, n_days=252, bars_per_day=1)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test 1: Unoptimized
        print("\n1. Unoptimized (baseline):")
        path_baseline = tmpdir / "baseline.parquet"
        df.write_parquet(path_baseline)

        mem_before = get_memory_usage_mb()
        feed_baseline = PolarsDataFeed(
            price_path=path_baseline,
            asset_id="SYM0001",
        )
        # Trigger initialization
        _ = feed_baseline.get_next_event()
        mem_after = get_memory_usage_mb()
        mem_baseline = mem_after - mem_before

        print(f"  Memory usage: {mem_baseline:.2f} MB")

        # Test 2: Optimized (categorical + compression)
        print("\n2. Optimized (categorical + zstd):")
        path_optimized = tmpdir / "optimized.parquet"
        write_optimized_parquet(
            df,
            path_optimized,
            compression="zstd",
            use_categorical=True,
        )

        mem_before = get_memory_usage_mb()
        feed_optimized = PolarsDataFeed(
            price_path=path_optimized,
            asset_id="SYM0001",
            use_categorical=True,
        )
        _ = feed_optimized.get_next_event()
        mem_after = get_memory_usage_mb()
        mem_optimized = mem_after - mem_before

        print(f"  Memory usage: {mem_optimized:.2f} MB")

        # Summary
        print("\n" + "-" * 80)
        print("Memory Target Summary:")
        print("-" * 80)
        print(f"Target: <2000 MB for 250 symbols × 252 days")
        print(f"Baseline: {mem_baseline:.2f} MB")
        print(f"Optimized: {mem_optimized:.2f} MB")
        print(f"Reduction: {(1 - mem_optimized/mem_baseline)*100:.1f}%")

        target_met = mem_optimized < 2000
        print(f"\nTarget met: {'✓ YES' if target_met else '✗ NO'}")

        return {
            "target_mb": 2000,
            "baseline_mb": mem_baseline,
            "optimized_mb": mem_optimized,
            "target_met": target_met,
        }


def benchmark_lazy_evaluation():
    """Verify lazy evaluation doesn't load data until needed."""
    print("\n" + "=" * 80)
    print("BENCHMARK 5: Lazy Evaluation")
    print("=" * 80)

    df = create_large_dataset(n_symbols=250, n_days=252, bars_per_day=1)

    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        path = tmpdir / "data.parquet"
        df.write_parquet(path)

        # Measure memory before and after feed creation
        print("\n1. Creating PolarsDataFeed (should not load data):")
        mem_before = get_memory_usage_mb()
        feed = PolarsDataFeed(price_path=path, asset_id="SYM0001")
        mem_after = get_memory_usage_mb()
        mem_construction = mem_after - mem_before

        print(f"  Memory delta after construction: {mem_construction:.2f} MB")
        print(f"  Feed initialized: {feed._initialized}")

        # Measure memory after first event (triggers loading)
        print("\n2. Getting first event (triggers data loading):")
        mem_before = get_memory_usage_mb()
        _ = feed.get_next_event()
        mem_after = get_memory_usage_mb()
        mem_first_event = mem_after - mem_before

        print(f"  Memory delta after first event: {mem_first_event:.2f} MB")
        print(f"  Feed initialized: {feed._initialized}")

        # Summary
        print("\n" + "-" * 80)
        print("Lazy Evaluation Summary:")
        print("-" * 80)
        print(f"Construction memory: {mem_construction:.2f} MB (should be ~0)")
        print(f"First event memory: {mem_first_event:.2f} MB (actual data load)")

        is_lazy = mem_construction < 10  # Allow 10MB overhead for structures
        print(f"\nLazy evaluation verified: {'✓ YES' if is_lazy else '✗ NO'}")

        return {
            "construction_mb": mem_construction,
            "first_event_mb": mem_first_event,
            "is_lazy": is_lazy,
        }


def run_all_benchmarks():
    """Run all optimization benchmarks."""
    print("\n" + "=" * 80)
    print("POLARS OPTIMIZATION BENCHMARKS")
    print("=" * 80)

    results = {}

    # Run benchmarks
    results["compression"] = benchmark_compression()
    results["categorical"] = benchmark_categorical_encoding()
    results["partitioning"] = benchmark_partitioning()
    results["memory_target"] = benchmark_memory_target()
    results["lazy_eval"] = benchmark_lazy_evaluation()

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    # Compression
    print("\n1. Compression (zstd vs uncompressed):")
    comp = results["compression"]
    zstd_size = comp["zstd"]["size_mb"]
    none_size = comp["none"]["size_mb"]
    print(f"   Size reduction: {(1 - zstd_size/none_size)*100:.1f}%")

    # Categorical
    print("\n2. Categorical encoding:")
    cat = results["categorical"]
    print(f"   Memory reduction: {cat['memory_reduction_pct']:.1f}%")

    # Partitioning
    print("\n3. Partitioning:")
    part = results["partitioning"]
    print(f"   Query speedup: {part['speedup']:.2f}x")

    # Memory target
    print("\n4. Memory target:")
    mem = results["memory_target"]
    print(f"   Target: <2000 MB")
    print(f"   Actual: {mem['optimized_mb']:.2f} MB")
    print(f"   Status: {'✓ PASS' if mem['target_met'] else '✗ FAIL'}")

    # Lazy evaluation
    print("\n5. Lazy evaluation:")
    lazy = results["lazy_eval"]
    print(f"   Status: {'✓ PASS' if lazy['is_lazy'] else '✗ FAIL'}")

    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
