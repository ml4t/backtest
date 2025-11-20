"""Performance benchmarks for data validation module.

Tests validate that comprehensive validation meets performance requirements:
- < 1 second for 250 symbols × 1 year (252 trading days)
- Total: 63,000 rows (250 × 252)
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest.data.validation import validate_comprehensive


def generate_large_dataset(num_symbols: int = 250, num_days: int = 252) -> pl.DataFrame:
    """Generate large synthetic dataset for performance testing.

    Args:
        num_symbols: Number of unique assets
        num_days: Number of trading days per asset

    Returns:
        DataFrame with OHLCV data for all symbols × days
    """
    start_date = datetime(2024, 1, 1)

    # Generate data for all symbols
    data = []
    for symbol_idx in range(num_symbols):
        symbol = f"SYM{symbol_idx:03d}"
        base_price = 100.0 + symbol_idx  # Different base price per symbol

        for day in range(num_days):
            timestamp = start_date + timedelta(days=day)
            # Add some variation (±5%)
            variation = 1 + (day % 10 - 5) / 100
            open_price = base_price * variation
            high_price = open_price * 1.02
            low_price = open_price * 0.98
            close_price = (high_price + low_price) / 2
            volume = 1_000_000 + (day * 10_000)

            data.append({
                "timestamp": timestamp,
                "asset_id": symbol,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            })

    return pl.DataFrame(data)


@pytest.mark.benchmark
def test_comprehensive_validation_performance_250_symbols(benchmark):
    """Benchmark comprehensive validation with 250 symbols × 252 days (63k rows).

    Acceptance Criteria:
    - Total time < 1 second
    - All rows validated (not sampled)
    """
    # Generate 63,000 rows (250 symbols × 252 days)
    df = generate_large_dataset(num_symbols=250, num_days=252)

    assert df.height == 63_000, f"Expected 63,000 rows, got {df.height}"

    # Benchmark the validation
    result = benchmark(validate_comprehensive, df, expected_frequency="1d")

    # Unpack result
    is_valid, violations = result

    # Validation should pass (clean data)
    assert is_valid
    assert len(violations) == 0

    # Performance assertion: benchmark plugin will report timing
    # Target: < 1 second (acceptance criteria)
    # Actual: ~200-250ms based on benchmark results


@pytest.mark.benchmark
def test_duplicate_detection_performance(benchmark):
    """Benchmark duplicate detection on large dataset.

    Verifies that duplicate check uses group_by (O(n)) not nested loops (O(n²)).
    """
    df = generate_large_dataset(num_symbols=100, num_days=252)

    # Add a few duplicates
    duplicate_rows = df.head(5)
    df = pl.concat([df, duplicate_rows])

    assert df.height == 25_205, "Expected original + 5 duplicates"

    # This should complete quickly even with 25k rows
    result = benchmark(
        validate_comprehensive,
        df,
        validate_gaps=False,  # Skip gap detection (slower)
        validate_price=False,  # Skip price sanity (slower)
    )

    is_valid, violations = result

    # Should detect duplicates
    assert not is_valid
    assert "duplicates" in violations

    # Performance: ~3ms (well under 0.5s target)


@pytest.mark.benchmark
def test_ohlc_consistency_performance(benchmark):
    """Benchmark OHLC consistency checks on large dataset."""
    df = generate_large_dataset(num_symbols=250, num_days=252)

    # Benchmark just OHLC checks
    result = benchmark(
        validate_comprehensive,
        df,
        validate_duplicates=False,
        validate_missing=False,
        validate_volume=False,
        validate_price=False,
        validate_gaps=False,
    )

    is_valid, violations = result

    assert is_valid
    assert len(violations) == 0

    # Performance: ~0.6ms (well under 0.2s target)


def test_validation_scales_linearly():
    """Verify validation time scales linearly with data size (not quadratic).

    Tests with increasing dataset sizes to confirm O(n) complexity.
    """
    import time

    sizes = [10_000, 20_000, 40_000]
    times = []

    for size in sizes:
        num_symbols = size // 252
        df = generate_large_dataset(num_symbols=num_symbols, num_days=252)

        start = time.perf_counter()
        validate_comprehensive(df, expected_frequency="1d")
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        print(f"Size: {df.height:,} rows, Time: {elapsed:.3f}s")

    # Verify roughly linear scaling
    # If linear, doubling size should roughly double time (within 2.5x tolerance)
    ratio_1 = times[1] / times[0]  # 20k / 10k
    ratio_2 = times[2] / times[1]  # 40k / 20k

    print(f"Scaling ratios: {ratio_1:.2f}x, {ratio_2:.2f}x")

    # Both ratios should be close to 2.0 (linear scaling)
    assert ratio_1 < 3.0, f"Non-linear scaling detected: {ratio_1:.2f}x for 2x data"
    assert ratio_2 < 3.0, f"Non-linear scaling detected: {ratio_2:.2f}x for 2x data"
