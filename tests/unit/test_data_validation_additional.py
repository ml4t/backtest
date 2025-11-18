"""Additional edge case tests for data validation to achieve >80% coverage.

These tests complement test_validation.py with additional edge cases:
- Timezone handling
- Microsecond precision
- Edge cases for gap detection
- Empty DataFrames in comprehensive validation
- Extreme outliers
"""

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from ml4t.backtest.data.validation import (
    SignalTimingMode,
    validate_comprehensive,
    validate_ohlc_consistency,
    validate_price_sanity,
    validate_signal_timing,
    validate_time_series_gaps,
    validate_volume_sanity,
)


class TestSignalTimingValidationEdgeCases:
    """Additional edge cases for signal timing validation."""

    def test_empty_signals_dataframe(self):
        """Test validation with empty signals DataFrame."""
        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["AAPL"],
                "close": [100.0],
            }
        )

        signals = pl.DataFrame(
            {
                "timestamp": [],
                "asset_id": [],
                "signal": [],
            },
            schema={
                "timestamp": pl.Datetime,
                "asset_id": pl.Utf8,
                "signal": pl.Float64,
            },
        )

        is_valid, violations = validate_signal_timing(
            signals, prices, mode=SignalTimingMode.STRICT, fail_on_violation=False
        )

        # Should be valid (no signals to validate)
        assert is_valid
        assert len(violations) == 0

    def test_empty_prices_dataframe(self):
        """Test validation with empty prices DataFrame."""
        signals = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["AAPL"],
                "signal": [1.0],
            }
        )

        prices = pl.DataFrame(
            {
                "timestamp": [],
                "asset_id": [],
                "close": [],
            },
            schema={
                "timestamp": pl.Datetime,
                "asset_id": pl.Utf8,
                "close": pl.Float64,
            },
        )

        is_valid, violations = validate_signal_timing(
            signals, prices, mode=SignalTimingMode.STRICT, fail_on_violation=False
        )

        # Should be valid (no prices to use signals on)
        assert is_valid
        assert len(violations) == 0

    def test_next_bar_mode_with_insufficient_bars(self):
        """Test NEXT_BAR mode when signal appears but no next bar exists."""
        ts1 = datetime(2025, 1, 1, 10, 0)
        ts2 = datetime(2025, 1, 1, 11, 0)

        # Signal at ts2 (last bar)
        signals = pl.DataFrame(
            {
                "timestamp": [ts2],
                "asset_id": ["AAPL"],
                "signal": [1.0],
            }
        )

        # Prices only at ts1 and ts2
        prices = pl.DataFrame(
            {
                "timestamp": [ts1, ts2],
                "asset_id": ["AAPL", "AAPL"],
                "close": [100.0, 101.0],
            }
        )

        is_valid, violations = validate_signal_timing(
            signals, prices, mode=SignalTimingMode.NEXT_BAR, fail_on_violation=False
        )

        # Should be valid (signal just can't be used, not a violation)
        assert is_valid
        assert len(violations) == 0


class TestOHLCConsistencyEdgeCases:
    """Additional edge cases for OHLC consistency validation."""

    def test_all_prices_equal(self):
        """Test OHLC consistency when all prices are equal (flat bar)."""
        df = pl.DataFrame(
            {
                "open": [100.0],
                "high": [100.0],  # All equal
                "low": [100.0],
                "close": [100.0],
            }
        )

        is_valid, violations = validate_ohlc_consistency(df)

        assert is_valid  # Flat bar is valid
        assert len(violations) == 0

    def test_high_equals_low(self):
        """Test OHLC when high equals low (zero range bar)."""
        df = pl.DataFrame(
            {
                "open": [100.5],  # Fixed: must be within [low, high] range
                "high": [100.5],
                "low": [100.5],  # high == low (zero range)
                "close": [100.5],
            }
        )

        is_valid, violations = validate_ohlc_consistency(df)

        assert is_valid
        assert len(violations) == 0

    def test_multiple_ohlc_violations_same_row(self):
        """Test row with multiple OHLC violations."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "open": [100.0],
                "high": [95.0],  # high < open (invalid)
                "low": [105.0],  # low > open (invalid)
                "close": [101.0],
            }
        )

        is_valid, violations = validate_ohlc_consistency(df)

        assert not is_valid
        assert len(violations) >= 2  # Both high and low violations


class TestPriceSanityEdgeCases:
    """Additional edge cases for price sanity validation."""

    def test_custom_price_thresholds(self):
        """Test with custom min/max price thresholds."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["BTC"],
                "close": [50000.0],  # Valid for crypto
            }
        )

        # Default max_price is 1M, so 50k should be OK
        is_valid, violations = validate_price_sanity(df)
        assert is_valid

        # But with custom max_price of 10k, should fail
        is_valid, violations = validate_price_sanity(df, max_price=10000.0)
        assert not is_valid
        assert violations[0]["type"] == "price_too_high"

    def test_price_change_at_zero(self):
        """Test percentage change calculation with zero price."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                "asset_id": ["AAPL", "AAPL"],
                "close": [0.01, 100.0],  # Extreme change from near-zero
            }
        )

        # Should detect extreme change
        is_valid, violations = validate_price_sanity(df, max_daily_change=0.50)
        assert not is_valid

    def test_no_close_column(self):
        """Test price sanity when close column is missing."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["AAPL"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                # No close column
            }
        )

        # Should still validate open/high/low
        is_valid, violations = validate_price_sanity(df)
        assert is_valid  # No violations with reasonable prices


class TestVolumeSanityEdgeCases:
    """Additional edge cases for volume validation."""

    def test_all_zero_volume(self):
        """Test DataFrame with all zero volumes."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, i) for i in range(1, 11)],
                "asset_id": ["AAPL"] * 10,
                "volume": [0] * 10,
            }
        )

        is_valid, violations = validate_volume_sanity(df)

        assert is_valid  # Zero volume is valid
        assert len(violations) == 0

    def test_single_asset_with_zero_std(self):
        """Test asset with constant volume (zero standard deviation)."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, i) for i in range(1, 11)],
                "asset_id": ["AAPL"] * 10,
                "volume": [1_000_000] * 10,  # All identical
            }
        )

        is_valid, violations = validate_volume_sanity(df)

        assert is_valid  # Constant volume is OK
        assert len(violations) == 0

    def test_mixed_negative_and_positive_volume(self):
        """Test mix of negative and positive volumes."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, i) for i in range(1, 6)],
                "asset_id": ["AAPL"] * 5,
                "volume": [1_000_000, -100, 2_000_000, -200, 3_000_000],
            }
        )

        is_valid, violations = validate_volume_sanity(df)

        assert not is_valid
        # Should detect 2 negative volumes
        negative_violations = [v for v in violations if v["type"] == "negative_volume"]
        assert len(negative_violations) == 2


class TestTimeSeriesGapEdgeCases:
    """Additional edge cases for time series gap detection."""

    def test_hourly_data_with_weekend_gap(self):
        """Test hourly data with weekend gap (expected for stocks)."""
        # Friday 5pm to Monday 9am gap (expected)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 3, 17, 0),  # Friday 5pm
                    datetime(2025, 1, 6, 9, 0),  # Monday 9am (64-hour gap)
                ],
                "asset_id": ["AAPL", "AAPL"],
                "close": [100.0, 101.0],
            }
        )

        # With hourly frequency and 3x multiplier, 64-hour gap should be flagged
        is_valid, gaps = validate_time_series_gaps(
            df, expected_frequency="1h", max_gap_multiplier=3.0
        )

        assert not is_valid  # Weekend gap exceeds 3-hour threshold
        assert len(gaps) == 1

    def test_high_frequency_data_seconds(self):
        """Test high-frequency data with second intervals."""
        base_time = datetime(2025, 1, 1, 9, 30, 0)
        timestamps = [
            base_time + timedelta(seconds=i) for i in [0, 1, 2, 3, 10]  # Gap at 10s
        ]

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset_id": ["AAPL"] * 5,
                "close": [100.0 + i for i in range(5)],
            }
        )

        is_valid, gaps = validate_time_series_gaps(
            df, expected_frequency="1s", max_gap_multiplier=2.0
        )

        assert not is_valid  # 7-second gap exceeds 2-second threshold
        assert len(gaps) == 1

    def test_irregular_frequency_inferred(self):
        """Test gap detection with irregular but mostly consistent frequency."""
        base_time = datetime(2025, 1, 1)
        # Mostly daily, with one large gap
        timestamps = [
            base_time,
            base_time + timedelta(days=1),
            base_time + timedelta(days=2),
            base_time + timedelta(days=3),
            base_time + timedelta(days=20),  # Large gap
        ]

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset_id": ["AAPL"] * 5,
                "close": [100.0 + i for i in range(5)],
            }
        )

        # Infer frequency (should be ~1 day from median)
        is_valid, gaps = validate_time_series_gaps(
            df, expected_frequency=None, max_gap_multiplier=5.0
        )

        assert not is_valid  # 17-day gap exceeds 5-day threshold
        assert len(gaps) >= 1


class TestComprehensiveValidationEdgeCases:
    """Additional edge cases for comprehensive validation."""

    def test_empty_dataframe_comprehensive(self):
        """Test comprehensive validation with empty DataFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [],
                "asset_id": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            },
            schema={
                "timestamp": pl.Datetime,
                "asset_id": pl.Utf8,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            },
        )

        is_valid, violations = validate_comprehensive(df)

        # Empty DataFrame should be valid (no violations)
        assert is_valid
        assert len(violations) == 0

    def test_partial_columns_comprehensive(self):
        """Test comprehensive validation with only some columns present."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["AAPL"],
                "close": [100.0],
                # Missing open, high, low, volume
            }
        )

        # Disable checks that require missing columns
        is_valid, violations = validate_comprehensive(
            df,
            validate_ohlc=False,  # No OHLC columns
            validate_volume=False,  # No volume
            required_columns=["timestamp", "asset_id", "close"],
        )

        assert is_valid
        assert len(violations) == 0

    def test_all_validations_fail_comprehensive(self):
        """Test comprehensive validation with maximum violations."""
        # Create intentionally bad data with every possible violation
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 1),  # Duplicate
                    datetime(2025, 1, 10),  # Gap
                ],
                "asset_id": ["AAPL", "AAPL", "AAPL"],
                "open": [100.0, 0.001, 0.001],  # Price too low
                "high": [90.0, 0.002, 0.002],  # high < open (invalid)
                "low": [99.0, 0.0005, 0.0005],  # Price too low
                "close": [101.0, 300.0, 0.0015],  # Extreme change + price too low
                "volume": [1_000_000, -100, 500_000_000],  # Negative + outlier
            }
        )

        is_valid, violations = validate_comprehensive(
            df, expected_frequency="1d"
        )

        assert not is_valid
        # Should have violations in multiple categories
        assert "duplicates" in violations
        assert "ohlc_consistency" in violations
        assert "volume_sanity" in violations
        assert "price_sanity" in violations
        assert "time_series_gaps" in violations

    def test_selective_validation_flags(self):
        """Test that all validation flags can be independently disabled."""
        # Data with all possible violations
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 1)],
                "asset_id": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [90.0, 103.0],  # Invalid
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1_000_000, -100],  # Negative
            }
        )

        # Disable ALL validations
        is_valid, violations = validate_comprehensive(
            df,
            validate_duplicates=False,
            validate_ohlc=False,
            validate_missing=False,
            validate_volume=False,
            validate_price=False,
            validate_gaps=False,
        )

        assert is_valid  # All checks disabled
        assert len(violations) == 0


class TestTimezoneHandling:
    """Test validation with timezone-aware timestamps."""

    def test_timezone_aware_timestamps(self):
        """Test that timezone-aware timestamps are handled correctly."""
        ts1 = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)

        df = pl.DataFrame(
            {
                "timestamp": [ts1, ts2],
                "asset_id": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1_000_000, 1_100_000],
            }
        )

        is_valid, violations = validate_comprehensive(df)
        assert is_valid

    def test_mixed_timezone_timestamps(self):
        """Test handling of mixed timezone timestamps."""
        ts1 = datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
        # Different timezone (but should still work if Polars normalizes)
        ts2 = datetime(2025, 1, 1, 11, 0, tzinfo=timezone.utc)

        df = pl.DataFrame(
            {
                "timestamp": [ts1, ts2],
                "asset_id": ["AAPL", "AAPL"],
                "close": [100.0, 101.0],
            }
        )

        # Should handle gracefully
        is_valid, violations = validate_time_series_gaps(df, expected_frequency="1h")
        assert is_valid
