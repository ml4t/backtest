"""Unit tests for data validation module."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest.data.validation import (
    SignalTimingMode,
    SignalTimingViolation,
    _parse_frequency_to_timedelta,
    validate_comprehensive,
    validate_missing_data,
    validate_no_duplicate_timestamps,
    validate_ohlc_consistency,
    validate_price_sanity,
    validate_signal_timing,
    validate_time_series_gaps,
    validate_volume_sanity,
)


class TestSignalTimingValidation:
    """Test signal timing validation to prevent look-ahead bias."""

    def test_strict_mode_valid(self):
        """Test STRICT mode with aligned timestamps (valid)."""
        ts1 = datetime(2025, 1, 1, 10, 0)
        ts2 = datetime(2025, 1, 1, 11, 0)

        signals = pl.DataFrame(
            {
                "timestamp": [ts1, ts2],
                "asset_id": ["AAPL", "AAPL"],
                "signal": [1.0, -1.0],
            }
        )

        prices = pl.DataFrame(
            {
                "timestamp": [ts1, ts2],
                "asset_id": ["AAPL", "AAPL"],
                "close": [100.0, 101.0],
            }
        )

        is_valid, violations = validate_signal_timing(
            signals, prices, mode=SignalTimingMode.STRICT, fail_on_violation=False
        )

        assert is_valid
        assert len(violations) == 0

    def test_next_bar_mode_valid(self):
        """Test NEXT_BAR mode with signal used on next bar (valid)."""
        ts1 = datetime(2025, 1, 1, 10, 0)
        ts2 = datetime(2025, 1, 1, 11, 0)
        ts3 = datetime(2025, 1, 1, 12, 0)

        # Signal appears at ts1, should be used at ts2 or later
        signals = pl.DataFrame(
            {
                "timestamp": [ts1],
                "asset_id": ["AAPL"],
                "signal": [1.0],
            }
        )

        prices = pl.DataFrame(
            {
                "timestamp": [ts1, ts2, ts3],
                "asset_id": ["AAPL", "AAPL", "AAPL"],
                "close": [100.0, 101.0, 102.0],
            }
        )

        is_valid, violations = validate_signal_timing(
            signals, prices, mode=SignalTimingMode.NEXT_BAR, fail_on_violation=False
        )

        assert is_valid
        assert len(violations) == 0

    def test_lookahead_bias_detected(self):
        """Test detection of look-ahead bias (signal used before available).

        In NEXT_BAR mode, if signal appears at 11:00, it can only be used starting
        from 12:00. If there's no 12:00 bar, the signal can't be used without
        look-ahead bias.
        """
        ts1 = datetime(2025, 1, 1, 10, 0)
        ts2 = datetime(2025, 1, 1, 11, 0)
        # No ts3 - this is the key: signal at 11:00 has no "next bar" to use it on

        signals = pl.DataFrame(
            {
                "timestamp": [ts1],  # Signal at 10:00
                "asset_id": ["AAPL"],
                "signal": [1.0],
            }
        )

        prices = pl.DataFrame(
            {
                "timestamp": [ts2],  # Only price at 11:00 (AFTER signal)
                "asset_id": ["AAPL"],
                "close": [101.0],
            }
        )

        # In NEXT_BAR mode, signal at 10:00 would need to be used at 11:00+
        # But there are no prices after 11:00, so this is actually valid
        # (signal just wasn't used)
        #
        # For true look-ahead bias test, we need signals that come AFTER prices
        signals_lookahead = pl.DataFrame(
            {
                "timestamp": [ts2],  # Signal at 11:00
                "asset_id": ["AAPL"],
                "signal": [1.0],
            }
        )

        prices_before = pl.DataFrame(
            {
                "timestamp": [ts1],  # Only price at 10:00 (BEFORE signal)
                "asset_id": ["AAPL"],
                "close": [100.0],
            }
        )

        # If we're in STRICT mode and signal is at 11:00 but price is at 10:00,
        # this would require using future signal for past decision
        is_valid, violations = validate_signal_timing(
            signals_lookahead,
            prices_before,
            mode=SignalTimingMode.STRICT,
            fail_on_violation=False,
        )

        assert not is_valid
        assert len(violations) > 0
        assert violations[0]["severity"] == "CRITICAL"

    def test_lookahead_bias_raises_exception(self):
        """Test that look-ahead bias raises exception when fail_on_violation=True."""
        ts1 = datetime(2025, 1, 1, 10, 0)
        ts2 = datetime(2025, 1, 1, 11, 0)

        # Signal at 11:00, price at 10:00 - would require using future signal
        signals = pl.DataFrame(
            {
                "timestamp": [ts2],  # Signal at 11:00
                "asset_id": ["AAPL"],
                "signal": [1.0],
            }
        )

        prices = pl.DataFrame(
            {
                "timestamp": [ts1],  # Price at 10:00 ONLY (before signal)
                "asset_id": ["AAPL"],
                "close": [100.0],
            }
        )

        with pytest.raises(SignalTimingViolation) as exc_info:
            validate_signal_timing(
                signals, prices, mode=SignalTimingMode.STRICT, fail_on_violation=True
            )

        assert "Look-ahead bias detected" in str(exc_info.value)

    def test_custom_lag_mode(self):
        """Test CUSTOM mode with N-bar lag."""
        ts1 = datetime(2025, 1, 1, 10, 0)
        ts2 = datetime(2025, 1, 1, 11, 0)
        ts3 = datetime(2025, 1, 1, 12, 0)
        ts4 = datetime(2025, 1, 1, 13, 0)

        # Signal at ts1, with 2-bar lag should be used at ts3
        signals = pl.DataFrame(
            {
                "timestamp": [ts1],
                "asset_id": ["AAPL"],
                "signal": [1.0],
            }
        )

        prices = pl.DataFrame(
            {
                "timestamp": [ts1, ts2, ts3, ts4],
                "asset_id": ["AAPL"] * 4,
                "close": [100.0, 101.0, 102.0, 103.0],
            }
        )

        is_valid, violations = validate_signal_timing(
            signals,
            prices,
            mode=SignalTimingMode.CUSTOM,
            custom_lag_bars=2,
            fail_on_violation=False,
        )

        assert is_valid
        assert len(violations) == 0

    def test_signal_after_all_prices(self):
        """Test signal that appears after all price data (not a violation)."""
        ts1 = datetime(2025, 1, 1, 10, 0)
        ts2 = datetime(2025, 1, 1, 11, 0)
        ts3 = datetime(2025, 1, 1, 12, 0)

        # Signal at ts3, but prices only up to ts2
        signals = pl.DataFrame(
            {
                "timestamp": [ts3],
                "asset_id": ["AAPL"],
                "signal": [1.0],
            }
        )

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

        # Should be valid (signal just wasn't used, not a timing violation)
        assert is_valid
        assert len(violations) == 0

    def test_multiple_assets(self):
        """Test validation across multiple assets."""
        ts1 = datetime(2025, 1, 1, 10, 0)
        ts2 = datetime(2025, 1, 1, 11, 0)

        signals = pl.DataFrame(
            {
                "timestamp": [ts1, ts1],
                "asset_id": ["AAPL", "MSFT"],
                "signal": [1.0, -1.0],
            }
        )

        prices = pl.DataFrame(
            {
                "timestamp": [ts1, ts2, ts1, ts2],
                "asset_id": ["AAPL", "AAPL", "MSFT", "MSFT"],
                "close": [100.0, 101.0, 200.0, 201.0],
            }
        )

        is_valid, violations = validate_signal_timing(
            signals, prices, mode=SignalTimingMode.NEXT_BAR, fail_on_violation=False
        )

        assert is_valid
        assert len(violations) == 0


class TestNoDuplicateTimestamps:
    """Test duplicate timestamp detection."""

    def test_no_duplicates(self):
        """Test DataFrame without duplicates."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1, 10, 0),
                    datetime(2025, 1, 1, 11, 0),
                ],
                "asset_id": ["AAPL", "AAPL"],
                "close": [100.0, 101.0],
            }
        )

        is_valid, duplicates = validate_no_duplicate_timestamps(df)

        assert is_valid
        assert len(duplicates) == 0

    def test_duplicates_detected(self):
        """Test detection of duplicate timestamps."""
        ts = datetime(2025, 1, 1, 10, 0)

        df = pl.DataFrame(
            {
                "timestamp": [ts, ts, ts],  # 3 rows with same timestamp
                "asset_id": ["AAPL", "AAPL", "AAPL"],
                "close": [100.0, 100.5, 101.0],
            }
        )

        is_valid, duplicates = validate_no_duplicate_timestamps(df)

        assert not is_valid
        assert len(duplicates) == 1
        assert duplicates[0]["count"] == 3

    def test_duplicates_different_assets(self):
        """Test that same timestamp for different assets is OK."""
        ts = datetime(2025, 1, 1, 10, 0)

        df = pl.DataFrame(
            {
                "timestamp": [ts, ts],
                "asset_id": ["AAPL", "MSFT"],  # Different assets
                "close": [100.0, 200.0],
            }
        )

        is_valid, duplicates = validate_no_duplicate_timestamps(df)

        assert is_valid  # Same timestamp, different assets is OK
        assert len(duplicates) == 0


class TestOHLCConsistency:
    """Test OHLC price relationship validation."""

    def test_valid_ohlc(self):
        """Test valid OHLC relationships."""
        df = pl.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
            }
        )

        is_valid, violations = validate_ohlc_consistency(df)

        assert is_valid
        assert len(violations) == 0

    def test_high_less_than_close(self):
        """Test detection of high < close (invalid)."""
        df = pl.DataFrame(
            {
                "open": [100.0],
                "high": [100.5],  # High less than close
                "low": [99.0],
                "close": [101.0],  # Close higher than high
            }
        )

        is_valid, violations = validate_ohlc_consistency(df)

        assert not is_valid
        assert len(violations) > 0
        assert violations[0]["type"] == "invalid_high"

    def test_low_greater_than_close(self):
        """Test detection of low > close (invalid)."""
        df = pl.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [101.5],  # Low greater than close
                "close": [101.0],
            }
        )

        is_valid, violations = validate_ohlc_consistency(df)

        assert not is_valid
        assert len(violations) > 0
        assert violations[0]["type"] == "invalid_low"

    def test_non_positive_prices(self):
        """Test detection of non-positive prices."""
        df = pl.DataFrame(
            {
                "open": [0.0],  # Non-positive
                "high": [102.0],
                "low": [-1.0],  # Negative
                "close": [101.0],
            }
        )

        is_valid, violations = validate_ohlc_consistency(df)

        assert not is_valid
        assert len(violations) >= 2  # At least open=0 and low<0


class TestMissingData:
    """Test missing data detection."""

    def test_no_missing_data(self):
        """Test DataFrame without missing data."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["AAPL"],
                "close": [100.0],
            }
        )

        is_valid, missing = validate_missing_data(
            df, required_columns=["timestamp", "asset_id", "close"]
        )

        assert is_valid
        assert len(missing) == 0

    def test_null_values_detected(self):
        """Test detection of null values."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                "asset_id": ["AAPL", None],  # Null value
                "close": [100.0, 101.0],
            }
        )

        is_valid, missing = validate_missing_data(
            df, required_columns=["timestamp", "asset_id", "close"]
        )

        assert not is_valid
        assert len(missing) == 1
        assert missing[0]["column"] == "asset_id"
        assert missing[0]["count"] == 1

    def test_missing_column(self):
        """Test detection of completely missing column."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "close": [100.0],
            }
        )

        is_valid, missing = validate_missing_data(
            df, required_columns=["timestamp", "asset_id", "close"]
        )

        assert not is_valid
        assert len(missing) == 1
        assert missing[0]["column"] == "asset_id"
        assert missing[0]["type"] == "missing_column"


class TestVolumeSanityValidation:
    """Test volume sanity validation."""

    def test_valid_volumes(self):
        """Test validation passes with valid volume data."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                "asset_id": ["AAPL", "AAPL"],
                "volume": [1_000_000, 2_000_000],
            }
        )

        is_valid, violations = validate_volume_sanity(df)

        assert is_valid
        assert len(violations) == 0

    def test_negative_volume_detected(self):
        """Test detection of negative volume (critical error)."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                "asset_id": ["AAPL", "AAPL"],
                "volume": [1_000_000, -100],
            }
        )

        is_valid, violations = validate_volume_sanity(df)

        assert not is_valid
        assert len(violations) == 1
        assert violations[0]["type"] == "negative_volume"
        assert violations[0]["severity"] == "CRITICAL"
        assert violations[0]["volume"] == -100

    def test_volume_outlier_detected(self):
        """Test detection of extreme volume outliers."""
        # More normal volumes, then one extreme outlier
        # With more data points, the outlier won't inflate std as much
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, i + 1) for i in range(20)
                ],
                "asset_id": ["AAPL"] * 20,
                "volume": [1_000_000] * 19 + [500_000_000],  # 500x outlier
            }
        )

        # Use stricter threshold to ensure detection
        is_valid, violations = validate_volume_sanity(df, max_outlier_std=3.0)

        assert not is_valid
        assert len(violations) >= 1
        assert violations[0]["type"] == "volume_outlier"
        assert violations[0]["severity"] == "WARNING"

    def test_zero_volume_allowed(self):
        """Test that zero volume is valid (not negative)."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["AAPL"],
                "volume": [0],
            }
        )

        is_valid, violations = validate_volume_sanity(df)

        assert is_valid
        assert len(violations) == 0


class TestTimeSeriesGapValidation:
    """Test time series gap detection."""

    def test_no_gaps_in_daily_data(self):
        """Test validation passes with complete daily data."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1) + timedelta(days=i) for i in range(5)
                ],
                "asset_id": ["AAPL"] * 5,
                "close": [100.0 + i for i in range(5)],
            }
        )

        is_valid, gaps = validate_time_series_gaps(df, expected_frequency="1d")

        assert is_valid
        assert len(gaps) == 0

    def test_gap_detected_in_daily_data(self):
        """Test detection of missing days in daily data."""
        # Missing days 2, 3, 4 (jump from day 1 to day 5)
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 5),  # Gap of 4 days
                ],
                "asset_id": ["AAPL", "AAPL"],
                "close": [100.0, 104.0],
            }
        )

        is_valid, gaps = validate_time_series_gaps(df, expected_frequency="1d", max_gap_multiplier=2.0)

        assert not is_valid
        assert len(gaps) == 1
        assert gaps[0]["type"] == "time_series_gap"
        assert gaps[0]["severity"] == "WARNING"
        assert gaps[0]["asset_id"] == "AAPL"

    def test_gap_detection_with_inferred_frequency(self):
        """Test gap detection with auto-inferred frequency from median."""
        # Daily data with one large gap
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 2),
                    datetime(2025, 1, 3),
                    datetime(2025, 1, 10),  # 7-day gap
                ],
                "asset_id": ["AAPL"] * 4,
                "close": [100.0, 101.0, 102.0, 109.0],
            }
        )

        is_valid, gaps = validate_time_series_gaps(df, expected_frequency=None, max_gap_multiplier=3.0)

        assert not is_valid
        assert len(gaps) == 1

    def test_empty_dataframe(self):
        """Test validation handles empty DataFrame."""
        df = pl.DataFrame(
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

        is_valid, gaps = validate_time_series_gaps(df)

        assert is_valid
        assert len(gaps) == 0

    def test_single_row_dataframe(self):
        """Test validation handles single-row DataFrame (no gaps possible)."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["AAPL"],
                "close": [100.0],
            }
        )

        is_valid, gaps = validate_time_series_gaps(df)

        assert is_valid
        assert len(gaps) == 0

    def test_multi_asset_gap_detection(self):
        """Test gap detection works independently for multiple assets."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 2),  # AAPL: no gap
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 10),  # MSFT: 9-day gap
                ],
                "asset_id": ["AAPL", "AAPL", "MSFT", "MSFT"],
                "close": [100.0, 101.0, 200.0, 209.0],
            }
        )

        is_valid, gaps = validate_time_series_gaps(df, expected_frequency="1d", max_gap_multiplier=3.0)

        assert not is_valid
        assert len(gaps) == 1
        assert gaps[0]["asset_id"] == "MSFT"


class TestPriceSanityValidation:
    """Test price sanity validation."""

    def test_valid_prices(self):
        """Test validation passes with reasonable prices."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                "asset_id": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
            }
        )

        is_valid, violations = validate_price_sanity(df)

        assert is_valid
        assert len(violations) == 0

    def test_price_too_low_detected(self):
        """Test detection of suspiciously low prices."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["AAPL"],
                "open": [0.001],  # Below default min_price of 0.01
                "high": [0.002],
                "low": [0.0005],
                "close": [0.0015],
            }
        )

        is_valid, violations = validate_price_sanity(df, min_price=0.01)

        assert not is_valid
        assert len(violations) == 4  # All 4 OHLC columns below minimum
        for v in violations:
            assert v["type"] == "price_too_low"
            assert v["severity"] == "WARNING"

    def test_price_too_high_detected(self):
        """Test detection of suspiciously high prices."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["AAPL"],
                "close": [2_000_000.0],  # Above default max_price of 1M
            }
        )

        is_valid, violations = validate_price_sanity(df, max_price=1_000_000.0)

        assert not is_valid
        assert len(violations) == 1
        assert violations[0]["type"] == "price_too_high"
        assert violations[0]["price"] == 2_000_000.0

    def test_extreme_price_change_detected(self):
        """Test detection of extreme percentage changes."""
        # 100% increase in one bar (from 100 to 200)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                "asset_id": ["AAPL", "AAPL"],
                "close": [100.0, 200.0],
            }
        )

        is_valid, violations = validate_price_sanity(df, max_daily_change=0.20)  # 20% max

        assert not is_valid
        assert len(violations) == 1
        assert violations[0]["type"] == "extreme_price_change"
        assert violations[0]["severity"] == "WARNING"
        assert violations[0]["pct_change"] > 0.20

    def test_normal_price_change_allowed(self):
        """Test that normal price changes are not flagged."""
        # 5% increase (well within 50% default threshold)
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                "asset_id": ["AAPL", "AAPL"],
                "close": [100.0, 105.0],
            }
        )

        is_valid, violations = validate_price_sanity(df)

        assert is_valid
        assert len(violations) == 0


class TestComprehensiveValidation:
    """Test comprehensive validation orchestrator."""

    def test_all_validations_pass(self):
        """Test comprehensive validation with clean data."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                "asset_id": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1_000_000, 1_100_000],
            }
        )

        is_valid, violations = validate_comprehensive(df, expected_frequency="1d")

        assert is_valid
        assert len(violations) == 0

    def test_multiple_violations_detected(self):
        """Test comprehensive validation detects multiple types of violations."""
        # Create data with multiple issues:
        # - Duplicate timestamp
        # - OHLC inconsistency
        # - Negative volume
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2025, 1, 1),
                    datetime(2025, 1, 1),  # Duplicate
                ],
                "asset_id": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [90.0, 103.0],  # First high < low (invalid)
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1_000_000, -100],  # Negative volume
            }
        )

        is_valid, violations = validate_comprehensive(df)

        assert not is_valid
        assert len(violations) >= 2  # Should have duplicates and volume violations
        assert "duplicates" in violations
        assert "volume_sanity" in violations

    def test_selective_validation(self):
        """Test that individual validations can be disabled."""
        # Data with duplicate timestamps but otherwise valid
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 1)],
                "asset_id": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [100.0, 101.0],
            }
        )

        # Disable duplicate check
        is_valid, violations = validate_comprehensive(
            df, validate_duplicates=False
        )

        assert is_valid  # Should pass because duplicate check is disabled
        assert len(violations) == 0

    def test_violations_grouped_by_category(self):
        """Test that violations are properly categorized."""
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["AAPL"],
                "open": [100.0],
                "high": [90.0],  # Invalid: high < open
                "low": [99.0],
                "close": [101.0],
                "volume": [-100],  # Negative
            }
        )

        is_valid, violations = validate_comprehensive(df)

        assert not is_valid
        assert "ohlc_consistency" in violations
        assert "volume_sanity" in violations
        assert isinstance(violations["ohlc_consistency"], list)
        assert isinstance(violations["volume_sanity"], list)


class TestFrequencyParsing:
    """Test frequency string parsing helper."""

    def test_parse_seconds(self):
        """Test parsing seconds frequency."""
        td = _parse_frequency_to_timedelta("30s")
        assert td == timedelta(seconds=30)

    def test_parse_minutes(self):
        """Test parsing minutes frequency."""
        td = _parse_frequency_to_timedelta("5m")
        assert td == timedelta(minutes=5)

    def test_parse_hours(self):
        """Test parsing hours frequency."""
        td = _parse_frequency_to_timedelta("4h")
        assert td == timedelta(hours=4)

    def test_parse_days(self):
        """Test parsing days frequency."""
        td = _parse_frequency_to_timedelta("1d")
        assert td == timedelta(days=1)

    def test_parse_weeks(self):
        """Test parsing weeks frequency."""
        td = _parse_frequency_to_timedelta("2w")
        assert td == timedelta(weeks=2)

    def test_invalid_frequency_string(self):
        """Test handling of invalid frequency strings."""
        with pytest.raises(ValueError, match="Cannot parse frequency string"):
            _parse_frequency_to_timedelta("invalid")

        with pytest.raises(ValueError, match="Cannot parse frequency string"):
            _parse_frequency_to_timedelta("1x")
