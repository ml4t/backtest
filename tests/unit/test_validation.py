"""Unit tests for data validation module."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest.data.validation import (
    SignalTimingMode,
    SignalTimingViolation,
    validate_missing_data,
    validate_no_duplicate_timestamps,
    validate_ohlc_consistency,
    validate_signal_timing,
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
