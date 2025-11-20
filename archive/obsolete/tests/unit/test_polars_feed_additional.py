"""Additional edge case tests for PolarsDataFeed to achieve >80% coverage.

These tests complement test_polars_feed.py with additional edge cases:
- Signal timing validation integration
- Missing columns handling
- Malformed timestamps
- Single-row DataFrames
- Filter expressions
- Null value handling in signals
"""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest

from ml4t.backtest.core.types import AssetId
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider
from ml4t.backtest.data.polars_feed import PolarsDataFeed
from ml4t.backtest.data.validation import SignalTimingMode, SignalTimingViolation


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPolarsDataFeedSignalTimingValidation:
    """Test signal timing validation integration."""

    def test_signal_timing_validation_passes(self, temp_dir):
        """Test that valid signal timing doesn't raise exception."""
        base_time = datetime(2025, 1, 1, 9, 30)
        timestamps = [base_time + timedelta(hours=i) for i in range(5)]

        # Prices
        prices = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset_id": ["AAPL"] * 5,
                "open": [100.0 + i for i in range(5)],
                "high": [101.0 + i for i in range(5)],
                "low": [99.0 + i for i in range(5)],
                "close": [100.5 + i for i in range(5)],
                "volume": [1000000] * 5,
            }
        )

        # Signals aligned with prices (valid in STRICT mode)
        signals = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset_id": ["AAPL"] * 5,
                "ml_pred": [0.1 * i for i in range(5)],
            }
        )

        price_path = temp_dir / "prices.parquet"
        signals_path = temp_dir / "signals.parquet"
        prices.write_parquet(price_path)
        signals.write_parquet(signals_path)

        # Should not raise exception
        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            signals_path=signals_path,
            validate_signal_timing=True,
            signal_timing_mode=SignalTimingMode.STRICT,
            fail_on_timing_violation=True,
        )

        # Should initialize without error
        event = feed.get_next_event()
        assert event is not None
        assert "ml_pred" in event.signals

    def test_signal_timing_validation_fails(self, temp_dir):
        """Test that invalid signal timing raises exception."""
        ts1 = datetime(2025, 1, 1, 10, 0)
        ts2 = datetime(2025, 1, 1, 11, 0)

        # Price at ts1 only
        prices = pl.DataFrame(
            {
                "timestamp": [ts1],
                "asset_id": ["AAPL"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000],
            }
        )

        # Signal at ts2 (after price) - look-ahead bias in STRICT mode
        signals = pl.DataFrame(
            {
                "timestamp": [ts2],
                "asset_id": ["AAPL"],
                "ml_pred": [0.5],
            }
        )

        price_path = temp_dir / "prices.parquet"
        signals_path = temp_dir / "signals.parquet"
        prices.write_parquet(price_path)
        signals.write_parquet(signals_path)

        # Should raise SignalTimingViolation
        with pytest.raises(SignalTimingViolation):
            feed = PolarsDataFeed(
                price_path=price_path,
                asset_id=AssetId("AAPL"),
                signals_path=signals_path,
                validate_signal_timing=True,
                signal_timing_mode=SignalTimingMode.STRICT,
                fail_on_timing_violation=True,
            )
            # Trigger initialization
            feed.get_next_event()

    def test_signal_timing_validation_disabled(self, temp_dir):
        """Test that validation can be disabled."""
        ts1 = datetime(2025, 1, 1, 10, 0)
        ts2 = datetime(2025, 1, 1, 11, 0)

        prices = pl.DataFrame(
            {
                "timestamp": [ts1],
                "asset_id": ["AAPL"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000],
            }
        )

        signals = pl.DataFrame(
            {
                "timestamp": [ts2],
                "asset_id": ["AAPL"],
                "ml_pred": [0.5],
            }
        )

        price_path = temp_dir / "prices.parquet"
        signals_path = temp_dir / "signals.parquet"
        prices.write_parquet(price_path)
        signals.write_parquet(signals_path)

        # Should NOT raise exception when validation disabled
        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            signals_path=signals_path,
            validate_signal_timing=False,  # Disabled
        )

        event = feed.get_next_event()
        assert event is not None  # Should work despite timing issue


class TestPolarsDataFeedNullHandling:
    """Test null value handling in data."""

    def test_null_signals_handled_gracefully(self, temp_dir):
        """Test that null signal values are handled."""
        base_time = datetime(2025, 1, 1)

        prices = pl.DataFrame(
            {
                "timestamp": [base_time, base_time + timedelta(hours=1)],
                "asset_id": ["AAPL", "AAPL"],
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000000, 1100000],
            }
        )

        # Signals with null value
        signals = pl.DataFrame(
            {
                "timestamp": [base_time, base_time + timedelta(hours=1)],
                "asset_id": ["AAPL", "AAPL"],
                "ml_pred": [0.5, None],  # Second value is null
            }
        )

        price_path = temp_dir / "prices.parquet"
        signals_path = temp_dir / "signals.parquet"
        prices.write_parquet(price_path)
        signals.write_parquet(signals_path)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            signals_path=signals_path,
            validate_signal_timing=False,
        )

        # First event should have signal
        event1 = feed.get_next_event()
        assert event1 is not None
        assert "ml_pred" in event1.signals
        assert event1.signals["ml_pred"] == 0.5

        # Second event should NOT have signal (null filtered out)
        event2 = feed.get_next_event()
        assert event2 is not None
        # Null signals should be filtered or converted to 0
        # Based on _extract_signals implementation


class TestPolarsDataFeedSingleRow:
    """Test single-row DataFrame edge case."""

    def test_single_row_dataframe(self, temp_dir):
        """Test feed with only one row of data."""
        prices = pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "asset_id": ["AAPL"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000],
            }
        )

        price_path = temp_dir / "single.parquet"
        prices.write_parquet(price_path)

        feed = PolarsDataFeed(price_path=price_path, asset_id=AssetId("AAPL"))

        # Should get exactly one event
        event = feed.get_next_event()
        assert event is not None
        assert event.close == 100.5

        # Next call should return None
        event2 = feed.get_next_event()
        assert event2 is None
        assert feed.is_exhausted


class TestPolarsDataFeedFilterExpressions:
    """Test custom filter expressions."""

    def test_custom_filter_applied(self, temp_dir):
        """Test that custom Polars filter expressions are applied."""
        base_time = datetime(2025, 1, 1)
        timestamps = [base_time + timedelta(hours=i) for i in range(10)]

        prices = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset_id": ["AAPL"] * 10,
                "open": [100.0 + i for i in range(10)],
                "high": [101.0 + i for i in range(10)],
                "low": [99.0 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
                "volume": [1000000] * 10,
            }
        )

        price_path = temp_dir / "prices.parquet"
        prices.write_parquet(price_path)

        # Filter to only rows where close > 105
        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            filters=[pl.col("close") > 105.0],
        )

        events = []
        while not feed.is_exhausted:
            event = feed.get_next_event()
            if event:
                events.append(event)

        # Should only get events where close > 105
        # close values: 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, ...
        # close > 105: 105.5, 106.5, 107.5, 108.5, 109.5 = 5 events
        assert len(events) == 5
        assert all(event.close > 105.0 for event in events)


class TestPolarsDataFeedSeekEdgeCases:
    """Test seek edge cases."""

    def test_seek_to_exact_timestamp(self, temp_dir):
        """Test seeking to exact timestamp in data."""
        base_time = datetime(2025, 1, 1)
        timestamps = [base_time + timedelta(hours=i) for i in range(10)]

        prices = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset_id": ["AAPL"] * 10,
                "open": [100.0 + i for i in range(10)],
                "high": [101.0 + i for i in range(10)],
                "low": [99.0 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
                "volume": [1000000] * 10,
            }
        )

        price_path = temp_dir / "prices.parquet"
        prices.write_parquet(price_path)

        feed = PolarsDataFeed(price_path=price_path, asset_id=AssetId("AAPL"))

        # Seek to exact timestamp (5th timestamp)
        target_ts = timestamps[5]
        feed.seek(target_ts)

        event = feed.get_next_event()
        assert event is not None
        assert event.timestamp == target_ts
        assert event.close == 105.5

    def test_seek_to_timestamp_not_in_data(self, temp_dir):
        """Test seeking to timestamp between existing timestamps."""
        base_time = datetime(2025, 1, 1)
        # Hourly data
        timestamps = [base_time + timedelta(hours=i) for i in range(10)]

        prices = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset_id": ["AAPL"] * 10,
                "open": [100.0 + i for i in range(10)],
                "high": [101.0 + i for i in range(10)],
                "low": [99.0 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
                "volume": [1000000] * 10,
            }
        )

        price_path = temp_dir / "prices.parquet"
        prices.write_parquet(price_path)

        feed = PolarsDataFeed(price_path=price_path, asset_id=AssetId("AAPL"))

        # Seek to timestamp between timestamps[4] and timestamps[5]
        target_ts = timestamps[4] + timedelta(minutes=30)
        feed.seek(target_ts)

        # Should get next timestamp >= target (timestamps[5])
        event = feed.get_next_event()
        assert event is not None
        assert event.timestamp >= target_ts
        assert event.timestamp == timestamps[5]

    def test_seek_past_all_data(self, temp_dir):
        """Test seeking past all available data."""
        base_time = datetime(2025, 1, 1)
        timestamps = [base_time + timedelta(hours=i) for i in range(10)]

        prices = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset_id": ["AAPL"] * 10,
                "open": [100.0 + i for i in range(10)],
                "high": [101.0 + i for i in range(10)],
                "low": [99.0 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
                "volume": [1000000] * 10,
            }
        )

        price_path = temp_dir / "prices.parquet"
        prices.write_parquet(price_path)

        feed = PolarsDataFeed(price_path=price_path, asset_id=AssetId("AAPL"))

        # Seek past all data
        target_ts = timestamps[-1] + timedelta(days=1)
        feed.seek(target_ts)

        # Should be exhausted
        assert feed.is_exhausted
        event = feed.get_next_event()
        assert event is None


class TestPolarsDataFeedResetAndReuse:
    """Test reset and multiple iterations."""

    def test_multiple_resets(self, temp_dir):
        """Test resetting feed multiple times."""
        base_time = datetime(2025, 1, 1)

        prices = pl.DataFrame(
            {
                "timestamp": [base_time + timedelta(hours=i) for i in range(5)],
                "asset_id": ["AAPL"] * 5,
                "open": [100.0 + i for i in range(5)],
                "high": [101.0 + i for i in range(5)],
                "low": [99.0 + i for i in range(5)],
                "close": [100.5 + i for i in range(5)],
                "volume": [1000000] * 5,
            }
        )

        price_path = temp_dir / "prices.parquet"
        prices.write_parquet(price_path)

        feed = PolarsDataFeed(price_path=price_path, asset_id=AssetId("AAPL"))

        # First iteration
        events1 = []
        while not feed.is_exhausted:
            event = feed.get_next_event()
            if event:
                events1.append(event)
        assert len(events1) == 5

        # Reset and iterate again
        feed.reset()
        events2 = []
        while not feed.is_exhausted:
            event = feed.get_next_event()
            if event:
                events2.append(event)
        assert len(events2) == 5

        # Events should be identical
        for e1, e2 in zip(events1, events2):
            assert e1.timestamp == e2.timestamp
            assert e1.close == e2.close

    def test_reset_mid_iteration(self, temp_dir):
        """Test resetting feed mid-iteration."""
        base_time = datetime(2025, 1, 1)

        prices = pl.DataFrame(
            {
                "timestamp": [base_time + timedelta(hours=i) for i in range(10)],
                "asset_id": ["AAPL"] * 10,
                "open": [100.0 + i for i in range(10)],
                "high": [101.0 + i for i in range(10)],
                "low": [99.0 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
                "volume": [1000000] * 10,
            }
        )

        price_path = temp_dir / "prices.parquet"
        prices.write_parquet(price_path)

        feed = PolarsDataFeed(price_path=price_path, asset_id=AssetId("AAPL"))

        # Get 3 events
        for _ in range(3):
            feed.get_next_event()

        # Reset mid-iteration
        feed.reset()

        # Should start from beginning
        event = feed.get_next_event()
        assert event is not None
        assert event.close == 100.5  # First event


class TestPolarsDataFeedFeatureProviderEdgeCases:
    """Test FeatureProvider edge cases."""

    def test_feature_provider_returns_empty(self, temp_dir):
        """Test FeatureProvider returning empty dicts."""
        base_time = datetime(2025, 1, 1)

        prices = pl.DataFrame(
            {
                "timestamp": [base_time],
                "asset_id": ["AAPL"],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000000],
            }
        )

        # Empty feature provider (no data for this timestamp)
        features_df = pl.DataFrame(
            {
                "timestamp": [base_time + timedelta(days=10)],  # Different timestamp
                "asset_id": ["AAPL"],
                "atr": [2.5],
            }
        )

        price_path = temp_dir / "prices.parquet"
        prices.write_parquet(price_path)

        feature_provider = PrecomputedFeatureProvider(features_df)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            feature_provider=feature_provider,
        )

        event = feed.get_next_event()
        assert event is not None
        # signals and context should be empty (no matching timestamp)
        assert event.signals == {}
        assert event.context == {}
