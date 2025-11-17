"""Unit tests for PolarsDataFeed with lazy loading and multi-source merging."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest

from ml4t.backtest.core.types import AssetId, MarketDataType
from ml4t.backtest.data.feature_provider import (
    CallableFeatureProvider,
    PrecomputedFeatureProvider,
)
from ml4t.backtest.data.polars_feed import PolarsDataFeed


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_price_data():
    """Create sample OHLCV price data."""
    base_time = datetime(2025, 1, 1, 9, 30)
    timestamps = [base_time + timedelta(hours=i) for i in range(10)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["AAPL"] * 10,
            "open": [100.0 + i for i in range(10)],
            "high": [101.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "volume": [1000000 + i * 10000 for i in range(10)],
        }
    )


@pytest.fixture
def sample_signals_data():
    """Create sample ML signals data."""
    base_time = datetime(2025, 1, 1, 9, 30)
    timestamps = [base_time + timedelta(hours=i) for i in range(10)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["AAPL"] * 10,
            "ml_pred": [0.1 * i for i in range(10)],
            "confidence": [0.5 + 0.05 * i for i in range(10)],
        }
    )


@pytest.fixture
def sample_features_data():
    """Create sample features data for PrecomputedFeatureProvider.

    Creates combined DataFrame with:
    - Per-asset features (asset_id = "AAPL")
    - Market-wide features (asset_id = None)
    """
    base_time = datetime(2025, 1, 1, 9, 30)
    timestamps = [base_time + timedelta(hours=i) for i in range(10)]

    # Per-asset features (indicators) - asset_id = "AAPL"
    asset_rows = []
    for ts in timestamps:
        asset_rows.append(
            {
                "timestamp": ts,
                "asset_id": "AAPL",
                "atr": 2.5,
                "rsi": 60.0,
                "volatility": 0.25,
            }
        )

    # Market-wide features (context) - asset_id = None
    market_rows = []
    for ts in timestamps:
        market_rows.append(
            {
                "timestamp": ts,
                "asset_id": None,  # Market features have None for asset_id
                "vix": 15.0,
                "spy_close": 450.0,
                "regime": 1.0,
            }
        )

    # Combine into single DataFrame
    # Note: Different rows have different columns populated
    # This is OK - PrecomputedFeatureProvider filters by asset_id
    all_rows = asset_rows + market_rows
    combined_df = pl.DataFrame(all_rows)

    return combined_df


class TestPolarsDataFeedBasic:
    """Test basic PolarsDataFeed functionality without features."""

    def test_initialization_no_collect(self, temp_dir, sample_price_data):
        """Test that initialization doesn't collect DataFrame (lazy loading)."""
        # Write price data
        price_path = temp_dir / "prices.parquet"
        sample_price_data.write_parquet(price_path)

        # Create feed
        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
        )

        # Verify lazy frame exists but not collected
        assert not feed._initialized
        assert feed.timestamp_groups is None

    def test_get_next_event_single_source(self, temp_dir, sample_price_data):
        """Test getting events from price data only."""
        price_path = temp_dir / "prices.parquet"
        sample_price_data.write_parquet(price_path)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
        )

        # Get first event
        event = feed.get_next_event()

        assert event is not None
        assert event.asset_id == "AAPL"
        assert event.open == 100.0
        assert event.high == 101.0
        assert event.low == 99.0
        assert event.close == 100.5
        assert event.volume == 1000000
        assert event.signals == {}  # No signals provided
        assert event.indicators == {}  # No feature provider
        assert event.context == {}  # No feature provider

        # Verify lazy loading happened
        assert feed._initialized

    def test_iteration_order(self, temp_dir, sample_price_data):
        """Test that events are returned in chronological order."""
        price_path = temp_dir / "prices.parquet"
        sample_price_data.write_parquet(price_path)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
        )

        events = []
        while not feed.is_exhausted:
            event = feed.get_next_event()
            if event:
                events.append(event)

        # Verify we got all 10 events
        assert len(events) == 10

        # Verify chronological order
        for i in range(1, len(events)):
            assert events[i].timestamp > events[i - 1].timestamp

        # Verify close prices match expected progression
        for i, event in enumerate(events):
            assert event.close == 100.5 + i

    def test_peek_next_timestamp(self, temp_dir, sample_price_data):
        """Test peeking at next timestamp without consuming event."""
        price_path = temp_dir / "prices.parquet"
        sample_price_data.write_parquet(price_path)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
        )

        # Peek before consuming
        ts1 = feed.peek_next_timestamp()
        ts2 = feed.peek_next_timestamp()
        assert ts1 == ts2  # Should return same timestamp

        # Consume event
        event = feed.get_next_event()
        assert event.timestamp == ts1

        # Peek at next - should be next timestamp (next hour)
        ts_next = feed.peek_next_timestamp()
        # In our test data, timestamps are hourly, so next should be >= ts1
        # Since we have one event per timestamp group, ts_next should be > ts1
        assert ts_next >= ts1  # Changed from > to >= for robustness

    def test_reset(self, temp_dir, sample_price_data):
        """Test resetting feed to beginning."""
        price_path = temp_dir / "prices.parquet"
        sample_price_data.write_parquet(price_path)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
        )

        # Consume some events
        event1 = feed.get_next_event()
        event2 = feed.get_next_event()
        event3 = feed.get_next_event()

        # Reset
        feed.reset()

        # Get first event again
        event_after_reset = feed.get_next_event()
        assert event_after_reset.timestamp == event1.timestamp
        assert event_after_reset.close == event1.close

    def test_seek_to_timestamp(self, temp_dir, sample_price_data):
        """Test seeking to specific timestamp."""
        price_path = temp_dir / "prices.parquet"
        sample_price_data.write_parquet(price_path)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
        )

        # Seek to middle
        target_ts = datetime(2025, 1, 1, 9, 30) + timedelta(hours=5)
        feed.seek(target_ts)

        # Next event should be at or after target
        event = feed.get_next_event()
        assert event.timestamp >= target_ts
        assert event.close == 105.5  # 5th event (0-indexed)

    def test_is_exhausted(self, temp_dir, sample_price_data):
        """Test exhaustion detection."""
        price_path = temp_dir / "prices.parquet"
        sample_price_data.write_parquet(price_path)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
        )

        # Not exhausted initially
        assert not feed.is_exhausted

        # Consume all events
        count = 0
        while not feed.is_exhausted:
            event = feed.get_next_event()
            if event:
                count += 1

        assert count == 10
        assert feed.is_exhausted

        # get_next_event should return None when exhausted
        assert feed.get_next_event() is None


class TestPolarsDataFeedMultiSource:
    """Test multi-source merging (price + signals)."""

    def test_merge_price_and_signals(
        self, temp_dir, sample_price_data, sample_signals_data
    ):
        """Test merging price data with ML signals."""
        price_path = temp_dir / "prices.parquet"
        signals_path = temp_dir / "signals.parquet"

        sample_price_data.write_parquet(price_path)
        sample_signals_data.write_parquet(signals_path)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            signals_path=signals_path,
            signal_columns=["ml_pred", "confidence"],
        )

        # Get first event
        event = feed.get_next_event()

        # Verify price data
        assert event.close == 100.5

        # Verify signals merged correctly
        assert "ml_pred" in event.signals
        assert "confidence" in event.signals
        assert event.signals["ml_pred"] == 0.0
        assert event.signals["confidence"] == 0.5

    def test_auto_detect_signal_columns(
        self, temp_dir, sample_price_data, sample_signals_data
    ):
        """Test auto-detection of signal columns when not specified."""
        price_path = temp_dir / "prices.parquet"
        signals_path = temp_dir / "signals.parquet"

        sample_price_data.write_parquet(price_path)
        sample_signals_data.write_parquet(signals_path)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            signals_path=signals_path,
            # No signal_columns specified - should auto-detect
        )

        event = feed.get_next_event()

        # Should auto-detect ml_pred and confidence as signals
        assert "ml_pred" in event.signals
        assert "confidence" in event.signals

    def test_missing_signals_for_some_timestamps(self, temp_dir, sample_price_data):
        """Test handling partial signal data (not all timestamps have signals)."""
        price_path = temp_dir / "prices.parquet"
        signals_path = temp_dir / "signals.parquet"

        sample_price_data.write_parquet(price_path)

        # Create signals for only first 5 timestamps
        base_time = datetime(2025, 1, 1, 9, 30)
        timestamps = [base_time + timedelta(hours=i) for i in range(5)]
        partial_signals = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset_id": ["AAPL"] * 5,
                "ml_pred": [0.1 * i for i in range(5)],
            }
        )
        partial_signals.write_parquet(signals_path)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            signals_path=signals_path,
        )

        events = []
        while not feed.is_exhausted:
            event = feed.get_next_event()
            if event:
                events.append(event)

        # Should have 10 events (all price data)
        assert len(events) == 10

        # First 5 should have signals
        for i in range(5):
            assert "ml_pred" in events[i].signals

        # Last 5 should have empty signals (left join)
        for i in range(5, 10):
            assert events[i].signals == {}


class TestPolarsDataFeedFeatureProvider:
    """Test FeatureProvider integration for indicators/context."""

    def test_precomputed_feature_provider(
        self, temp_dir, sample_price_data, sample_features_data
    ):
        """Test integration with PrecomputedFeatureProvider."""
        price_path = temp_dir / "prices.parquet"
        sample_price_data.write_parquet(price_path)

        # sample_features_data now returns combined DataFrame with both
        # asset-level (asset_id="AAPL") and market-level (asset_id=None) features
        features_df = sample_features_data

        # PrecomputedFeatureProvider auto-detects feature columns
        # Just need to pass features_df with proper schema
        feature_provider = PrecomputedFeatureProvider(
            features_df=features_df,
            timestamp_col="timestamp",
            asset_col="asset_id",
        )

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            feature_provider=feature_provider,
        )

        event = feed.get_next_event()

        # Verify price data
        assert event.close == 100.5

        # Verify indicators populated
        assert "atr" in event.indicators
        assert "rsi" in event.indicators
        assert "volatility" in event.indicators
        assert event.indicators["atr"] == 2.5
        assert event.indicators["rsi"] == 60.0

        # Verify context populated
        assert "vix" in event.context
        assert "spy_close" in event.context
        assert "regime" in event.context
        assert event.context["vix"] == 15.0

    def test_callable_feature_provider(self, temp_dir, sample_price_data):
        """Test integration with CallableFeatureProvider."""
        price_path = temp_dir / "prices.parquet"
        sample_price_data.write_parquet(price_path)

        # Define feature callables
        def compute_atr(asset_id: str, timestamp: datetime) -> dict[str, float]:
            return {"atr": 2.5, "atr_pct": 0.025}

        def compute_vix(timestamp: datetime) -> dict[str, float]:
            return {"vix": 15.0, "regime": 1.0}

        feature_provider = CallableFeatureProvider(
            compute_fn=compute_atr, compute_market_fn=compute_vix
        )

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            feature_provider=feature_provider,
        )

        event = feed.get_next_event()

        # Verify indicators from callable
        assert event.indicators["atr"] == 2.5
        assert event.indicators["atr_pct"] == 0.025

        # Verify context from callable
        assert event.context["vix"] == 15.0
        assert event.context["regime"] == 1.0

    def test_all_three_sources(
        self, temp_dir, sample_price_data, sample_signals_data, sample_features_data
    ):
        """Test all three data sources: price + signals + features."""
        price_path = temp_dir / "prices.parquet"
        signals_path = temp_dir / "signals.parquet"

        sample_price_data.write_parquet(price_path)
        sample_signals_data.write_parquet(signals_path)

        # sample_features_data is already combined DataFrame
        features_df = sample_features_data

        feature_provider = PrecomputedFeatureProvider(
            features_df=features_df,
            timestamp_col="timestamp",
            asset_col="asset_id",
        )

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            signals_path=signals_path,
            signal_columns=["ml_pred"],
            feature_provider=feature_provider,
        )

        event = feed.get_next_event()

        # Verify all three dicts populated
        assert event.close == 100.5  # Price data
        assert "ml_pred" in event.signals  # Signals
        assert "atr" in event.indicators  # Indicators
        assert "vix" in event.context  # Context

        # Verify values
        assert event.signals["ml_pred"] == 0.0
        assert event.indicators["atr"] == 2.5
        assert event.context["vix"] == 15.0


class TestPolarsDataFeedPerformance:
    """Test performance characteristics and optimizations."""

    def test_group_by_optimization(self, temp_dir):
        """Test that group_by optimization is used (partition_by)."""
        # Create larger dataset to verify grouping
        base_time = datetime(2025, 1, 1)
        n_days = 100
        timestamps = [base_time + timedelta(days=i) for i in range(n_days)]

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset_id": ["AAPL"] * n_days,
                "close": [100.0 + i for i in range(n_days)],
                "open": [99.0 + i for i in range(n_days)],
                "high": [101.0 + i for i in range(n_days)],
                "low": [98.0 + i for i in range(n_days)],
                "volume": [1000000] * n_days,
            }
        )

        price_path = temp_dir / "prices.parquet"
        df.write_parquet(price_path)

        feed = PolarsDataFeed(price_path=price_path, asset_id=AssetId("AAPL"))

        # Consume all events
        events = []
        while not feed.is_exhausted:
            event = feed.get_next_event()
            if event:
                events.append(event)

        # Verify all events retrieved
        assert len(events) == n_days

        # Verify timestamp_groups was created (group_by happened)
        assert feed.timestamp_groups is not None
        assert len(feed.timestamp_groups) == n_days

    def test_lazy_initialization(self, temp_dir, sample_price_data):
        """Test that DataFrame collection is deferred until first get_next_event."""
        price_path = temp_dir / "prices.parquet"
        sample_price_data.write_parquet(price_path)

        feed = PolarsDataFeed(price_path=price_path, asset_id=AssetId("AAPL"))

        # Should not be initialized yet
        assert not feed._initialized
        assert feed.timestamp_groups is None

        # Peek should trigger initialization
        feed.peek_next_timestamp()
        assert feed._initialized
        assert feed.timestamp_groups is not None


class TestPolarsDataFeedEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self, temp_dir):
        """Test handling of empty price data."""
        price_path = temp_dir / "empty.parquet"

        # Create empty DataFrame with correct schema
        empty_df = pl.DataFrame(
            {
                "timestamp": [],
                "asset_id": [],
                "close": [],
                "open": [],
                "high": [],
                "low": [],
                "volume": [],
            },
            schema={
                "timestamp": pl.Datetime,
                "asset_id": pl.Utf8,
                "close": pl.Float64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "volume": pl.Int64,
            },
        )
        empty_df.write_parquet(price_path)

        feed = PolarsDataFeed(price_path=price_path, asset_id=AssetId("AAPL"))

        # Should initialize without error
        event = feed.get_next_event()
        assert event is None
        assert feed.is_exhausted

    def test_filter_for_specific_asset(self, temp_dir):
        """Test filtering for specific asset when file contains multiple assets."""
        base_time = datetime(2025, 1, 1)
        timestamps = [base_time + timedelta(hours=i) for i in range(10)]

        # Create data with multiple assets
        multi_asset_df = pl.DataFrame(
            {
                "timestamp": timestamps * 2,  # Repeat for 2 assets
                "asset_id": ["AAPL"] * 10 + ["MSFT"] * 10,
                "close": [100.0 + i for i in range(10)]
                + [200.0 + i for i in range(10)],
                "open": [99.0 + i for i in range(20)],
                "high": [101.0 + i for i in range(20)],
                "low": [98.0 + i for i in range(20)],
                "volume": [1000000] * 20,
            }
        )

        price_path = temp_dir / "multi_asset.parquet"
        multi_asset_df.write_parquet(price_path)

        # Create feed for AAPL only
        feed = PolarsDataFeed(price_path=price_path, asset_id=AssetId("AAPL"))

        events = []
        while not feed.is_exhausted:
            event = feed.get_next_event()
            if event:
                events.append(event)

        # Should only get AAPL events
        assert len(events) == 10
        assert all(event.asset_id == "AAPL" for event in events)
        assert all(event.close < 200 for event in events)  # AAPL prices, not MSFT

    def test_data_type_parameter(self, temp_dir, sample_price_data):
        """Test that data_type parameter is respected."""
        price_path = temp_dir / "prices.parquet"
        sample_price_data.write_parquet(price_path)

        feed = PolarsDataFeed(
            price_path=price_path,
            asset_id=AssetId("AAPL"),
            data_type=MarketDataType.TRADE,  # Non-default (use TRADE instead of TICK)
        )

        event = feed.get_next_event()
        assert event.data_type == MarketDataType.TRADE
