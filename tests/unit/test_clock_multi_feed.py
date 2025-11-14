"""Test Clock synchronization with multiple data feeds."""

from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from qengine.core.clock import Clock, ClockMode
from qengine.core.event import MarketEvent, SignalEvent
from qengine.core.types import MarketDataType
from qengine.data.feed import DataFeed, SignalSource


class MockDataFeed(DataFeed):
    """Mock data feed for testing."""

    def __init__(self, events: List[MarketEvent]):
        """Initialize with a list of events."""
        self.events = events
        self.index = 0
        self._is_exhausted = False

    def get_next_event(self) -> MarketEvent | None:
        """Get the next event."""
        if self.index >= len(self.events):
            self._is_exhausted = True
            return None
        event = self.events[self.index]
        self.index += 1
        return event

    def peek_next_timestamp(self) -> datetime | None:
        """Peek at the next timestamp without consuming."""
        if self.index >= len(self.events):
            return None
        return self.events[self.index].timestamp

    @property
    def is_exhausted(self) -> bool:
        """Check if the feed has no more events."""
        return self.index >= len(self.events)

    def seek(self, timestamp: datetime) -> None:
        """Seek to the given timestamp."""
        # Simple linear search for mock - advance to first event at or after timestamp
        while self.index < len(self.events) and self.events[self.index].timestamp < timestamp:
            self.index += 1

    def reset(self) -> None:
        """Reset the feed to the beginning."""
        self.index = 0
        self._is_exhausted = False


class MockSignalSource(SignalSource):
    """Mock signal source for testing."""

    def __init__(self, signals: List[SignalEvent]):
        """Initialize with a list of signal events."""
        self.signals = signals
        self.index = 0

    def get_next_signal(self) -> SignalEvent | None:
        """Get the next signal."""
        if self.index >= len(self.signals):
            return None
        signal = self.signals[self.index]
        self.index += 1
        return signal

    def peek_next_timestamp(self) -> datetime | None:
        """Peek at the next timestamp without consuming."""
        if self.index >= len(self.signals):
            return None
        return self.signals[self.index].timestamp

    @property
    def is_exhausted(self) -> bool:
        """Check if the source has no more signals."""
        return self.index >= len(self.signals)

    def reset(self) -> None:
        """Reset the signal source to the beginning."""
        self.index = 0


class TestClockMultiFeed:
    """Test suite for Clock with multiple data feeds."""

    def test_single_feed(self):
        """Test Clock with a single data feed."""
        base_time = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        events = [
            MarketEvent(
                timestamp=base_time + timedelta(minutes=i),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0 + i,
                volume=1000,
            )
            for i in range(5)
        ]

        feed = MockDataFeed(events)
        clock = Clock(mode=ClockMode.BACKTEST, calendar=None)
        clock.add_data_feed(feed)

        # Verify events come out in order
        for i, expected_event in enumerate(events):
            event = clock.get_next_event()
            assert event is not None
            assert event.timestamp == expected_event.timestamp
            assert event.asset_id == expected_event.asset_id
            assert event.price == expected_event.price

        # No more events
        assert clock.get_next_event() is None

    def test_two_feeds_interleaved(self):
        """Test Clock with two feeds having interleaved timestamps."""
        base_time = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Feed 1: AAPL at even minutes
        feed1_events = [
            MarketEvent(
                timestamp=base_time + timedelta(minutes=i * 2),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0 + i,
                volume=1000,
            )
            for i in range(3)
        ]

        # Feed 2: GOOGL at odd minutes
        feed2_events = [
            MarketEvent(
                timestamp=base_time + timedelta(minutes=i * 2 + 1),
                asset_id="GOOGL",
                data_type=MarketDataType.BAR,
                price=2800.0 + i,
                volume=500,
            )
            for i in range(3)
        ]

        feed1 = MockDataFeed(feed1_events)
        feed2 = MockDataFeed(feed2_events)

        clock = Clock(mode=ClockMode.BACKTEST, calendar=None)
        clock.add_data_feed(feed1)
        clock.add_data_feed(feed2)

        # Expected order: AAPL(0), GOOGL(1), AAPL(2), GOOGL(3), AAPL(4), GOOGL(5)
        expected_order = ["AAPL", "GOOGL", "AAPL", "GOOGL", "AAPL", "GOOGL"]
        expected_minutes = [0, 1, 2, 3, 4, 5]

        for expected_asset, expected_minute in zip(expected_order, expected_minutes):
            event = clock.get_next_event()
            assert event is not None
            assert event.asset_id == expected_asset
            expected_time = base_time + timedelta(minutes=expected_minute)
            assert event.timestamp == expected_time

        # No more events
        assert clock.get_next_event() is None

    def test_identical_timestamps(self):
        """Test Clock with multiple feeds having identical timestamps."""
        base_time = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Both feeds have events at the same timestamps
        timestamps = [base_time + timedelta(minutes=i) for i in range(3)]

        feed1_events = [
            MarketEvent(
                timestamp=ts,
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0 + i,
                volume=1000,
            )
            for i, ts in enumerate(timestamps)
        ]

        feed2_events = [
            MarketEvent(
                timestamp=ts,
                asset_id="GOOGL",
                data_type=MarketDataType.BAR,
                price=2800.0 + i,
                volume=500,
            )
            for i, ts in enumerate(timestamps)
        ]

        feed1 = MockDataFeed(feed1_events)
        feed2 = MockDataFeed(feed2_events)

        clock = Clock(mode=ClockMode.BACKTEST, calendar=None)
        clock.add_data_feed(feed1)
        clock.add_data_feed(feed2)

        # With identical timestamps, order should be deterministic (FIFO based on which feed was added first)
        # We should get all events, 6 total (3 from each feed)
        events_received = []
        while True:
            event = clock.get_next_event()
            if event is None:
                break
            events_received.append((event.timestamp, event.asset_id))

        assert len(events_received) == 6

        # Check that we got all events from both feeds
        aapl_count = sum(1 for _, asset in events_received if asset == "AAPL")
        googl_count = sum(1 for _, asset in events_received if asset == "GOOGL")
        assert aapl_count == 3
        assert googl_count == 3

        # Check chronological order is maintained
        for i in range(len(events_received) - 1):
            assert events_received[i][0] <= events_received[i + 1][0]

    def test_different_frequencies(self):
        """Test Clock with feeds at different frequencies (tick vs daily)."""
        base_time = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Feed 1: Tick data every second for first minute
        tick_events = [
            MarketEvent(
                timestamp=base_time + timedelta(seconds=i),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0 + i * 0.01,
                volume=100,
            )
            for i in range(10)  # 10 tick events
        ]

        # Feed 2: Daily data
        daily_events = [
            MarketEvent(
                timestamp=base_time,  # Same as first tick
                asset_id="SPY",
                data_type=MarketDataType.BAR,
                price=450.0,
                volume=1000000,
                close=450.0,
            ),
            MarketEvent(
                timestamp=base_time + timedelta(days=1),
                asset_id="SPY",
                data_type=MarketDataType.BAR,
                price=451.0,
                volume=1000000,
                close=451.0,
            ),
        ]

        tick_feed = MockDataFeed(tick_events)
        daily_feed = MockDataFeed(daily_events)

        clock = Clock(mode=ClockMode.BACKTEST, calendar=None)
        clock.add_data_feed(tick_feed)
        clock.add_data_feed(daily_feed)

        events_received = []
        while True:
            event = clock.get_next_event()
            if event is None:
                break
            events_received.append(event)
            # Stop after first day to avoid processing next day's SPY
            if len(events_received) >= 11:  # 10 ticks + 1 daily
                break

        # Should get the daily SPY event first (or interleaved with first tick)
        # Then all tick events
        assert len(events_received) == 11
        spy_count = sum(1 for e in events_received if e.asset_id == "SPY")
        aapl_count = sum(1 for e in events_received if e.asset_id == "AAPL")
        assert spy_count == 1  # One daily event
        assert aapl_count == 10  # Ten tick events

    def test_clock_with_end_time(self):
        """Test Clock respects end_time parameter."""
        base_time = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        end_time = base_time + timedelta(minutes=2, seconds=30)

        events = [
            MarketEvent(
                timestamp=base_time + timedelta(minutes=i),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0 + i,
                volume=1000,
            )
            for i in range(5)  # 5 events over 5 minutes
        ]

        feed = MockDataFeed(events)
        clock = Clock(
            mode=ClockMode.BACKTEST,
            calendar=None,
            start_time=base_time,
            end_time=end_time,
        )
        clock.add_data_feed(feed)

        events_received = []
        while True:
            event = clock.get_next_event()
            if event is None:
                break
            events_received.append(event)

        # Should only get events up to end_time (first 3 events: 0, 1, 2 minutes)
        assert len(events_received) == 3
        assert all(e.timestamp <= end_time for e in events_received)

    def test_empty_feeds(self):
        """Test Clock with empty feeds."""
        feed1 = MockDataFeed([])
        feed2 = MockDataFeed([])

        clock = Clock(mode=ClockMode.BACKTEST, calendar=None)
        clock.add_data_feed(feed1)
        clock.add_data_feed(feed2)

        assert clock.get_next_event() is None

    def test_three_feeds_complex(self):
        """Test Clock with three feeds having complex timing."""
        base_time = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

        # Feed 1: Every 2 minutes
        feed1_events = [
            MarketEvent(
                timestamp=base_time + timedelta(minutes=i * 2),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0 + i,
                volume=1000,
            )
            for i in range(3)
        ]

        # Feed 2: Every 3 minutes
        feed2_events = [
            MarketEvent(
                timestamp=base_time + timedelta(minutes=i * 3),
                asset_id="GOOGL",
                data_type=MarketDataType.BAR,
                price=2800.0 + i,
                volume=500,
            )
            for i in range(2)
        ]

        # Feed 3: Every minute
        feed3_events = [
            MarketEvent(
                timestamp=base_time + timedelta(minutes=i),
                asset_id="MSFT",
                data_type=MarketDataType.BAR,
                price=400.0 + i,
                volume=800,
            )
            for i in range(5)
        ]

        feed1 = MockDataFeed(feed1_events)
        feed2 = MockDataFeed(feed2_events)
        feed3 = MockDataFeed(feed3_events)

        clock = Clock(mode=ClockMode.BACKTEST, calendar=None)
        clock.add_data_feed(feed1)
        clock.add_data_feed(feed2)
        clock.add_data_feed(feed3)

        # Collect all events
        events_received = []
        while True:
            event = clock.get_next_event()
            if event is None:
                break
            events_received.append(event)

        # Total: 3 AAPL + 2 GOOGL + 5 MSFT = 10 events
        assert len(events_received) == 10

        # Verify chronological order
        for i in range(len(events_received) - 1):
            assert events_received[i].timestamp <= events_received[i + 1].timestamp

        # Count by asset
        aapl = sum(1 for e in events_received if e.asset_id == "AAPL")
        googl = sum(1 for e in events_received if e.asset_id == "GOOGL")
        msft = sum(1 for e in events_received if e.asset_id == "MSFT")
        assert aapl == 3
        assert googl == 2
        assert msft == 5


class TestClockMarketHours:
    """Test Clock market hours and calendar functionality."""

    def test_is_market_open_no_calendar(self):
        """Test that market is always open when no calendar is configured."""
        clock = Clock(mode=ClockMode.BACKTEST)
        assert clock.is_market_open == True

    def test_is_trading_day_no_calendar(self):
        """Test that all days are trading days when no calendar is configured."""
        clock = Clock(mode=ClockMode.BACKTEST)
        test_date = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)
        assert clock.is_trading_day(test_date) == True

    def test_next_market_open_no_calendar(self):
        """Test that next_market_open returns None when no calendar is configured."""
        clock = Clock(mode=ClockMode.BACKTEST)
        assert clock.next_market_open is None

    def test_next_market_close_no_calendar(self):
        """Test that next_market_close returns None when no calendar is configured."""
        clock = Clock(mode=ClockMode.BACKTEST)
        assert clock.next_market_close is None

    def test_market_calendar_initialization(self):
        """Test Clock with market calendar configured."""
        # Setup clock with NYSE calendar
        start = datetime(2024, 1, 2, 14, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc)

        clock = Clock(mode=ClockMode.BACKTEST, calendar="NYSE", start_time=start, end_time=end)

        # Verify calendar was initialized
        assert clock.calendar is not None
        assert clock.trading_sessions is not None
        assert clock.calendar_name == "NYSE"

    def test_clock_mode_property(self):
        """Test clock mode property access."""
        clock_backtest = Clock(mode=ClockMode.BACKTEST)
        assert clock_backtest.mode == ClockMode.BACKTEST

        clock_live = Clock(mode=ClockMode.LIVE)
        assert clock_live.mode == ClockMode.LIVE

    def test_clock_time_bounds(self):
        """Test clock respects start and end time bounds."""
        start = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, 17, 0, tzinfo=timezone.utc)

        clock = Clock(mode=ClockMode.BACKTEST, start_time=start, end_time=end)

        assert clock.start_time == start
        assert clock.end_time == end
        assert clock.current_time == start


class TestClockAdvanceTo:
    """Test Clock advance_to functionality."""

    def test_advance_to_requires_backtest_mode(self):
        """Test that advance_to only works in backtest mode."""
        clock = Clock(mode=ClockMode.LIVE)
        target = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

        with pytest.raises(RuntimeError, match="Can only advance time in backtest mode"):
            clock.advance_to(target)

    def test_advance_to_cannot_go_backward(self):
        """Test that advance_to prevents going back in time."""
        clock = Clock(mode=ClockMode.BACKTEST)

        # Create a simple feed and advance time
        events = [
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0,
                close=150.0,
                volume=1000,
            )
        ]
        feed = MockDataFeed(events)
        clock.add_data_feed(feed)

        # Get first event to advance time
        clock.get_next_event()

        # Try to go back in time
        earlier = datetime(2023, 12, 31, 10, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="Cannot go back in time"):
            clock.advance_to(earlier)

    def test_advance_to_with_no_events_skipped(self):
        """Test advance_to when no events need to be skipped."""
        clock = Clock(mode=ClockMode.BACKTEST)

        # Create events
        events = [
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0,
                close=150.0,
                volume=1000,
            )
        ]
        feed = MockDataFeed(events)
        clock.add_data_feed(feed)

        # Advance to time before any events
        target = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)
        skipped = clock.advance_to(target)

        assert skipped == 0
        assert clock.current_time == target

    def test_advance_to_raises_error_when_skipping_events_not_allowed(self):
        """Test that advance_to raises error when events would be skipped and skip_events=False."""
        clock = Clock(mode=ClockMode.BACKTEST)

        # Create events
        events = [
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0,
                close=150.0,
                volume=1000,
            ),
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=151.0,
                close=151.0,
                volume=1000,
            ),
        ]
        feed = MockDataFeed(events)
        clock.add_data_feed(feed)

        # Try to advance past events without allowing skips
        target = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="Would skip .* event\\(s\\)"):
            clock.advance_to(target, skip_events=False)

    def test_advance_to_skips_events_when_allowed(self):
        """Test that advance_to skips events when skip_events=True."""
        clock = Clock(mode=ClockMode.BACKTEST)

        # Create events
        events = [
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0,
                close=150.0,
                volume=1000,
            ),
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=151.0,
                close=151.0,
                volume=1000,
            ),
        ]
        feed = MockDataFeed(events)
        clock.add_data_feed(feed)

        # Advance past events with skipping allowed
        target = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
        skipped = clock.advance_to(target, skip_events=True)

        # Only 1 event is skipped because Clock uses lazy loading:
        # add_data_feed only loads the first event (10:00) into the queue.
        # The second event (11:00) won't be loaded until the first is processed.
        assert skipped == 1
        assert clock.current_time == target


class TestClockResetAndStats:
    """Test Clock reset and statistics functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears clock state."""
        clock = Clock(
            mode=ClockMode.BACKTEST,
            start_time=datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc),
        )

        # Create and process some events
        events = [
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0,
                close=150.0,
                volume=1000,
            ),
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=151.0,
                close=151.0,
                volume=1000,
            ),
        ]
        feed = MockDataFeed(events)
        clock.add_data_feed(feed)

        # Process first event
        event = clock.get_next_event()
        assert event is not None

        # Reset clock
        clock.reset()

        # Verify state is reset
        assert clock.current_time == clock.start_time
        stats = clock.stats
        assert stats["events_processed"] == 0
        assert stats["queue_size"] > 0  # Feed should be re-primed

    def test_stats_returns_correct_information(self):
        """Test that stats property returns accurate clock statistics."""
        start_time = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        clock = Clock(mode=ClockMode.BACKTEST, start_time=start_time)

        # Add a data feed
        events = [
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0,
                close=150.0,
                volume=1000,
            )
        ]
        feed = MockDataFeed(events)
        clock.add_data_feed(feed)

        stats = clock.stats

        assert stats["current_time"] == start_time
        assert stats["events_processed"] == 0
        assert stats["queue_size"] == 1  # One event in queue
        assert stats["data_feeds"] == 1
        assert stats["signal_sources"] == 0
        assert stats["mode"] == ClockMode.BACKTEST.value

    def test_stats_updates_after_processing_events(self):
        """Test that stats updates correctly after processing events."""
        clock = Clock(mode=ClockMode.BACKTEST)

        events = [
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0,
                close=150.0,
                volume=1000,
            ),
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=151.0,
                close=151.0,
                volume=1000,
            ),
        ]
        feed = MockDataFeed(events)
        clock.add_data_feed(feed)

        # Process one event
        event1 = clock.get_next_event()
        assert event1 is not None

        stats = clock.stats
        assert stats["events_processed"] == 1
        assert stats["queue_size"] == 1
        assert stats["current_time"] == event1.timestamp

    def test_repr_returns_formatted_string(self):
        """Test that __repr__ returns a properly formatted string."""
        start_time = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        clock = Clock(mode=ClockMode.BACKTEST, start_time=start_time)

        repr_str = repr(clock)

        assert "Clock" in repr_str
        assert "backtest" in repr_str.lower()
        assert "events=0" in repr_str


class TestClockSignalSources:
    """Test Clock signal source integration."""

    def test_add_signal_source_increments_count(self):
        """Test that adding a signal source is tracked in stats."""
        clock = Clock(mode=ClockMode.BACKTEST)

        # Create signal source
        signals = [
            SignalEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                signal_value=1.0,
                model_id="test_model",
            )
        ]
        signal_source = MockSignalSource(signals)
        clock.add_signal_source(signal_source)

        # Verify signal source is tracked
        stats = clock.stats
        assert stats["signal_sources"] == 1
        assert stats["queue_size"] == 1

    def test_mixed_feeds_and_signal_sources(self):
        """Test clock with both data feeds and signal sources."""
        clock = Clock(mode=ClockMode.BACKTEST)

        # Add data feed
        market_events = [
            MarketEvent(
                timestamp=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
                asset_id="AAPL",
                data_type=MarketDataType.BAR,
                price=150.0,
                close=150.0,
                volume=1000,
            )
        ]
        data_feed = MockDataFeed(market_events)
        clock.add_data_feed(data_feed)

        # Add signal source
        signals = [
            SignalEvent(
                timestamp=datetime(2024, 1, 1, 9, 45, tzinfo=timezone.utc),
                asset_id="SIGNAL_MODEL",
                signal_value=1.0,
                model_id="test_model",
            )
        ]
        signal_source = MockSignalSource(signals)
        clock.add_signal_source(signal_source)

        # Verify both are tracked
        stats = clock.stats
        assert stats["data_feeds"] == 1
        assert stats["signal_sources"] == 1
        assert stats["queue_size"] == 2  # Both events in queue

        # Signal should come first (earlier timestamp)
        first_event = clock.get_next_event()
        assert first_event.asset_id == "SIGNAL_MODEL"

        second_event = clock.get_next_event()
        assert second_event.asset_id == "AAPL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
