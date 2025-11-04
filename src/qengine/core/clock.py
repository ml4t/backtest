"""Time management and synchronization for QEngine."""

import heapq
import logging
from datetime import datetime
from enum import Enum

import pandas_market_calendars as mcal

from qengine.core.event import Event
from qengine.data.feed import DataFeed, SignalSource


class ClockMode(Enum):
    """Clock operation modes."""

    BACKTEST = "backtest"  # Historical simulation
    PAPER = "paper"  # Paper trading with real-time data
    LIVE = "live"  # Live trading


class Clock:
    """
    Master time keeper for the simulation.

    The Clock is responsible for:
    - Advancing simulation time
    - Coordinating multiple data sources
    - Ensuring point-in-time correctness
    - Managing trading calendar
    """

    def __init__(
        self,
        mode: ClockMode = ClockMode.BACKTEST,
        calendar: str | None = "NYSE",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ):
        """
        Initialize the Clock.

        Args:
            mode: Operating mode (backtest, paper, live)
            calendar: Market calendar name (e.g., 'NYSE', 'NASDAQ')
            start_time: Simulation start time
            end_time: Simulation end time
        """
        self.mode = mode
        self.calendar_name = calendar
        self.start_time = start_time
        self.end_time = end_time
        self.logger = logging.getLogger(__name__)

        # Current simulation time
        self._current_time = start_time

        # Data sources
        self._data_feeds: list[DataFeed] = []
        self._signal_sources: list[SignalSource] = []

        # Event queue (min heap by timestamp)
        # Store tuples of (timestamp, sequence, event, source) for stable ordering
        self._event_queue: list[tuple[datetime, int, Event, object]] = []
        self._sequence_counter = 0  # Ensures FIFO when timestamps are identical

        # Market calendar
        if calendar:
            self.calendar = mcal.get_calendar(calendar)
            if start_time and end_time:
                self.trading_sessions = self.calendar.schedule(
                    start_date=start_time.date(),
                    end_date=end_time.date(),
                )
            else:
                self.trading_sessions = None
        else:
            self.calendar = None
            self.trading_sessions = None

        # Statistics
        self._events_processed = 0
        self._ticks_processed = 0

    def add_data_feed(self, feed: DataFeed) -> None:
        """
        Add a data feed to the clock.

        Args:
            feed: Data feed to add
        """
        self._data_feeds.append(feed)
        self._prime_feed(feed)

    def add_signal_source(self, source: SignalSource) -> None:
        """
        Add a signal source to the clock.

        Args:
            source: Signal source to add
        """
        self._signal_sources.append(source)
        self._prime_signal_source(source)

    def _prime_feed(self, feed: DataFeed) -> None:
        """
        Prime a data feed by adding its first event to the queue.

        Args:
            feed: Data feed to prime
        """
        next_event = feed.get_next_event()
        if next_event:
            # Store tuple of (timestamp, sequence, event, source) for stable ordering
            heapq.heappush(
                self._event_queue, (next_event.timestamp, self._sequence_counter, next_event, feed)
            )
            self._sequence_counter += 1

    def _prime_signal_source(self, source: SignalSource) -> None:
        """
        Prime a signal source.

        Args:
            source: Signal source to prime
        """
        next_signal = source.get_next_signal()
        if next_signal:
            # Store tuple of (timestamp, sequence, event, source) for stable ordering
            heapq.heappush(
                self._event_queue,
                (next_signal.timestamp, self._sequence_counter, next_signal, source),
            )
            self._sequence_counter += 1

    def get_next_event(self) -> Event | None:
        """
        Get the next event across all data sources.

        Returns:
            Next event in chronological order or None
        """
        if not self._event_queue:
            return None

        # Get event with earliest timestamp and its source
        # Now unpacking 4 elements: timestamp, sequence, event, source
        timestamp, sequence, event, source = heapq.heappop(self._event_queue)

        # Update current time (ensures PIT correctness)
        self._current_time = timestamp

        # Check if we're past end time
        if self.end_time and timestamp > self.end_time:
            return None

        # Replenish queue from the source that provided this event
        self._replenish_queue(event, source)

        self._events_processed += 1

        return event

    def _replenish_queue(self, last_event: Event, source) -> None:
        """
        Add the next event from the source that provided the last event.

        For correct multi-feed synchronization, only the feed that just provided
        an event needs to be replenished. The heap maintains chronological order
        across all feeds.

        Args:
            last_event: The event that was just processed
            source: The data feed or signal source that provided the last event
        """
        from qengine.data.feed import DataFeed, SignalSource

        # Only replenish from the source that provided the last event
        # The heap ensures proper chronological ordering across all feeds
        if isinstance(source, DataFeed):
            if not source.is_exhausted:
                next_timestamp = source.peek_next_timestamp()
                if next_timestamp and (not self.end_time or next_timestamp <= self.end_time):
                    next_event = source.get_next_event()
                    if next_event:
                        heapq.heappush(
                            self._event_queue,
                            (next_event.timestamp, self._sequence_counter, next_event, source),
                        )
                        self._sequence_counter += 1
        elif isinstance(source, SignalSource):
            next_timestamp = source.peek_next_timestamp()
            if next_timestamp and (not self.end_time or next_timestamp <= self.end_time):
                next_signal = source.get_next_signal()
                if next_signal:
                    heapq.heappush(
                        self._event_queue,
                        (next_signal.timestamp, self._sequence_counter, next_signal, source),
                    )
                    self._sequence_counter += 1

    @property
    def current_time(self) -> datetime | None:
        """Get the current simulation time."""
        return self._current_time

    @property
    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.

        Returns:
            True if market is open at current time
        """
        if not self.calendar or not self._current_time:
            return True  # Assume always open if no calendar

        # Check if current time is within a trading session
        current_date = self._current_time.date()
        if current_date in self.trading_sessions.index:
            session = self.trading_sessions.loc[current_date]
            market_open = session["market_open"]
            market_close = session["market_close"]

            # Convert to timezone-aware if needed
            if self._current_time.tzinfo:
                return market_open <= self._current_time <= market_close
            return (
                market_open.replace(tzinfo=None)
                <= self._current_time
                <= market_close.replace(tzinfo=None)
            )

        return False

    @property
    def next_market_open(self) -> datetime | None:
        """
        Get the next market open time.

        Returns:
            Next market open datetime or None
        """
        if not self.calendar or not self._current_time:
            return None

        current_date = self._current_time.date()
        future_sessions = self.trading_sessions[self.trading_sessions.index >= current_date]

        for _date, session in future_sessions.iterrows():
            market_open = session["market_open"]
            if market_open > self._current_time:
                return market_open

        return None

    @property
    def next_market_close(self) -> datetime | None:
        """
        Get the next market close time.

        Returns:
            Next market close datetime or None
        """
        if not self.calendar or not self._current_time:
            return None

        current_date = self._current_time.date()
        if current_date in self.trading_sessions.index:
            market_close = self.trading_sessions.loc[current_date]["market_close"]
            if market_close > self._current_time:
                return market_close

        # Look for next session
        future_sessions = self.trading_sessions[self.trading_sessions.index > current_date]
        if not future_sessions.empty:
            return future_sessions.iloc[0]["market_close"]

        return None

    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if a given date is a trading day.

        Args:
            date: Date to check

        Returns:
            True if date is a trading day
        """
        if not self.trading_sessions:
            return True

        return date.date() in self.trading_sessions.index

    def advance_to(self, timestamp: datetime, skip_events: bool = False) -> int:
        """
        Advance the clock to a specific time.

        Used for jumping forward in time during backtesting.

        Args:
            timestamp: Target timestamp
            skip_events: If False (default), raise error if events would be skipped

        Returns:
            Number of events skipped

        Raises:
            ValueError: If skip_events=False and events would be skipped, or if trying to go back in time
            RuntimeError: If not in backtest mode
        """
        if self.mode != ClockMode.BACKTEST:
            raise RuntimeError("Can only advance time in backtest mode")

        if self._current_time is not None and timestamp < self._current_time:
            raise ValueError("Cannot go back in time")

        # Count events that would be skipped
        skipped_count = 0
        while self._event_queue and self._event_queue[0][0] < timestamp:
            skipped_count += 1
            if not skip_events:
                raise ValueError(
                    f"Would skip {skipped_count} event(s) before {timestamp}. "
                    f"Set skip_events=True to allow."
                )
            heapq.heappop(self._event_queue)

        self._current_time = timestamp

        if skipped_count > 0:
            self.logger.warning(f"Clock advanced to {timestamp}, skipped {skipped_count} events")

        return skipped_count

    def reset(self) -> None:
        """Reset the clock to initial state."""
        self._current_time = self.start_time
        self._event_queue.clear()
        self._events_processed = 0
        self._ticks_processed = 0

        # Reset all data feeds
        for feed in self._data_feeds:
            feed.reset()
            self._prime_feed(feed)

        # Reset all signal sources
        for source in self._signal_sources:
            source.reset()
            self._prime_signal_source(source)

    @property
    def stats(self) -> dict:
        """Get clock statistics."""
        return {
            "current_time": self._current_time,
            "events_processed": self._events_processed,
            "queue_size": len(self._event_queue),
            "data_feeds": len(self._data_feeds),
            "signal_sources": len(self._signal_sources),
            "mode": self.mode.value,
        }

    def __repr__(self) -> str:
        return (
            f"Clock(mode={self.mode.value}, "
            f"time={self._current_time}, "
            f"events={self._events_processed})"
        )
