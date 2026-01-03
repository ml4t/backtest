"""Tests for calendar integration in the Engine.

These tests verify that the Engine correctly handles trading session enforcement
when a calendar is configured with enforce_sessions=True.
"""

from datetime import datetime, timedelta

import polars as pl

from ml4t.backtest import DataFeed, Engine, OrderSide, Strategy
from ml4t.backtest.config import BacktestConfig, DataFrequency


class SimpleStrategy(Strategy):
    """Simple strategy that buys on every bar for testing."""

    def __init__(self):
        self.bars_processed = 0
        self.timestamps_seen = []

    def on_data(self, timestamp, data, context, broker):
        self.bars_processed += 1
        self.timestamps_seen.append(timestamp)

        # Buy if no position
        for asset in data:
            pos = broker.get_position(asset)
            if pos is None or pos.quantity == 0:
                broker.submit_order(asset, 10.0, OrderSide.BUY)


def generate_daily_data(start_date: datetime, n_bars: int, include_weekends: bool = True):
    """Generate daily price data, optionally including weekends."""
    dates = []
    prices = []
    current = start_date

    for i in range(n_bars):
        dates.append(current)
        prices.append(100.0 + i * 0.1)

        if include_weekends:
            current += timedelta(days=1)
        else:
            # Skip weekends
            current += timedelta(days=1)
            while current.weekday() >= 5:  # Saturday=5, Sunday=6
                current += timedelta(days=1)

    return pl.DataFrame(
        {
            "timestamp": dates,
            "asset": ["TEST"] * n_bars,
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000000] * n_bars,
        }
    )


def generate_minute_data(date: datetime, include_outside_hours: bool = True):
    """Generate minute-level data for a single day."""
    timestamps = []
    prices = []
    price = 100.0

    if include_outside_hours:
        # Pre-market (4:00 AM - 9:29 AM)
        for hour in range(4, 9):
            for minute in range(0, 60):
                ts = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                timestamps.append(ts)
                prices.append(price)
                price += 0.01
        for minute in range(0, 30):
            ts = date.replace(hour=9, minute=minute, second=0, microsecond=0)
            timestamps.append(ts)
            prices.append(price)
            price += 0.01

    # Regular trading hours (9:30 AM - 4:00 PM)
    for minute in range(30, 60):
        ts = date.replace(hour=9, minute=minute, second=0, microsecond=0)
        timestamps.append(ts)
        prices.append(price)
        price += 0.01
    for hour in range(10, 16):
        for minute in range(0, 60):
            ts = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            timestamps.append(ts)
            prices.append(price)
            price += 0.01

    if include_outside_hours:
        # After-hours (4:00 PM - 8:00 PM)
        for hour in range(16, 20):
            for minute in range(0, 60):
                ts = date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                timestamps.append(ts)
                prices.append(price)
                price += 0.01

    n_bars = len(timestamps)
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset": ["TEST"] * n_bars,
            "open": prices,
            "high": [p + 0.1 for p in prices],
            "low": [p - 0.1 for p in prices],
            "close": prices,
            "volume": [100000] * n_bars,
        }
    )


class TestCalendarEnforcementDaily:
    """Tests for daily data calendar enforcement."""

    def test_no_calendar_processes_all_bars(self):
        """Without calendar, all bars are processed (backward compatible)."""
        # Generate data including weekends
        start = datetime(2024, 1, 1)  # Monday
        df = generate_daily_data(start, 14, include_weekends=True)  # 2 weeks

        feed = DataFeed(prices_df=df)
        strategy = SimpleStrategy()

        # No config = no calendar enforcement
        engine = Engine(feed=feed, strategy=strategy)
        results = engine.run()

        # All 14 bars should be processed
        assert strategy.bars_processed == 14
        assert results["skipped_bars"] == 0

    def test_enforce_false_processes_all_bars(self):
        """With enforce_sessions=False, all bars are processed."""
        start = datetime(2024, 1, 1)  # Monday
        df = generate_daily_data(start, 14, include_weekends=True)

        feed = DataFeed(prices_df=df)
        strategy = SimpleStrategy()

        config = BacktestConfig(
            calendar="NYSE",
            enforce_sessions=False,  # Explicit False
            data_frequency=DataFrequency.DAILY,
        )
        engine = Engine(feed=feed, strategy=strategy, config=config)
        results = engine.run()

        # All 14 bars should be processed
        assert strategy.bars_processed == 14
        assert results["skipped_bars"] == 0

    def test_skip_weekend_data(self):
        """With enforce_sessions=True, weekend bars are skipped."""
        # Generate 2 weeks of data including weekends (14 days)
        # Start on Jan 8, 2024 (Monday, not a holiday) to avoid New Year's Day
        # Jan 8 (Mon), 9 (Tue), 10 (Wed), 11 (Thu), 12 (Fri),
        # 13 (Sat-skip), 14 (Sun-skip),
        # 15 (Mon-MLK Day holiday), 16 (Tue), 17 (Wed), 18 (Thu), 19 (Fri),
        # 20 (Sat-skip), 21 (Sun-skip)
        start = datetime(2024, 1, 8)  # Monday (not a holiday)
        df = generate_daily_data(start, 14, include_weekends=True)

        feed = DataFeed(prices_df=df)
        strategy = SimpleStrategy()

        config = BacktestConfig(
            calendar="NYSE",
            enforce_sessions=True,
            data_frequency=DataFrequency.DAILY,
        )
        engine = Engine(feed=feed, strategy=strategy, config=config)
        results = engine.run()

        # Should skip 4 weekend days + 1 MLK Day = 5 non-trading days
        # 14 days - 5 = 9 trading days
        assert results["skipped_bars"] == 5
        assert strategy.bars_processed == 9

        # Verify no weekend timestamps in processed bars
        for ts in strategy.timestamps_seen:
            assert ts.weekday() < 5, f"Weekend day {ts} was processed"

    def test_skip_holiday_data(self):
        """NYSE holidays are skipped when enforce_sessions=True."""
        # Use a date range that includes July 4th 2024 (Thursday, holiday)
        # July 1 (Mon), 2 (Tue), 3 (Wed), 4 (Thu-holiday), 5 (Fri)
        dates = [
            datetime(2024, 7, 1),
            datetime(2024, 7, 2),
            datetime(2024, 7, 3),
            datetime(2024, 7, 4),  # Independence Day - holiday
            datetime(2024, 7, 5),
        ]
        prices = [100.0 + i for i in range(5)]

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "asset": ["TEST"] * 5,
                "open": prices,
                "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [1000000] * 5,
            }
        )

        feed = DataFeed(prices_df=df)
        strategy = SimpleStrategy()

        config = BacktestConfig(
            calendar="NYSE",
            enforce_sessions=True,
            data_frequency=DataFrequency.DAILY,
        )
        engine = Engine(feed=feed, strategy=strategy, config=config)
        results = engine.run()

        # July 4th should be skipped
        assert results["skipped_bars"] == 1
        assert strategy.bars_processed == 4

        # Verify July 4th not in processed timestamps
        for ts in strategy.timestamps_seen:
            assert ts.day != 4, "July 4th was processed"


class TestCalendarEnforcementIntraday:
    """Tests for intraday data calendar enforcement.

    Note: Intraday session validation falls back to trading day check when
    timezone-naive timestamps are used (which is typical). Full intraday
    session enforcement requires timezone-aware timestamps matching the
    exchange's timezone.
    """

    def test_intraday_weekend_skipped(self):
        """Weekend intraday bars are skipped (via trading day fallback)."""
        # Saturday data - will be caught by trading day check
        saturday = datetime(2024, 1, 6, 10, 0)  # Saturday 10 AM
        timestamps = [
            saturday,
            saturday + timedelta(minutes=1),
            saturday + timedelta(minutes=2),
        ]
        prices = [100.0, 100.1, 100.2]
        n_bars = len(timestamps)

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset": ["TEST"] * n_bars,
                "open": prices,
                "high": [p + 0.1 for p in prices],
                "low": [p - 0.1 for p in prices],
                "close": prices,
                "volume": [100000] * n_bars,
            }
        )

        feed = DataFeed(prices_df=df)
        strategy = SimpleStrategy()

        config = BacktestConfig(
            calendar="NYSE",
            enforce_sessions=True,
            data_frequency=DataFrequency.MINUTE_1,
        )
        engine = Engine(feed=feed, strategy=strategy, config=config)
        results = engine.run()

        # All bars should be skipped (Saturday)
        assert strategy.bars_processed == 0
        assert results["skipped_bars"] == 3

    def test_intraday_holiday_skipped(self):
        """Holiday intraday bars are skipped (via trading day fallback)."""
        # July 4th 2024 - Independence Day
        july4 = datetime(2024, 7, 4, 10, 0)
        timestamps = [
            july4,
            july4 + timedelta(minutes=1),
        ]
        prices = [100.0, 100.1]
        n_bars = len(timestamps)

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset": ["TEST"] * n_bars,
                "open": prices,
                "high": [p + 0.1 for p in prices],
                "low": [p - 0.1 for p in prices],
                "close": prices,
                "volume": [100000] * n_bars,
            }
        )

        feed = DataFeed(prices_df=df)
        strategy = SimpleStrategy()

        config = BacktestConfig(
            calendar="NYSE",
            enforce_sessions=True,
            data_frequency=DataFrequency.MINUTE_1,
        )
        engine = Engine(feed=feed, strategy=strategy, config=config)
        results = engine.run()

        # All bars should be skipped (holiday)
        assert strategy.bars_processed == 0
        assert results["skipped_bars"] == 2

    def test_intraday_trading_day_processed(self):
        """Intraday bars on trading days are processed."""
        # Regular Tuesday
        tuesday = datetime(2024, 1, 2, 10, 0)
        timestamps = [
            tuesday,
            tuesday + timedelta(minutes=1),
            tuesday + timedelta(minutes=2),
        ]
        prices = [100.0, 100.1, 100.2]
        n_bars = len(timestamps)

        df = pl.DataFrame(
            {
                "timestamp": timestamps,
                "asset": ["TEST"] * n_bars,
                "open": prices,
                "high": [p + 0.1 for p in prices],
                "low": [p - 0.1 for p in prices],
                "close": prices,
                "volume": [100000] * n_bars,
            }
        )

        feed = DataFeed(prices_df=df)
        strategy = SimpleStrategy()

        config = BacktestConfig(
            calendar="NYSE",
            enforce_sessions=True,
            data_frequency=DataFrequency.MINUTE_1,
        )
        engine = Engine(feed=feed, strategy=strategy, config=config)
        results = engine.run()

        # All bars should be processed (regular trading day)
        assert strategy.bars_processed == 3
        assert results["skipped_bars"] == 0


class TestCalendarEdgeCases:
    """Tests for calendar edge cases."""

    def test_empty_data_with_calendar(self):
        """Empty data with calendar configured works correctly."""
        df = pl.DataFrame(
            {
                "timestamp": [],
                "asset": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        ).cast(
            {
                "timestamp": pl.Datetime,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Int64,
            }
        )

        feed = DataFeed(prices_df=df)
        strategy = SimpleStrategy()

        config = BacktestConfig(
            calendar="NYSE",
            enforce_sessions=True,
            data_frequency=DataFrequency.DAILY,
        )
        engine = Engine(feed=feed, strategy=strategy, config=config)
        results = engine.run()

        assert strategy.bars_processed == 0
        assert results["skipped_bars"] == 0

    def test_mixed_valid_invalid_days(self):
        """Mix of trading days and non-trading days."""
        dates = [
            datetime(2024, 1, 2),  # Tuesday - trading
            datetime(2024, 1, 3),  # Wednesday - trading
            datetime(2024, 1, 6),  # Saturday - skip
            datetime(2024, 1, 7),  # Sunday - skip
            datetime(2024, 1, 8),  # Monday - trading
            datetime(2024, 1, 15),  # MLK Day - holiday (skip)
        ]
        prices = [100.0 + i for i in range(6)]
        n_bars = len(dates)

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "asset": ["TEST"] * n_bars,
                "open": prices,
                "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [1000000] * n_bars,
            }
        )

        feed = DataFeed(prices_df=df)
        strategy = SimpleStrategy()

        config = BacktestConfig(
            calendar="NYSE",
            enforce_sessions=True,
            data_frequency=DataFrequency.DAILY,
        )
        engine = Engine(feed=feed, strategy=strategy, config=config)
        results = engine.run()

        # Should process 3 trading days, skip 3 (Sat, Sun, MLK Day)
        assert strategy.bars_processed == 3
        assert results["skipped_bars"] == 3

    def test_different_calendar_cme(self):
        """CME calendar has different trading hours."""
        # CME Equity trades longer hours than NYSE
        # Just verify calendar loads and enforcement works
        date = datetime(2024, 1, 2)
        df = generate_daily_data(date, 5, include_weekends=False)

        feed = DataFeed(prices_df=df)
        strategy = SimpleStrategy()

        config = BacktestConfig(
            calendar="CME_Equity",
            enforce_sessions=True,
            data_frequency=DataFrequency.DAILY,
        )
        engine = Engine(feed=feed, strategy=strategy, config=config)
        results = engine.run()

        # All 5 trading days should be processed (no weekends in data)
        assert strategy.bars_processed == 5
        assert results["skipped_bars"] == 0
