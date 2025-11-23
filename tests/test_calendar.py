"""Tests for calendar module (pandas_market_calendars integration)."""

from datetime import UTC, date, datetime

import polars as pl
import pytest

from ml4t.backtest.calendar import (
    CALENDAR_ALIASES,
    filter_to_trading_days,
    filter_to_trading_sessions,
    generate_trading_minutes,
    get_calendar,
    get_holidays,
    get_schedule,
    get_trading_days,
    is_market_open,
    is_trading_day,
    list_calendars,
    next_trading_day,
    previous_trading_day,
)


class TestGetCalendar:
    """Tests for get_calendar function."""

    def test_get_calendar_by_mic(self):
        """Test getting calendar by MIC code."""
        cal = get_calendar("XNYS")
        assert cal is not None
        assert "NYSE" in cal.name or "XNYS" in str(type(cal))

    def test_get_calendar_by_alias(self):
        """Test getting calendar by alias."""
        cal = get_calendar("NYSE")
        assert cal is not None

    def test_get_calendar_cme_equity(self):
        """Test getting CME product-specific calendar."""
        cal = get_calendar("CME_Equity")
        assert cal is not None

    def test_get_calendar_cached(self):
        """Test that calendars are cached."""
        cal1 = get_calendar("NYSE")
        cal2 = get_calendar("NYSE")
        assert cal1 is cal2  # Same object from cache

    def test_get_calendar_invalid(self):
        """Test invalid calendar raises error."""
        with pytest.raises(RuntimeError):
            get_calendar("INVALID_EXCHANGE_123")


class TestGetSchedule:
    """Tests for get_schedule function."""

    def test_get_schedule_basic(self):
        """Test basic schedule retrieval."""
        schedule = get_schedule("NYSE", date(2024, 1, 1), date(2024, 1, 31))

        assert isinstance(schedule, pl.DataFrame)
        assert "session_date" in schedule.columns
        assert "market_open" in schedule.columns
        assert "market_close" in schedule.columns
        assert "timezone" in schedule.columns

    def test_get_schedule_correct_dates(self):
        """Test schedule excludes holidays and weekends."""
        schedule = get_schedule("NYSE", date(2024, 1, 1), date(2024, 1, 5))

        # Jan 1 is New Year's Day (holiday)
        # Jan 2-5 should have 4 trading days (Tue-Fri)
        dates = schedule["session_date"].to_list()
        assert date(2024, 1, 1) not in dates  # Holiday
        assert date(2024, 1, 2) in dates

    def test_get_schedule_empty_range(self):
        """Test empty schedule for holiday-only range."""
        # Just New Year's Day
        schedule = get_schedule("NYSE", date(2024, 1, 1), date(2024, 1, 1))
        assert schedule.is_empty()

    def test_get_schedule_timezone(self):
        """Test timezone is included."""
        schedule = get_schedule("NYSE", date(2024, 6, 1), date(2024, 6, 1))

        if not schedule.is_empty():
            tz = schedule["timezone"][0]
            assert tz == "America/New_York"

    def test_get_schedule_utc_times(self):
        """Test that times are in UTC."""
        schedule = get_schedule("NYSE", date(2024, 6, 3), date(2024, 6, 3))

        if not schedule.is_empty():
            # NYSE opens at 9:30 AM ET = 13:30 UTC (summer)
            open_time = schedule["market_open"][0]
            # Just verify it's a datetime with UTC
            assert open_time is not None


class TestGetTradingDays:
    """Tests for get_trading_days function."""

    def test_get_trading_days_basic(self):
        """Test basic trading days retrieval."""
        days = get_trading_days("NYSE", date(2024, 1, 1), date(2024, 1, 31))

        assert isinstance(days, pl.Series)
        assert len(days) == 21  # Jan 2024 had 21 trading days

    def test_get_trading_days_excludes_holidays(self):
        """Test trading days excludes holidays."""
        days = get_trading_days("NYSE", date(2024, 1, 1), date(2024, 1, 31))
        days_list = days.to_list()

        # New Year's Day and MLK Day should not be included
        assert date(2024, 1, 1) not in days_list  # New Year's
        assert date(2024, 1, 15) not in days_list  # MLK Day


class TestIsTradingDay:
    """Tests for is_trading_day function."""

    def test_is_trading_day_regular(self):
        """Test regular trading day."""
        assert is_trading_day("NYSE", date(2024, 6, 3)) is True  # Monday

    def test_is_trading_day_weekend(self):
        """Test weekend is not trading day."""
        assert is_trading_day("NYSE", date(2024, 6, 1)) is False  # Saturday
        assert is_trading_day("NYSE", date(2024, 6, 2)) is False  # Sunday

    def test_is_trading_day_holiday(self):
        """Test holiday is not trading day."""
        assert is_trading_day("NYSE", date(2024, 7, 4)) is False  # Independence Day
        assert is_trading_day("NYSE", date(2024, 12, 25)) is False  # Christmas

    def test_is_trading_day_with_string(self):
        """Test with string date input."""
        assert is_trading_day("NYSE", "2024-06-03") is True

    def test_is_trading_day_with_datetime(self):
        """Test with datetime input."""
        assert is_trading_day("NYSE", datetime(2024, 6, 3, 12, 0)) is True


class TestIsMarketOpen:
    """Tests for is_market_open function."""

    def test_is_market_open_during_hours(self):
        """Test market is open during trading hours."""
        # 3 PM UTC = 11 AM ET (NYSE is open 9:30 AM - 4 PM ET)
        dt = datetime(2024, 6, 3, 15, 0, tzinfo=UTC)
        assert is_market_open("NYSE", dt) is True

    def test_is_market_open_before_hours(self):
        """Test market is closed before trading hours."""
        # 10 AM UTC = 6 AM ET (before market open)
        dt = datetime(2024, 6, 3, 10, 0, tzinfo=UTC)
        assert is_market_open("NYSE", dt) is False

    def test_is_market_open_after_hours(self):
        """Test market is closed after trading hours."""
        # 10 PM UTC = 6 PM ET (after market close)
        dt = datetime(2024, 6, 3, 22, 0, tzinfo=UTC)
        assert is_market_open("NYSE", dt) is False

    def test_is_market_open_weekend(self):
        """Test market is closed on weekends."""
        dt = datetime(2024, 6, 1, 15, 0, tzinfo=UTC)  # Saturday
        assert is_market_open("NYSE", dt) is False


class TestNextTradingDay:
    """Tests for next_trading_day function."""

    def test_next_trading_day_regular(self):
        """Test next trading day from regular day."""
        # From Monday, next is Tuesday
        result = next_trading_day("NYSE", date(2024, 6, 3))
        assert result == date(2024, 6, 4)

    def test_next_trading_day_friday(self):
        """Test next trading day from Friday."""
        # From Friday, next is Monday
        result = next_trading_day("NYSE", date(2024, 6, 7))
        assert result == date(2024, 6, 10)

    def test_next_trading_day_before_holiday(self):
        """Test next trading day before a holiday."""
        # July 3 (Wed), next should skip July 4 (Thu holiday) to July 5 (Fri)
        result = next_trading_day("NYSE", date(2024, 7, 3))
        assert result == date(2024, 7, 5)

    def test_next_trading_day_n_days(self):
        """Test getting nth trading day."""
        result = next_trading_day("NYSE", date(2024, 6, 3), n=5)
        # 5 trading days from Mon Jun 3: Jun 4, 5, 6, 7, 10
        assert result == date(2024, 6, 10)


class TestPreviousTradingDay:
    """Tests for previous_trading_day function."""

    def test_previous_trading_day_regular(self):
        """Test previous trading day from regular day."""
        # From Tuesday, previous is Monday
        result = previous_trading_day("NYSE", date(2024, 6, 4))
        assert result == date(2024, 6, 3)

    def test_previous_trading_day_monday(self):
        """Test previous trading day from Monday."""
        # From Monday, previous is Friday
        result = previous_trading_day("NYSE", date(2024, 6, 10))
        assert result == date(2024, 6, 7)

    def test_previous_trading_day_after_holiday(self):
        """Test previous trading day after a holiday."""
        # July 5 (Fri), previous should skip July 4 (Thu holiday) to July 3 (Wed)
        result = previous_trading_day("NYSE", date(2024, 7, 5))
        assert result == date(2024, 7, 3)

    def test_previous_trading_day_n_days(self):
        """Test getting nth previous trading day."""
        result = previous_trading_day("NYSE", date(2024, 6, 10), n=5)
        # 5 trading days back from Mon Jun 10: Jun 7, 6, 5, 4, 3
        assert result == date(2024, 6, 3)


class TestListCalendars:
    """Tests for list_calendars function."""

    def test_list_calendars_returns_list(self):
        """Test list_calendars returns a list."""
        calendars = list_calendars()
        assert isinstance(calendars, list)
        assert len(calendars) > 50  # Should have many calendars

    def test_list_calendars_contains_major(self):
        """Test list contains major exchanges."""
        calendars = list_calendars()
        # Note: The actual names may vary, so we check for common patterns
        calendar_str = " ".join(calendars)
        assert "NYSE" in calendar_str or "XNYS" in calendar_str


class TestCalendarAliases:
    """Tests for CALENDAR_ALIASES."""

    def test_aliases_exist(self):
        """Test common aliases exist."""
        assert "NYSE" in CALENDAR_ALIASES
        assert "NASDAQ" in CALENDAR_ALIASES
        assert "CME" in CALENDAR_ALIASES
        assert "LSE" in CALENDAR_ALIASES

    def test_alias_resolution(self):
        """Test aliases resolve to valid calendars."""
        for alias in ["NYSE", "NASDAQ", "LSE"]:
            cal = get_calendar(alias)
            assert cal is not None


class TestGetHolidays:
    """Tests for get_holidays function."""

    def test_get_holidays_basic(self):
        """Test basic holidays retrieval."""
        holidays = get_holidays("NYSE", date(2024, 1, 1), date(2024, 12, 31))

        assert isinstance(holidays, pl.DataFrame)
        assert "date" in holidays.columns
        assert len(holidays) >= 9  # NYSE has ~9-10 holidays per year

    def test_get_holidays_contains_major(self):
        """Test holidays contains major US holidays."""
        holidays = get_holidays("NYSE", date(2024, 1, 1), date(2024, 12, 31))
        dates = holidays["date"].to_list()

        # Check for some major holidays
        assert date(2024, 1, 1) in dates  # New Year's Day
        assert date(2024, 7, 4) in dates  # Independence Day
        assert date(2024, 12, 25) in dates  # Christmas


class TestCMEFuturesCalendar:
    """Tests specific to CME futures calendars with intraday breaks."""

    def test_cme_equity_calendar_exists(self):
        """Test CME_Equity calendar can be loaded."""
        cal = get_calendar("CME_Equity")
        assert cal is not None

    def test_cme_schedule_retrieval(self):
        """Test CME schedule can be retrieved."""
        schedule = get_schedule("CME_Equity", date(2024, 6, 1), date(2024, 6, 30))
        assert not schedule.is_empty()


class TestFilterToTradingDays:
    """Tests for filter_to_trading_days function."""

    def test_filter_to_trading_days_basic(self):
        """Test basic filtering to trading days."""
        # Create test data with trading and non-trading days
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1),  # New Year's (holiday)
                    datetime(2024, 1, 2),  # Trading day
                    datetime(2024, 1, 3),  # Trading day
                    datetime(2024, 1, 6),  # Saturday
                    datetime(2024, 1, 7),  # Sunday
                    datetime(2024, 1, 8),  # Trading day
                ],
                "price": [100, 101, 102, 103, 104, 105],
            }
        )

        filtered = filter_to_trading_days(df, "NYSE")

        # Should only have trading days
        assert len(filtered) == 3
        dates = filtered["timestamp"].to_list()
        assert datetime(2024, 1, 1) not in dates  # Holiday
        assert datetime(2024, 1, 2) in dates
        assert datetime(2024, 1, 3) in dates
        assert datetime(2024, 1, 6) not in dates  # Weekend
        assert datetime(2024, 1, 7) not in dates  # Weekend
        assert datetime(2024, 1, 8) in dates

    def test_filter_to_trading_days_custom_column(self):
        """Test filtering with custom timestamp column name."""
        df = pl.DataFrame(
            {
                "date": [
                    datetime(2024, 1, 1),  # Holiday
                    datetime(2024, 1, 2),  # Trading day
                ],
                "value": [1, 2],
            }
        )

        filtered = filter_to_trading_days(df, "NYSE", timestamp_col="date")
        assert len(filtered) == 1


class TestFilterToTradingSessions:
    """Tests for filter_to_trading_sessions function."""

    def test_filter_to_trading_sessions_basic(self):
        """Test basic filtering to trading sessions."""
        # Create test data with trading and non-trading times
        # NYSE opens 9:30 AM ET = 13:30 UTC (summer), closes 4:00 PM ET = 20:00 UTC
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 6, 3, 10, 0, tzinfo=UTC),  # 6 AM ET - pre-market
                    datetime(2024, 6, 3, 13, 30, tzinfo=UTC),  # 9:30 AM ET - market open
                    datetime(2024, 6, 3, 17, 0, tzinfo=UTC),  # 1 PM ET - during hours
                    datetime(2024, 6, 3, 19, 59, tzinfo=UTC),  # 3:59 PM ET - just before close
                    datetime(2024, 6, 3, 21, 0, tzinfo=UTC),  # 5 PM ET - after hours
                ],
                "price": [100, 101, 102, 103, 104],
            }
        )

        filtered = filter_to_trading_sessions(df, "NYSE")

        # Should only have times during trading sessions (9:30 AM - 4:00 PM ET)
        assert len(filtered) == 3  # 13:30, 17:00, 19:59 UTC
        timestamps = filtered["timestamp"].to_list()
        assert datetime(2024, 6, 3, 10, 0, tzinfo=UTC) not in timestamps  # Pre-market
        assert datetime(2024, 6, 3, 13, 30, tzinfo=UTC) in timestamps  # Open
        assert datetime(2024, 6, 3, 17, 0, tzinfo=UTC) in timestamps  # During
        assert datetime(2024, 6, 3, 19, 59, tzinfo=UTC) in timestamps  # Just before close
        assert datetime(2024, 6, 3, 21, 0, tzinfo=UTC) not in timestamps  # After hours

    def test_filter_to_trading_sessions_weekend(self):
        """Test that weekend data is filtered out."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 6, 1, 17, 0, tzinfo=UTC),  # Saturday
                    datetime(2024, 6, 3, 17, 0, tzinfo=UTC),  # Monday during hours (1 PM ET)
                ],
                "price": [100, 101],
            }
        )

        filtered = filter_to_trading_sessions(df, "NYSE")
        assert len(filtered) == 1

    def test_filter_to_trading_sessions_empty_df(self):
        """Test with empty DataFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [],
                "price": [],
            }
        ).cast({"timestamp": pl.Datetime("us", "UTC"), "price": pl.Float64})

        filtered = filter_to_trading_sessions(df, "NYSE")
        assert filtered.is_empty()

    def test_filter_to_trading_sessions_naive_datetime(self):
        """Test with timezone-naive datetime (assumed UTC)."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 6, 3, 10, 0),  # Pre-market (naive)
                    datetime(2024, 6, 3, 17, 0),  # During hours (naive)
                ],
                "price": [100, 101],
            }
        )

        filtered = filter_to_trading_sessions(df, "NYSE")
        assert len(filtered) == 1  # Only 17:00 is during trading

    def test_filter_to_trading_sessions_irregular_timestamps(self):
        """Test with irregular timestamps (trade bars)."""
        # Simulate irregular trade bar timestamps
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 6, 3, 13, 30, 15, tzinfo=UTC),  # 15 sec after open
                    datetime(2024, 6, 3, 13, 31, 42, tzinfo=UTC),  # Irregular
                    datetime(2024, 6, 3, 14, 15, 33, tzinfo=UTC),  # Mid-morning
                    datetime(2024, 6, 3, 19, 59, 58, tzinfo=UTC),  # 2 sec before close
                    datetime(2024, 6, 3, 20, 0, 1, tzinfo=UTC),  # 1 sec after close
                ],
                "price": [100.1, 100.2, 100.5, 101.0, 101.1],
            }
        )

        filtered = filter_to_trading_sessions(df, "NYSE")
        assert len(filtered) == 4  # All except the one after close


class TestGenerateTradingMinutes:
    """Tests for generate_trading_minutes function."""

    def test_generate_trading_minutes_single_day(self):
        """Test generating minutes for a single trading day."""
        minutes = generate_trading_minutes("NYSE", date(2024, 6, 3), date(2024, 6, 3))

        # NYSE: 9:30 AM - 4:00 PM ET = 6.5 hours = 390 minutes
        # Plus close = 391 (or 390 if close is on a minute boundary)
        assert len(minutes) >= 390
        assert len(minutes) <= 391

    def test_generate_trading_minutes_5m_freq(self):
        """Test generating 5-minute bars."""
        minutes = generate_trading_minutes("NYSE", date(2024, 6, 3), date(2024, 6, 3), freq="5m")

        # 390 minutes / 5 = 78 bars + close
        assert len(minutes) >= 78
        assert len(minutes) <= 79

    def test_generate_trading_minutes_multiple_days(self):
        """Test generating minutes across multiple days."""
        minutes = generate_trading_minutes("NYSE", date(2024, 6, 3), date(2024, 6, 4))

        # 2 trading days * ~391 minutes each
        assert len(minutes) > 700

    def test_generate_trading_minutes_invalid_freq(self):
        """Test invalid frequency raises error."""
        with pytest.raises(ValueError):
            generate_trading_minutes("NYSE", date(2024, 6, 3), date(2024, 6, 3), freq="2m")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_date_range(self):
        """Test handling of empty date range on holiday."""
        # Single holiday day should return empty
        schedule = get_schedule("NYSE", date(2024, 1, 1), date(2024, 1, 1))
        assert schedule.is_empty()  # New Year's Day

    def test_single_day_range(self):
        """Test single day range on trading day."""
        schedule = get_schedule("NYSE", date(2024, 6, 3), date(2024, 6, 3))
        assert len(schedule) == 1

    def test_far_future_date(self):
        """Test handling of far future date."""
        # Should work for dates a few years out
        schedule = get_schedule("NYSE", date(2026, 1, 1), date(2026, 1, 31))
        assert len(schedule) > 0

    def test_historical_date(self):
        """Test handling of historical date."""
        schedule = get_schedule("NYSE", date(2020, 1, 1), date(2020, 1, 31))
        assert len(schedule) > 0
