"""Tests for session alignment utilities."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from ml4t.backtest.sessions import (
    SessionConfig,
    align_to_sessions,
    assign_session_date,
    compute_session_pnl,
    session_date_for_timestamp,
)


class TestSessionConfig:
    """Tests for SessionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SessionConfig(calendar="NYSE")
        assert config.timezone == "UTC"
        assert config.session_start_time is None

    def test_get_session_start_hour_custom(self):
        """Test custom session start hour."""
        config = SessionConfig(calendar="CME_Equity", session_start_time="17:30")
        assert config.get_session_start_hour() == 17
        assert config.get_session_start_minute() == 30

    def test_get_session_start_hour_defaults(self):
        """Test default session start hours for various exchanges."""
        cme = SessionConfig(calendar="CME_Equity")
        assert cme.get_session_start_hour() == 17

        nyse = SessionConfig(calendar="NYSE")
        assert nyse.get_session_start_hour() == 9
        assert nyse.get_session_start_minute() == 30

        nymex = SessionConfig(calendar="NYMEX")
        assert nymex.get_session_start_hour() == 18

    def test_get_session_start_hour_unknown_calendar(self):
        """Test unknown calendar defaults to midnight."""
        config = SessionConfig(calendar="UNKNOWN_EXCHANGE")
        assert config.get_session_start_hour() == 0
        assert config.get_session_start_minute() == 0

    def test_get_session_start_minute_no_colon(self):
        """Test session start time without minutes."""
        config = SessionConfig(calendar="CME_Equity", session_start_time="17")
        assert config.get_session_start_hour() == 17
        assert config.get_session_start_minute() == 0

    def test_custom_morning_session_boundary_is_rejected(self):
        with pytest.raises(ValueError, match="custom morning session_start_time"):
            SessionConfig(calendar="CRYPTO", session_start_time="06:00")

    def test_standard_morning_session_boundary_can_be_repeated(self):
        config = SessionConfig(calendar="NYSE", session_start_time="09:30")

        assert config.get_session_start_hour() == 9
        assert config.get_session_start_minute() == 30

    def test_standard_morning_boundary_accepts_calendar_alias(self):
        config = SessionConfig(calendar="XNYS", session_start_time="09:30")

        assert config.get_session_start_hour() == 9
        assert config.get_session_start_minute() == 30

    def test_standard_morning_boundary_comes_from_exchange_calendar(self):
        config = SessionConfig(calendar="XTKS", session_start_time="09:00")

        assert config.get_session_start_hour() == 9
        assert config.get_session_start_minute() == 0

    def test_nonstandard_morning_boundary_names_the_calendar_open(self):
        with pytest.raises(ValueError, match=r"XTKS.*standard market open is 09:00"):
            SessionConfig(calendar="XTKS", session_start_time="09:30")

    def test_morning_boundary_without_exchange_calendar_has_a_concise_error(self):
        with pytest.raises(ValueError, match="requires exchange calendar metadata"):
            session_date_for_timestamp(
                datetime(2024, 1, 5, 16, 0),
                calendar=None,
                timezone="UTC",
                session_start_time="09:30",
                data_frequency="1m",
                timestamp_semantics="bar_close",
            )

    def test_utc_calendar_name_is_reported_as_an_unknown_exchange(self):
        with pytest.raises(ValueError, match="UTC.*standard market open"):
            SessionConfig(calendar="UTC", session_start_time="09:30")

    def test_unknown_calendar_has_a_concise_standard_open_error(self):
        with pytest.raises(ValueError, match="UNKNOWN_EXCHANGE.*standard market open"):
            SessionConfig(calendar="UNKNOWN_EXCHANGE", session_start_time="09:30")


class TestAssignSessionDate:
    """Tests for assign_session_date()."""

    @pytest.fixture
    def chicago_tz(self) -> ZoneInfo:
        return ZoneInfo("America/Chicago")

    def test_cme_evening_after_session_start(self, chicago_tz: ZoneInfo):
        """Test CME: 6pm CT on Monday -> Tuesday session."""
        ts = datetime(2024, 1, 8, 18, 0)  # Monday 6pm CT
        session_date = assign_session_date(ts, chicago_tz, 17, 0)

        # 6pm is after 5pm session start, so belongs to Tuesday (1/9)
        assert session_date == datetime(2024, 1, 9, 0, 0)

    def test_cme_evening_before_session_start(self, chicago_tz: ZoneInfo):
        """Test CME: 4pm CT on Monday -> Monday session."""
        ts = datetime(2024, 1, 8, 16, 0)  # Monday 4pm CT
        session_date = assign_session_date(ts, chicago_tz, 17, 0)

        # 4pm is before 5pm session start, so belongs to Monday (1/8)
        assert session_date == datetime(2024, 1, 8, 0, 0)

    def test_cme_exactly_at_session_start(self, chicago_tz: ZoneInfo):
        """Test CME: exactly 5pm CT -> next day's session."""
        ts = datetime(2024, 1, 8, 17, 0)  # Monday 5pm CT exactly
        session_date = assign_session_date(ts, chicago_tz, 17, 0)

        # Exactly at session start goes to next day
        assert session_date == datetime(2024, 1, 9, 0, 0)

    def test_cme_early_morning(self, chicago_tz: ZoneInfo):
        """Test CME: 3am CT on Tuesday -> Tuesday session."""
        ts = datetime(2024, 1, 9, 3, 0)  # Tuesday 3am CT
        session_date = assign_session_date(ts, chicago_tz, 17, 0)

        # 3am is before 5pm, so belongs to Tuesday (1/9)
        assert session_date == datetime(2024, 1, 9, 0, 0)

    def test_nyse_regular_hours(self):
        """Test NYSE: 10am ET -> same day session."""
        ny_tz = ZoneInfo("America/New_York")
        ts = datetime(2024, 1, 8, 10, 0)  # Monday 10am ET
        session_date = assign_session_date(ts, ny_tz, 9, 30)

        # 10am is after 9:30am session start, same day
        assert session_date == datetime(2024, 1, 8, 0, 0)

    def test_nyse_before_session(self):
        """Test NYSE: pre-market data remains in the same-day session."""
        ny_tz = ZoneInfo("America/New_York")
        ts = datetime(2024, 1, 8, 8, 0)  # Monday 8am ET
        session_date = assign_session_date(ts, ny_tz, 9, 30)

        assert session_date == datetime(2024, 1, 8, 0, 0)

    def test_timezone_aware_input(self, chicago_tz: ZoneInfo):
        """Test with timezone-aware timestamp input."""
        ts = datetime(2024, 1, 8, 18, 0, tzinfo=chicago_tz)
        session_date = assign_session_date(ts, chicago_tz, 17, 0)

        assert session_date == datetime(2024, 1, 9, 0, 0)

    def test_registered_calendar_alias_uses_authoritative_market_open(self):
        timestamp = datetime(2024, 1, 8, 18, 0)

        assert (
            session_date_for_timestamp(
                timestamp,
                calendar="CMES",
                timezone="America/Chicago",
                session_start_time=None,
                data_frequency="1m",
                timestamp_semantics="bar_close",
            )
            == datetime(2024, 1, 9).date()
        )


class TestComputeSessionPnL:
    """Tests for compute_session_pnl()."""

    @pytest.fixture
    def cme_config(self) -> SessionConfig:
        return SessionConfig(
            calendar="CME_Equity",
            timezone="America/Chicago",
            session_start_time="17:00",
        )

    def test_empty_equity_curve(self, cme_config: SessionConfig):
        """Test with empty equity curve."""
        result = compute_session_pnl([], cme_config)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0
        assert "session_date" in result.columns
        assert "pnl" in result.columns

    def test_nyse_premarket_and_regular_bars_share_the_same_session(self):
        new_york = ZoneInfo("America/New_York")
        equity_curve = [
            (datetime(2024, 1, 8, 8, 0, tzinfo=new_york), 100000.0),
            (datetime(2024, 1, 8, 10, 0, tzinfo=new_york), 100100.0),
        ]

        result = compute_session_pnl(
            equity_curve,
            SessionConfig(calendar="NYSE", timezone="America/New_York"),
        )

        assert result["session_date"].to_list() == [datetime(2024, 1, 8)]
        assert result["num_bars"].to_list() == [2]

    def test_single_session(self, cme_config: SessionConfig):
        """Test with single trading session."""
        # All bars on Monday after 5pm CT (Tuesday session)
        chicago = ZoneInfo("America/Chicago")
        equity_curve = [
            (datetime(2024, 1, 8, 18, 0, tzinfo=chicago), 100000.0),
            (datetime(2024, 1, 8, 20, 0, tzinfo=chicago), 100100.0),
            (datetime(2024, 1, 8, 22, 0, tzinfo=chicago), 100200.0),
        ]

        result = compute_session_pnl(equity_curve, cme_config)

        assert len(result) == 1
        assert result["open_equity"][0] == 100000.0
        assert result["close_equity"][0] == 100200.0
        assert result["pnl"][0] == 200.0

    def test_multi_session(self, cme_config: SessionConfig):
        """Test with multiple trading sessions."""
        chicago = ZoneInfo("America/Chicago")
        equity_curve = [
            # Monday evening -> Tuesday session
            (datetime(2024, 1, 8, 18, 0, tzinfo=chicago), 100000.0),
            (datetime(2024, 1, 8, 20, 0, tzinfo=chicago), 100100.0),
            # Tuesday morning (still Tuesday session)
            (datetime(2024, 1, 9, 8, 0, tzinfo=chicago), 100200.0),
            # Tuesday evening -> Wednesday session
            (datetime(2024, 1, 9, 18, 0, tzinfo=chicago), 100300.0),
            (datetime(2024, 1, 9, 20, 0, tzinfo=chicago), 100500.0),
        ]

        result = compute_session_pnl(equity_curve, cme_config)

        assert len(result) == 2  # Two sessions

        # First session (Tuesday)
        assert result["session_date"][0] == datetime(2024, 1, 9, 0, 0)
        assert result["open_equity"][0] == 100000.0
        assert result["close_equity"][0] == 100200.0
        assert result["num_bars"][0] == 3

        # Second session (Wednesday)
        assert result["session_date"][1] == datetime(2024, 1, 10, 0, 0)

    def test_high_low_equity(self, cme_config: SessionConfig):
        """Test high/low equity tracking within session."""
        chicago = ZoneInfo("America/Chicago")
        equity_curve = [
            (datetime(2024, 1, 8, 18, 0, tzinfo=chicago), 100000.0),
            (datetime(2024, 1, 8, 19, 0, tzinfo=chicago), 100500.0),  # High
            (datetime(2024, 1, 8, 20, 0, tzinfo=chicago), 99800.0),  # Low
            (datetime(2024, 1, 8, 21, 0, tzinfo=chicago), 100200.0),
        ]

        result = compute_session_pnl(equity_curve, cme_config)

        assert result["high_equity"][0] == 100500.0
        assert result["low_equity"][0] == 99800.0
        assert result["intra_session_dd"][0] < 0  # Should be negative

    def test_return_pct_calculation(self, cme_config: SessionConfig):
        """Test return percentage between sessions."""
        chicago = ZoneInfo("America/Chicago")
        equity_curve = [
            # Session 1
            (datetime(2024, 1, 8, 18, 0, tzinfo=chicago), 100000.0),
            (datetime(2024, 1, 8, 20, 0, tzinfo=chicago), 105000.0),
            # Session 2
            (datetime(2024, 1, 9, 18, 0, tzinfo=chicago), 105000.0),
            (datetime(2024, 1, 9, 20, 0, tzinfo=chicago), 110250.0),  # +5%
        ]

        result = compute_session_pnl(equity_curve, cme_config)

        # First session has 0 return (no previous close)
        assert result["return_pct"][0] == 0.0
        # Second session: (110250 - 105000) / 105000 = 0.05
        assert abs(result["return_pct"][1] - 0.05) < 0.001

    def test_cumulative_return(self, cme_config: SessionConfig):
        """Test cumulative return from first session."""
        chicago = ZoneInfo("America/Chicago")
        equity_curve = [
            (datetime(2024, 1, 8, 18, 0, tzinfo=chicago), 100000.0),
            (datetime(2024, 1, 8, 20, 0, tzinfo=chicago), 110000.0),  # +10%
            (datetime(2024, 1, 9, 18, 0, tzinfo=chicago), 110000.0),
            (datetime(2024, 1, 9, 20, 0, tzinfo=chicago), 121000.0),  # +21% total
        ]

        result = compute_session_pnl(equity_curve, cme_config)

        # Cumulative return: (close / initial_open) - 1
        assert abs(result["cumulative_return"][0] - 0.10) < 0.001
        assert abs(result["cumulative_return"][1] - 0.21) < 0.001


class TestAlignToSessions:
    """Tests for align_to_sessions()."""

    def test_basic_alignment(self):
        """Test basic session alignment on DataFrame."""
        chicago = ZoneInfo("America/Chicago")
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 8, 18, 0, tzinfo=chicago),
                    datetime(2024, 1, 8, 20, 0, tzinfo=chicago),
                    datetime(2024, 1, 9, 8, 0, tzinfo=chicago),
                ],
                "value": [1, 2, 3],
            }
        )

        config = SessionConfig(
            calendar="CME_Equity",
            timezone="America/Chicago",
            session_start_time="17:00",
        )

        result = align_to_sessions(df, config)

        assert "session_date" in result.columns
        # All three bars belong to Tuesday session
        assert result["session_date"][0] == datetime(2024, 1, 9, 0, 0)
        assert result["session_date"][1] == datetime(2024, 1, 9, 0, 0)
        assert result["session_date"][2] == datetime(2024, 1, 9, 0, 0)

    def test_custom_timestamp_column(self):
        """Test with custom timestamp column name."""
        chicago = ZoneInfo("America/Chicago")
        df = pl.DataFrame(
            {
                "my_time": [datetime(2024, 1, 8, 18, 0, tzinfo=chicago)],
                "value": [100],
            }
        )

        config = SessionConfig(
            calendar="CME_Equity",
            timezone="America/Chicago",
        )

        result = align_to_sessions(df, config, timestamp_col="my_time")

        assert "session_date" in result.columns
