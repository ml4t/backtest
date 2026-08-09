"""Session alignment for trading calendars.

This module provides session-aware P&L computation for exchanges with
non-standard trading sessions like CME futures (5pm CT Sunday - 4pm CT Friday).

Example:
    >>> from ml4t.backtest.sessions import SessionConfig, compute_session_pnl
    >>>
    >>> # CME futures: sessions start 5pm CT previous day
    >>> config = SessionConfig(
    ...     calendar="CME_Equity",
    ...     timezone="America/Chicago",
    ...     session_start_time="17:00",
    ... )
    >>>
    >>> # Compute session-aligned daily P&L
    >>> daily_pnl = compute_session_pnl(equity_curve, config)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import polars as pl
from ml4t.specs.market_data import TimestampSemantics

if TYPE_CHECKING:
    pass


@dataclass
class SessionConfig:
    """Configuration for trading session alignment.

    Attributes:
        calendar: Exchange calendar name (e.g., "CME_Equity", "NYSE")
        timezone: Calendar timezone (e.g., "America/Chicago", "America/New_York")
        session_start_time: Override an evening session boundary (e.g., "17:00" for CME),
            or restate the calendar's standard morning open. Custom morning boundaries are
            rejected because morning-start sessions use the local calendar date. Evening
            overrides require an evening-start calendar or no exchange calendar. If None,
            uses the calendar's authoritative standard open.
    """

    calendar: str
    timezone: str = "UTC"
    session_start_time: str | None = None  # Format: "HH:MM"

    def __post_init__(self) -> None:
        if self.session_start_time is None:
            return
        _validate_session_start_time(self.calendar, self.session_start_time)

    def get_session_start_hour(self) -> int:
        """Get session start hour (0-23)."""
        if self.session_start_time:
            parts = self.session_start_time.split(":")
            return int(parts[0])
        return _default_session_start(self.calendar).hour

    def get_session_start_minute(self) -> int:
        """Get session start minute (0-59)."""
        if self.session_start_time:
            parts = self.session_start_time.split(":")
            return int(parts[1]) if len(parts) > 1 else 0
        return _default_session_start(self.calendar).minute

    def get_session_timezone(self) -> ZoneInfo:
        """Return the authoritative calendar timezone or configured fallback."""
        from .calendar import get_calendar

        try:
            return ZoneInfo(str(get_calendar(self.calendar).tz))
        except RuntimeError:
            return ZoneInfo(self.timezone)


def _default_session_start(calendar: str) -> time:
    from .calendar import get_standard_market_open_time

    try:
        return get_standard_market_open_time(calendar)
    except ValueError:
        return time.min


def _validate_session_start_time(calendar: str | None, session_start_time: str) -> None:
    parts = session_start_time.split(":")
    start = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    time(*start)
    if calendar is None:
        if start[0] >= 12:
            return
        raise ValueError("morning session_start_time requires exchange calendar metadata")
    from .calendar import get_standard_market_open_time

    try:
        market_open = get_standard_market_open_time(calendar)
    except ValueError:
        if start[0] >= 12:
            return
        raise
    standard_open = (market_open.hour, market_open.minute)
    if start == standard_open or (start[0] >= 12 and market_open.hour >= 12):
        return
    boundary = "morning" if start[0] < 12 else "evening"
    raise ValueError(
        f"custom {boundary} session_start_time {session_start_time!r} is unsupported for "
        f"calendar {calendar!r}; its standard market open is {market_open:%H:%M}"
    )


def session_date_for_timestamp(
    timestamp: datetime,
    *,
    calendar: str | None,
    timezone: str | None,
    session_start_time: str | None,
    data_frequency: Any | None,
    timestamp_semantics: TimestampSemantics | str | None,
) -> date:
    """Return the exchange session date for one feed timestamp."""
    from .calendar import get_calendar, get_calendar_sessions_by_open_date
    from .config import DataFrequency, _to_backtest_frequency

    frequency = _to_backtest_frequency(data_frequency) if data_frequency is not None else None
    semantics = (
        timestamp_semantics
        if isinstance(timestamp_semantics, TimestampSemantics)
        else TimestampSemantics(timestamp_semantics)
        if timestamp_semantics is not None
        else None
    )
    if (
        semantics is TimestampSemantics.SESSION_LABEL
        or frequency is DataFrequency.DAILY
        or (semantics is None and frequency is None)
    ):
        return timestamp.date()

    data_timezone = ZoneInfo(timezone or "UTC")
    exchange_timezone = (
        ZoneInfo(str(get_calendar(calendar).tz)) if calendar is not None else data_timezone
    )
    event_time = (
        timestamp.replace(tzinfo=data_timezone)
        if timestamp.tzinfo is None
        else timestamp.astimezone(exchange_timezone)
    )
    if event_time.tzinfo != exchange_timezone:
        event_time = event_time.astimezone(exchange_timezone)
    if calendar is not None and session_start_time is None:
        event_time_utc = event_time.astimezone(ZoneInfo("UTC"))
        event_date = event_time.date()
        label_years = (
            (event_date.year, event_date.year + 1)
            if event_date.month == 12 and event_date.day == 31
            else (event_date.year,)
        )
        sessions = (
            session
            for year in label_years
            for session in get_calendar_sessions_by_open_date(calendar, year).get(event_date, ())
            if session.market_open <= event_time_utc
        )
        latest = max(sessions, key=lambda session: session.market_open, default=None)
        return latest.session_date if latest is not None else event_date

    if session_start_time is not None:
        _validate_session_start_time(calendar, session_start_time)
    session_config = SessionConfig(
        calendar=_normalize_session_calendar(calendar or "UTC"),
        timezone=str(exchange_timezone),
        session_start_time=session_start_time,
    )
    return assign_session_date(
        event_time,
        exchange_timezone,
        session_config.get_session_start_hour(),
        session_config.get_session_start_minute(),
    ).date()


def _normalize_session_calendar(calendar: str) -> str:
    normalized = calendar.upper()
    if normalized in {"NYSE", "XNYS", "AMEX"}:
        return "NYSE"
    if normalized == "NASDAQ":
        return "NASDAQ"
    if normalized in {"CME", "CME_EQUITY"}:
        return "CME_Equity"
    if normalized in {"CBOT", "NYMEX", "COMEX"}:
        return normalized
    return calendar


def compute_session_pnl(
    equity_curve: list[tuple[datetime, float]],
    session_config: SessionConfig,
) -> pl.DataFrame:
    """Compute daily P&L aligned to trading session boundaries.

    For exchanges like CME where sessions span midnight (5pm CT Sunday
    to 4pm CT Monday = Monday session), this correctly groups data
    by trading session rather than calendar day.

    Args:
        equity_curve: List of (timestamp, portfolio_value) tuples
        session_config: Session alignment configuration

    Returns:
        DataFrame with columns:
            session_date: Trading session date
            session_start: Session start timestamp
            session_end: Session end timestamp
            open_equity: Equity at session start
            close_equity: Equity at session end
            high_equity: Intra-session high
            low_equity: Intra-session low
            pnl: Session P&L (close - open)
            return_pct: Session return percentage
            intra_session_dd: Maximum intra-session drawdown
            num_bars: Number of bars in session
    """
    if not equity_curve:
        return _empty_session_pnl_df()

    # Build DataFrame with session assignment
    timestamps = [ts for ts, _ in equity_curve]
    values = [v for _, v in equity_curve]

    # Assign session dates
    session_dates = []
    tz = session_config.get_session_timezone()
    session_start_hour = session_config.get_session_start_hour()
    session_start_minute = session_config.get_session_start_minute()

    for ts in timestamps:
        session_date = assign_session_date(ts, tz, session_start_hour, session_start_minute)
        session_dates.append(session_date)

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "equity": values,
            "session_date": session_dates,
        }
    )

    # Group by session and compute metrics
    session_pnl = (
        df.group_by("session_date")
        .agg(
            [
                pl.col("timestamp").first().alias("session_start"),
                pl.col("timestamp").last().alias("session_end"),
                pl.col("equity").first().alias("open_equity"),
                pl.col("equity").last().alias("close_equity"),
                pl.col("equity").max().alias("high_equity"),
                pl.col("equity").min().alias("low_equity"),
                pl.len().alias("num_bars"),
            ]
        )
        .sort("session_date")
    )

    # Compute P&L and returns
    session_pnl = session_pnl.with_columns(
        [
            (pl.col("close_equity") - pl.col("open_equity")).alias("pnl"),
        ]
    )

    # Return percent (relative to previous session close)
    prev_close = session_pnl.select(pl.col("close_equity").shift(1)).to_series()
    return_pct = (session_pnl["close_equity"] - prev_close) / prev_close
    return_pct = return_pct.fill_null(0.0)

    # Intra-session drawdown
    intra_dd = (session_pnl["low_equity"] - session_pnl["high_equity"]) / session_pnl["high_equity"]

    # Cumulative return from first session
    initial = session_pnl["open_equity"][0] if len(session_pnl) > 0 else 1.0
    cum_return = (session_pnl["close_equity"] / initial) - 1.0

    session_pnl = session_pnl.with_columns(
        [
            return_pct.alias("return_pct"),
            intra_dd.alias("intra_session_dd"),
            cum_return.alias("cumulative_return"),
        ]
    )

    return session_pnl


def assign_session_date(
    timestamp: datetime,
    timezone: ZoneInfo,
    session_start_hour: int,
    session_start_minute: int = 0,
) -> datetime:
    """Assign a timestamp to its trading session date.

    For sessions that start in the evening (like CME at 5pm), timestamps before the
    session start belong to the current day's session, and timestamps after belong to
    the next day's session. Morning-start sessions use the local calendar date for both
    pre-market and regular-hours bars; their open time does not define a date boundary.

    Args:
        timestamp: Bar timestamp (may be tz-aware or naive)
        timezone: Session timezone
        session_start_hour: Hour when session starts (0-23)
        session_start_minute: Minute when session starts (0-59)

    Returns:
        Session date (as datetime with date only, time=00:00)

    Example:
        CME session starts 5pm CT:
        - 2024-01-08 17:00 CT -> session_date = 2024-01-09 (Tuesday session)
        - 2024-01-08 16:59 CT -> session_date = 2024-01-08 (Monday session)
        - 2024-01-07 18:00 CT -> session_date = 2024-01-08 (Monday session, Sunday evening)
    """
    # Convert to session timezone if needed
    if timestamp.tzinfo is None:
        ts_local = timestamp.replace(tzinfo=timezone)
    else:
        ts_local = timestamp.astimezone(timezone)

    session_start_time = time(session_start_hour, session_start_minute)
    bar_time = ts_local.time()

    # If session starts in evening (hour >= 12), timestamps after session start
    # belong to NEXT calendar day's session
    if session_start_hour >= 12:
        if bar_time >= session_start_time:
            # After session start -> next calendar day's session
            session_date = ts_local.date() + timedelta(days=1)
        else:
            # Before session start -> current calendar day's session
            session_date = ts_local.date()
    else:
        # Morning-start sessions use the local calendar date for regular and pre-market bars.
        session_date = ts_local.date()

    return datetime(session_date.year, session_date.month, session_date.day)


def align_to_sessions(
    df: pl.DataFrame,
    session_config: SessionConfig,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Add session_date column to any DataFrame with timestamps.

    Args:
        df: DataFrame with a timestamp column
        session_config: Session alignment configuration
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with added 'session_date' column
    """
    tz = session_config.get_session_timezone()
    session_start_hour = session_config.get_session_start_hour()
    session_start_minute = session_config.get_session_start_minute()

    session_dates = []
    for ts in df[timestamp_col]:
        session_date = assign_session_date(ts, tz, session_start_hour, session_start_minute)
        session_dates.append(session_date)

    return df.with_columns(pl.Series("session_date", session_dates))


def _empty_session_pnl_df() -> pl.DataFrame:
    """Return empty DataFrame with session P&L schema."""
    return pl.DataFrame(
        schema={
            "session_date": pl.Datetime,
            "session_start": pl.Datetime,
            "session_end": pl.Datetime,
            "open_equity": pl.Float64,
            "close_equity": pl.Float64,
            "high_equity": pl.Float64,
            "low_equity": pl.Float64,
            "pnl": pl.Float64,
            "return_pct": pl.Float64,
            "intra_session_dd": pl.Float64,
            "cumulative_return": pl.Float64,
            "num_bars": pl.Int32,
        }
    )
