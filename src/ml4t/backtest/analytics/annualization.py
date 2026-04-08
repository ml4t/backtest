"""Helpers for annualization and session-aware result semantics."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from ml4t.specs.market_data import FeedSpec, TimestampSemantics

from ..calendar import get_schedule
from ..config import DataFrequency
from ..sessions import SessionConfig

_ANNUALIZATION_FACTORS: dict[str, int] = {
    "crypto": 365,
    "NYSE": 252,
    "NASDAQ": 252,
    "CME_Equity": 252,
    "CME_Agriculture": 252,
    "CME_Globex_Energy_and_Metals": 252,
    "LSE": 253,
    "XETRA": 252,
    "TSX": 252,
    "HKEX": 252,
    "JPX": 245,
}

_BAR_MINUTES: dict[DataFrequency, float] = {
    DataFrequency.HOURLY: 60.0,
    DataFrequency.MINUTE_30: 30.0,
    DataFrequency.MINUTE_15: 15.0,
    DataFrequency.MINUTE_5: 5.0,
    DataFrequency.MINUTE_1: 1.0,
}

_DEFAULT_TRADING_DAYS_PER_YEAR = 252.0
_DEFAULT_SESSION_MINUTES = 390.0


def get_annualization_factor(calendar: str | None) -> int:
    """Get annualization factor for a trading calendar."""
    if calendar is None:
        return int(_DEFAULT_TRADING_DAYS_PER_YEAR)

    cal_upper = calendar.upper()
    for name, factor in _ANNUALIZATION_FACTORS.items():
        if name.upper() == cal_upper:
            return factor

    try:
        from pandas_market_calendars import get_calendar

        cal = get_calendar(calendar)
        schedule = cal.schedule("2024-01-01", "2024-12-31")
        return len(schedule)
    except Exception:
        return int(_DEFAULT_TRADING_DAYS_PER_YEAR)


def resolve_periods_per_year(
    data_frequency: DataFrequency | Any | None,
    *,
    calendar: str | None,
) -> float | None:
    """Resolve periods-per-year from configured feed cadence and calendar metadata."""
    frequency = _coerce_frequency(data_frequency)
    if frequency is None:
        return None

    if frequency == DataFrequency.DAILY:
        return float(get_annualization_factor(calendar))

    bar_minutes = _BAR_MINUTES.get(frequency)
    if bar_minutes is None:
        return None

    session_minutes = _session_minutes_per_day(calendar)
    if session_minutes is None:
        session_minutes = _DEFAULT_SESSION_MINUTES

    annual_days = float(get_annualization_factor(calendar))
    return float(annual_days * (session_minutes / bar_minutes))


def should_session_align(
    *,
    calendar: str | None,
    feed_spec: FeedSpec | Any | None = None,
    timestamps: Sequence[datetime] | None = None,
) -> bool:
    """Determine whether result aggregation should align to trading sessions."""
    spec = FeedSpec.from_any(feed_spec) if feed_spec is not None else None
    semantics = spec.timestamp_semantics if spec is not None else None
    if semantics is not None and not isinstance(semantics, TimestampSemantics):
        semantics = TimestampSemantics(str(semantics))

    if semantics == TimestampSemantics.SESSION_LABEL:
        return False

    resolved_calendar = calendar if calendar is not None else (spec.calendar if spec else None)
    if resolved_calendar is None:
        return False

    session_start_time = spec.session_start_time if spec is not None else None
    session_config = SessionConfig(
        calendar=resolved_calendar,
        session_start_time=session_start_time,
    )
    if session_config.get_session_start_hour() < 12:
        return False

    if semantics in {TimestampSemantics.EVENT_TIME, TimestampSemantics.BAR_CLOSE}:
        return True

    return not (timestamps and _timestamps_look_date_labeled(timestamps))


def _coerce_frequency(data_frequency: DataFrequency | Any | None) -> DataFrequency | None:
    if data_frequency is None:
        return None
    if isinstance(data_frequency, DataFrequency):
        return data_frequency
    try:
        return DataFrequency(str(data_frequency))
    except ValueError:
        return None


def _session_minutes_per_day(calendar: str | None) -> float | None:
    if calendar is None:
        return None
    if calendar.upper() == "CRYPTO":
        return 24.0 * 60.0

    schedule = get_schedule(calendar, "2024-01-02", "2024-01-12", include_breaks=True)
    if schedule.is_empty():
        return None

    row = schedule.row(0, named=True)
    minutes = (row["market_close"] - row["market_open"]).total_seconds() / 60.0
    break_start = row.get("break_start")
    break_end = row.get("break_end")
    if break_start is not None and break_end is not None:
        minutes -= (break_end - break_start).total_seconds() / 60.0

    return minutes if minutes > 0 else None


def _timestamps_look_date_labeled(timestamps: Sequence[datetime]) -> bool:
    return all(
        ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0
        for ts in timestamps
    )
