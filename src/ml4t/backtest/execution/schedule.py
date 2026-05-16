"""Rebalance schedule resolution utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Any

import polars as pl

from ml4t.specs.market_data import FeedSpec, TimestampSemantics

from ..calendar import get_schedule
from ..config import DataFrequency, _to_backtest_frequency
from ..sessions import SessionConfig, assign_session_date


class RebalanceCadence(str, Enum):
    """Supported rebalance cadences."""

    EVERY_BAR = "every_bar"
    EVERY_SESSION = "every_session"
    FIXED_N_SESSIONS = "fixed_n_sessions"
    WEEKLY = "weekly"
    MONTH_END = "month_end"
    EXPLICIT_TIMESTAMPS = "explicit_timestamps"


@dataclass(frozen=True)
class RebalanceSchedule:
    """Describe when a strategy or executor should rebalance."""

    cadence: RebalanceCadence = RebalanceCadence.EVERY_BAR
    every_n: int = 1
    timestamps: tuple[datetime, ...] = ()

    def __post_init__(self) -> None:
        if self.cadence == RebalanceCadence.FIXED_N_SESSIONS and self.every_n < 1:
            raise ValueError("RebalanceSchedule.every_n must be >= 1")
        if self.cadence == RebalanceCadence.EXPLICIT_TIMESTAMPS and not self.timestamps:
            raise ValueError("Explicit timestamp schedules require at least one timestamp")

    @classmethod
    def every_bar(cls) -> RebalanceSchedule:
        return cls(cadence=RebalanceCadence.EVERY_BAR)

    @classmethod
    def every_session(cls) -> RebalanceSchedule:
        return cls(cadence=RebalanceCadence.EVERY_SESSION)

    @classmethod
    def fixed_n_sessions(cls, n: int) -> RebalanceSchedule:
        return cls(cadence=RebalanceCadence.FIXED_N_SESSIONS, every_n=n)

    @classmethod
    def weekly(cls) -> RebalanceSchedule:
        return cls(cadence=RebalanceCadence.WEEKLY)

    @classmethod
    def month_end(cls) -> RebalanceSchedule:
        return cls(cadence=RebalanceCadence.MONTH_END)

    @classmethod
    def explicit_timestamps(cls, timestamps: Sequence[datetime]) -> RebalanceSchedule:
        return cls(
            cadence=RebalanceCadence.EXPLICIT_TIMESTAMPS,
            timestamps=tuple(sorted({_coerce_timestamp(ts) for ts in timestamps})),
        )


def resolve_rebalance_timestamps(
    available_timestamps: Sequence[datetime] | pl.Series,
    schedule: RebalanceSchedule | RebalanceCadence | str,
    *,
    feed_spec: FeedSpec | Any | None = None,
    calendar: str | None = None,
    timezone: str | None = None,
    session_start_time: str | None = None,
    data_frequency: Any | None = None,
    timestamp_semantics: TimestampSemantics | str | None = None,
) -> pl.Series:
    """Resolve rebalance timestamps from available bars and schedule semantics."""
    ts_list = _normalize_timestamps(available_timestamps)
    if not ts_list:
        return pl.Series("timestamp", [], dtype=pl.Datetime("us"))

    schedule = _coerce_schedule(schedule)
    cadence = schedule.cadence

    if cadence == RebalanceCadence.EVERY_BAR:
        return pl.Series("timestamp", ts_list)

    if cadence == RebalanceCadence.EXPLICIT_TIMESTAMPS:
        explicit = set(schedule.timestamps)
        return pl.Series("timestamp", [ts for ts in ts_list if ts in explicit])

    metadata = _resolve_schedule_metadata(
        ts_list,
        feed_spec=feed_spec,
        calendar=calendar,
        timezone=timezone,
        session_start_time=session_start_time,
        data_frequency=data_frequency,
        timestamp_semantics=timestamp_semantics,
    )
    session_config = _build_session_config(
        ts_list,
        calendar=metadata["calendar"],
        timezone=metadata["timezone"],
        timezone_explicit=metadata["timezone_explicit"],
        session_start_time=metadata["session_start_time"],
    )
    session_dates, session_closes = _resolve_sessions(
        ts_list,
        session_config=session_config,
        timestamp_semantics=metadata["timestamp_semantics"],
    )

    if cadence == RebalanceCadence.EVERY_SESSION:
        return pl.Series("timestamp", session_closes)

    if cadence == RebalanceCadence.FIXED_N_SESSIONS:
        return pl.Series("timestamp", session_closes[:: schedule.every_n])

    grouped: dict[tuple[int, int], datetime] = {}
    for session_date, ts in zip(session_dates, session_closes, strict=False):
        if cadence == RebalanceCadence.WEEKLY:
            key = session_date.isocalendar()[:2]
        elif cadence == RebalanceCadence.MONTH_END:
            key = (session_date.year, session_date.month)
        else:
            raise ValueError(f"Unsupported rebalance cadence: {cadence}")
        grouped[key] = ts

    return pl.Series("timestamp", list(grouped.values()))


def _normalize_timestamps(available_timestamps: Sequence[datetime] | pl.Series) -> list[datetime]:
    if isinstance(available_timestamps, pl.Series):
        if available_timestamps.is_empty():
            return []
        return sorted({_coerce_timestamp(ts) for ts in available_timestamps.to_list()})
    return sorted({_coerce_timestamp(ts) for ts in available_timestamps})


def _coerce_timestamp(value: datetime | date) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    raise TypeError(f"Unsupported timestamp type: {type(value).__name__}")


def _coerce_schedule(schedule: RebalanceSchedule | RebalanceCadence | str) -> RebalanceSchedule:
    if isinstance(schedule, RebalanceSchedule):
        return schedule
    if isinstance(schedule, str):
        schedule = RebalanceCadence(schedule)
    return RebalanceSchedule(cadence=schedule)


def _resolve_schedule_metadata(
    timestamps: Sequence[datetime],
    *,
    feed_spec: FeedSpec | Any | None,
    calendar: str | None,
    timezone: str | None,
    session_start_time: str | None,
    data_frequency: Any | None,
    timestamp_semantics: TimestampSemantics | str | None,
) -> dict[str, Any]:
    spec = FeedSpec.from_any(feed_spec) if feed_spec is not None else None

    resolved_calendar = calendar if calendar is not None else (spec.calendar if spec else None)
    resolved_timezone = timezone if timezone is not None else (spec.timezone if spec else None)
    timezone_explicit = resolved_timezone is not None
    if resolved_timezone is None:
        resolved_timezone = "UTC"
    resolved_session_start = (
        session_start_time
        if session_start_time is not None
        else (spec.session_start_time if spec else None)
    )
    resolved_frequency = (
        data_frequency if data_frequency is not None else (spec.data_frequency if spec else None)
    )
    semantics = (
        timestamp_semantics
        if timestamp_semantics is not None
        else (spec.timestamp_semantics if spec else None)
    )

    if semantics is None:
        semantics = _infer_timestamp_semantics(timestamps, resolved_frequency)
    elif not isinstance(semantics, TimestampSemantics):
        semantics = TimestampSemantics(str(semantics))

    return {
        "calendar": resolved_calendar,
        "timezone": resolved_timezone,
        "timezone_explicit": timezone_explicit,
        "session_start_time": resolved_session_start,
        "data_frequency": resolved_frequency,
        "timestamp_semantics": semantics,
    }


def _infer_timestamp_semantics(
    timestamps: Sequence[datetime],
    data_frequency: Any | None,
) -> TimestampSemantics:
    if data_frequency is not None:
        frequency = _to_backtest_frequency(data_frequency)
        if frequency == DataFrequency.DAILY and _timestamps_look_date_labeled(timestamps):
            return TimestampSemantics.SESSION_LABEL

    if _timestamps_look_date_labeled(timestamps):
        return TimestampSemantics.SESSION_LABEL

    return TimestampSemantics.EVENT_TIME


def _timestamps_look_date_labeled(timestamps: Sequence[datetime]) -> bool:
    return all(
        ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0
        for ts in timestamps
    )


def _build_session_config(
    timestamps: Sequence[datetime],
    *,
    calendar: str | None,
    timezone: str,
    timezone_explicit: bool,
    session_start_time: str | None,
) -> SessionConfig:
    if calendar is None:
        return SessionConfig(
            calendar="UTC", timezone=timezone, session_start_time=session_start_time
        )

    inferred_timezone = timezone
    if not timezone_explicit:
        schedule = get_schedule(calendar, timestamps[0].date(), timestamps[-1].date())
        if not schedule.is_empty():
            inferred_timezone = schedule["timezone"][0]

    return SessionConfig(
        calendar=_normalize_session_calendar(calendar),
        timezone=inferred_timezone,
        session_start_time=session_start_time,
    )


def _normalize_session_calendar(calendar: str) -> str:
    normalized = calendar.upper()
    if normalized in {"NYSE", "XNYS", "AMEX"}:
        return "NYSE"
    if normalized == "NASDAQ":
        return "NASDAQ"
    if normalized in {"CME", "CME_EQUITY"}:
        return "CME_Equity"
    if normalized == "CBOT":
        return "CBOT"
    if normalized == "NYMEX":
        return "NYMEX"
    if normalized == "COMEX":
        return "COMEX"
    return calendar


def _resolve_sessions(
    timestamps: Sequence[datetime],
    *,
    session_config: SessionConfig,
    timestamp_semantics: TimestampSemantics,
) -> tuple[list[datetime], list[datetime]]:
    tz = _session_config_timezone(session_config)
    session_start_hour = session_config.get_session_start_hour()
    session_start_minute = session_config.get_session_start_minute()

    session_closes: dict[datetime, datetime] = {}
    for ts in timestamps:
        if timestamp_semantics == TimestampSemantics.SESSION_LABEL:
            session_date = _session_label_date(ts, tz)
        else:
            session_date = assign_session_date(ts, tz, session_start_hour, session_start_minute)
        session_closes[session_date] = ts
    session_dates = list(session_closes.keys())
    return session_dates, list(session_closes.values())


def _session_label_date(timestamp: datetime, timezone) -> datetime:
    ts_local = timestamp if timestamp.tzinfo is None else timestamp.astimezone(timezone)
    return datetime(ts_local.year, ts_local.month, ts_local.day)


def _session_config_timezone(session_config: SessionConfig):
    from zoneinfo import ZoneInfo

    return ZoneInfo(session_config.timezone)
