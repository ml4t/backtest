"""Rebalance schedule resolution utilities."""

from __future__ import annotations

import warnings
from calendar import monthrange
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
from ml4t.specs.market_data import FeedSpec, TimestampSemantics

from ..calendar import get_schedule
from ..config import DataFrequency, _to_backtest_frequency
from ..sessions import session_date_for_timestamp


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
    _timestamp_set: frozenset[datetime] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.cadence == RebalanceCadence.FIXED_N_SESSIONS and self.every_n < 1:
            raise ValueError("RebalanceSchedule.every_n must be >= 1")
        timestamps = tuple(sorted({_coerce_timestamp(ts) for ts in self.timestamps}))
        if self.cadence == RebalanceCadence.EXPLICIT_TIMESTAMPS and not timestamps:
            raise ValueError("Explicit timestamp schedules require at least one timestamp")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "_timestamp_set", frozenset(timestamps))

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
            timestamps=tuple(timestamps),
        )


def is_rebalance_timestamp(
    timestamp: datetime,
    schedule: RebalanceSchedule | RebalanceCadence | str,
    *,
    session_index: int,
    calendar: str | None = None,
    timezone: str | None = None,
    session_start_time: str | None = None,
    data_frequency: Any | None = None,
    timestamp_semantics: TimestampSemantics | str | None = None,
    is_session_close: bool | None = None,
) -> bool:
    """Evaluate a schedule from current calendar metadata without future feed timestamps."""
    resolved = _coerce_schedule(schedule)
    return _evaluate_rebalance_timestamp(
        timestamp,
        resolved,
        session_index=session_index,
        calendar=calendar,
        timezone=timezone,
        session_start_time=session_start_time,
        data_frequency=data_frequency,
        timestamp_semantics=timestamp_semantics,
        is_session_close=is_session_close,
        session_date=None,
    )


def _evaluate_rebalance_timestamp(
    timestamp: datetime,
    schedule: RebalanceSchedule,
    *,
    session_index: int,
    calendar: str | None,
    timezone: str | None,
    session_start_time: str | None,
    data_frequency: Any | None,
    timestamp_semantics: TimestampSemantics | str | None,
    is_session_close: bool | None,
    session_date: date | None,
) -> bool:
    """Evaluate one resolved schedule with an optional precomputed session date."""
    resolved = schedule
    cadence = resolved.cadence
    if cadence is RebalanceCadence.EVERY_BAR:
        return True
    if cadence is RebalanceCadence.EXPLICIT_TIMESTAMPS:
        return timestamp in resolved._timestamp_set
    _validate_schedule_calendar(resolved, calendar)
    if is_session_close is None:
        is_session_close = _is_session_close_timestamp(
            timestamp,
            calendar=calendar,
            timezone=timezone,
            data_frequency=data_frequency,
            timestamp_semantics=timestamp_semantics,
        )
    if not is_session_close:
        return False
    if cadence is RebalanceCadence.EVERY_SESSION:
        return True
    if cadence is RebalanceCadence.FIXED_N_SESSIONS:
        return (session_index - 1) % resolved.every_n == 0

    if session_date is None:
        session_date = session_date_for_timestamp(
            timestamp,
            calendar=calendar,
            timezone=timezone,
            session_start_time=session_start_time,
            data_frequency=data_frequency,
            timestamp_semantics=timestamp_semantics,
        )
    period_start, period_end = _period_bounds(cadence, session_date)

    return session_date == _calendar_period_end(calendar, period_start, period_end)


def _is_session_close_timestamp(
    timestamp: datetime,
    *,
    calendar: str | None,
    timezone: str | None,
    data_frequency: Any | None,
    timestamp_semantics: TimestampSemantics | str | None,
) -> bool:
    frequency, semantics = _normalize_schedule_metadata(data_frequency, timestamp_semantics)
    if semantics is TimestampSemantics.SESSION_LABEL or frequency is DataFrequency.DAILY:
        return True
    if timestamp.time() == time.min and semantics is None and frequency is None:
        return True
    if semantics is None and frequency is None:
        _warn_missing_boundary_metadata()
        return True
    if calendar is None:
        raise ValueError("intraday session schedules require calendar metadata or is_session_close")

    source_timezone = ZoneInfo(timezone or "UTC")
    localized = (
        timestamp.replace(tzinfo=source_timezone)
        if timestamp.tzinfo is None
        else timestamp.astimezone(source_timezone)
    )
    event_time = localized.astimezone(ZoneInfo("UTC"))
    event_date = localized.date()
    closes = _calendar_closes(
        calendar,
        event_date - timedelta(days=1),
        event_date + timedelta(days=1),
    )
    return event_time in closes


@lru_cache(maxsize=512)
def _calendar_closes(calendar: str, start: date, end: date) -> frozenset[datetime]:
    return frozenset(get_schedule(calendar, start, end)["market_close"])


@lru_cache(maxsize=512)
def _calendar_period_end(calendar: str, start: date, end: date) -> date | None:
    schedule = get_schedule(calendar, start, end)
    return None if schedule.is_empty() else schedule["session_date"][-1]


def _warn_missing_boundary_metadata() -> None:
    warnings.warn(
        "session cadence has no data_frequency or timestamp_semantics; events are treated as "
        "daily session closes",
        UserWarning,
        stacklevel=2,
        skip_file_prefixes=(str(Path(__file__).parents[1]),),
    )


def _session_requires_close(
    schedule: RebalanceSchedule,
    session_date: date,
    session_index: int,
    calendar: str | None,
) -> bool:
    cadence = schedule.cadence
    if cadence is RebalanceCadence.EVERY_SESSION:
        return True
    if cadence is RebalanceCadence.FIXED_N_SESSIONS:
        return (session_index - 1) % schedule.every_n == 0
    if cadence not in {RebalanceCadence.WEEKLY, RebalanceCadence.MONTH_END}:
        return False
    _validate_schedule_calendar(schedule, calendar)
    if calendar is None:
        raise ValueError("weekly and month_end schedules require calendar metadata")
    period_start, period_end = _period_bounds(cadence, session_date)
    return session_date == _calendar_period_end(calendar, period_start, period_end)


def _period_bounds(cadence: RebalanceCadence, session_date: date) -> tuple[date, date]:
    if cadence is RebalanceCadence.WEEKLY:
        period_start = session_date - timedelta(days=session_date.weekday())
        return period_start, period_start + timedelta(days=6)
    if cadence is RebalanceCadence.MONTH_END:
        return (
            session_date.replace(day=1),
            session_date.replace(day=monthrange(session_date.year, session_date.month)[1]),
        )
    raise ValueError(f"Unsupported period cadence: {cadence}")


@lru_cache(maxsize=512)
def _calendar_close_for_session(calendar: str, session_date: date) -> datetime | None:
    schedule = get_schedule(calendar, session_date, session_date)
    return None if schedule.is_empty() else schedule["market_close"][0]


def _event_time_utc(timestamp: datetime, timezone: str | None) -> datetime:
    source_timezone = ZoneInfo(timezone or "UTC")
    localized = (
        timestamp.replace(tzinfo=source_timezone)
        if timestamp.tzinfo is None
        else timestamp.astimezone(source_timezone)
    )
    return localized.astimezone(ZoneInfo("UTC"))


def _session_reached_expected_close(
    calendar: str | None,
    session_date: date,
    last_event_time: datetime | None,
    timezone: str | None,
) -> bool:
    if last_event_time is None or calendar is None:
        return True
    expected = _calendar_close_for_session(calendar, session_date)
    return expected is None or _event_time_utc(last_event_time, timezone) >= expected


def _raise_close_alignment_error(
    cadence: RebalanceCadence,
    calendar: str | None,
    session_date: date,
) -> None:
    expected = _calendar_close_for_session(calendar, session_date) if calendar is not None else None
    expected_text = expected.isoformat() if expected is not None else "unavailable"
    raise ValueError(
        f"{cadence.value} resolved no session closes for required session {session_date} on "
        f"calendar {calendar!r}; expected market_close {expected_text}. Verify that intraday "
        "timestamps align with the exchange close and declared timestamp semantics"
    )


def _validate_schedule_calendar(
    schedule: RebalanceSchedule,
    calendar: str | None,
) -> None:
    if (
        schedule.cadence in {RebalanceCadence.WEEKLY, RebalanceCadence.MONTH_END}
        and calendar is None
    ):
        raise ValueError("weekly and month_end schedules require calendar metadata")


class _OnlineRebalanceEvaluator:
    """Evaluate a schedule causally and diagnose missed required session closes."""

    def __init__(
        self,
        schedule: RebalanceSchedule,
        *,
        calendar: str | None,
        timezone: str | None,
        session_start_time: str | None,
        data_frequency: Any | None,
        timestamp_semantics: TimestampSemantics | str | None,
    ) -> None:
        _validate_schedule_calendar(schedule, calendar)
        self.schedule = schedule
        self.calendar = calendar
        self.timezone = timezone
        self.session_start_time = session_start_time
        self.data_frequency = data_frequency
        self.timestamp_semantics = timestamp_semantics
        self._session_date: date | None = None
        self._session_index = 0
        self._required_close_matched = False
        self._last_event_time: datetime | None = None

    def evaluate(self, timestamp: datetime, *, is_session_close: bool | None = None) -> bool:
        session_date = session_date_for_timestamp(
            timestamp,
            calendar=self.calendar,
            timezone=self.timezone,
            session_start_time=self.session_start_time,
            data_frequency=self.data_frequency,
            timestamp_semantics=self.timestamp_semantics,
        )
        if (
            self.calendar is not None
            and _calendar_close_for_session(self.calendar, session_date) is None
        ):
            return False
        if self._session_date is not None and session_date < self._session_date:
            raise ValueError(
                f"session date moved backward from {self._session_date} to {session_date}; "
                "start a new evaluator or call TargetWeightExecutor.reset() before another run"
            )
        if session_date != self._session_date:
            if (
                self._session_date is not None
                and _session_requires_close(
                    self.schedule,
                    self._session_date,
                    self._session_index,
                    self.calendar,
                )
                and not self._required_close_matched
                and _session_reached_expected_close(
                    self.calendar,
                    self._session_date,
                    self._last_event_time,
                    self.timezone,
                )
            ):
                _raise_close_alignment_error(
                    self.schedule.cadence,
                    self.calendar,
                    self._session_date,
                )
            self._session_date = session_date
            self._session_index += 1
            self._required_close_matched = False
        self._last_event_time = timestamp
        matched = is_rebalance_timestamp(
            timestamp,
            self.schedule,
            session_index=self._session_index,
            calendar=self.calendar,
            timezone=self.timezone,
            session_start_time=self.session_start_time,
            data_frequency=self.data_frequency,
            timestamp_semantics=self.timestamp_semantics,
            is_session_close=is_session_close,
        )
        if matched:
            self._required_close_matched = True
        return matched

    @property
    def has_observations(self) -> bool:
        """Return whether at least one event has been evaluated."""
        return self._session_date is not None

    def validate_completed_run(self) -> None:
        """Validate the final observed session after every event has been evaluated."""
        if (
            self._session_date is not None
            and _session_requires_close(
                self.schedule,
                self._session_date,
                self._session_index,
                self.calendar,
            )
            and not self._required_close_matched
            and _session_reached_expected_close(
                self.calendar,
                self._session_date,
                self._last_event_time,
                self.timezone,
            )
        ):
            _raise_close_alignment_error(
                self.schedule.cadence,
                self.calendar,
                self._session_date,
            )


def _normalize_schedule_metadata(
    data_frequency: Any | None,
    timestamp_semantics: TimestampSemantics | str | None,
) -> tuple[DataFrequency | None, TimestampSemantics | None]:
    frequency = _to_backtest_frequency(data_frequency) if data_frequency is not None else None
    semantics = (
        timestamp_semantics
        if isinstance(timestamp_semantics, TimestampSemantics)
        else TimestampSemantics(timestamp_semantics)
        if timestamp_semantics is not None
        else None
    )
    return frequency, semantics


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
        return pl.Series("timestamp", [ts for ts in ts_list if ts in schedule._timestamp_set])

    metadata = _resolve_schedule_metadata(
        ts_list,
        feed_spec=feed_spec,
        calendar=calendar,
        timezone=timezone,
        session_start_time=session_start_time,
        data_frequency=data_frequency,
        timestamp_semantics=timestamp_semantics,
    )
    _validate_schedule_calendar(schedule, metadata["calendar"])
    session_indices: dict[date, int] = {}
    matched_sessions: set[date] = set()
    last_event_by_session: dict[date, datetime] = {}
    resolved: list[datetime] = []
    for timestamp in ts_list:
        session_date = session_date_for_timestamp(
            timestamp,
            calendar=metadata["calendar"],
            timezone=metadata["timezone"],
            session_start_time=metadata["session_start_time"],
            data_frequency=metadata["data_frequency"],
            timestamp_semantics=metadata["timestamp_semantics"],
        )
        if (
            metadata["calendar"] is not None
            and _calendar_close_for_session(metadata["calendar"], session_date) is None
        ):
            continue
        session_index = session_indices.setdefault(session_date, len(session_indices) + 1)
        last_event_by_session[session_date] = timestamp
        if _evaluate_rebalance_timestamp(
            timestamp,
            schedule,
            session_index=session_index,
            calendar=metadata["calendar"],
            timezone=metadata["timezone"],
            session_start_time=metadata["session_start_time"],
            data_frequency=metadata["data_frequency"],
            timestamp_semantics=metadata["timestamp_semantics"],
            is_session_close=None,
            session_date=session_date,
        ):
            resolved.append(timestamp)
            matched_sessions.add(session_date)
    is_intraday = (
        metadata["timestamp_semantics"] is not TimestampSemantics.SESSION_LABEL
        and _to_backtest_frequency(metadata["data_frequency"]) is not DataFrequency.DAILY
    )
    if is_intraday:
        missing_required_sessions = [
            session_date
            for session_date, session_index in session_indices.items()
            if _session_requires_close(
                schedule,
                session_date,
                session_index,
                metadata["calendar"],
            )
            and session_date not in matched_sessions
            and _session_reached_expected_close(
                metadata["calendar"],
                session_date,
                last_event_by_session[session_date],
                metadata["timezone"],
            )
        ]
        if missing_required_sessions:
            _raise_close_alignment_error(
                cadence,
                metadata["calendar"],
                missing_required_sessions[0],
            )
    return pl.Series("timestamp", resolved)


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
