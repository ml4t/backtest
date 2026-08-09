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

from ..calendar import get_calendar_sessions
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
    _instant_sets_by_timezone: dict[str | None, frozenset[datetime]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        if self.cadence == RebalanceCadence.FIXED_N_SESSIONS and self.every_n < 1:
            raise ValueError("RebalanceSchedule.every_n must be >= 1")
        timestamps = tuple(
            sorted(
                {_coerce_timestamp(ts) for ts in self.timestamps},
                key=lambda timestamp: (
                    _event_time_utc(timestamp, None),
                    timestamp.tzinfo is not None,
                ),
            )
        )
        if self.cadence == RebalanceCadence.EXPLICIT_TIMESTAMPS and not timestamps:
            raise ValueError("Explicit timestamp schedules require at least one timestamp")
        object.__setattr__(self, "timestamps", timestamps)

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
        event_time = _event_time_utc(timestamp, timezone)
        return event_time in _explicit_schedule_instants(resolved, timezone)
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

    localized = _localize_event_time(timestamp, timezone)
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
    return frozenset(
        session.market_close
        for year in range(start.year, end.year + 1)
        for session_date, session in get_calendar_sessions(calendar, year).items()
        if start <= session_date <= end
    )


@lru_cache(maxsize=512)
def _calendar_period_end(calendar: str, start: date, end: date) -> date | None:
    sessions = (
        session_date
        for year in range(start.year, end.year + 1)
        for session_date in get_calendar_sessions(calendar, year)
        if start <= session_date <= end
    )
    return max(sessions, default=None)


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


def _completed_period_ends_between(
    cadence: RebalanceCadence,
    previous_session: date,
    current_session: date,
    calendar: str,
) -> tuple[date, ...]:
    """Return exchange period ends completed before the current observed session."""
    if cadence not in {RebalanceCadence.WEEKLY, RebalanceCadence.MONTH_END}:
        return ()
    period_start, period_end = _period_bounds(cadence, previous_session)
    current_period_start, _ = _period_bounds(cadence, current_session)
    completed: list[date] = []
    while period_start < current_period_start:
        expected = _calendar_period_end(calendar, period_start, period_end)
        if expected is not None:
            completed.append(expected)
        period_start, period_end = _period_bounds(cadence, period_end + timedelta(days=1))
    return tuple(completed)


@lru_cache(maxsize=512)
def _calendar_close_for_session(calendar: str, session_date: date) -> datetime | None:
    session = get_calendar_sessions(calendar, session_date.year).get(session_date)
    return None if session is None else session.market_close


def _localize_event_time(timestamp: datetime, timezone: str | None) -> datetime:
    source_timezone = ZoneInfo(timezone or "UTC")
    return (
        timestamp.replace(tzinfo=source_timezone)
        if timestamp.tzinfo is None
        else timestamp.astimezone(source_timezone)
    )


def _event_time_utc(timestamp: datetime, timezone: str | None) -> datetime:
    return _localize_event_time(timestamp, timezone).astimezone(ZoneInfo("UTC"))


def _explicit_schedule_instants(
    schedule: RebalanceSchedule,
    timezone: str | None,
) -> frozenset[datetime]:
    cached = schedule._instant_sets_by_timezone.get(timezone)
    if cached is not None:
        return cached
    normalized = frozenset(
        _event_time_utc(timestamp, timezone) for timestamp in schedule.timestamps
    )
    return schedule._instant_sets_by_timezone.setdefault(timezone, normalized)


def _raise_explicit_alignment_error(
    scheduled: datetime,
    nearest_observed: datetime,
    timezone: str | None,
) -> None:
    scheduled_instant = _event_time_utc(scheduled, timezone).isoformat()
    nearest_instant = _event_time_utc(nearest_observed, timezone).isoformat()
    raise ValueError(
        f"explicit_timestamps schedule instant {scheduled_instant} did not match an observed feed "
        f"event within the observed instant window; nearest observed instant is {nearest_instant}. "
        f"Naive timestamps use timezone {timezone or 'UTC'!r}"
    )


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


def _raise_missing_period_end_error(
    cadence: RebalanceCadence,
    calendar: str,
    period_end: date,
    next_observed_session: date,
) -> None:
    raise ValueError(
        f"{cadence.value} feed is missing calendar period-end session {period_end} on calendar "
        f"{calendar!r} before observed session {next_observed_session}; verify feed completeness"
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
        self._observed_event_count = 0
        self._observed_exchange_session = False
        self._matched_period_ends: set[date] = set()
        self._explicit_schedule_by_instant = {
            _event_time_utc(scheduled, timezone): scheduled for scheduled in schedule.timestamps
        }
        self._explicit_sorted_instants = tuple(sorted(self._explicit_schedule_by_instant))
        self._matched_explicit_instants: set[datetime] = set()
        self._first_explicit_instant: datetime | None = None
        self._last_explicit_instant: datetime | None = None
        self._last_explicit_event: datetime | None = None
        self._explicit_cursor = 0
        self._nearest_explicit_events: dict[datetime, datetime] = {}

    def evaluate(self, timestamp: datetime, *, is_session_close: bool | None = None) -> bool:
        self._observed_event_count += 1
        if self.schedule.cadence is RebalanceCadence.EVERY_BAR:
            return True
        if self.schedule.cadence is RebalanceCadence.EXPLICIT_TIMESTAMPS:
            event_time = _event_time_utc(timestamp, self.timezone)
            if self._last_explicit_instant is not None and event_time < self._last_explicit_instant:
                raise ValueError(
                    f"event timestamp moved backward from "
                    f"{self._last_explicit_instant.isoformat()} to {event_time.isoformat()}; "
                    "start a new evaluator or call TargetWeightExecutor.reset() before another run"
                )
            if self._first_explicit_instant is None:
                self._first_explicit_instant = event_time
            while (
                self._explicit_cursor < len(self._explicit_sorted_instants)
                and self._explicit_sorted_instants[self._explicit_cursor] <= event_time
            ):
                scheduled_time = self._explicit_sorted_instants[self._explicit_cursor]
                if scheduled_time >= self._first_explicit_instant:
                    nearest_event = timestamp
                    if self._last_explicit_instant is not None:
                        previous_distance = abs(
                            (scheduled_time - self._last_explicit_instant).total_seconds()
                        )
                        current_distance = abs((event_time - scheduled_time).total_seconds())
                        if previous_distance <= current_distance:
                            nearest_event = self._last_explicit_event
                    if nearest_event is not None:
                        self._nearest_explicit_events[scheduled_time] = nearest_event
                self._explicit_cursor += 1
            self._last_explicit_instant = event_time
            self._last_explicit_event = timestamp
            if event_time not in self._explicit_schedule_by_instant:
                return False
            self._matched_explicit_instants.add(event_time)
            return True
        session_date = session_date_for_timestamp(
            timestamp,
            calendar=self.calendar,
            timezone=self.timezone,
            session_start_time=self.session_start_time,
            data_frequency=self.data_frequency,
            timestamp_semantics=self.timestamp_semantics,
        )
        if (
            is_session_close is not True
            and self.calendar is not None
            and _calendar_close_for_session(self.calendar, session_date) is None
        ):
            return False
        self._observed_exchange_session = True
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
            ):
                _raise_close_alignment_error(
                    self.schedule.cadence,
                    self.calendar,
                    self._session_date,
                )
            if self._session_date is not None and self.calendar is not None:
                for expected_period_end in _completed_period_ends_between(
                    self.schedule.cadence,
                    self._session_date,
                    session_date,
                    self.calendar,
                ):
                    if expected_period_end not in self._matched_period_ends:
                        _raise_missing_period_end_error(
                            self.schedule.cadence,
                            self.calendar,
                            expected_period_end,
                            session_date,
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
            if self.schedule.cadence in {
                RebalanceCadence.WEEKLY,
                RebalanceCadence.MONTH_END,
            }:
                self._matched_period_ends.add(session_date)
        return matched

    @property
    def has_observations(self) -> bool:
        """Return whether at least one event has been evaluated."""
        return self._observed_event_count > 0

    def validate_completed_run(self) -> None:
        """Validate the final observed session after every event has been evaluated."""
        if self.schedule.cadence is RebalanceCadence.EVERY_BAR:
            return
        if self.schedule.cadence is RebalanceCadence.EXPLICIT_TIMESTAMPS:
            if self._first_explicit_instant is None or self._last_explicit_instant is None:
                return
            unmatched = [
                instant
                for instant in self._explicit_sorted_instants
                if instant not in self._matched_explicit_instants
                and self._first_explicit_instant <= instant <= self._last_explicit_instant
            ]
            if unmatched:
                scheduled_instant = unmatched[0]
                scheduled = self._explicit_schedule_by_instant[scheduled_instant]
                nearest = self._nearest_explicit_events[scheduled_instant]
                _raise_explicit_alignment_error(scheduled, nearest, self.timezone)
            return
        if (
            self._observed_event_count > 0
            and self.calendar is not None
            and not self._observed_exchange_session
        ):
            raise ValueError(
                f"{self.schedule.cadence.value} observed no exchange sessions for calendar "
                f"{self.calendar!r}; verify the calendar and feed timestamps"
            )
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
        metadata = _resolve_schedule_metadata(
            ts_list,
            feed_spec=feed_spec,
            calendar=calendar,
            timezone=timezone,
            session_start_time=session_start_time,
            data_frequency=data_frequency,
            timestamp_semantics=timestamp_semantics,
        )
        resolved_timezone = metadata["timezone"]
        observed_by_instant = {
            _event_time_utc(timestamp, resolved_timezone): timestamp for timestamp in ts_list
        }
        scheduled_instants = {
            scheduled: _event_time_utc(scheduled, resolved_timezone)
            for scheduled in schedule.timestamps
        }
        first_instant = _event_time_utc(ts_list[0], resolved_timezone)
        last_instant = _event_time_utc(ts_list[-1], resolved_timezone)
        unmatched = sorted(
            (
                scheduled
                for scheduled, instant in scheduled_instants.items()
                if instant not in observed_by_instant and first_instant <= instant <= last_instant
            ),
            key=lambda scheduled: scheduled_instants[scheduled],
        )
        if unmatched:
            scheduled = unmatched[0]
            scheduled_time = scheduled_instants[scheduled]
            nearest = min(
                ts_list,
                key=lambda timestamp: abs(
                    (_event_time_utc(timestamp, resolved_timezone) - scheduled_time).total_seconds()
                ),
            )
            _raise_explicit_alignment_error(scheduled, nearest, resolved_timezone)
        scheduled_instant_set = frozenset(scheduled_instants.values())
        return pl.Series(
            "timestamp",
            [
                timestamp
                for timestamp in ts_list
                if _event_time_utc(timestamp, resolved_timezone) in scheduled_instant_set
            ],
        )

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
    if not session_indices:
        raise ValueError(
            f"{cadence.value} observed no exchange sessions for calendar "
            f"{metadata['calendar']!r}; verify the calendar and feed timestamps"
        )
    if metadata["calendar"] is not None:
        observed_sessions = tuple(session_indices)
        for previous_session, current_session in zip(
            observed_sessions,
            observed_sessions[1:],
            strict=False,
        ):
            for expected_period_end in _completed_period_ends_between(
                cadence,
                previous_session,
                current_session,
                metadata["calendar"],
            ):
                if expected_period_end not in session_indices:
                    _raise_missing_period_end_error(
                        cadence,
                        metadata["calendar"],
                        expected_period_end,
                        current_session,
                    )
    is_intraday = (
        metadata["timestamp_semantics"] is not TimestampSemantics.SESSION_LABEL
        and _to_backtest_frequency(metadata["data_frequency"]) is not DataFrequency.DAILY
    )
    if is_intraday:
        final_session = next(reversed(session_indices))
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
            and (
                session_date != final_session
                or _session_reached_expected_close(
                    metadata["calendar"],
                    session_date,
                    last_event_by_session[session_date],
                    metadata["timezone"],
                )
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
