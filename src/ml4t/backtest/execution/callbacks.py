"""Run-scoped callbacks on declared calendar clocks and observed market events."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from ml4t.specs import LifecyclePhase, TimestampSemantics

from ..calendar import get_schedule
from ..sessions import SessionConfig, _is_right_labeled_bar, session_date_for_timestamp
from .schedule import (
    RebalanceCadence,
    RebalanceSchedule,
    _evaluate_rebalance_timestamp,
    is_rebalance_timestamp,
)

if TYPE_CHECKING:
    from ..runtime import RuntimeServices


@dataclass(frozen=True)
class _Registration:
    callback: Callable[..., Any]
    phase: LifecyclePhase
    at: str
    offset: timedelta
    schedule: RebalanceSchedule
    include_start_session: bool


class SessionScheduler:
    """Schedule callbacks through the Engine's existing transactional dispatcher.

    Pre-open clocks are explicit calendar events with no invented market observation.
    Explicit intrabar clock callbacks run before an observation at the same instant,
    with no new quote, fill or account sample. Other callbacks require an actual
    price event at the requested instant. Offsets
    are elapsed minutes from the calendar endpoint, not ordinal price observations.
    """

    def __init__(self, runtime: RuntimeServices) -> None:
        self._runtime = runtime
        self._registrations: list[_Registration] = []
        self._initial_count = 0
        self._prepared = False
        self._defer_pre_at_close = False
        self._pre_events: list[tuple[datetime, int]] = []
        self._pre_index = 0
        self._events: dict[datetime, list[int]] = {}
        self._session_indexes: dict[date, int] = {}
        self._event_sessions: dict[tuple[datetime, int], date] = {}
        self._event_opens: dict[tuple[datetime, int], datetime] = {}

    def register(
        self,
        callback: Callable[..., Any],
        *,
        phase: LifecyclePhase = LifecyclePhase.MARKET_EVENT,
        at: str = "bar",
        minutes_before: int = 0,
        minutes_after: int = 0,
        schedule: RebalanceSchedule | None = None,
        include_start_session: bool = False,
    ) -> None:
        """Register before execution or during on_start/on_prepare, in stable order."""
        if self._prepared:
            raise RuntimeError("Register callbacks before market execution starts")
        if not callable(callback):
            raise TypeError("callback must be callable")
        phase = LifecyclePhase(phase)
        if phase not in {
            LifecyclePhase.PRE_OPEN,
            LifecyclePhase.OPENING_AUCTION,
            LifecyclePhase.MARKET_EVENT,
            LifecyclePhase.INTRABAR,
            LifecyclePhase.CLOSE,
        }:
            raise ValueError("Unsupported scheduled callback phase")
        if at not in {"bar", "clock", "session_open", "session_close"}:
            raise ValueError("at must be bar, clock, session_open or session_close")
        for value in [minutes_before, minutes_after]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Minute offsets must be nonnegative integers")
        if minutes_before and minutes_after:
            raise ValueError("Choose minutes_before or minutes_after")
        offset = timedelta(minutes=minutes_after - minutes_before)
        if schedule is not None and not isinstance(schedule, RebalanceSchedule):
            raise TypeError("schedule must be a RebalanceSchedule")
        if at == "clock" and (
            phase is not LifecyclePhase.INTRABAR
            or offset
            or schedule is None
            or schedule.cadence is not RebalanceCadence.EXPLICIT_TIMESTAMPS
        ):
            raise ValueError(
                "clock callbacks require intrabar phase and explicit physical timestamps"
            )
        if at == "bar" and offset:
            raise ValueError("Bar callbacks do not accept elapsed offsets")
        if phase is LifecyclePhase.PRE_OPEN and (at != "session_open" or offset > timedelta(0)):
            raise ValueError("pre_open callbacks require session_open with a nonpositive offset")
        if not isinstance(include_start_session, bool) or (
            include_start_session and phase is not LifecyclePhase.PRE_OPEN
        ):
            raise ValueError("include_start_session is a Boolean option for pre_open callbacks")
        if phase is LifecyclePhase.OPENING_AUCTION and (at != "session_open" or offset):
            raise ValueError("opening_auction callbacks require the actual session_open")
        self._registrations.append(
            _Registration(
                callback,
                phase,
                at,
                offset,
                schedule or RebalanceSchedule.every_bar(),
                include_start_session,
            )
        )
        if self._runtime._engine is not None:
            self._validate()

    def _validate(self) -> None:
        engine = self._runtime._engine
        assert engine is not None
        if engine.config.resolved_calendar is None and any(
            r.at in {"session_open", "session_close"} for r in self._registrations
        ):
            raise ValueError("Session callbacks require a declared calendar")
        if (
            self._registrations
            and engine.config.resolved_timestamp_semantics is TimestampSemantics.SESSION_LABEL
        ):
            raise ValueError("Scheduled callbacks require event instants, not session labels")

        if any(r.phase is LifecyclePhase.OPENING_AUCTION for r in self._registrations) and (
            engine.config.resolved_timestamp_semantics is not TimestampSemantics.EVENT_TIME
            or _is_right_labeled_bar(
                engine.config.resolved_timestamp_semantics,
                engine.config.resolved_feed_spec.bar_label,
            )
        ):
            raise ValueError("opening_auction callbacks require event_time opening observations")

    def _bind(self) -> None:
        self._initial_count = len(self._registrations)
        self._validate()

    def _prepare(self) -> None:
        self._validate()
        engine = self._runtime._engine
        assert engine is not None
        self._defer_pre_at_close = _is_right_labeled_bar(
            engine.config.resolved_timestamp_semantics, engine.config.resolved_feed_spec.bar_label
        )
        self._prepared = True
        registrations = self._registrations
        if not any(
            r.at != "bar"
            or r.schedule.cadence
            not in {RebalanceCadence.EVERY_BAR, RebalanceCadence.EXPLICIT_TIMESTAMPS}
            for r in registrations
        ):
            return
        timestamps = [t for t in engine.feed.timestamps if self._runtime._accepts(t)]
        first, last = timestamps[0], timestamps[-1]
        timezone = ZoneInfo(engine.config.resolved_timezone)

        def feed_time(timestamp):
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone)
            return (
                timestamp.astimezone(first.tzinfo)
                if first.tzinfo is not None
                else timestamp.astimezone(timezone).replace(tzinfo=None)
            )

        for index, registration in enumerate(registrations):
            if registration.at != "clock":
                continue
            for scheduled in registration.schedule.timestamps:
                instant = feed_time(scheduled)
                if not self._runtime._accepts(instant):
                    continue
                if instant > last:
                    raise ValueError("Clock callback is after the last captured market event")
                self._pre_events.append((instant, index))
        if not any(
            r.at in {"session_open", "session_close"}
            or r.at == "bar"
            and r.schedule.cadence
            not in {RebalanceCadence.EVERY_BAR, RebalanceCadence.EXPLICIT_TIMESTAMPS}
            for r in registrations
        ):
            self._pre_events.sort()
            if self._runtime._assets is not None:
                self._runtime._assets._prepare_points([time for time, _ in self._pre_events])
            return
        calendar = engine.config.resolved_calendar
        if calendar is None:
            raise ValueError("Session cadence requires a declared calendar")
        clock = SessionConfig(
            calendar,
            engine.config.resolved_timezone,
            engine.config.resolved_session_start_time,
            engine.config.resolved_timestamp_semantics,
            engine.config.resolved_feed_spec.bar_label,
            engine.config.resolved_data_frequency,
        )
        first_session = clock.date_for_timestamp(first).date()

        sessions = get_schedule(
            calendar, first.date() - timedelta(days=1), last.date() + timedelta(days=1)
        )
        session_index = 0
        for row in sessions.iter_rows(named=True):
            opened, closed = feed_time(row["market_open"]), feed_time(row["market_close"])
            if closed < first or opened > last:
                continue
            session_index += 1
            self._session_indexes[row["session_date"]] = session_index
            for index, registration in enumerate(registrations):
                if registration.at in {"bar", "clock"}:
                    continue
                anchor = opened if registration.at == "session_open" else closed
                timestamp = anchor + registration.offset
                start_session_pre = (
                    registration.include_start_session
                    and row["session_date"] == first_session
                    and self._runtime.start is not None
                    and timestamp < self._runtime.start
                )
                if not self._runtime._accepts(timestamp) and not start_session_pre:
                    continue
                if not is_rebalance_timestamp(
                    closed,
                    registration.schedule,
                    session_index=session_index,
                    calendar=calendar,
                    timezone=engine.config.resolved_timezone,
                    timestamp_semantics=TimestampSemantics.BAR_CLOSE,
                    is_session_close=True,
                ):
                    continue
                self._event_sessions[(timestamp, index)] = row["session_date"]
                self._event_opens[(timestamp, index)] = opened
                if registration.phase is LifecyclePhase.PRE_OPEN:
                    self._pre_events.append((timestamp, index))
                else:
                    self._events.setdefault(timestamp, []).append(index)
        self._pre_events.sort(
            key=lambda event: (event[0], self._registrations[event[1]].at != "clock", event[1])
        )
        if self._runtime._assets is not None:
            self._runtime._assets._prepare_points(
                [time for time, _ in self._pre_events]
                + list(self._events)
                + list(self._event_opens.values())
            )

    def _invoke(self, index: int, timestamp: datetime, data: Any, context: dict) -> None:
        engine = self._runtime._engine
        assert engine is not None
        registration = self._registrations[index]
        engine._advance_account_events(timestamp)
        with engine.broker._scheduled_callback_clock(
            timestamp,
            session_date=self._event_sessions.get((timestamp, index)),
            session_open=self._event_opens.get((timestamp, index)),
            has_price_event=registration.at != "clock",
        ):
            engine.lifecycle_dispatcher.dispatch_registered(
                registration.phase,
                registration.callback,
                engine.broker,
                timestamp,
                data,
                context,
                engine.broker,
                event_time=timestamp,
            )

    def _before_event(self, timestamp: datetime, *, has_price_event: bool) -> None:
        market = self._runtime.market
        while self._pre_index < len(self._pre_events):
            instant, index = self._pre_events[self._pre_index]
            if instant > timestamp or (
                instant == timestamp
                and has_price_event
                and self._defer_pre_at_close
                and self._registrations[index].at != "clock"
            ):
                break
            self._pre_index += 1
            market._asof, market._inclusive = instant, False
            self._invoke(index, instant, {}, {})

    def _after_event(self, timestamp: datetime) -> None:
        while self._pre_index < len(self._pre_events):
            instant, index = self._pre_events[self._pre_index]
            if instant != timestamp:
                break
            self._pre_index += 1
            market = self._runtime.market
            market._asof, market._inclusive = instant, True
            self._invoke(index, instant, {}, {})

    def _on_event(self, timestamp: datetime, data: Any, context: dict, *, opening: bool) -> None:
        if not data:
            return
        engine = self._runtime._engine
        assert engine is not None
        indexes = list(self._events.get(timestamp, ()))
        if not opening:
            indexes += [i for i, r in enumerate(self._registrations) if r.at == "bar"]
        for index in sorted(indexes):
            registration = self._registrations[index]
            if (registration.phase is LifecyclePhase.OPENING_AUCTION) != opening:
                continue
            if (
                registration.at == "bar"
                and registration.schedule.cadence is not RebalanceCadence.EVERY_BAR
            ):
                session_date = session_date_for_timestamp(
                    timestamp,
                    calendar=engine.config.resolved_calendar,
                    timezone=engine.config.resolved_timezone,
                    session_start_time=engine.config.resolved_session_start_time,
                    data_frequency=engine.config.resolved_data_frequency,
                    timestamp_semantics=engine.config.resolved_timestamp_semantics,
                    bar_label=engine.config.resolved_feed_spec.bar_label,
                )
                if not _evaluate_rebalance_timestamp(
                    timestamp,
                    registration.schedule,
                    session_index=self._session_indexes.get(session_date, 0),
                    calendar=engine.config.resolved_calendar,
                    timezone=engine.config.resolved_timezone,
                    session_start_time=engine.config.resolved_session_start_time,
                    data_frequency=engine.config.resolved_data_frequency,
                    timestamp_semantics=engine.config.resolved_timestamp_semantics,
                    is_session_close=None,
                    session_date=session_date,
                ):
                    continue
            self._invoke(index, timestamp, data, context)

    def _unbind(self) -> None:
        del self._registrations[self._initial_count :]
        self._prepared = False
        self._defer_pre_at_close = False
        self._pre_events.clear()
        self._pre_index = 0
        self._events.clear()
        self._session_indexes.clear()
        self._event_sessions.clear()
        self._event_opens.clear()
