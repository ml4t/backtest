"""Date/time factories compiled onto native calendar observations and scheduler."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite
from numbers import Real
from typing import Any

import pandas as pd
from ml4t.specs import LifecyclePhase

from ...calendar import get_calendar, get_schedule
from ...execution.schedule import RebalanceSchedule
from ...sessions import SessionConfig, _is_right_labeled_bar
from ._scope import callback_scope


@dataclass(frozen=True)
class _DateRule:
    kind: str
    days_offset: int = 0


def _date_rule(kind: str, days_offset: int, maximum: int) -> _DateRule:
    if (
        not isinstance(days_offset, Real)
        or not isfinite(days_offset)
        or int(days_offset) != days_offset
        or not 0 <= days_offset < maximum
    ):
        raise ValueError(f"days_offset must be an integer from 0 to {maximum - 1}")
    return _DateRule(kind, int(days_offset))


class date_rules:
    @staticmethod
    def every_day() -> _DateRule:
        return _DateRule("every_day")

    @staticmethod
    def week_start(days_offset: int = 0) -> _DateRule:
        return _date_rule("week_start", days_offset, 5)

    @staticmethod
    def week_end(days_offset: int = 0) -> _DateRule:
        return _date_rule("week_end", days_offset, 5)

    @staticmethod
    def month_start(days_offset: int = 0) -> _DateRule:
        return _date_rule("month_start", days_offset, 23)

    @staticmethod
    def month_end(days_offset: int = 0) -> _DateRule:
        return _date_rule("month_end", days_offset, 23)


@dataclass(frozen=True)
class _TimeRule:
    kind: str
    offset: timedelta = timedelta(0)


def _time_rule(kind, offset, hours, minutes):
    if offset is not None and (hours is not None or minutes is not None):
        raise ValueError("Use offset or hours/minutes, not both")
    if offset is None:
        offset = (
            timedelta(minutes=1)
            if hours is None and minutes is None
            else timedelta(hours=hours or 0, minutes=minutes or 0)
        )
    if not isinstance(offset, timedelta) or not timedelta(minutes=1) <= offset <= timedelta(
        hours=12
    ):
        raise ValueError("offset must be between one minute and twelve hours")
    return _TimeRule(kind, offset)


class time_rules:
    @staticmethod
    def every_minute() -> _TimeRule:
        return _TimeRule("every_minute")

    @staticmethod
    def market_open(
        offset: timedelta | None = None, hours: int | None = None, minutes: int | None = None
    ) -> _TimeRule:
        return _time_rule("market_open", offset, hours, minutes)

    @staticmethod
    def market_close(
        offset: timedelta | None = None, hours: int | None = None, minutes: int | None = None
    ) -> _TimeRule:
        return _time_rule("market_close", offset, hours, minutes)


class calendars:
    US_EQUITIES = "NYSE"
    US_FUTURES = "us_futures"


@dataclass(frozen=True)
class _Schedule:
    callback: Callable[..., Any]
    date_rule: _DateRule
    time_rule: _TimeRule
    half_days: bool
    calendar: str | None


def _selected_dates(rows, rule):
    if rule.kind == "every_day":
        return {row["session_date"] for row in rows}
    groups = defaultdict(list)
    for row in rows:
        date = row["session_date"]
        key = date.isocalendar()[:2] if rule.kind.startswith("week") else (date.year, date.month)
        groups[key].append(date)
    index = -rule.days_offset - 1 if rule.kind.endswith("end") else rule.days_offset
    return {dates[index] for dates in groups.values() if len(dates) > rule.days_offset}


def _register(context, runtime, data, before_trading_start):
    """Called during native on_start so native unbind owns registration cleanup."""
    engine = runtime._engine
    assert engine is not None
    config = engine.config
    if not context._schedules and before_trading_start is None:
        return
    calendar = config.resolved_calendar
    if calendar is None:
        raise ValueError(
            "Zipline session callbacks require a declared calendar; calendar=None disables them"
        )
    points = [t for t in engine.feed.timestamps if runtime._accepts(t)]
    if not points:
        return
    first, last = points[0], points[-1]
    start, end = first.date() - timedelta(days=45), last.date() + timedelta(days=45)
    rows = get_schedule(calendar, start, end, include_breaks=True).to_dicts()
    source_calendar = get_calendar(calendar)
    legacy_futures = source_calendar.name == "us_futures"
    schedule = source_calendar.schedule(start_date=start, end_date=end)
    early = {t.date() for t in source_calendar.early_closes(schedule).index}
    clock = SessionConfig(
        calendar,
        config.resolved_timezone,
        config.resolved_session_start_time,
        config.resolved_timestamp_semantics,
        config.resolved_feed_spec.bar_label,
        config.resolved_data_frequency,
    )
    by_session = defaultdict(list)
    for time in points:
        date = clock.date_for_timestamp(time).date()
        by_session[date].append(time)
    frequency = getattr(config.resolved_data_frequency, "value", config.resolved_data_frequency)
    daily = frequency in {"daily", "1d"}
    if not daily and frequency != "1m":
        raise ValueError("Zipline scheduling requires declared daily or one-minute observations")
    if not daily and not _is_right_labeled_bar(
        config.resolved_timestamp_semantics, config.resolved_feed_spec.bar_label
    ):
        raise ValueError(
            "Zipline minute scheduling requires explicitly right-labelled minute observations"
        )
    if daily and any(len(times) != 1 for times in by_session.values()):
        raise ValueError("Prepare one daily observation timestamp per session before scheduling")

    regular_by_session = {}
    if not daily:
        for row in rows:
            opened, closed = pd.Timestamp(row["market_open"]), pd.Timestamp(row["market_close"])
            regular = []
            for time in by_session.get(row["session_date"], []):
                stamp = pd.Timestamp(time)
                instant = (
                    stamp.tz_localize(config.resolved_timezone).tz_convert("UTC")
                    if stamp.tz is None
                    else stamp.tz_convert("UTC")
                )
                if not opened < instant <= closed:
                    continue
                if row.get("break_start") is not None and pd.Timestamp(
                    row["break_start"]
                ) < instant <= pd.Timestamp(row["break_end"]):
                    continue
                regular.append((time, instant))
            regular_by_session[row["session_date"]] = regular

    def wrapper(callback, *, pre=False):
        def invoke(_timestamp, _bars, _native_context, _broker):
            previous = context._in_before_trading_start
            context._in_before_trading_start = pre
            try:
                with callback_scope(context):
                    callback(context, data)
            finally:
                context._in_before_trading_start = previous

        return invoke

    config.metadata["migration_schedule"] = {
        "calendar": calendar,
        "daily_ignores_time_rule": daily,
        "before_trading_start_minutes_before_session_open": 45
        if before_trading_start is not None and not legacy_futures
        else None,
        "before_trading_start_clock": "08:45 America/New_York"
        if before_trading_start is not None and legacy_futures
        else None,
        "rules": [],
    }
    if before_trading_start is not None:
        if legacy_futures:
            clocks = [
                pd.Timestamp(date).replace(hour=8, minute=45).tz_localize("America/New_York")
                for date in by_session
            ]
            runtime.scheduler.register(
                wrapper(before_trading_start, pre=True),
                phase=LifecyclePhase.INTRABAR,
                at="clock",
                schedule=RebalanceSchedule.explicit_timestamps(clocks),
            )
        else:
            runtime.scheduler.register(
                wrapper(before_trading_start, pre=True),
                phase=LifecyclePhase.PRE_OPEN,
                at="session_open",
                minutes_before=45,
                include_start_session=True,
            )
    for plan in context._schedules:
        if plan.calendar is not None and get_calendar(plan.calendar).name != source_calendar.name:
            raise ValueError("Schedule calendar must match the declared execution calendar")
        selected = _selected_dates(rows, plan.date_rule)
        if not plan.half_days:
            selected -= early
        targets = []
        for row in rows:
            date = row["session_date"]
            observed = by_session.get(date, [])
            if date not in selected or not observed:
                continue
            if daily:
                targets.append(observed[0])
                continue
            opened, closed = pd.Timestamp(row["market_open"]), pd.Timestamp(row["market_close"])
            if legacy_futures:
                opened += timedelta(hours=12, minutes=30)
                closed -= timedelta(hours=1)

            eligible = regular_by_session.get(date, [])
            if plan.time_rule.kind == "every_minute":
                if eligible:
                    targets.append(eligible[0][0])
                continue
            target = (
                opened + plan.time_rule.offset
                if plan.time_rule.kind == "market_open"
                else closed - plan.time_rule.offset
            )
            targets.extend(time for time, instant in eligible if instant == target)
        config.metadata["migration_schedule"]["rules"].append(
            {
                "date_rule": plan.date_rule.kind,
                "days_offset": plan.date_rule.days_offset,
                "time_rule": plan.time_rule.kind,
                "offset_ns": pd.Timedelta(plan.time_rule.offset).value,
                "half_days": plan.half_days,
                "observed_callback_count": len(targets),
            }
        )
        if targets:
            runtime.scheduler.register(
                wrapper(plan.callback),
                phase=LifecyclePhase.MARKET_EVENT,
                schedule=RebalanceSchedule.explicit_timestamps(targets),
            )
