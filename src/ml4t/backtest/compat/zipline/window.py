"""Translate legacy session dates into a borrowed native physical trading window."""

from contextlib import contextmanager
from datetime import timedelta

import pandas as pd

from ...calendar import get_schedule
from ...sessions import SessionConfig


def _date(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Run dates require finite session labels")
    if stamp.tz is not None or stamp != stamp.normalize():
        raise ValueError(
            "Run dates require timezone-naive midnight session labels; use RuntimeServices for physical instants"
        )
    return stamp.date()


def _utc(value, timezone):
    stamp = pd.Timestamp(value)
    return (
        stamp.tz_localize(timezone).tz_convert("UTC")
        if stamp.tz is None
        else stamp.tz_convert("UTC")
    )


def _physical(value, naive, timezone):
    stamp = pd.Timestamp(value)
    return stamp.tz_convert(timezone).tz_localize(None) if naive else stamp


def _window(feed, config, runtime, start, end):
    if runtime._engine is not None:
        raise RuntimeError("RuntimeServices already belongs to an active run")
    points = [point for point in feed.timestamps if runtime._accepts(point)]
    if not points:
        raise ValueError("Run window contains no feed events")
    timezone = config.resolved_timezone
    first, last = points[0], points[-1]
    bounded_start, bounded_end = runtime.start, runtime.end
    mode = "physical_observations"
    if start is not None or end is not None:
        calendar = config.resolved_calendar
        if calendar is None:
            raise ValueError(
                "Legacy session start/end require a declared calendar; use physical RuntimeServices bounds for calendar=None"
            )
        clock = SessionConfig(
            calendar,
            timezone,
            config.resolved_session_start_time,
            config.resolved_timestamp_semantics,
            config.resolved_feed_spec.bar_label,
            config.resolved_data_frequency,
        )
        first_date = _date(start) if start is not None else clock.date_for_timestamp(first).date()
        last_date = _date(end) if end is not None else clock.date_for_timestamp(last).date()
        if first_date > last_date:
            raise ValueError("start must not exceed end")
        schedule = get_schedule(calendar, first_date, last_date)
        if schedule.is_empty():
            raise ValueError("Session run window contains no calendar sessions")
        opened = schedule["market_open"][0] + timedelta(minutes=1)
        closed = schedule["market_close"][-1]
        proposed_start = _physical(opened, first.tzinfo is None, timezone)
        proposed_end = _physical(closed, first.tzinfo is None, timezone)
        for label, supplied, existing, proposed in [
            ("start", start, runtime.start, proposed_start),
            ("end", end, runtime.end, proposed_end),
        ]:
            if supplied is not None and existing is not None and existing != proposed:
                raise ValueError(f"Session {label} conflicts with the native RuntimeServices bound")
        bounded_start = (
            proposed_start if start is not None or runtime.start is None else runtime.start
        )
        bounded_end = proposed_end if end is not None or runtime.end is None else runtime.end
        if bounded_start > bounded_end:
            raise ValueError("Session bounds conflict with the native trading window")
        if not any(bounded_start <= point <= bounded_end for point in points):
            raise ValueError("Session run window contains no feed events")
        mode = "session_dates"
    frequency = getattr(config.resolved_data_frequency, "value", config.resolved_data_frequency)
    environment = {
        "arena": "backtest",
        "platform": "ml4t.backtest",
        "capital_base": config.initial_cash,
        "data_frequency": {"1d": "daily", "daily": "daily", "1m": "minute"}.get(
            frequency, frequency
        ),
        "start": _utc(bounded_start if bounded_start is not None else first, timezone),
        "end": _utc(bounded_end if bounded_end is not None else last, timezone),
    }
    return bounded_start, bounded_end, environment, mode


@contextmanager
def _borrow_window(runtime, start, end):
    previous = runtime.start, runtime.end
    runtime.start, runtime.end = start, end
    try:
        yield
    finally:
        runtime.start, runtime.end = previous
