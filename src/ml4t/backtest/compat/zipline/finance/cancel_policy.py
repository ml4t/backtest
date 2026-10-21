"""Cancellation parameters compiled to the native session-end policy."""

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from ....calendar import get_calendar, get_schedule
from ....sessions import SessionConfig


@dataclass(frozen=True)
class EODCancel:
    warn_on_cancel: bool = True


@dataclass(frozen=True)
class NeverCancel:
    pass


def _configure_cancellation(engine, policy: EODCancel) -> None:
    config, runtime = engine.config, engine.runtime
    assert runtime is not None
    calendar = config.resolved_calendar
    if calendar is None:
        raise ValueError("EODCancel requires a declared calendar")
    points = [point for point in engine.feed.timestamps if runtime._accepts(point)]
    if not points:
        raise ValueError("EODCancel requires market observations")
    clock = SessionConfig(
        calendar,
        config.resolved_timezone,
        config.resolved_session_start_time,
        config.resolved_timestamp_semantics,
        config.resolved_feed_spec.bar_label,
        config.resolved_data_frequency,
    )
    sessions = {clock.date_for_timestamp(point).date() for point in points}
    future = get_calendar(calendar).name == "us_futures"
    bounds, closes = {}, []
    for row in get_schedule(calendar, min(sessions), max(sessions)).iter_rows(named=True):
        if row["session_date"] not in sessions:
            continue
        opened, closed = row["market_open"], row["market_close"]
        if future:
            opened += timedelta(hours=12, minutes=30)
            closed -= timedelta(hours=1)
            bounds[row["session_date"]] = (opened, closed)
        stamp = pd.Timestamp(closed)
        if points[0].tzinfo is None:
            stamp = stamp.tz_convert(config.resolved_timezone).tz_localize(None)
        closes.append(stamp)
    if set(closes) - set(points):
        raise ValueError("EODCancel requires each selected session execution-close observation")
    engine.broker.configure_session_cancellation(
        bounds=bounds, warn_on_cancel=policy.warn_on_cancel
    )
    config.metadata["migration_cancel_policy"] = {
        "method": "native_session_end",
        "warn_on_cancel": policy.warn_on_cancel,
        "execution_closes": [point.isoformat() for point in closes],
    }
