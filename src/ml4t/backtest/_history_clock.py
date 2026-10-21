"""Select declared calendar windows without creating executable market events."""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from ml4t.specs import BarGridAnchor, BarLabel, PartialBarPolicy

from .calendar import get_schedule
from .sessions import _is_right_labeled_bar

if TYPE_CHECKING:
    from .config import BacktestConfig


@lru_cache(maxsize=64)
def _schedule(calendar: str, start: date, end: date):
    return tuple(get_schedule(calendar, start, end, include_breaks=True).iter_rows(named=True))


def _sessions(config: BacktestConfig, asof: datetime, count: int, inclusive: bool):
    calendar = config.resolved_calendar
    if calendar is None:
        raise ValueError("session and expected_bar history require a declared calendar")
    right = _is_right_labeled_bar(
        config.resolved_timestamp_semantics, config.resolved_feed_spec.bar_label
    )

    def clock(value):
        if value is None:
            return None
        if asof.tzinfo is not None:
            return value.astimezone(asof.tzinfo)
        return value.astimezone(ZoneInfo(config.resolved_timezone)).replace(tzinfo=None)

    # Calendar days are only a search bound. Membership always comes from actual sessions.
    days = max(14, count * 3)
    earliest = asof.date().toordinal() - 1
    while True:
        start = asof.date() - timedelta(days=min(days, earliest))
        rows = []
        for row in _schedule(calendar, start, asof.date() + timedelta(days=2)):
            opened = clock(row["market_open"])
            if opened < asof or opened == asof and inclusive and not right:
                rows.append(
                    {
                        name: clock(row.get(name))
                        for name in ("market_open", "market_close", "break_start", "break_end")
                    }
                )
        if len(rows) >= count or days >= earliest:
            return rows[-count:]
        days *= 2


def _intervals(row):
    opened, closed = row["market_open"], row["market_close"]
    start, end = row["break_start"], row["break_end"]
    if start is not None and end is not None and opened < start < end < closed:
        return [(opened, start), (end, closed)]
    return [(opened, closed)]


def session_times(
    config: BacktestConfig,
    asof: datetime,
    count: int,
    inclusive: bool,
    observed: Sequence[datetime],
) -> list[datetime]:
    rows = _sessions(config, asof, count, inclusive)
    right = _is_right_labeled_bar(
        config.resolved_timestamp_semantics, config.resolved_feed_spec.bar_label
    )
    intervals = [pair for row in rows for pair in _intervals(row)]
    if not intervals:
        return []
    start = bisect_left(observed, intervals[0][0])
    end = bisect_right(observed, asof) if inclusive else bisect_left(observed, asof)
    return [
        t
        for t in observed[start:end]
        if any(
            opened < t <= closed if right else opened <= t < closed for opened, closed in intervals
        )
    ]


def _grid_parameters(config: BacktestConfig):
    feed = config.resolved_feed_spec
    if feed.time_contract_version != 2 or any(
        getattr(feed, field) is None
        for field in ("data_frequency", "bar_grid_anchor", "bar_label", "partial_bar_policy")
    ):
        raise ValueError("expected_bar history requires a complete version-2 bar grid")
    if feed.bar_grid_anchor != BarGridAnchor.SESSION_OPEN or feed.bar_type not in (None, "time"):
        raise ValueError("expected_bar history requires a session-open fixed-time grid")
    raw_frequency = feed.data_frequency
    if not config._explicit_data_frequency and config.feed_spec is not None:
        raw_frequency = config.feed_spec.data_frequency or raw_frequency
    value = str(getattr(raw_frequency, "value", raw_frequency)).lower()
    daily = value in {"1d", "d", "daily"}
    match = re.fullmatch(r"(\d+)?(minutes?|min|m|hours?|h|seconds?|s)", value)
    if not daily and match is None:
        raise ValueError(f"Unsupported fixed bar grid data_frequency: {value!r}")
    step = None
    if match:
        amount = int(match[1] or 1)
        if amount <= 0:
            raise ValueError("Bar grid data_frequency must be positive")
        seconds = 3600 if match[2].startswith("h") else 60 if match[2].startswith("m") else 1
        step = timedelta(seconds=amount * seconds)
    right = feed.bar_label == BarLabel.RIGHT
    partial = feed.partial_bar_policy == PartialBarPolicy.INCLUDE
    return step, right, partial


def expected_times(
    config: BacktestConfig,
    asof: datetime,
    count: int,
    inclusive: bool,
    observed: Sequence[datetime],
    *,
    session_close_offset: timedelta | None = None,
) -> list[datetime]:
    step, right, partial = _grid_parameters(config)
    daily = step is None
    if session_close_offset is not None and (
        not isinstance(session_close_offset, timedelta)
        or session_close_offset > timedelta(0)
        or not daily
        or not right
    ):
        raise ValueError(
            "session_close_offset requires a nonpositive timedelta and daily right labels"
        )
    # Each session contributes at least one eligible bar for daily inputs; sparse or
    # oversized intraday intervals may require additional earlier sessions.
    session_count = min(count + 1, 14)
    while True:
        sessions = _sessions(config, asof, session_count, inclusive)
        times = []
        for row in reversed(sessions):
            if daily:
                opened, closed = row["market_open"], row["market_close"]
                if session_close_offset is not None:
                    if session_close_offset <= opened - closed:
                        raise ValueError("History close must follow the declared session open")
                    closed += session_close_offset
                segments = [(opened, closed)]
            else:
                segments = _intervals(row)
            for opened, closed in reversed(segments):
                interval = closed - opened if daily else step
                assert interval is not None
                full, remainder = divmod(closed - opened, interval)
                size = full + int(bool(remainder) and partial)
                if right:
                    last = min(size, int((asof - opened) // interval))
                    if remainder and partial and (closed < asof or closed == asof and inclusive):
                        last = size
                    if not inclusive and min(opened + last * interval, closed) == asof:
                        last -= 1
                    for index in range(last, max(0, last - (count - len(times))), -1):
                        times.append(min(opened + index * interval, closed))
                else:
                    last = min(size - 1, int((asof - opened) // interval))
                    if not inclusive and opened + last * interval == asof:
                        last -= 1
                    for index in range(last, max(-1, last - (count - len(times))), -1):
                        times.append(opened + index * interval)
                if len(times) >= count:
                    break
            if len(times) >= count:
                break
        if len(times) >= count or len(sessions) < session_count:
            break
        # A partial-excluding interval larger than every session has no grid points.
        if (
            step is not None
            and not partial
            and all(
                closed - opened < step for row in sessions for opened, closed in _intervals(row)
            )
        ):
            raise ValueError("The declared grid has no complete bars in the selected sessions")
        session_count *= 2
    result = sorted(times)
    expected = set(result)
    start = bisect_left(observed, result[0]) if result else 0
    end = bisect_right(observed, asof) if inclusive else bisect_left(observed, asof)
    if result and any(t not in expected for t in observed[start:end]):
        raise ValueError(
            "Observed price timestamp is off-grid for the declared expected_bar history"
        )
    return result
