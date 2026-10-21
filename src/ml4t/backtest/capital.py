"""External capital events and returns between independently marked valuations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

import polars as pl

from ._temporal import temporal_rows


@dataclass(frozen=True)
class CapitalEvent:
    timestamp: datetime
    event_id: str
    kind: str
    value: float


@dataclass(frozen=True)
class CapitalChange:
    timestamp: datetime
    event_id: str
    kind: str
    value: float
    cash_delta: float
    nav_before: float
    nav_after: float


def index_capital_events(
    frame: pl.DataFrame | None, feed_timestamps: list[datetime]
) -> dict[datetime, list[CapitalEvent]]:
    if frame is None:
        return {}
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("capital_flows must be a Polars DataFrame")
    if not {"timestamp", "event_id", "kind", "value"} <= set(frame.columns):
        raise ValueError("capital_flows requires timestamp, event_id, kind and value")
    if not isinstance(frame.schema["timestamp"], pl.Datetime):
        raise ValueError("capital flow timestamp must be a datetime column")
    canonical = {timestamp: timestamp for timestamp in feed_timestamps}
    events: dict[datetime, list[CapitalEvent]] = {}
    seen = set()
    for row in temporal_rows(frame.sort("timestamp", maintain_order=True), named=True):
        timestamp = canonical.get(row["timestamp"])
        if timestamp is None:
            raise ValueError("capital flow timestamp must match a feed event timestamp")
        event_id = row["event_id"]
        if not isinstance(event_id, str) or not event_id or event_id in seen:
            raise ValueError("capital flow event_id must be nonempty and unique")
        seen.add(event_id)
        if row["kind"] not in {"delta", "target"}:
            raise ValueError("capital flow kind must be delta or target")
        value = row["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("capital flow value must be finite")
        if row["kind"] == "target" and value < 0:
            raise ValueError("target capital must be nonnegative")
        known_at = row.get("known_at", timestamp)
        if not isinstance(known_at, datetime):
            raise ValueError("capital flow known_at must be a datetime when supplied")
        try:
            future = known_at > timestamp
        except TypeError as exc:
            raise ValueError(
                "capital flow known_at and timestamp require compatible timezones"
            ) from exc
        if future:
            raise ValueError("capital flow cannot be applied before known_at")
        events.setdefault(timestamp, []).append(
            CapitalEvent(timestamp, event_id, row["kind"], value)
        )
    return events


def flow_adjusted_returns(
    equity_curve: list[tuple[datetime, float]],
    initial_capital: float,
    changes: list[CapitalChange],
) -> list[float]:
    """Chain pre-flow and post-flow valuation ratios; external cash earns no return."""
    indexed: dict[datetime, list[CapitalChange]] = {}
    timestamps = {timestamp for timestamp, _ in equity_curve}
    for change in changes:
        if change.timestamp not in timestamps:
            raise ValueError("capital change has no corresponding account observation")
        indexed.setdefault(change.timestamp, []).append(change)

    def ratio(end: float, start: float) -> float:
        if start == 0 and end == 0:
            return 1.0
        if start <= 0 or not isfinite(end / start):
            raise ValueError("exact investment return requires a positive starting valuation")
        return end / start

    previous = initial_capital
    returns = []
    for timestamp, equity in equity_curve:
        factor = 1.0
        for change in indexed.get(timestamp, []):
            factor *= ratio(change.nav_before, previous)
            previous = change.nav_after
        factor *= ratio(equity, previous)
        returns.append(factor - 1.0)
        previous = equity
    return returns
