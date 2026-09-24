"""Validated funding events and cash-flow records for perpetual positions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

import polars as pl


@dataclass(frozen=True)
class FundingEvent:
    """One asset's scheduled rate or cash amount at a feed timestamp."""

    timestamp: datetime
    asset: str
    rate: float | None = None
    amount_per_unit: float | None = None


@dataclass(frozen=True)
class FundingPayment:
    """Applied cash flow, including zero for an event with no held position."""

    timestamp: datetime
    asset: str
    quantity: float
    mark_price: float | None
    multiplier: float
    rate: float | None
    amount_per_unit: float | None
    cash_delta: float


def index_funding_events(
    frame: pl.DataFrame | None,
    feed_timestamps: list[datetime],
    feed_assets: set[str],
) -> dict[datetime, list[FundingEvent]]:
    """Reject ambiguous inputs before a backtest can change portfolio state."""
    if frame is None:
        return {}
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("funding input must be a Polars DataFrame")
    required = {"timestamp", "asset"}
    if not required <= set(frame.columns):
        raise ValueError("funding input requires timestamp and asset columns")
    if "rate" not in frame.columns and "amount_per_unit" not in frame.columns:
        raise ValueError("funding input requires rate or amount_per_unit")
    if not isinstance(frame.schema["timestamp"], pl.Datetime):
        raise ValueError("funding timestamp must be a datetime column")

    canonical_timestamps = {timestamp: timestamp for timestamp in feed_timestamps}
    indexed: dict[datetime, list[FundingEvent]] = {}
    seen: set[tuple[datetime, str]] = set()
    for row in frame.iter_rows(named=True):
        timestamp = canonical_timestamps.get(row["timestamp"])
        if timestamp is None:
            raise ValueError("funding timestamp must match a feed event timestamp")
        asset = row["asset"]
        if not isinstance(asset, str) or not asset or asset not in feed_assets:
            raise ValueError(f"funding asset {asset!r} is absent from the feed")
        key = (timestamp, asset)
        if key in seen:
            raise ValueError(f"duplicate funding event for {asset} at {timestamp.isoformat()}")
        seen.add(key)

        rate = row.get("rate")
        amount = row.get("amount_per_unit")
        if (rate is None) == (amount is None):
            raise ValueError("funding event requires exactly one of rate or amount_per_unit")
        value = rate if rate is not None else amount
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("funding rate or amount_per_unit must be finite")
        indexed.setdefault(timestamp, []).append(
            FundingEvent(timestamp, asset, rate=rate, amount_per_unit=amount)
        )
    for events in indexed.values():
        events.sort(key=lambda event: event.asset)
    return indexed
