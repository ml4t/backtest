"""Consolidate causal intraday read windows without emitting executable bars."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import TYPE_CHECKING

import polars as pl

from ._history_clock import _grid_parameters, _intervals, _sessions
from ._temporal import temporal_frame, temporal_rows

if TYPE_CHECKING:
    from .runtime import MarketDataView


def session_bars(
    market: MarketDataView,
    assets: Sequence[str],
    fields: Sequence[str],
    *,
    bar_count: int,
    forward_fill: bool | Mapping[str, bool],
    finite_prices_only: bool,
) -> pl.DataFrame:
    market._check_active()
    if isinstance(bar_count, bool) or not isinstance(bar_count, int) or bar_count <= 0:
        raise ValueError("bar_count must be a positive integer")
    if set(fields) - {"price", "open", "high", "low", "close", "volume"}:
        raise ValueError("Session bars require OHLCV/price fields")
    config = market._history_config
    step, right, partial = _grid_parameters(config)
    if step is None:
        raise ValueError("Session bars require a declared intraday source grid")
    asof = market._asof
    assert asof is not None
    sessions = _sessions(config, asof, bar_count, market._inclusive)
    layouts = []
    expected = set()
    for session in sessions:
        times = []
        for opened, closed in _intervals(session):
            full, remainder = divmod(closed - opened, step)
            size = full + int(bool(remainder) and partial)
            for index in range(size):
                instant = min(opened + (index + int(right)) * step, closed)
                if instant < asof or instant == asof and market._inclusive:
                    times.append(instant)
        layouts.append((session, times))
        expected.update(times)
    if expected:
        start = bisect_left(market._timestamps, min(expected))
        end = (
            bisect_right(market._timestamps, asof)
            if market._inclusive
            else bisect_left(market._timestamps, asof)
        )
        if any(time not in expected for time in market._timestamps[start:end]):
            raise ValueError("Observed price timestamp is off-grid for declared session bars")
    result = []
    for session, times in layouts:
        frame = market._query(
            assets,
            fields,
            times,
            forward_fill=forward_fill,
            finite_prices_only=finite_prices_only,
        )
        rows = {asset: [] for asset in assets}
        for row in temporal_rows(frame, named=True):
            rows[row["asset"]].append(row)
        closed = session["market_close"]
        for asset in assets:
            selected = rows[asset]
            observed = 0
            for instant in times:
                raw = market._row(asset, instant)
                if raw is not None and any(market._available(field, raw) for field in fields):
                    observed += 1
            record = {
                "timestamp": min(asof, closed),
                "asset": asset,
                "window_start": times[0] if times else session["market_open"],
                "session_closed": closed < asof or closed == asof and market._inclusive,
                "observed_bars": observed,
                "expected_bars": len(times),
            }
            for field in fields:
                values = [row[field] for row in selected if row[field] is not None]
                if not values:
                    value = None
                elif field == "volume":
                    if any(value < 0 for value in values):
                        raise ValueError("Session volume must be nonnegative")
                    value = (
                        sum(values)
                        if len(values) == len(times) and all(isfinite(value) for value in values)
                        else None
                    )
                elif field == "open":
                    value = values[0]
                elif field == "high":
                    value = max(values)
                elif field == "low":
                    value = min(values)
                else:
                    value = values[-1]
                record[field] = value
            result.append(record)
    clock = market._frame.schema[market._timestamp_col]
    return temporal_frame(
        result,
        schema={
            "timestamp": clock,
            "asset": pl.String,
            "window_start": clock,
            "session_closed": pl.Boolean,
            "observed_bars": pl.UInt64,
            "expected_bars": pl.UInt64,
            **{
                field: market._frame.schema.get(market._column(field), pl.Float64)
                for field in fields
            },
        },
    )
