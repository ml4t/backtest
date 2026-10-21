"""Opt-in simulation services over the existing feed and broker."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from datetime import datetime
from math import isfinite
from typing import TYPE_CHECKING, Any

import polars as pl

from ._temporal import temporal_frame, temporal_row, temporal_rows
from .execution.callbacks import SessionScheduler

if TYPE_CHECKING:
    from .engine import Engine


class MarketDataView:
    """Read-only raw observations with explicit causal windows and stale views."""

    def __init__(self, runtime: RuntimeServices, engine: Engine) -> None:
        self._runtime = runtime
        feed = engine.feed
        self._timestamp_col = feed.feed_spec.timestamp_col
        self._entity_col = feed._entity_col
        frame = feed.prices
        if frame is None:
            raise ValueError("Runtime market services require a prices frame")
        self._frame = frame
        self._columns = set(self._frame.columns)
        self._roles = {
            name: getattr(feed.feed_spec, f"{name}_col", name)
            for name in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "bid",
                "ask",
                "mid",
                "bid_size",
                "ask_size",
            ]
        }
        self._roles["price"] = getattr(feed.feed_spec, "price_col", None) or self._roles["close"]
        if self._roles["price"] not in self._columns:
            self._roles["price"] = self._roles["close"]
        self._rows: dict[str, dict[datetime, int]] = {}
        self._times: dict[str, list[datetime]] = {}
        for field, column in runtime.known_at_cols.items():
            if self._column(field) not in self._columns:
                raise ValueError(f"Known-at declaration has no input field {field!r}")
            if column not in self._columns or not isinstance(
                self._frame.schema[column], pl.Datetime
            ):
                raise ValueError(f"Known-at column {column!r} must contain datetimes")
        index_columns = list(
            dict.fromkeys([self._timestamp_col, self._entity_col, *runtime.known_at_cols.values()])
        )
        for index, row in enumerate(temporal_rows(self._frame.select(index_columns), named=True)):
            asset, timestamp = row[self._entity_col], row[self._timestamp_col]
            rows = self._rows.setdefault(asset, {})
            if timestamp in rows:
                raise ValueError(f"Duplicate market observation for {asset!r} at {timestamp}")
            for column in runtime.known_at_cols.values():
                if row[column] is not None:
                    try:
                        _ = row[column] <= timestamp
                    except TypeError as exc:
                        raise ValueError(
                            "Known-at and observation timezones must be compatible"
                        ) from exc
            rows[timestamp] = index
        for asset, rows in self._rows.items():
            self._times[asset] = sorted(rows)
        self._timestamps = sorted({t for times in self._times.values() for t in times})
        self._asof: datetime | None = None
        self._inclusive = False

    def _row(self, asset: str, timestamp: datetime) -> dict[str, Any] | None:
        index = self._rows.get(asset, {}).get(timestamp)
        return temporal_row(self._frame, index) if index is not None else None

    def _column(self, field: str) -> str:
        return self._roles.get(field, field)

    def _available(self, field: str, row: dict[str, Any]) -> bool:
        assert self._asof is not None
        column = self._column(field)
        declaration = self._runtime.known_at_cols.get(field)
        if declaration is None:
            declaration = next(
                (v for k, v in self._runtime.known_at_cols.items() if self._column(k) == column),
                None,
            )
        known = row.get(declaration) if declaration else row[self._timestamp_col]
        if known is None:
            return False
        if row[self._timestamp_col] == self._asof and not self._inclusive:
            return False
        return known <= self._asof

    def _query(
        self,
        assets: Sequence[str],
        fields: Sequence[str],
        times: list[datetime],
        *,
        forward_fill: bool,
    ) -> pl.DataFrame:
        if self._runtime._engine is None or self._runtime._market is not self or self._asof is None:
            raise RuntimeError("market services require an active run")
        if isinstance(assets, str) or isinstance(fields, str) or not assets or not fields:
            raise ValueError("assets and fields must be nonempty sequences")
        if len(set(assets)) != len(assets) or len(set(fields)) != len(fields):
            raise ValueError("assets and fields cannot contain duplicates")
        if set(fields) & {"timestamp", "asset"}:
            raise ValueError("Requested fields collide with observation keys")
        for field in fields:
            if self._column(field) not in self._columns and field not in self._roles:
                raise ValueError(f"Unknown market field {field!r}")
            if forward_fill and self._column(field) in {
                self._column("volume"),
                self._column("bid_size"),
                self._column("ask_size"),
            }:
                raise ValueError(f"Forward filling {field!r} would invent tradable volume")
        records = []
        for timestamp in times:
            for asset in assets:
                row = self._row(asset, timestamp)
                result: dict[str, Any] = {"timestamp": timestamp, "asset": asset}
                for field in fields:
                    candidate = row
                    observed = timestamp
                    if (
                        candidate is None
                        or not self._available(field, candidate)
                        or candidate.get(self._column(field)) is None
                    ):
                        candidate = None
                        if forward_fill:
                            asset_times = self._times.get(asset, [])
                            for index in range(bisect_right(asset_times, timestamp) - 1, -1, -1):
                                past = asset_times[index]
                                value = self._row(asset, past)
                                assert value is not None
                                if (
                                    self._available(field, value)
                                    and value.get(self._column(field)) is not None
                                ):
                                    candidate, observed = value, past
                                    break
                    result[field] = (
                        candidate.get(self._column(field)) if candidate is not None else None
                    )
                    if forward_fill:
                        result[f"{field}_observed_at"] = observed if candidate is not None else None
                records.append(result)
        schema = {
            "timestamp": self._frame.schema[self._timestamp_col],
            "asset": self._frame.schema[self._entity_col],
        }
        schema.update({f: self._frame.schema.get(self._column(f), pl.Float64) for f in fields})
        if forward_fill:
            schema.update(
                {f"{f}_observed_at": self._frame.schema[self._timestamp_col] for f in fields}
            )
        return temporal_frame(records, schema=schema)

    def history(
        self,
        assets: Sequence[str],
        fields: Sequence[str],
        *,
        bar_count: int,
        forward_fill: bool = False,
    ) -> pl.DataFrame:
        """Return the last N observed feed-price instants, retaining missing asset rows.

        Window units are observations, not elapsed minutes or sessions. Known-at
        declarations mask unavailable fields. Forward filling is explicit and retains
        the source timestamp; it never changes executable feed observations.
        """
        if self._runtime._engine is None or self._runtime._market is not self or self._asof is None:
            raise RuntimeError("market services require an active run")
        if isinstance(bar_count, bool) or not isinstance(bar_count, int) or bar_count <= 0:
            raise ValueError("bar_count must be a positive integer")
        end = bisect_right(self._timestamps, self._asof)
        if not self._inclusive and end and self._timestamps[end - 1] == self._asof:
            end -= 1
        return self._query(
            assets,
            fields,
            self._timestamps[max(0, end - bar_count) : end],
            forward_fill=forward_fill,
        )

    def current(
        self, assets: Sequence[str], fields: Sequence[str], *, forward_fill: bool = False
    ) -> pl.DataFrame:
        """Return values at the active instant, with no implicit stale executable quote."""
        if self._runtime._engine is None or self._runtime._market is not self or self._asof is None:
            raise RuntimeError("market services require an active run")
        times = [self._asof] if self._inclusive else []
        return self._query(assets, fields, times, forward_fill=forward_fill)

    def _past_rows(self, asset: str):
        if self._runtime._engine is None or self._runtime._market is not self or self._asof is None:
            raise RuntimeError("market services require an active run")
        times = self._times.get(asset, [])
        end = bisect_right(times, self._asof)
        if not self._inclusive and end and times[end - 1] == self._asof:
            end -= 1
        for index in range(end - 1, -1, -1):
            timestamp = times[index]
            row = self._row(asset, timestamp)
            assert row is not None
            yield timestamp, row

    def last_observed(self, asset: str, field: str = "price") -> datetime | None:
        """Return the source instant of the last known non-null field observation."""
        for timestamp, row in self._past_rows(asset):
            if self._available(field, row) and row.get(self._column(field)) is not None:
                return timestamp
        return None

    def last_traded(self, asset: str) -> datetime | None:
        """Return the last known finite-price bar with actual positive observed volume."""
        if self._column("volume") not in self._columns:
            raise ValueError("last_traded requires observed volume; use last_observed for quotes")
        for timestamp, row in self._past_rows(asset):
            price, volume = row.get(self._column("price")), row.get(self._column("volume"))
            if (
                self._available("price", row)
                and self._available("volume", row)
                and price is not None
                and isfinite(price)
                and volume is not None
                and isfinite(volume)
                and volume > 0
            ):
                return timestamp
        return None

    def can_trade(self, asset: str, *, require_volume: bool = False) -> bool:
        """Check current known positive quotes, optionally requiring observed positive volume.

        This tests data eligibility, not account buying power or order-specific controls.
        A stale history value is never an executable quote for this native method.
        """
        fields = ["price", "volume"] if require_volume else ["price"]
        frame = self.current([asset], fields)
        if frame.is_empty():
            return False
        price = frame["price"][0]
        if price is None or not isfinite(price) or price <= 0:
            return False
        if require_volume:
            volume = frame["volume"][0]
            return volume is not None and isfinite(volume) and volume > 0
        return True


class RuntimeServices:
    """Explicit run-scoped services without changing Strategy attributes or context.

    Feed rows before ``start`` supply warmup history, not trading callbacks or account
    samples. Bounds are inclusive actual event instants. No network access occurs.
    """

    def __init__(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        known_at_cols: dict[str, str] | None = None,
    ) -> None:
        if start is not None and not isinstance(start, datetime):
            raise TypeError("start must be a datetime")
        if end is not None and not isinstance(end, datetime):
            raise TypeError("end must be a datetime")
        if start is not None and end is not None and start > end:
            raise ValueError("start must not exceed end")
        self.start, self.end = start, end
        self.known_at_cols = dict(known_at_cols or {})
        self.scheduler = SessionScheduler(self)
        self._engine: Engine | None = None
        self._market: MarketDataView | None = None

    @property
    def market(self) -> MarketDataView:
        """The active causal view; unavailable after normal or exceptional run cleanup."""
        if self._market is None:
            raise RuntimeError("market services require an active run")
        return self._market

    def _bind(self, engine: Engine) -> None:
        if self._engine is not None:
            raise RuntimeError("RuntimeServices already belongs to an active run")
        market = MarketDataView(self, engine)
        eligible = [t for t in engine.feed.timestamps if self._accepts(t)]
        if not eligible:
            raise ValueError("RuntimeServices window contains no feed events")
        for events in [engine._capital_events, engine._funding_events]:
            if any(not self._accepts(t) for t in events):
                raise ValueError(
                    "Cash events outside the trading window require explicit preparation"
                )
        market._asof = self.start if self.start is not None else eligible[0]
        self._engine, self._market = engine, market
        try:
            self.scheduler._bind()
        except BaseException:
            self._engine, self._market = None, None
            raise

    def _accepts(self, timestamp: datetime) -> bool:
        return (self.start is None or timestamp >= self.start) and (
            self.end is None or timestamp <= self.end
        )

    def _advance(self, timestamp: datetime, data: dict[str, dict]) -> dict[str, dict]:
        market = self.market
        market._asof, market._inclusive = timestamp, True
        if not self.known_at_cols:
            return data
        visible = {asset: dict(values) for asset, values in data.items()}
        for asset, values in visible.items():
            row = market._row(asset, timestamp)
            if row is None:
                continue
            for field in [
                "price",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "bid",
                "ask",
                "mid",
                "bid_size",
                "ask_size",
            ]:
                raw = market._column(field)
                source = (
                    field
                    if raw in market._columns
                    else "close"
                    if field in {"price", "open", "high", "low"}
                    else field
                )
                if not market._available(source, row):
                    values[field] = None
        return visible

    def _unbind(self) -> None:
        self.scheduler._unbind()
        self._engine, self._market = None, None
