"""Run-scoped eligibility over an offline, caller-owned identity source."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from importlib import import_module
from math import isfinite
from typing import TYPE_CHECKING, Any, Protocol, cast

import pandas as pd
import polars as pl
from ml4t.specs import FeedSpec

from ._temporal import timestamp_values
from .controls import OrderControlContext, TradingControlViolation

if TYPE_CHECKING:
    from .engine import Engine
    from .runtime import RuntimeServices


class InstrumentSource(Protocol):
    """Data's public causal registry queries, without a mandatory Data dependency."""

    def lifetimes(
        self,
        on: date | datetime | str | None = None,
        *,
        as_of: datetime | str | None = None,
        universe: str | None = None,
    ) -> pl.DataFrame: ...

    def resolve(
        self,
        symbol: str,
        on: date | datetime | str,
        exchange: str | None = None,
        *,
        as_of: datetime | str | None = None,
    ) -> Any: ...


class InstrumentView:
    """Delegate source selection at the simulation cutoff; never infer a ticker identity."""

    def __init__(
        self,
        runtime: RuntimeServices,
        engine: Engine,
        source: InstrumentSource,
        feed_spec: FeedSpec,
    ) -> None:
        self._runtime = runtime
        self._engine: Engine | None = engine
        self._source = source
        self._spec = feed_spec
        entity = feed_spec.entity_col or "asset"
        if isinstance(entity, (tuple, list)) and len(entity) == 1:
            entity = entity[0]
        if not isinstance(entity, str):
            raise ValueError("Instrument queries require one declared stable entity column")
        self._entity = entity
        assert engine.feed.prices is not None
        self._dtype = engine.feed.prices.schema[engine.feed._entity_col]
        self._ids_at = lru_cache(maxsize=128)(self._load_ids)
        self._query_points = self._prepare_query_points(engine)

    def _prepare_query_points(self, engine: Engine) -> dict[int, date]:
        self._query_points: dict[int, date] = {}
        self._prepare_points(
            [
                t
                for t in engine.feed.timestamps
                if self._runtime.end is None or t <= self._runtime.end
            ]
        )
        return self._query_points

    def _prepare_points(self, timestamps: list[datetime]) -> None:
        # Data owns conversion from physical instants to registry session labels.
        # Batch its public helper instead of constructing a calendar per query.
        if type(self._source).__module__ != "ml4t.data.assets.registry":
            return
        assets = import_module("ml4t.data.assets")
        if (
            type(self._source) is not assets.InstrumentRegistry
            or cast(Any, self._source).identities.schema["valid_from"] != pl.Date
            or self._spec.timestamp_semantics != "session_label"
        ):
            return
        times = sorted({self._instant(t).value for t in timestamps} - self._query_points.keys())
        if not times:
            return
        points = pl.Series("timestamp", times, dtype=pl.Int64).cast(pl.Datetime("ns", "UTC"))
        labels = import_module("ml4t.data.core.revisions").session_labels_at(
            points, self._spec.with_overrides(timestamp_semantics="event_time", bar_label=None)
        )
        self._query_points.update(zip(times, labels.to_list(), strict=True))

    def _point(self, on: date | datetime | str) -> date | datetime | str:
        return (
            self._query_points.get(self._instant(on).value, on) if isinstance(on, datetime) else on
        )

    def _owner(self) -> Engine:
        engine = self._engine
        if engine is None or self._runtime._engine is not engine:
            raise RuntimeError("instrument services require their active run")
        return engine

    def _instant(self, value: datetime) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(self._owner().config.timezone or "UTC")
        return stamp.tz_convert("UTC")

    def _clock(self) -> pd.Timestamp:
        self._owner()
        stamp = self._runtime.market._asof
        assert stamp is not None
        return self._instant(stamp)

    def _read(
        self, on: date | datetime | str, as_of: datetime, universe: str | None = None
    ) -> pl.DataFrame:
        self._owner()
        point = self._point(on)
        frame = self._source.lifetimes(on=point, as_of=as_of, universe=universe)
        if not isinstance(frame, pl.DataFrame) or self._entity not in frame:
            raise ValueError(
                "Instrument source must return its declared entity column in a Polars frame"
            )
        if frame.schema[self._entity] != self._dtype:
            raise ValueError("Instrument and price stable identity dtypes must match")
        if frame[self._entity].null_count() or frame[self._entity].is_duplicated().any():
            raise ValueError("Active instrument identities must be known and unique")
        if frame.height:
            if (
                "available_at" not in frame
                or not isinstance(frame.schema["available_at"], pl.Datetime)
                or frame.schema["available_at"].time_zone is None
            ):
                raise ValueError("Causal instrument facts require aware available_at instants")
            if any(
                value is None or value > as_of for value in timestamp_values(frame["available_at"])
            ):
                raise ValueError("Instrument source returned unavailable future or unknown facts")
        return frame.clone()

    def _load_ids(self, on: datetime, as_of: datetime) -> frozenset[Any]:
        return frozenset(self._read(on, as_of)[self._entity].to_list())

    def _active_at(self, asset: Any, on: datetime, as_of: datetime) -> bool:
        self._owner()
        return asset in self._ids_at(self._instant(on), self._instant(as_of))

    def active(
        self, *, on: date | datetime | str | None = None, universe: str | None = None
    ) -> pl.DataFrame:
        """Return Data-selected identities/membership at an explicit point, known now."""
        clock = self._clock()
        return self._read(clock if on is None else on, clock, universe)

    def resolve(
        self, symbol: str, *, exchange: str | None = None, on: date | datetime | str | None = None
    ) -> Any:
        """Resolve dated symbols through Data with the current known-at cutoff."""
        clock = self._clock()
        point = clock if on is None else on
        asset = self._source.resolve(symbol, self._point(point), exchange, as_of=clock)
        frame = self._read(point, clock)
        ids = frame[self._entity].to_list()
        if asset not in ids or "symbol" not in frame or "exchange" not in frame:
            raise ValueError("Resolved instrument lacks matching declared identity facts")
        row = frame.row(ids.index(asset), named=True)
        if row["symbol"] != symbol or (exchange is not None and row["exchange"] != exchange):
            raise ValueError("Resolved instrument conflicts with dated symbol or exchange facts")
        return asset

    def is_active(self, asset: Any) -> bool:
        """Test declared identity coverage, separately from executable quotes."""
        clock = self._clock()
        return asset in self._ids_at(clock, clock)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "feed_spec": self._spec.to_dict(),
            "as_of": self._clock().isoformat(),
            "inactive_position_policy": self._runtime.inactive_position_policy,
        }

    def validate(self, request: OrderControlContext) -> None:
        clock = self._clock()
        earliest, _ = self._owner().broker._preopen_order_eligibility()
        point = clock if earliest is None else self._instant(earliest)
        if not self._active_at(request.asset, point, clock):
            raise TradingControlViolation(
                f"Instrument {request.asset!r} has no declared active identity at {point}, known at {clock}"
            )

    @staticmethod
    def _has_quote(values: dict[str, Any]) -> bool:
        for field in ["price", "open", "high", "low", "close", "vwap", "bid", "ask", "mid"]:
            value = values.get(field)
            if value is not None and (not isinstance(value, (int, float)) or isfinite(value)):
                return True
        return any(
            isinstance(values.get(field), (int, float))
            and isfinite(values[field])
            and values[field] > 0
            for field in ["volume", "bid_size", "ask_size"]
        )

    def _before_event(self, timestamp: datetime, data: dict[str, dict]) -> None:
        engine = self._owner()
        if self._runtime.inactive_position_policy == "require_flat":
            inactive = [
                asset
                for asset, position in engine.broker.get_positions().items()
                if position.quantity != 0 and not self.is_active(asset)
            ]
            if inactive:
                raise ValueError(
                    f"Positions lack active instrument coverage: {inactive}; close before expiry or explicitly retain the last eligible mark"
                )
        for asset, values in data.items():
            if self._has_quote(values) and not self.is_active(asset):
                raise ValueError(
                    f"Observed quote conflicts with inactive or unknown instrument {asset!r}"
                )
        for order in engine.broker.get_orders(pending_only=True):
            clock = self._clock()
            point = (
                max(clock, self._instant(order._eligible_from))
                if order._eligible_from is not None
                else clock
            )
            if not self._active_at(order.asset, point, clock):
                engine.broker._cancel_order_at(
                    order.order_id, timestamp=timestamp, reason="instrument_inactive"
                )

    def _close(self) -> None:
        self._ids_at.cache_clear()
        self._query_points.clear()
        self._engine = None
