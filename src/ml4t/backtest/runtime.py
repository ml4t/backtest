"""Opt-in simulation services over the existing feed and broker."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import polars as pl
from ml4t.specs import FeedSpec, TimestampSemantics, WindowUnit

from ._history_clock import expected_times, session_times
from ._instruments import InstrumentSource, InstrumentView
from ._temporal import temporal_frame, temporal_row, temporal_rows
from .auxiliary import AuxiliaryData
from .execution.callbacks import SessionScheduler
from .futures_lifecycle import FuturesLifecyclePolicy

if TYPE_CHECKING:
    from .engine import Engine


class PreparedHistory(Protocol):
    """Read interface for Data's causal prepared-history output."""

    @property
    def prices(self) -> pl.DataFrame: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...


class HistoryProvider(Protocol):
    """Offline preparation with the actual inclusive availability cutoff.

    Before-event queries additionally pass ``observation_before``. The provider
    must exclude those observations before calculating derived values, while
    preserving availability equality at ``as_of``.
    """

    def __call__(self, *, as_of: datetime) -> PreparedHistory: ...


def _request_history(
    provider: HistoryProvider, *, as_of: datetime, observation_before: datetime | None
) -> PreparedHistory:
    if observation_before is None:
        return provider(as_of=as_of)
    return cast(Callable[..., PreparedHistory], provider)(
        as_of=as_of, observation_before=observation_before
    )


class MarketDataView:
    """Read-only raw observations with explicit causal windows and stale views."""

    def __init__(
        self,
        runtime: RuntimeServices,
        engine: Engine,
        *,
        prices: pl.DataFrame | None = None,
        feed_spec: FeedSpec | None = None,
        owner: MarketDataView | None = None,
        known_at_cols: dict[str, str] | None = None,
        quote_fields: bool = True,
        eligible_cols: dict[str, str] | None = None,
    ) -> None:
        self._runtime = runtime
        self._owner = owner if owner is not None else self
        self._clock_asof: datetime | None = None
        self._clock_inclusive = False
        self._quote_fields = quote_fields
        self._eligible_cols = dict(eligible_cols or {})
        self._prepared: tuple[datetime, bool, MarketDataView] | None = None
        self._preparation_metadata: dict[str, Any] = {}
        self._known_at_cols = dict(
            runtime.known_at_cols if known_at_cols is None else known_at_cols
        )
        feed = engine.feed
        spec = feed_spec if feed_spec is not None else feed.feed_spec
        self._timestamp_col = spec.timestamp_col
        entity_col = spec.entity_col
        if not isinstance(entity_col, str):
            raise ValueError("Runtime history requires one stable entity column")
        self._entity_col = entity_col
        frame = prices if prices is not None else feed.prices
        if frame is None:
            raise ValueError("Runtime market services require a prices frame")
        self._frame = frame
        self._history_config = engine.config
        self._snapshot_cutoff: datetime | None = None
        self._columns = set(self._frame.columns)
        self._roles = (
            {
                name: getattr(spec, f"{name}_col", name)
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
            if quote_fields
            else {}
        )
        if quote_fields:
            self._roles["price"] = spec.price_col or self._roles["close"]
            if self._roles["price"] not in self._columns:
                self._roles["price"] = self._roles["close"]
        self._rows: dict[str, dict[datetime, int]] = {}
        self._times: dict[str, list[datetime]] = {}
        for field, column in self._eligible_cols.items():
            if (
                self._column(field) not in self._columns
                or column not in self._columns
                or self._frame.schema[column] != pl.Boolean
            ):
                raise ValueError(
                    "Feature eligibility requires existing fields and boolean eligibility columns"
                )
        for field, column in self._known_at_cols.items():
            if self._column(field) not in self._columns:
                raise ValueError(f"Known-at declaration has no input field {field!r}")
            if column not in self._columns or not isinstance(
                self._frame.schema[column], pl.Datetime
            ):
                raise ValueError(f"Known-at column {column!r} must contain datetimes")
        index_columns = list(
            dict.fromkeys([self._timestamp_col, self._entity_col, *self._known_at_cols.values()])
        )
        for index, row in enumerate(temporal_rows(self._frame.select(index_columns), named=True)):
            asset, timestamp = row[self._entity_col], row[self._timestamp_col]
            rows = self._rows.setdefault(asset, {})
            if timestamp in rows:
                raise ValueError(f"Duplicate market observation for {asset!r} at {timestamp}")
            for column in self._known_at_cols.values():
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

    @property
    def _asof(self) -> datetime | None:
        return self._owner._clock_asof

    @_asof.setter
    def _asof(self, value: datetime | None) -> None:
        self._clock_asof = value

    @property
    def _inclusive(self) -> bool:
        return self._owner._clock_inclusive if self._quote_fields else True

    @_inclusive.setter
    def _inclusive(self, value: bool) -> None:
        self._clock_inclusive = value

    def _row(self, asset: str, timestamp: datetime) -> dict[str, Any] | None:
        index = self._rows.get(asset, {}).get(timestamp)
        return temporal_row(self._frame, index) if index is not None else None

    def _column(self, field: str) -> str:
        return self._roles.get(field, field)

    def _published(self, field: str, row: dict[str, Any]) -> bool:
        assert self._asof is not None
        column = self._column(field)
        declaration = self._known_at_cols.get(field)
        if declaration is None:
            declaration = next(
                (v for k, v in self._known_at_cols.items() if self._column(k) == column),
                None,
            )
        known = row.get(declaration) if declaration else row[self._timestamp_col]
        if known is None:
            return False
        if row[self._timestamp_col] == self._asof and not self._inclusive:
            return False
        return known <= self._asof

    def _ineligible(self, field: str, row: dict[str, Any]) -> bool:
        column = self._eligible_cols.get(field)
        return column is not None and self._published(field, row) and row.get(column) is not True

    def _available(self, field: str, row: dict[str, Any]) -> bool:
        column = self._eligible_cols.get(field)
        if not self._published(field, row) or (column is not None and row.get(column) is not True):
            return False
        instruments = self._runtime._assets
        if self._quote_fields and instruments is not None:
            assert self._asof is not None
            return instruments._active_at(
                row[self._entity_col], row[self._timestamp_col], self._asof
            )
        return True

    def _check_active(self) -> None:
        if (
            self._runtime._engine is None
            or self._runtime._market is not self._owner
            or self._asof is None
        ):
            raise RuntimeError("market services require an active run")
        if self._snapshot_cutoff is not None and self._snapshot_cutoff != self._asof:
            raise RuntimeError(
                "Prepared source snapshot is no longer active; request it at this cutoff"
            )

    def _query(
        self,
        assets: Sequence[str],
        fields: Sequence[str],
        times: list[datetime],
        *,
        forward_fill: bool | Mapping[str, bool],
        max_age: timedelta | None = None,
        finite_prices_only: bool = False,
    ) -> pl.DataFrame:
        self._check_active()
        if not isinstance(finite_prices_only, bool):
            raise TypeError("finite_prices_only must be Boolean")
        price_columns = (
            {
                self._column(name)
                for name in ["price", "open", "high", "low", "close", "vwap", "bid", "ask", "mid"]
            }
            if finite_prices_only and self._quote_fields
            else set()
        )

        def nonfinite_price(row, field):
            return self._column(field) in price_columns and not isfinite(row[self._column(field)])

        assert self._asof is not None
        if isinstance(assets, str) or isinstance(fields, str) or not assets or not fields:
            raise ValueError("assets and fields must be nonempty sequences")
        if len(set(assets)) != len(assets) or len(set(fields)) != len(fields):
            raise ValueError("assets and fields cannot contain duplicates")
        if set(fields) & {"timestamp", "asset"}:
            raise ValueError("Requested fields collide with observation keys")
        if max_age is not None and (not isinstance(max_age, timedelta) or max_age < timedelta(0)):
            raise ValueError("max_age must be a nonnegative timedelta")
        if isinstance(forward_fill, bool):
            fill_fields = list(fields) if forward_fill else []
        elif isinstance(forward_fill, Mapping):
            if set(forward_fill) - set(fields) or any(
                not isinstance(enabled, bool) for enabled in forward_fill.values()
            ):
                raise ValueError("forward_fill must map requested field names to booleans")
            fill_fields = [field for field in fields if forward_fill.get(field, False)]
        else:
            raise ValueError("forward_fill must be Boolean or a per-field mapping")
        if {f"{field}_observed_at" for field in fill_fields} & set(fields):
            raise ValueError(
                "Forward-fill provenance columns collide with requested payload fields"
            )
        for field in fields:
            if self._column(field) not in self._columns and field not in self._roles:
                raise ValueError(f"Unknown market field {field!r}")
            if (
                field in fill_fields
                and self._quote_fields
                and self._column(field)
                in {
                    self._column("volume"),
                    self._column("bid_size"),
                    self._column("ask_size"),
                }
            ):
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
                        or finite_prices_only
                        and nonfinite_price(candidate, field)
                    ):
                        candidate = None
                        if field in fill_fields and not (
                            row is not None and self._ineligible(field, row)
                        ):
                            asset_times = self._times.get(asset, [])
                            for index in range(bisect_right(asset_times, timestamp) - 1, -1, -1):
                                past = asset_times[index]
                                value = self._row(asset, past)
                                if value is not None and self._ineligible(field, value):
                                    break
                                assert value is not None
                                if (
                                    self._available(field, value)
                                    and value.get(self._column(field)) is not None
                                    and not (finite_prices_only and nonfinite_price(value, field))
                                ):
                                    candidate, observed = value, past
                                    break
                    expired = max_age is not None and timestamp - observed > max_age
                    result[field] = (
                        candidate.get(self._column(field))
                        if candidate is not None and not expired
                        else None
                    )
                    if field in fill_fields:
                        result[f"{field}_observed_at"] = observed if candidate is not None else None
                records.append(result)
        schema = {
            "timestamp": self._frame.schema[self._timestamp_col],
            "asset": self._frame.schema[self._entity_col],
        }
        schema.update({f: self._frame.schema.get(self._column(f), pl.Float64) for f in fields})
        if fill_fields:
            schema.update(
                {f"{f}_observed_at": self._frame.schema[self._timestamp_col] for f in fill_fields}
            )
        return temporal_frame(records, schema=schema)

    def _observed_high_low(self, asset: str) -> tuple[float | None, float | None]:
        self._check_active()
        if self._asof is None or not self._inclusive:
            return None, None
        row = self._row(asset, self._asof)
        if row is None:
            return None, None
        high = row.get(self._column("high")) if self._available("high", row) else None
        low = row.get(self._column("low")) if self._available("low", row) else None
        return high, low

    def _prepared_view(self) -> MarketDataView:
        runtime = self._runtime
        engine, asof = runtime._engine, self._asof
        if engine is None or runtime._market is not self._owner or asof is None:
            raise RuntimeError("market services require an active run")
        if runtime.history_provider is None:
            raise ValueError("view='prepared' requires a causal history_provider")
        if self._prepared is not None and self._prepared[:2] == (asof, self._inclusive):
            return self._prepared[2]
        from ._history_source import prepare_history_source

        view = prepare_history_source(
            runtime, runtime.history_provider, known_at_cols=self._known_at_cols
        )
        frame, metadata = view._frame, view._preparation_metadata
        spec = FeedSpec.from_any(metadata["feed_spec"])
        if spec.timestamp_semantics != engine.config.resolved_timestamp_semantics:
            raise ValueError("Prepared history cannot reinterpret the observation timestamp role")
        if spec.resolve_calendar() != engine.config.resolved_calendar:
            raise ValueError("Prepared history calendar conflicts with the execution feed")
        source_spec = engine.feed.feed_spec
        for name in ("bar_label", "bar_grid_anchor", "partial_bar_policy", "session_start_time"):
            if getattr(spec, name) != getattr(source_spec, name):
                raise ValueError(f"Prepared history cannot change the declared {name}")
        for row in temporal_rows(frame.select(view._timestamp_col, view._entity_col), named=True):
            instant, asset = row[view._timestamp_col], row[view._entity_col]
            if instant not in self._rows.get(asset, {}):
                raise ValueError("Prepared history must retain observed execution keys")
        view._history_config = engine.config
        self._prepared = (asof, self._inclusive, view)
        return view

    @property
    def prepared_metadata(self) -> dict[str, Any]:
        """Copy the active provider's basis, cutoff and source preparation declarations."""
        self._check_active()
        if self._preparation_metadata:
            return deepcopy(self._preparation_metadata)
        return deepcopy(self._prepared_view()._preparation_metadata)

    def history(
        self,
        assets: Sequence[str],
        fields: Sequence[str],
        *,
        bar_count: int | None = None,
        unit: WindowUnit | str = WindowUnit.OBSERVATION,
        duration: timedelta | None = None,
        forward_fill: bool | Mapping[str, bool] = False,
        max_age: timedelta | None = None,
        view: Literal["raw", "prepared"] = "raw",
        finite_prices_only: bool = False,
        session_close_offset: timedelta | None = None,
    ) -> pl.DataFrame:
        """Select causal raw observations or declared grid points, without resampling.

        ``bar_count`` counts observation instants, expected bars or exchange sessions.
        Elapsed windows instead require ``duration`` and select ``(asof-duration, asof]``.
        Session windows retain each observed instant, without aggregating fields.
        Grid gaps remain null unless the caller explicitly requests a stale read view.
        An explicit nonpositive ``session_close_offset`` selects earlier daily
        right-labelled expected endpoints; it never moves an observed quote or
        changes the calendar, execution clock or default grid.
        """
        if view == "prepared":
            return self._prepared_view().history(
                assets,
                fields,
                bar_count=bar_count,
                unit=unit,
                duration=duration,
                forward_fill=forward_fill,
                max_age=max_age,
                finite_prices_only=finite_prices_only,
                session_close_offset=session_close_offset,
            )
        if view != "raw":
            raise ValueError("history view must be raw or prepared")
        self._check_active()
        assert self._asof is not None
        unit = WindowUnit(unit)
        if session_close_offset is not None and unit is not WindowUnit.EXPECTED_BAR:
            raise ValueError("session_close_offset requires expected_bar daily history")
        if unit is WindowUnit.ELAPSED:
            if (
                bar_count is not None
                or not isinstance(duration, timedelta)
                or duration <= timedelta(0)
            ):
                raise ValueError("elapsed history requires a positive duration and no bar_count")
        elif (
            duration is not None
            or isinstance(bar_count, bool)
            or not isinstance(bar_count, int)
            or bar_count <= 0
        ):
            raise ValueError("bar_count must be a positive integer, without duration")
        end = bisect_right(self._timestamps, self._asof)
        if not self._inclusive and end and self._timestamps[end - 1] == self._asof:
            end -= 1
        if unit is WindowUnit.ELAPSED:
            assert duration is not None
            times = self._timestamps[
                bisect_right(self._timestamps, self._asof - duration, hi=end) : end
            ]
        elif unit is WindowUnit.OBSERVATION:
            assert bar_count is not None
            times = self._timestamps[max(0, end - bar_count) : end]
        else:
            assert bar_count is not None
            engine = self._runtime._engine
            assert engine is not None
            if unit is WindowUnit.EXPECTED_BAR:
                times = expected_times(
                    self._history_config,
                    self._asof,
                    bar_count,
                    self._inclusive,
                    self._timestamps,
                    session_close_offset=session_close_offset,
                )
            else:
                times = session_times(
                    self._history_config, self._asof, bar_count, self._inclusive, self._timestamps
                )
        return self._query(
            assets,
            fields,
            times,
            forward_fill=forward_fill,
            max_age=max_age,
            finite_prices_only=finite_prices_only,
        )

    def session_bars(
        self,
        assets: Sequence[str],
        fields: Sequence[str],
        *,
        bar_count: int,
        view: Literal["raw", "prepared"] = "raw",
        forward_fill: bool | Mapping[str, bool] = False,
        finite_prices_only: bool = False,
    ) -> pl.DataFrame:
        """Aggregate available intraday OHLCV through the simulation clock.

        The window counts declared exchange sessions and includes the current
        partial session. Timestamp is the observed cutoff, capped at session
        close. Session/grid coverage accompanies each read-only row; missing
        volume never becomes a known zero or a complete volume sum.
        """
        from ._session_bars import session_bars

        if view not in {"raw", "prepared"}:
            raise ValueError("history view must be raw or prepared")
        return session_bars(
            self._prepared_view() if view == "prepared" else self,
            assets,
            fields,
            bar_count=bar_count,
            forward_fill=forward_fill,
            finite_prices_only=finite_prices_only,
        )

    def current(
        self,
        assets: Sequence[str],
        fields: Sequence[str],
        *,
        forward_fill: bool | Mapping[str, bool] = False,
        max_age: timedelta | None = None,
        view: Literal["raw", "prepared"] = "raw",
        finite_prices_only: bool = False,
    ) -> pl.DataFrame:
        """Return values at the active instant, with no implicit stale executable quote."""
        if view == "prepared":
            return self._prepared_view().current(
                assets,
                fields,
                forward_fill=forward_fill,
                max_age=max_age,
                finite_prices_only=finite_prices_only,
            )
        if view != "raw":
            raise ValueError("current view must be raw or prepared")
        self._check_active()
        assert self._asof is not None
        stale_view = forward_fill is True or (
            isinstance(forward_fill, Mapping)
            and any(value is True for value in forward_fill.values())
        )
        times = [self._asof] if self._inclusive or stale_view else []
        return self._query(
            assets,
            fields,
            times,
            forward_fill=forward_fill,
            max_age=max_age,
            finite_prices_only=finite_prices_only,
        )

    def _past_rows(self, asset: str):
        self._check_active()
        assert self._asof is not None
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

    def observation_age(self, asset: str, field: str = "price") -> timedelta | None:
        """Elapsed time since a known non-null observation, or None without history."""
        if self._column(field) not in self._columns and field not in self._roles:
            raise ValueError(f"Unknown market field {field!r}")
        observed = self.last_observed(asset, field)
        assert self._asof is not None
        return None if observed is None else self._asof - observed

    def is_stale(
        self, asset: str, field: str = "price", *, max_age: timedelta = timedelta(0)
    ) -> bool:
        """Test recency separately from quote validity and execution eligibility.

        No observed history returns False; can_trade separately rejects missing
        or unusable current quotes. This method never enables stale execution.
        """
        if not isinstance(max_age, timedelta) or max_age < timedelta(0):
            raise ValueError("max_age must be a nonnegative timedelta")
        age = self.observation_age(asset, field)
        return age is not None and age > max_age


class FeatureDataView:
    """Causal prepared fields sharing the runtime cutoff, without executable quote roles."""

    def __init__(self, view: MarketDataView) -> None:
        self._view = view

    def current(
        self,
        assets: Sequence[str],
        fields: Sequence[str],
        *,
        forward_fill: bool | Mapping[str, bool] = False,
        max_age: timedelta | None = None,
    ) -> pl.DataFrame:
        return self._view.current(assets, fields, forward_fill=forward_fill, max_age=max_age)

    def history(
        self,
        assets: Sequence[str],
        fields: Sequence[str],
        *,
        bar_count: int | None = None,
        unit: WindowUnit | str = WindowUnit.OBSERVATION,
        duration: timedelta | None = None,
        forward_fill: bool | Mapping[str, bool] = False,
        max_age: timedelta | None = None,
    ) -> pl.DataFrame:
        if WindowUnit(unit) not in {WindowUnit.OBSERVATION, WindowUnit.ELAPSED}:
            raise ValueError(
                "Feature history requires observation or elapsed units; prepare feature grids in Engineer"
            )
        return self._view.history(
            assets,
            fields,
            bar_count=bar_count,
            unit=unit,
            duration=duration,
            forward_fill=forward_fill,
            max_age=max_age,
        )

    def last_observed(self, asset: str, field: str) -> datetime | None:
        return self._view.last_observed(asset, field)


class AuxiliaryDataView(FeatureDataView):
    """Research-only fields from one frozen source at the active decision cutoff."""

    def __init__(self, view: MarketDataView, source: AuxiliaryData) -> None:
        super().__init__(view)
        self._source = source

    def _fields(self, fields: Sequence[str]) -> None:
        if set(fields) - self._source.known_at_cols.keys():
            raise ValueError("Auxiliary queries require declared payload fields")

    def current(
        self,
        assets: Sequence[str],
        fields: Sequence[str],
        *,
        forward_fill: bool | Mapping[str, bool] = False,
        max_age: timedelta | None = None,
    ) -> pl.DataFrame:
        self._fields(fields)
        return super().current(assets, fields, forward_fill=forward_fill, max_age=max_age)

    def history(
        self,
        assets: Sequence[str],
        fields: Sequence[str],
        *,
        bar_count: int | None = None,
        unit: WindowUnit | str = WindowUnit.OBSERVATION,
        duration: timedelta | None = None,
        forward_fill: bool | Mapping[str, bool] = False,
        max_age: timedelta | None = None,
    ) -> pl.DataFrame:
        self._fields(fields)
        return super().history(
            assets,
            fields,
            bar_count=bar_count,
            unit=unit,
            duration=duration,
            forward_fill=forward_fill,
            max_age=max_age,
        )

    def last_observed(self, asset: str, field: str) -> datetime | None:
        self._fields([field])
        return super().last_observed(asset, field)

    @property
    def assets(self) -> tuple[Any, ...]:
        """Stable keys with at least one published observation known so far."""
        self._view._check_active()
        return tuple(
            asset
            for asset in self._view._times
            if any(
                self._view._available(field, row)
                for _, row in self._view._past_rows(asset)
                for field in self._source.known_at_cols
            )
        )

    @property
    def metadata(self) -> dict[str, Any]:
        self._view._check_active()
        return self._source.metadata


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
        execution_windows: Sequence[tuple[datetime, datetime]] | None = None,
        known_at_cols: dict[str, str] | None = None,
        history_provider: HistoryProvider | None = None,
        history_sources: Mapping[str, HistoryProvider] | None = None,
        history_assets: Mapping[str, Sequence[str]] | None = None,
        signal_known_at_cols: dict[str, str] | None = None,
        context_known_at_cols: dict[str, str] | None = None,
        signal_eligible_cols: dict[str, str] | None = None,
        auxiliary: Sequence[AuxiliaryData] = (),
        instruments: InstrumentSource | None = None,
        instrument_feed_spec: FeedSpec | Any | None = None,
        inactive_position_policy: Literal["retain_last_mark", "require_flat"] = "retain_last_mark",
        futures_lifecycle: Sequence[FuturesLifecyclePolicy] = (),
    ) -> None:
        if start is not None and not isinstance(start, datetime):
            raise TypeError("start must be a datetime")
        if end is not None and not isinstance(end, datetime):
            raise TypeError("end must be a datetime")
        if start is not None and end is not None and start > end:
            raise ValueError("start must not exceed end")
        self.start, self.end = start, end
        self.execution_windows = execution_windows
        self.known_at_cols = dict(known_at_cols or {})
        self.signal_known_at_cols = dict(signal_known_at_cols or {})
        self.context_known_at_cols = dict(context_known_at_cols or {})
        self.signal_eligible_cols = dict(signal_eligible_cols or {})
        if history_provider is not None and not callable(history_provider):
            raise TypeError("history_provider must be callable")
        self.history_provider = history_provider
        self.history_sources = dict(history_sources or {})
        if any(
            not isinstance(name, str) or not name or not callable(provider)
            for name, provider in self.history_sources.items()
        ):
            raise TypeError("history_sources requires nonempty source names and callable providers")
        self._history_views: dict[str, tuple[datetime, bool, MarketDataView]] = {}
        self.history_assets = {name: tuple(keys) for name, keys in (history_assets or {}).items()}
        if any(
            name not in self.history_sources
            or isinstance((history_assets or {})[name], str)
            or any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys)
            for name, keys in self.history_assets.items()
        ):
            raise ValueError("history_assets requires unique research keys for registered sources")
        if inactive_position_policy not in {"retain_last_mark", "require_flat"}:
            raise ValueError("inactive_position_policy must be retain_last_mark or require_flat")
        if instruments is None and (
            instrument_feed_spec is not None or inactive_position_policy != "retain_last_mark"
        ):
            raise ValueError("Instrument metadata and policies require instruments")
        if instruments is not None and (
            instrument_feed_spec is None
            or not callable(getattr(instruments, "lifetimes", None))
            or not callable(getattr(instruments, "resolve", None))
        ):
            raise ValueError(
                "instruments requires causal lifetimes/resolve methods and instrument_feed_spec"
            )
        prepared_auxiliary = tuple(auxiliary)
        if any(not isinstance(source, AuxiliaryData) for source in prepared_auxiliary):
            raise TypeError("auxiliary must contain frozen AuxiliaryData sources")
        if len({source.source_id for source in prepared_auxiliary}) != len(prepared_auxiliary):
            raise ValueError("Duplicate auxiliary source identifiers")
        self._auxiliary_sources = {source.source_id: source for source in prepared_auxiliary}
        self._auxiliary_views: dict[str, AuxiliaryDataView] = {}
        self.instruments = instruments
        self.instrument_feed_spec = (
            FeedSpec.from_any(instrument_feed_spec) if instruments is not None else None
        )
        self.inactive_position_policy = inactive_position_policy
        self.scheduler = SessionScheduler(self)
        self.futures_lifecycle = tuple(futures_lifecycle)
        if any(not isinstance(policy, FuturesLifecyclePolicy) for policy in self.futures_lifecycle):
            raise TypeError("futures_lifecycle requires FuturesLifecyclePolicy declarations")
        if len({policy.asset for policy in self.futures_lifecycle}) != len(self.futures_lifecycle):
            raise ValueError("Choose one controlling lifecycle boundary per dated contract")
        if len({policy.event_id for policy in self.futures_lifecycle}) != len(
            self.futures_lifecycle
        ):
            raise ValueError("Lifecycle event IDs must be unique")
        self._engine: Engine | None = None
        self._market: MarketDataView | None = None
        self._signals: FeatureDataView | None = None
        self._assets: InstrumentView | None = None

    @property
    def execution_windows(self) -> tuple[tuple[datetime, datetime], ...] | None:
        """Inclusive physical event intervals; excluded feed rows remain read-only history."""
        return self._execution_windows

    @execution_windows.setter
    def execution_windows(self, value: Sequence[tuple[datetime, datetime]] | None) -> None:
        if getattr(self, "_engine", None) is not None:
            raise RuntimeError("Configure execution windows before an active run")
        try:
            windows = None if value is None else tuple(tuple(pair) for pair in value)
        except TypeError as exc:
            raise TypeError("Execution windows require pairs of physical datetimes") from exc
        previous = None
        for pair in windows or ():
            if len(pair) != 2 or any(not isinstance(time, datetime) for time in pair):
                raise TypeError("Each execution window requires two physical datetimes")
            opened, closed = pair
            try:
                invalid = opened > closed or previous is not None and opened <= previous
            except TypeError as exc:
                raise ValueError("Execution window timezones must be compatible") from exc
            if invalid:
                raise ValueError("Execution windows must be ordered, nonoverlapping intervals")
            previous = closed
        self._execution_windows = windows
        self._execution_window_starts = tuple(pair[0] for pair in windows or ())

    def history_source(self, source_id: str) -> MarketDataView:
        """Read an explicitly prepared independent grid at this run's actual cutoff.

        Providers own aggregation, adjustment and source availability. These rows
        never enter the execution feed or replace executable quotes.
        """
        from ._history_source import prepare_history_source

        market = self.market
        market._check_active()
        if source_id not in self.history_sources:
            raise ValueError(f"No prepared history source {source_id!r}")
        assert market._asof is not None
        cached = self._history_views.get(source_id)
        if cached is not None and cached[:2] == (market._asof, market._inclusive):
            return cached[2]
        view = prepare_history_source(
            self,
            self.history_sources[source_id],
            research_keys=self.history_assets.get(source_id, ()),
        )
        self._history_views[source_id] = (market._asof, market._inclusive, view)
        return view

    @property
    def assets(self) -> InstrumentView:
        """Causal identity queries over the declared offline source during this run."""
        if self._engine is None:
            raise RuntimeError("instrument services require an active run")
        if self._assets is None:
            raise ValueError("Instrument queries require instruments and instrument_feed_spec")
        return self._assets

    @property
    def market(self) -> MarketDataView:
        """The active causal view; unavailable after normal or exceptional run cleanup."""
        if self._market is None:
            raise RuntimeError("market services require an active run")
        return self._market

    @property
    def signals(self) -> FeatureDataView:
        """Prepared per-asset fields with independent publication times and no quote aliases."""
        if self._engine is None:
            raise RuntimeError("signal services require an active run")
        if self._signals is None:
            raise ValueError("Signal queries require a signals frame")
        return self._signals

    def auxiliary(self, source_id: str) -> AuxiliaryDataView:
        """Look up the exact registered offline source; never fetch or substitute data."""
        if self._engine is None:
            raise RuntimeError("auxiliary services require an active run")
        if source_id not in self._auxiliary_views:
            raise ValueError(f"No prepared auxiliary source registered for {source_id!r}")
        return self._auxiliary_views[source_id]

    @contextmanager
    def auxiliary_scope(self, sources: Sequence[AuxiliaryData]) -> Iterator[RuntimeServices]:
        """Temporarily include frozen sources while preserving the caller's registry.

        Enter before Engine.run; normal binding/unbinding owns the active views.
        Repeated references to the same source object share one registry entry.
        A different source with an existing identifier is a conflict.
        """
        if self._engine is not None:
            raise RuntimeError("Auxiliary input scope requires an inactive runtime")
        previous = dict(self._auxiliary_sources)
        try:
            for source in sources:
                if not isinstance(source, AuxiliaryData):
                    raise TypeError("auxiliary scope requires frozen AuxiliaryData sources")
                existing = self._auxiliary_sources.get(source.source_id)
                if existing is not None and existing is not source:
                    raise ValueError("Duplicate auxiliary source identifier in scope")
                self._auxiliary_sources[source.source_id] = source
            yield self
        finally:
            self._auxiliary_sources.clear()
            self._auxiliary_sources.update(previous)

    def _bind(self, engine: Engine) -> None:
        if self._engine is not None:
            raise RuntimeError("RuntimeServices already belongs to an active run")
        if engine.config.resolved_timestamp_semantics not in {
            TimestampSemantics.EVENT_TIME,
            TimestampSemantics.BAR_CLOSE,
        }:
            raise ValueError(
                "RuntimeServices requires declared timestamp_semantics='event_time' or 'bar_close'; "
                "prepare session labels as actual instants before binding physical runtime cutoffs"
            )
        market = MarketDataView(self, engine)
        signals = None
        if engine.feed.signals is not None:
            known_at = dict(self.signal_known_at_cols)
            eligible = dict(self.signal_eligible_cols)
            metadata = engine.feed.feature_metadata
            if metadata is not None:
                for declared, supplied, name in (
                    (metadata["projection"]["signal_known_at_cols"], known_at, "publication"),
                    (metadata["projection"]["signal_eligible_cols"], eligible, "eligibility"),
                ):
                    for field, column in declared.items():
                        if field in supplied and supplied[field] != column:
                            raise ValueError(f"Runtime {name} mapping conflicts with feature_spec")
                        supplied[field] = column
            entity = engine.feed._entity_col
            prices = engine.feed.prices
            assert prices is not None
            if engine.feed.signals.schema[entity] != prices.schema[entity]:
                raise ValueError("Signal and price stable asset key dtypes must match")
            signals = FeatureDataView(
                MarketDataView(
                    self,
                    engine,
                    prices=engine.feed.signals,
                    feed_spec=engine.feed.feed_spec,
                    owner=market,
                    known_at_cols=known_at,
                    quote_fields=False,
                    eligible_cols=eligible,
                )
            )
        elif self.signal_known_at_cols or self.signal_eligible_cols:
            raise ValueError("Signal known-at declarations require a signals frame")
        if self.context_known_at_cols:
            frame = engine.feed.context
            if frame is None:
                raise ValueError("Context known-at declarations require a context frame")
            time = engine.feed.feed_spec.timestamp_col
            if frame[time].is_duplicated().any():
                raise ValueError("Context publication requires unique timestamps")
            for field, column in self.context_known_at_cols.items():
                if (
                    field not in frame
                    or column not in frame
                    or not isinstance(frame.schema[column], pl.Datetime)
                ):
                    raise ValueError(
                        "Context known-at fields require existing fields and datetime publication columns"
                    )
                for row in temporal_rows(
                    frame.select(list(dict.fromkeys([time, column]))), named=True
                ):
                    if row[column] is not None:
                        try:
                            _ = row[column] <= row[time]
                        except TypeError as exc:
                            raise ValueError(
                                "Context publication and event timezones must be compatible"
                            ) from exc
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
        self._signals = signals
        try:
            for source_id, source in self._auxiliary_sources.items():
                view = MarketDataView(
                    self,
                    engine,
                    prices=source.frame,
                    feed_spec=source.feed_spec,
                    owner=market,
                    known_at_cols=source.known_at_cols,
                    eligible_cols=source.eligible_cols,
                    quote_fields=False,
                )
                if view._timestamps:
                    assert market._asof is not None
                    try:
                        _ = view._timestamps[0] <= market._asof
                    except TypeError as exc:
                        raise ValueError(
                            "Auxiliary and simulation clock timezones must be compatible"
                        ) from exc
                self._auxiliary_views[source_id] = AuxiliaryDataView(view, source)
            if self.instruments is not None:
                assert self.instrument_feed_spec is not None
                self._assets = InstrumentView(
                    self, engine, self.instruments, self.instrument_feed_spec
                )
                self._assets.active()
                engine.broker.register_trading_control(self._assets)
            self.scheduler._bind()
            for policy in self.futures_lifecycle:
                policy._check_engine(engine)
                if eligible[0] <= policy.effective_at <= eligible[-1]:
                    if not self._accepts(policy.effective_at):
                        raise ValueError("Lifecycle boundary falls in an excluded execution window")
                    policy._register(self)
            for policy in self.futures_lifecycle:
                engine.broker.register_trading_control(policy)
            engine.broker._set_observed_high_low_reader(market._observed_high_low)
        except BaseException:
            self.scheduler._unbind()
            for policy in self.futures_lifecycle:
                engine.broker.unregister_trading_control(policy)
            if self._assets is not None:
                engine.broker.unregister_trading_control(self._assets)
                self._assets._close()
                self._assets = None
            self._engine, self._market = None, None
            self._signals = None
            self._auxiliary_views.clear()
            self._history_views.clear()
            raise

    def _accepts(self, timestamp: datetime) -> bool:
        if not (
            (self.start is None or timestamp >= self.start)
            and (self.end is None or timestamp <= self.end)
        ):
            return False
        if self._execution_windows is None:
            return True
        index = bisect_right(self._execution_window_starts, timestamp) - 1
        return index >= 0 and timestamp <= self._execution_windows[index][1]

    def _advance(self, timestamp: datetime, data: dict[str, dict]) -> dict[str, dict]:
        market = self.market
        market._asof, market._inclusive = timestamp, True
        if self._assets is not None:
            self._assets._before_event(timestamp, data)
        signal_view = self._signals._view if self._signals is not None else None
        signal_fields = (
            signal_view._known_at_cols.keys() | signal_view._eligible_cols.keys()
            if signal_view is not None
            else set()
        )
        if not self.known_at_cols and not signal_fields and self._assets is None:
            return data
        visible = {asset: dict(values) for asset, values in data.items()}
        for asset, values in visible.items():
            if signal_fields:
                assert signal_view is not None
                view = signal_view
                signal = view._row(asset, timestamp)
                if signal is not None:
                    delivered = dict(values.get("signals", {}))
                    for field in signal_fields:
                        if not view._available(field, signal):
                            delivered[field] = None
                    values["signals"] = delivered
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

    def _mask_context(self, timestamp: datetime, context: dict[str, Any]) -> dict[str, Any]:
        if not self.context_known_at_cols:
            return context
        engine = self._engine
        assert engine is not None
        frame = engine.feed._slice_for_timestamp(
            engine.feed.context, engine.feed._context_ranges_by_ts, timestamp
        )
        if frame is None or not frame.height:
            return context
        row = temporal_row(frame, 0)
        result = dict(context)
        for field, column in self.context_known_at_cols.items():
            if row[column] is None or row[column] > timestamp:
                result[field] = None
        return result

    def _unbind(self) -> None:
        if self._engine is not None:
            self._engine.broker._set_observed_high_low_reader(None)
            for policy in self.futures_lifecycle:
                self._engine.broker.unregister_trading_control(policy)
        self.scheduler._unbind()
        if self._assets is not None:
            assert self._engine is not None
            self._engine.broker.unregister_trading_control(self._assets)
            self._assets._close()
            self._assets = None
        self._engine, self._market = None, None
        self._signals = None
        self._auxiliary_views.clear()
        self._history_views.clear()
