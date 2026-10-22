"""Pandas container projection over causal native market reads."""

from collections.abc import Callable, Mapping, Sequence
from math import isfinite
from typing import Any, Literal

import pandas as pd
from ml4t.specs import WindowUnit

from ..._temporal import temporal_rows
from ...runtime import RuntimeServices
from ...sessions import SessionConfig
from .assets import Asset, Future
from .continuous import ContinuousFuture, PreparedFutures
from .csv import PreparedCSV
from .window import _daily_history_close_offset


class BarData:
    """Project current/history without a separate price store or execution path.

    With a calendar, daily labels are native session dates. Calendar-free reads
    and minute output preserve physical UTC instants. Daily reads on minute
    execution may project native causal session aggregation. Adjustment policy
    remains with the declared prepared source.
    """

    def __init__(
        self,
        runtime: RuntimeServices,
        *,
        history_view: Literal["raw", "prepared"] = "raw",
        history_unit: WindowUnit | str | None = None,
        history_sources: Mapping[str, str] | None = None,
        csv_sources: Sequence[PreparedCSV] = (),
        restriction: Callable[[Asset, Any], bool] | None = None,
        read_policy: Literal["native", "zipline"] = "native",
        continuous_futures: Sequence[PreparedFutures] = (),
    ) -> None:
        if history_view not in {"raw", "prepared"}:
            raise ValueError("history_view must be raw or prepared")
        if read_policy not in {"native", "zipline"}:
            raise ValueError("read_policy must be native or zipline")
        self._runtime = runtime
        self._futures = {source.handle: source for source in continuous_futures}
        self._futures_cache: dict[ContinuousFuture, tuple[Any, bool, Any]] = {}
        self._csv_sources = tuple(csv_sources)
        self._restriction = restriction
        self._read_policy = read_policy
        self._history_sources = dict(history_sources or {})
        if any(
            frequency not in {"1d", "1m"} or not isinstance(name, str) or not name
            for frequency, name in self._history_sources.items()
        ):
            raise ValueError("history_sources maps 1d/1m to explicit native source names")
        self._history_view = history_view
        self._history_unit = None if history_unit is None else WindowUnit(history_unit)
        if self._history_unit not in {None, WindowUnit.OBSERVATION, WindowUnit.EXPECTED_BAR}:
            raise ValueError("Zipline bar_count history requires observation or expected_bar units")

    @staticmethod
    def _value(value):
        return float("nan") if value is None else value

    @staticmethod
    def _selection(assets, fields, *, research=False):
        valid = (Asset, str) if research else (Asset,)
        one_asset, one_field = isinstance(assets, valid), isinstance(fields, str)
        chosen = [assets] if one_asset else list(assets)
        names = [fields] if one_field else list(fields)
        if not chosen or any(not isinstance(asset, valid) for asset in chosen):
            raise TypeError("BarData requires an Asset or a nonempty sequence of declared Assets")
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("BarData fields must be a string or nonempty sequence of field names")
        legacy_ids = [
            asset.legacy_sid
            for asset in chosen
            if isinstance(asset, Asset) and asset.legacy_sid is not None
        ]
        if len(set(legacy_ids)) != len(legacy_ids):
            raise ValueError("Legacy SID mapping is ambiguous across native asset keys")
        if len(set(chosen)) != len(chosen) or len(set(names)) != len(names):
            raise ValueError("BarData asset and field selections cannot contain duplicates")
        return chosen, names, one_asset, one_field

    def _market_current(self, assets, fields, **kwargs):
        market = self._runtime.market
        market._check_active()
        if (
            self._read_policy == "zipline"
            and not market._inclusive
            and not (set(fields) == {"price"} and self._history_view == "raw")
        ):
            engine = self._runtime._engine
            assert engine is not None
            unit = self._history_unit or (
                WindowUnit.OBSERVATION
                if engine.config.resolved_calendar is None
                else WindowUnit.EXPECTED_BAR
            )
            return market.history(
                assets,
                fields,
                bar_count=1,
                unit=unit,
                view=self._history_view,
                session_close_offset=(
                    _daily_history_close_offset(engine.config)
                    if unit is WindowUnit.EXPECTED_BAR
                    else None
                ),
                **kwargs,
            )
        return market.current(assets, fields, **kwargs)

    def _current_value(self, asset, name):
        if isinstance(asset, ContinuousFuture):
            contract, _ = self._continuous_state(asset)
            if name == "contract":
                return contract
            return float("nan") if contract is None else self._current_value(contract, name)
        if isinstance(asset, Asset) and name == "last_traded":
            return self._last_traded_value(asset)
        key = asset.key if isinstance(asset, Asset) else asset
        market_field = isinstance(asset, Asset) and name in {
            "price",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "last_traded",
        }
        candidates = [
            source
            for source in self._csv_sources
            if name in source.fields and key in source.source.frame["asset"]
        ]
        if not market_field and candidates:
            if len(candidates) > 1:
                raise ValueError(
                    "CSV field/asset is ambiguous across original sources; query runtime.auxiliary by source identity"
                )
            value = self._runtime.auxiliary(candidates[0].source.source_id).current(
                [key], [name], forward_fill=True
            )[name][0]
            return self._value(value)
        if isinstance(asset, str):
            raise ValueError("Research key/field requires a registered frozen CSV source")
        if not market_field and any(name in source.fields for source in self._csv_sources):
            return float("nan")
        frame = self._market_current(
            [key],
            [name],
            forward_fill=self._stale_fields([name]),
            finite_prices_only=self._read_policy == "zipline",
        )
        rows = list(temporal_rows(frame, named=True))
        return self._market_value(asset, name, rows[0].get(name) if rows else None)

    def _stale_fields(self, names, ffill=None):
        selected = self._read_policy == "zipline" if ffill is None else ffill
        if not isinstance(selected, bool):
            raise TypeError("ffill must be Boolean")
        return {"price": selected} if "price" in names else False

    def _alive(self, asset, timestamp=None):
        market = self._runtime.market
        assert market._asof is not None
        source = self._runtime._assets
        return source is None or source._active_at(
            asset.key, timestamp if timestamp is not None else market._asof, market._asof
        )

    def _market_value(self, asset, field, value, timestamp=None):
        if isinstance(asset, ContinuousFuture):
            return self._value(value)
        if self._read_policy == "zipline" and not self._alive(asset, timestamp):
            return 0 if field == "volume" else float("nan")
        return self._value(value)

    def _last_traded_value(self, asset):
        traded = self._runtime.market.last_traded(asset.key)
        if traded is None:
            return pd.NaT
        engine = self._runtime._engine
        assert engine is not None
        frequency = getattr(
            engine.config.resolved_data_frequency, "value", engine.config.resolved_data_frequency
        )
        daily = self._read_policy == "zipline" and frequency in {"daily", "1d"}
        return self._labels([traded], "1d" if daily else "1m")[0]

    @property
    def fetcher_assets(self) -> list[Asset]:
        if not self._csv_sources:
            return []
        source = self._csv_sources[-1]
        known = self._runtime.auxiliary(source.source.source_id).assets
        return sorted(
            {asset for asset in source.df["sid"] if isinstance(asset, Asset) and asset.key in known}
        )

    def can_trade(self, assets: Asset | Sequence[Asset]) -> bool | pd.Series:
        market = self._runtime.market
        if isinstance(assets, Asset):
            chosen, one = [assets], True
        else:
            chosen, one = list(assets), False
        if any(not isinstance(asset, Asset) for asset in chosen):
            raise TypeError("Tradability queries require declared Assets")

        def tradable(asset):
            if isinstance(asset, ContinuousFuture):
                asset, _ = self._continuous_state(asset)
                if asset is None:
                    return False
            return (
                self._known_price_tradable(asset)
                if self._read_policy == "zipline"
                else market.can_trade(asset.key)
            ) and not (self._restriction is not None and self._restriction(asset, market._asof))

        flags = [tradable(asset) for asset in chosen]
        return flags[0] if one else pd.Series(flags, index=chosen, dtype=bool)

    def _known_price_tradable(self, asset):
        if not self._alive(asset):
            return False
        frame = self._market_current(
            [asset.key], ["price"], forward_fill=True, finite_prices_only=True
        )
        price = frame["price"][0] if not frame.is_empty() else None
        return price is not None and isfinite(price) and price > 0

    def is_stale(self, assets: Asset | Sequence[Asset]) -> bool | pd.Series:
        market = self._runtime.market
        if isinstance(assets, Asset):
            chosen, one = [assets], True
        else:
            chosen, one = list(assets), False
        if any(not isinstance(asset, Asset) for asset in chosen):
            raise TypeError("Staleness queries require declared Assets")

        def stale(asset):
            if isinstance(asset, ContinuousFuture):
                asset, _ = self._continuous_state(asset)
                if asset is None:
                    return False
            if self._read_policy == "native":
                return market.is_stale(asset.key)
            if not self._alive(asset):
                return False
            observed = self._market_current([asset.key], ["volume"])
            volume = None if observed.is_empty() else observed["volume"][0]
            return (
                not (volume is not None and volume > 0)
                and market.last_traded(asset.key) is not None
            )

        flags = [bool(stale(asset)) for asset in chosen]
        return flags[0] if one else pd.Series(flags, index=chosen, dtype=bool)

    def current(
        self, assets: Asset | str | Sequence[Asset | str], fields: str | Sequence[str]
    ) -> Any:
        chosen, names, one_asset, one_field = self._selection(
            assets, fields, research=bool(self._csv_sources)
        )
        rows = {}
        individual = bool(self._csv_sources) or any(
            isinstance(asset, ContinuousFuture) for asset in chosen
        )
        if not individual:
            raw = [name for name in names if name != "last_traded"]
            if raw:
                frame = self._market_current(
                    [asset.key for asset in chosen],
                    raw,
                    forward_fill=self._stale_fields(raw),
                    finite_prices_only=self._read_policy == "zipline",
                )
                rows = {row["asset"]: row for row in temporal_rows(frame, named=True)}
        result = pd.DataFrame(
            [
                [
                    self._current_value(asset, name)
                    if individual
                    else self._last_traded_value(asset)
                    if name == "last_traded"
                    else self._market_value(asset, name, rows.get(asset.key, {}).get(name))
                    for name in names
                ]
                for asset in chosen
            ],
            index=pd.Index(chosen),
            columns=pd.Index(names, dtype=object),
        )
        if "last_traded" in names and all(isinstance(asset, Asset) for asset in chosen):
            engine = self._runtime._engine
            assert engine is not None
            frequency = getattr(
                engine.config.resolved_data_frequency,
                "value",
                engine.config.resolved_data_frequency,
            )
            daily = (
                self._read_policy == "zipline"
                and frequency in {"daily", "1d"}
                and engine.config.resolved_calendar is not None
            )
            result["last_traded"] = pd.Series(
                result["last_traded"],
                index=result.index,
                dtype="datetime64[ns]" if daily else "datetime64[ns, UTC]",
            )
        if one_asset and one_field:
            value = result.iloc[0, 0]
            if names[0] == "contract":
                return value
            return float("nan") if value is None else value
        if one_asset:
            row = result.iloc[0].copy()
            row.name = chosen[0].symbol if isinstance(chosen[0], Asset) else chosen[0]
            return row
        if one_field:
            return result[names[0]]
        return result

    def _continuous_source(self, handle: ContinuousFuture) -> PreparedFutures:
        source = self._futures.get(handle)
        if source is None or source.handle.query != handle.query:
            raise ValueError("Continuous query must match its registered PreparedFutures source")
        return source

    def _continuous_state(self, handle: ContinuousFuture) -> tuple[Future | None, list[Future]]:
        self._runtime.market._check_active()
        source = self._continuous_source(handle)
        market = self._runtime.market
        cached = self._futures_cache.get(handle)
        if cached is None or cached[:2] != (market._asof, market._inclusive):
            assert market._asof is not None
            cutoff = pd.Timestamp(market._asof)
            if cutoff.tz is None:
                assert self._runtime._engine is not None
                cutoff = cutoff.tz_localize(self._runtime._engine.config.resolved_timezone)
            state = source.at(cutoff, before_observation=not market._inclusive)
            cached = (market._asof, market._inclusive, state)
            self._futures_cache[handle] = cached
        return cached[2]

    def current_chain(self, continuous_future: ContinuousFuture) -> list[Future]:
        """Return the prepared dated maturities from the selected front plus offset."""
        if not isinstance(continuous_future, ContinuousFuture):
            raise TypeError("current_chain requires a registered ContinuousFuture")
        return list(self._continuous_state(continuous_future)[1])

    def _labels(self, times, frequency, config=None):
        engine = self._runtime._engine
        assert engine is not None
        config = engine.config if config is None else config
        if frequency == "1m" or config.resolved_calendar is None:
            stamps = pd.DatetimeIndex(times)
            if stamps.tz is None:
                stamps = stamps.tz_localize(config.resolved_timezone)
            return stamps.tz_convert("UTC").as_unit("ns")
        session = SessionConfig(
            config.resolved_calendar,
            config.resolved_timezone,
            config.resolved_session_start_time,
            config.resolved_timestamp_semantics,
            config.resolved_feed_spec.bar_label,
            config.resolved_data_frequency,
        )
        return pd.DatetimeIndex([session.date_for_timestamp(t).date() for t in times]).as_unit("ns")

    def history(
        self,
        assets: Asset | Sequence[Asset],
        fields: str | Sequence[str],
        bar_count: int,
        frequency: str,
        ffill: bool | None = None,
    ) -> pd.DataFrame | pd.Series:
        chosen, names, one_asset, one_field = self._selection(assets, fields)
        if set(names) - {"price", "open", "high", "low", "close", "volume"}:
            raise ValueError(
                "Prepared bar history supports OHLCV/price; other domains need declared preparation"
            )
        self._runtime.market._check_active()
        engine = self._runtime._engine
        assert engine is not None
        market = self._runtime.market
        source = self._history_sources.get(frequency)
        continuous = [asset for asset in chosen if isinstance(asset, ContinuousFuture)]
        for asset in continuous:
            self._continuous_source(asset)
        if continuous and source is None:
            raise ValueError(
                "Continuous history requires registered handles and a named Data history source"
            )
        if source is not None:
            market = self._runtime.history_source(source)
        config = market._history_config
        declared = config.resolved_feed_spec.data_frequency
        canonical = {"daily": "1d", "d": "1d", "1d": "1d", "minute": "1m", "1m": "1m"}
        source_frequency = canonical.get(str(getattr(declared, "value", declared)))
        aggregate = (
            self._read_policy == "zipline"
            and source is None
            and frequency == "1d"
            and source_frequency == "1m"
        )
        if frequency not in {"1d", "1m"} or source_frequency != frequency and not aggregate:
            raise ValueError(
                "Requested history frequency must match its explicitly prepared source grid; "
                "prepare cross-frequency bars in Data before the run"
            )
        unit = self._history_unit
        if unit is None:
            unit = (
                WindowUnit.OBSERVATION
                if config.resolved_calendar is None
                else WindowUnit.EXPECTED_BAR
            )
        if aggregate:
            frame = market.session_bars(
                [asset.key for asset in chosen],
                names,
                bar_count=bar_count,
                view=self._history_view,
                forward_fill=self._stale_fields(names, ffill),
                finite_prices_only=True,
            )
        else:
            frame = market.history(
                [asset.key for asset in chosen],
                names,
                bar_count=bar_count,
                unit=unit,
                view="raw" if source is not None else self._history_view,
                forward_fill=self._stale_fields(names, ffill),
                finite_prices_only=self._read_policy == "zipline",
                session_close_offset=(
                    _daily_history_close_offset(config)
                    if self._read_policy == "zipline" and unit is WindowUnit.EXPECTED_BAR
                    else None
                ),
            )
        rows = list(temporal_rows(frame, named=True))
        times = list(dict.fromkeys(row["timestamp"] for row in rows))
        source_times = market._timestamps
        first = (
            frame["window_start"][0]
            if aggregate and not frame.is_empty()
            else times[0]
            if times
            else None
        )
        if len(times) != bar_count or (
            first is not None and source_times and first < source_times[0]
        ):
            raise ValueError(
                "History window starts before captured source coverage; supply warmup bars"
            )
        labels = self._labels(times, frequency, config)
        lookup = {(row["timestamp"], row["asset"]): row for row in rows}
        if one_field:
            result = pd.DataFrame(
                [
                    [
                        self._market_value(
                            asset, names[0], lookup[(time, asset.key)][names[0]], time
                        )
                        for asset in chosen
                    ]
                    for time in times
                ],
                index=labels,
                columns=chosen,
                dtype="float64",
            )
            return result[chosen[0]] if one_asset else result
        # Published Zipline3.2 uses this MultiIndex even for a single asset and
        # leaves its asset level unnamed. Preserve actual output, not its prose.
        result = pd.DataFrame(
            [
                [
                    self._market_value(asset, name, lookup[(time, asset.key)][name], time)
                    for name in names
                ]
                for time in times
                for asset in chosen
            ],
            index=pd.MultiIndex.from_product(
                [labels, chosen], names=["date" if frequency == "1d" else "date_time", None]
            ),
            columns=pd.Index(names, name="fields", dtype=object),
            dtype="float64",
        )
        return result.sort_index()
