"""Pandas container projection over causal native market reads."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import pandas as pd
from ml4t.specs import WindowUnit

from ..._temporal import temporal_rows
from ...runtime import RuntimeServices
from ...sessions import SessionConfig
from .assets import Asset
from .csv import PreparedCSV


class BarData:
    """Project current/history without a separate price store or execution path.

    With a calendar, daily labels are native session dates. Calendar-free reads
    and minute output preserve physical UTC instants. Requested history frequency must match the prepared
    source grid; this boundary never resamples raw data or computes adjustments.
    """

    def __init__(
        self,
        runtime: RuntimeServices,
        *,
        history_view: Literal["raw", "prepared"] = "raw",
        history_unit: WindowUnit | str | None = None,
        csv_sources: Sequence[PreparedCSV] = (),
        restriction: Callable[[Asset, Any], bool] | None = None,
    ) -> None:
        if history_view not in {"raw", "prepared"}:
            raise ValueError("history_view must be raw or prepared")
        self._runtime = runtime
        self._csv_sources = tuple(csv_sources)
        self._restriction = restriction
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

    def _current_value(self, asset, name):
        key = asset.key if isinstance(asset, Asset) else asset
        market_field = isinstance(asset, Asset) and name in {
            "price",
            "open",
            "high",
            "low",
            "close",
            "volume",
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
        frame = self._runtime.market.current([key], [name])
        rows = list(temporal_rows(frame, named=True))
        return self._value(rows[0][name] if rows else None)

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
        flags = [
            market.can_trade(asset.key)
            and not (self._restriction is not None and self._restriction(asset, market._asof))
            for asset in chosen
        ]
        return flags[0] if one else pd.Series(flags, index=chosen, dtype=bool)

    def current(
        self, assets: Asset | str | Sequence[Asset | str], fields: str | Sequence[str]
    ) -> Any:
        chosen, names, one_asset, one_field = self._selection(
            assets, fields, research=bool(self._csv_sources)
        )
        rows = {}
        if not self._csv_sources:
            frame = self._runtime.market.current([asset.key for asset in chosen], names)
            rows = {row["asset"]: row for row in temporal_rows(frame, named=True)}
        result = pd.DataFrame(
            [
                [
                    self._current_value(asset, name)
                    if self._csv_sources
                    else self._value(rows.get(asset.key, {}).get(name))
                    for name in names
                ]
                for asset in chosen
            ],
            index=pd.Index(chosen),
            columns=pd.Index(names, dtype=object),
        )
        if one_asset and one_field:
            value = result.iloc[0, 0]
            return float("nan") if value is None else value
        if one_asset:
            row = result.iloc[0].copy()
            row.name = chosen[0].symbol if isinstance(chosen[0], Asset) else chosen[0]
            return row
        if one_field:
            return result[names[0]]
        return result

    def _labels(self, times, frequency):
        engine = self._runtime._engine
        assert engine is not None
        config = engine.config
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
    ) -> pd.DataFrame | pd.Series:
        chosen, names, one_asset, one_field = self._selection(assets, fields)
        if set(names) - {"price", "open", "high", "low", "close", "volume"}:
            raise ValueError(
                "Prepared bar history supports OHLCV/price; other domains need declared preparation"
            )
        self._runtime.market._check_active()
        engine = self._runtime._engine
        assert engine is not None
        declared = engine.config.resolved_feed_spec.data_frequency
        canonical = {"daily": "1d", "d": "1d", "1d": "1d", "minute": "1m", "1m": "1m"}
        if (
            frequency not in {"1d", "1m"}
            or canonical.get(str(getattr(declared, "value", declared))) != frequency
        ):
            raise ValueError(
                "Requested history frequency must match its explicitly prepared source grid; "
                "prepare cross-frequency bars in Data before the run"
            )
        unit = self._history_unit
        if unit is None:
            unit = (
                WindowUnit.OBSERVATION
                if engine.config.resolved_calendar is None
                else WindowUnit.EXPECTED_BAR
            )
        frame = self._runtime.market.history(
            [asset.key for asset in chosen],
            names,
            bar_count=bar_count,
            unit=unit,
            view=self._history_view,
        )
        rows = list(temporal_rows(frame, named=True))
        times = list(dict.fromkeys(row["timestamp"] for row in rows))
        source_times = self._runtime.market._timestamps
        if len(times) != bar_count or (times and source_times and times[0] < source_times[0]):
            raise ValueError(
                "History window starts before captured source coverage; supply warmup bars"
            )
        labels = self._labels(times, frequency)
        lookup = {(row["timestamp"], row["asset"]): row for row in rows}
        if one_field:
            result = pd.DataFrame(
                [
                    [self._value(lookup[(time, asset.key)][names[0]]) for asset in chosen]
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
                [self._value(lookup[(time, asset.key)][name]) for name in names]
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
