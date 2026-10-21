"""Offline CSV callback adaptation over frozen native research inputs."""

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from ..._feature_signals import _pandas_frame
from ...auxiliary import AuxiliaryData
from .assets import Asset
from .lookup import PreparedAssetLookup, _symbol_parts


def _instants(values: pd.Series, timezone: str, date_format: str | None = None):
    clock = pd.DatetimeIndex(pd.to_datetime(values, format=date_format, errors="coerce"))
    if clock.tz is None:
        clock = clock.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return clock.tz_convert("UTC").as_unit("ns")


def _callback_name(callback):
    if callback is None:
        return None
    return f"{getattr(callback, '__module__', type(callback).__module__)}.{getattr(callback, '__qualname__', type(callback).__qualname__)}"


class PreparedCSV:
    """Prepare a frozen caller-loaded frame, with no execution-time I/O.

    Dates identify physical observations; publication is a separate required
    column. Preprocessing runs before date/symbol mapping and postprocessing
    receives a UTC DatetimeIndex and immutable Asset or research-key sids.
    """

    def __init__(
        self,
        source_id: str,
        frame: pd.DataFrame,
        *,
        available_at_col: str,
        assets: Mapping[str, Asset] | PreparedAssetLookup | None = None,
        pre_func: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        post_func: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        date_column: str = "date",
        date_format: str | None = None,
        timezone: str = "UTC",
        symbol: str | None = None,
        mask: bool = True,
        symbol_column: str = "symbol",
        country_code: str | None = None,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Load CSV bytes before preparation and pass a Pandas DataFrame")
        if not isinstance(mask, bool):
            raise TypeError("CSV mask must be Boolean")
        if symbol is not None and (not isinstance(symbol, str) or not symbol):
            raise ValueError("CSV research symbol must be a nonempty string")
        if symbol is None and assets is None:
            raise ValueError("Supply dated asset lookup or a fixed research symbol")
        prepared = frame.copy(deep=True)
        if pre_func is not None:
            prepared = pre_func(prepared)
        if not isinstance(prepared, pd.DataFrame) or not prepared.columns.is_unique:
            raise ValueError("CSV preprocessing must return a frame with unique columns")
        required = {date_column, available_at_col}
        if symbol is None:
            required.add(symbol_column)
        if required - set(prepared):
            raise ValueError(
                "CSV preparation is missing declared date, publication or symbol columns"
            )
        if "dt" in prepared and date_column != "dt":
            raise ValueError(
                "CSV dt is reserved for the parsed observation clock; rename the payload field"
            )
        if date_column == available_at_col or available_at_col in {
            "sid",
            "asset",
            "timestamp",
            "dt",
            symbol_column,
        }:
            raise ValueError("CSV publication and observation/identity columns must be distinct")
        clock = _instants(prepared[date_column], timezone, date_format)
        dropped_dates = int(clock.isna().sum())
        prepared = prepared.loc[~clock.isna()].copy()
        clock = clock[~clock.isna()]
        prepared["dt"] = clock
        missing = []

        def resolve(ticker, point):
            if symbol is not None:
                return symbol
            if not isinstance(ticker, str):
                return None
            if isinstance(assets, PreparedAssetLookup):
                try:
                    return assets.resolve(ticker, point, country_code=country_code)
                except ValueError:
                    return None
            assert isinstance(assets, Mapping)
            choices = [
                asset
                for name, asset in assets.items()
                if _symbol_parts(name) == _symbol_parts(ticker)
            ]
            if len(choices) > 1:
                raise ValueError("CSV symbol mapping is ambiguous")
            if not choices:
                return None
            if not isinstance(choices[0], Asset):
                raise TypeError("CSV symbol mapping values must be declared Assets")
            return choices[0]

        ids = []
        for _, row in prepared.iterrows():
            ticker = row.get(symbol_column)
            asset = resolve(ticker, row["dt"])
            if asset is None:
                missing.append(ticker)
                if not mask and (not isinstance(ticker, str) or not ticker):
                    raise ValueError("Unmasked CSV research identifiers must be nonempty strings")
                asset = None if mask else ticker
            ids.append(asset)
        prepared["sid"] = ids
        prepared = prepared.loc[prepared["sid"].notna()].copy()
        prepared = prepared.drop(
            columns=[
                *([date_column] if date_column != "dt" else []),
                *([symbol_column] if symbol is None else []),
            ]
        )
        prepared = prepared.set_index("dt").sort_index(kind="stable")
        if post_func is not None:
            prepared = post_func(prepared.copy(deep=True))
        if (
            not isinstance(prepared, pd.DataFrame)
            or not prepared.columns.is_unique
            or not isinstance(prepared.index, pd.DatetimeIndex)
            or prepared.index.tz is None
            or prepared.index.hasnans
            or {"sid", available_at_col} - set(prepared)
        ):
            raise ValueError(
                "CSV postprocessing requires aware observation index, sid and publication"
            )
        if any(
            not isinstance(key, Asset | str) or (isinstance(key, str) and not key)
            for key in prepared["sid"]
        ):
            raise TypeError(
                "CSV postprocessing must retain declared Assets or research identifiers"
            )
        self._df = prepared.copy(deep=True)
        self._df.index = pd.DatetimeIndex(self._df.index).tz_convert("UTC").as_unit("ns")
        payload = self._df.drop(columns="sid").copy(deep=True)
        if {"timestamp", "asset"} & set(payload):
            raise ValueError("CSV payload conflicts with native timestamp/asset keys")
        payload.insert(
            0,
            "asset",
            pd.Index(
                [key.key if isinstance(key, Asset) else key for key in prepared["sid"]],
                dtype=object,
            ),
        )
        payload.insert(0, "timestamp", self._df.index)
        payload[available_at_col] = _instants(payload[available_at_col], timezone)
        payload = payload.reset_index(drop=True)
        self._fields = tuple(
            name for name in payload if name not in {"timestamp", "asset", available_at_col}
        )
        if not self._fields:
            raise ValueError("CSV preparation requires at least one payload field")
        preparation = {
            "original_source_id": source_id,
            "transport": "caller_loaded_offline",
            "date_column": date_column,
            "date_format": date_format,
            "timezone": timezone,
            "date_policy": "physical_instant_no_implicit_session_roll",
            "publication_col": available_at_col,
            "pre_func": _callback_name(pre_func),
            "post_func": _callback_name(post_func),
            "symbol_column": symbol_column,
            "symbol": symbol,
            "country_code": country_code,
            "mask": mask,
            "unmapped_symbols": missing,
            "invalid_dates_dropped": dropped_dates,
            "symbol_lookup": assets.metadata
            if isinstance(assets, PreparedAssetLookup)
            else "explicit_static_mapping",
        }
        self._source = AuxiliaryData(
            source_id,
            _pandas_frame(payload),
            feed_spec={
                "time_contract_version": 2,
                "timestamp_semantics": "event_time",
                "calendar": None,
                "entity_col": "asset",
            },
            known_at_cols=dict.fromkeys(self._fields, available_at_col),
            preparation=preparation,
        )

    @property
    def source(self) -> AuxiliaryData:
        return self._source

    @property
    def fields(self) -> tuple[str, ...]:
        return self._fields

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy(deep=True)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._source.metadata
