"""Pandas presentation of observed native accounts, fills and recorded variables."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import pandas as pd

from ..._temporal import temporal_rows, timestamp_values
from ...result import BacktestResult
from .assets import Asset, Future


def to_performance_frame(
    result: BacktestResult, *, assets: Sequence[Asset] | None = None
) -> pd.DataFrame:
    """Project a native result without rerunning decisions or reconstructing holdings.

    Migration runs retain their declared asset identifiers and scoped order prefix
    in result metadata. Other native runs use stable keys with unknown display
    symbols unless explicit assets are supplied. Rows retain native observation
    frequency; this function does not aggregate minute rows into daily packets.
    """
    metadata = result.config.metadata if result.config is not None else {}
    declaration = metadata.get("migration_result", {})
    if assets is None:
        declared = declaration.get("assets")
        assets = (
            [(Future(**row) if "root_symbol" in row else Asset(**row)) for row in declared]
            if declared is not None
            else [Asset(key) for key in sorted({fill.asset for fill in result.fills})]
        )
    if any(not isinstance(asset, Asset) for asset in assets):
        raise TypeError("assets must contain migration Assets")
    by_key = {asset.key: asset for asset in assets}
    if len(by_key) != len(assets):
        raise ValueError("Duplicate native asset keys in performance mapping")
    sids = [asset.legacy_sid for asset in assets if asset.legacy_sid is not None]
    if len(set(sids)) != len(sids):
        raise ValueError("Ambiguous legacy SID in performance mapping")
    missing = {fill.asset for fill in result.fills} - by_key.keys()
    if missing:
        raise ValueError(f"Performance asset mapping is missing {sorted(missing)}")
    prefix = declaration.get("order_id_prefix")
    return _performance(
        result, by_key, lambda order_id: f"{prefix}:{order_id}" if prefix else order_id
    )


def _performance(
    result, assets: Mapping[str, Asset], order_id: Callable[[str], str]
) -> pd.DataFrame:
    frame = result.to_portfolio_state_dataframe()
    rows = list(temporal_rows(frame, named=True))
    index = pd.DatetimeIndex([row["timestamp"] for row in rows])
    if index.tz is None:
        index = index.tz_localize(result.config.resolved_timezone)
    performance = pd.DataFrame(
        {
            "portfolio_value": [row["equity"] for row in rows],
            "cash": [row["cash"] for row in rows],
            "positions_value": [row.get("positions_value", row["net_exposure"]) for row in rows],
            "returns": result.to_returns_series().to_list(),
        },
        index=index.tz_convert("UTC").as_unit("ns"),
    )
    if result.instrument_accounting is not None:
        for name in (
            "initial_margin_required",
            "maintenance_margin_required",
            "available_funds",
            "futures_unsettled_pnl",
            "settlement_holds",
            "cash_buffer",
            "margin_breach",
        ):
            performance[name] = [row[name] for row in rows]
    if result.benchmark is not None:
        benchmark = result.to_benchmark_dataframe()
        benchmark_index = pd.DatetimeIndex(timestamp_values(benchmark["timestamp"])).as_unit("ns")
        period_returns = pd.Series(benchmark["benchmark_return"].to_list(), index=benchmark_index)
        performance["benchmark_return"] = period_returns.reindex(performance.index)
        performance["benchmark_period_return"] = ((1 + period_returns).cumprod() - 1).reindex(
            performance.index
        )
        performance.attrs["benchmark_metadata"] = deepcopy(result.benchmark_metadata)
    transactions: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for fill in result.fills:
        timestamp = pd.Timestamp(fill.timestamp)
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize(result.config.resolved_timezone)
        timestamp = timestamp.tz_convert("UTC")
        transactions.setdefault(timestamp, []).append(
            {
                "sid": assets[fill.asset],
                "amount": fill.quantity * (1 if fill.side.value == "buy" else -1),
                "dt": timestamp,
                "price": fill.price,
                "order_id": order_id(fill.order_id),
                "native_order_id": fill.order_id,
                "commission": fill.commission,
            }
        )
    performance["transactions"] = [
        transactions.get(timestamp, []) for timestamp in performance.index
    ]
    if result.recorded_variables is not None:
        records = result.to_recorded_records()
        by_observation = {row["bar_index"]: row["values"] for row in records}
        names = set().union(*(row["values"] for row in records)) if records else set()
        for name in sorted(names):
            if name in performance.columns:
                raise ValueError(f"Recorded name {name!r} conflicts with a native account field")
            performance[name] = [
                by_observation.get(row["observation_id"], {}).get(name) for row in rows
            ]
    performance.attrs["backtest_result"] = result
    return performance
