"""Shared input preparation for real-strategy framework comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ml4t.backtest.calendar import filter_to_trading_days, filter_to_trading_sessions

FX_COMPARISON_SCOPE = (
    "USD-quoted pairs from the frozen target stream, preserving native USD valuation in every "
    "required engine"
)


def comparison_scope(spec: Mapping[str, Any]) -> str:
    """Describe the engine input projection applied to a frozen bundle."""
    case_study = spec["backtest_config"].get("metadata", {}).get("case_study")
    return FX_COMPARISON_SCOPE if case_study == "fx_pairs" else "complete frozen bundle"


def _filter_comparison_universe(
    frame: pl.DataFrame,
    spec: Mapping[str, Any],
) -> pl.DataFrame:
    if comparison_scope(spec) == FX_COMPARISON_SCOPE:
        return frame.filter(pl.col("symbol").str.ends_with("_USD"))
    return frame


def filter_comparison_market(
    market: pl.DataFrame,
    spec: Mapping[str, Any],
) -> pl.DataFrame:
    """Apply the production session contract before any compared engine runs."""
    calendar = spec["backtest_config"]["calendar"]
    market = _filter_comparison_universe(market, spec)
    if "close" in market.columns:
        market = market.drop_nulls("close").with_columns(
            pl.coalesce(pl.col(column), pl.col("close")).alias(column)
            for column in ("open", "high", "low")
            if column in market.columns
        )
    if "volume" in market.columns:
        market = market.with_columns(pl.col("volume").fill_null(0.0))
    if not bool(calendar.get("enforce_sessions", False)):
        return market
    calendar_id = str(calendar["calendar"])
    if str(calendar["data_frequency"]).lower() == "daily":
        return filter_to_trading_days(market, calendar_id)
    return filter_to_trading_sessions(
        market,
        calendar_id,
        naive_tz=str(calendar.get("timezone", "UTC")),
    )


def filter_comparison_targets(
    targets: pl.DataFrame,
    spec: Mapping[str, Any],
) -> pl.DataFrame:
    """Apply the same native-asset comparison universe to frozen targets."""
    return _filter_comparison_universe(targets, spec)
