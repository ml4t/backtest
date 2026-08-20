"""Shared input preparation for real-strategy framework comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import polars as pl

from ml4t.backtest.calendar import filter_to_trading_days, filter_to_trading_sessions


def filter_comparison_market(
    market: pl.DataFrame,
    spec: Mapping[str, Any],
) -> pl.DataFrame:
    """Apply the production session contract before any compared engine runs."""
    calendar = spec["backtest_config"]["calendar"]
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
