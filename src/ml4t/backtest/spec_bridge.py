"""Pure conversions from artifact specifications into runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ml4t.specs.market_data import FeedSpec, MarketDataSpec


def market_data_spec_to_feed_spec(spec: MarketDataSpec | Mapping[str, Any]) -> FeedSpec:
    """Project a market-data spec onto the backtest runtime feed contract."""
    return FeedSpec.from_any(spec)


def market_data_spec_to_runtime_metadata(
    spec: MarketDataSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Extract runtime scheduling and annualization metadata from a market-data spec."""
    feed_spec = market_data_spec_to_feed_spec(spec)
    return {
        "calendar": feed_spec.calendar,
        "timezone": feed_spec.timezone,
        "data_frequency": feed_spec.data_frequency,
        "timestamp_semantics": feed_spec.timestamp_semantics,
        "session_start_time": feed_spec.session_start_time,
        "bar_type": feed_spec.bar_type,
    }


__all__ = ["market_data_spec_to_feed_spec", "market_data_spec_to_runtime_metadata"]
