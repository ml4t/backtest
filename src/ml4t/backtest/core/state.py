"""State owners shared by Broker orchestration components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..types import Fill, Order, Trade


@dataclass(slots=True)
class MarketState:
    """Canonical point-in-time market data for the current bar."""

    time: datetime | None = None
    prices: dict[str, float] = field(default_factory=dict)
    opens: dict[str, float] = field(default_factory=dict)
    highs: dict[str, float] = field(default_factory=dict)
    lows: dict[str, float] = field(default_factory=dict)
    closes: dict[str, float] = field(default_factory=dict)
    volumes: dict[str, float] = field(default_factory=dict)
    bids: dict[str, float] = field(default_factory=dict)
    asks: dict[str, float] = field(default_factory=dict)
    mids: dict[str, float] = field(default_factory=dict)
    bid_sizes: dict[str, float] = field(default_factory=dict)
    ask_sizes: dict[str, float] = field(default_factory=dict)
    signals: dict[str, dict[str, float]] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)
    asset_bars_seen: dict[str, int] = field(default_factory=dict)
    bar_index: int = 0


@dataclass(slots=True)
class OrderState:
    """Canonical order lifecycle and partial-fill state."""

    orders: list[Order] = field(default_factory=list)
    pending: list[Order] = field(default_factory=list)
    counter: int = 0
    current_bar: list[Order] = field(default_factory=list)
    current_bar_ids: set[str] = field(default_factory=set)
    partial_quantities: dict[str, float] = field(default_factory=dict)
    filled_this_bar: set[str] = field(default_factory=set)


@dataclass(slots=True)
class RiskState:
    """Canonical position-rule configuration and per-bar risk state."""

    position_rules: Any = None
    position_rules_by_asset: dict[str, Any] = field(default_factory=dict)
    pending_exits: dict[str, dict[str, Any]] = field(default_factory=dict)
    stop_exits_this_bar: set[str] = field(default_factory=set)
    positions_created_this_bar: set[str] = field(default_factory=set)


@dataclass(slots=True)
class ExecutionJournal:
    """Canonical append-only fill and trade records."""

    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
