"""State owners shared by Broker orchestration components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ..controls import AccountControlContext
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
    vwaps: dict[str, float] = field(default_factory=dict)
    bids: dict[str, float] = field(default_factory=dict)
    asks: dict[str, float] = field(default_factory=dict)
    mids: dict[str, float] = field(default_factory=dict)
    bid_sizes: dict[str, float] = field(default_factory=dict)
    ask_sizes: dict[str, float] = field(default_factory=dict)
    signals: dict[str, dict[str, float]] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)
    last_price_times: dict[str, datetime] | None = None
    asset_bars_seen: dict[str, int] = field(default_factory=dict)
    bar_index: int = 0
    observed_high_low: Callable[[str], tuple[float | None, float | None]] | None = None


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
    volume_consumed_by_asset: dict[str, float] = field(default_factory=dict)
    control_counts: dict[tuple[str, date], int] = field(default_factory=dict)


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
    recorded_values: dict[str, str] | None = None
    recorded_rows: list[tuple[datetime, int, str, str]] | None = None
    corporate_receivable_snapshots: dict[tuple[datetime, int | None], float] | None = None
    completed_corporate_entitlements: list[dict[str, Any]] | None = None
    futures_settlements: list[dict[str, Any]] | None = None
    futures_roll_intents: list[dict[str, Any]] | None = None
    accounting_snapshots: dict[tuple[datetime, int | None], AccountControlContext] | None = None
    fill_snapshots: (
        list[tuple[int, datetime, tuple[float, float, float, float, int], list[tuple[Any, ...]]]]
        | None
    ) = None
