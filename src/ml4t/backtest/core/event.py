"""Event system for ml4t.backtest."""

import logging
from abc import ABC
from datetime import datetime
from typing import Any

from ml4t.backtest.core.types import (
    AssetId,
    EventType,
    MarketDataType,
    OrderId,
    OrderSide,
    OrderType,
    Price,
    Quantity,
    TimeInForce,
    Volume,
)

logger = logging.getLogger(__name__)


class Event(ABC):
    """Base class for all events in the system."""

    def __init__(
        self,
        timestamp: datetime,
        event_type: EventType,
        metadata: dict[str, Any] | None = None,
    ):
        self.timestamp = timestamp
        self.event_type = event_type
        self.metadata = metadata or {}

    def __lt__(self, other: "Event") -> bool:
        """Compare events by timestamp for priority queue."""
        return self.timestamp < other.timestamp

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(timestamp={self.timestamp})"


class MarketEvent(Event):
    """Market data event (trade, quote, or bar).

    This event carries two types of auxiliary data beyond OHLCV prices:

    1. **signals**: Per-asset numerical features (ML predictions, technical indicators, etc.)
       - Examples: 'ml_score', 'rsi_14', 'atr_20', 'momentum_20', 'predicted_return'
       - Use case: All per-asset data for trading decisions and risk management
       - Populated by: ML model inference, feature engineering pipeline, or precomputed features
       - Note: Signals are just numbers - user code decides if they're for entry, exit, or sizing

    2. **context**: Market-wide context and regime data (same for all assets at timestamp)
       - Examples: 'vix', 'spy_return', 'market_regime', 'sector_performance'
       - Use case: Context-dependent strategy logic (VIX filtering, sector rotation)
       - Populated by: Market data providers or precomputed market features

    Both dictionaries are optional and default to empty dicts for backward compatibility.

    Attributes:
        timestamp: Event timestamp
        asset_id: Asset identifier
        data_type: Type of market data (BAR, TRADE, QUOTE)
        price: Last trade price (for TRADE events)
        size: Trade size (for TRADE events)
        bid_price: Best bid price (for QUOTE events)
        ask_price: Best ask price (for QUOTE events)
        bid_size: Bid size (for QUOTE events)
        ask_size: Ask size (for QUOTE events)
        open: Open price (for BAR events)
        high: High price (for BAR events)
        low: Low price (for BAR events)
        close: Close price (for BAR events)
        volume: Volume (for BAR events)
        signals: Per-asset numerical features (ML predictions, indicators, etc.)
        context: Market-wide context and regime data
        metadata: Additional event metadata

    Examples:
        >>> # ML strategy usage
        >>> ml_score = event.signals.get('ml_score', 0.0)
        >>> if ml_score > 0.7:
        ...     strategy.buy(event.asset_id)
        >>>
        >>> # Risk management using technical indicators
        >>> atr = event.signals.get('atr_20', 0.0)
        >>> stop_loss = entry_price - 2.0 * atr  # Volatility-scaled stop
        >>>
        >>> # Context-dependent logic
        >>> vix = event.context.get('vix', 0.0)
        >>> if vix > 30:
        ...     # High volatility regime - tighten stops or skip trades
        ...     pass
    """

    def __init__(
        self,
        timestamp: datetime,
        asset_id: AssetId,
        data_type: MarketDataType,
        price: Price | None = None,
        size: Quantity | None = None,
        bid_price: Price | None = None,
        ask_price: Price | None = None,
        bid_size: Quantity | None = None,
        ask_size: Quantity | None = None,
        open: Price | None = None,
        high: Price | None = None,
        low: Price | None = None,
        close: Price | None = None,
        volume: Volume | None = None,
        signals: dict[str, float] | None = None,
        context: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(timestamp, EventType.MARKET, metadata)
        self.asset_id = asset_id
        self.data_type = data_type
        self.price = price
        self.size = size
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.signals = signals or {}
        self.context = context or {}


class SignalEvent(Event):
    """ML model signal event."""

    def __init__(
        self,
        timestamp: datetime,
        asset_id: AssetId,
        signal_value: float,
        model_id: str,
        confidence: float | None = None,
        features: dict[str, Any] | None = None,
        ts_event: datetime | None = None,
        ts_arrival: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(timestamp, EventType.SIGNAL, metadata)
        self.asset_id = asset_id
        self.signal_value = signal_value
        self.model_id = model_id
        self.confidence = confidence
        self.features = features or {}
        self.ts_event = ts_event
        self.ts_arrival = ts_arrival or timestamp


class OrderEvent(Event):
    """Order submission event."""

    def __init__(
        self,
        timestamp: datetime,
        order_id: OrderId,
        asset_id: AssetId,
        order_type: OrderType,
        side: OrderSide,
        quantity: Quantity,
        limit_price: Price | None = None,
        stop_price: Price | None = None,
        time_in_force: TimeInForce = TimeInForce.DAY,
        parent_order_id: OrderId | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(timestamp, EventType.ORDER, metadata)
        self.order_id = order_id
        self.asset_id = asset_id
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.parent_order_id = parent_order_id


class FillEvent(Event):
    """Order fill/execution event."""

    def __init__(
        self,
        timestamp: datetime,
        order_id: OrderId,
        trade_id: str,
        asset_id: AssetId,
        side: OrderSide,
        fill_quantity: Quantity,
        fill_price: Price,
        commission: float = 0.0,
        slippage: float = 0.0,
        market_impact: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(timestamp, EventType.FILL, metadata)
        self.order_id = order_id
        self.trade_id = trade_id
        self.asset_id = asset_id
        self.side = side
        self.fill_quantity = fill_quantity
        self.fill_price = fill_price
        self.commission = commission
        self.slippage = slippage
        self.market_impact = market_impact

    @property
    def total_cost(self) -> float:
        """Total transaction cost including all fees."""
        return self.commission + self.slippage + self.market_impact


class CorporateActionEvent(Event):
    """Corporate action event (split, dividend, etc)."""

    def __init__(
        self,
        timestamp: datetime,
        asset_id: AssetId,
        action_type: str,
        ex_date: datetime,
        record_date: datetime | None = None,
        payment_date: datetime | None = None,
        adjustment_factor: float | None = None,
        dividend_amount: float | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(timestamp, EventType.CORPORATE_ACTION, metadata)
        self.asset_id = asset_id
        self.action_type = action_type
        self.ex_date = ex_date
        self.record_date = record_date
        self.payment_date = payment_date
        self.adjustment_factor = adjustment_factor
        self.dividend_amount = dividend_amount


