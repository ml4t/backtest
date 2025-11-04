"""Core type definitions for QEngine."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import NewType, Union

# Time types
Timestamp = NewType("Timestamp", datetime)
Nanoseconds = NewType("Nanoseconds", int)

# Market data types
AssetId = NewType("AssetId", str)
Price = Union[float, Decimal]
Quantity = Union[float, int]
Volume = Union[float, int]

# Order types
OrderId = NewType("OrderId", str)
TradeId = NewType("TradeId", str)
PositionId = NewType("PositionId", str)

# Portfolio types
Cash = Union[float, Decimal]
Currency = NewType("Currency", str)


class EventType(Enum):
    """Types of events in the system."""

    MARKET = "market"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"
    CORPORATE_ACTION = "corporate_action"
    TIMER = "timer"
    CUSTOM = "custom"


class OrderType(Enum):
    """Types of orders."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"
    OCO = "oco"


class OrderStatus(Enum):
    """Order status lifecycle."""

    CREATED = "created"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side (buy/sell)."""

    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class TimeInForce(Enum):
    """Time-in-force constraints for orders."""

    DAY = "day"  # Valid for the day
    GTC = "gtc"  # Good till canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTD = "gtd"  # Good till date
    MOC = "moc"  # Market on close
    MOO = "moo"  # Market on open


class AssetType(Enum):
    """Types of tradeable assets."""

    EQUITY = "equity"
    FUTURE = "future"
    OPTION = "option"
    FOREX = "forex"
    CRYPTO = "crypto"
    BOND = "bond"
    COMMODITY = "commodity"
    INDEX = "index"


class BarType(Enum):
    """Types of price bars."""

    TICK = "tick"
    TIME = "time"
    VOLUME = "volume"
    DOLLAR = "dollar"
    TICK_IMBALANCE = "tick_imbalance"
    VOLUME_IMBALANCE = "volume_imbalance"


class MarketDataType(Enum):
    """Types of market data."""

    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"
    ORDERBOOK = "orderbook"
