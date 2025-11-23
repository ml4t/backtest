"""Core types for backtesting engine."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# === Enums ===


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ExecutionMode(Enum):
    """Order execution timing mode."""

    SAME_BAR = "same_bar"  # Orders fill at current bar's close (default)
    NEXT_BAR = "next_bar"  # Orders fill at next bar's open (like Backtrader)


# === Dataclasses ===


@dataclass
class Order:
    asset: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    trail_amount: float | None = None
    parent_id: str | None = None
    order_id: str = ""
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime | None = None
    filled_at: datetime | None = None
    filled_price: float | None = None
    filled_quantity: float = 0.0


@dataclass
class Position:
    asset: str
    quantity: float
    entry_price: float
    entry_time: datetime
    bars_held: int = 0
    # Risk tracking fields
    high_water_mark: float | None = None  # Highest price since entry (for longs)
    low_water_mark: float | None = None  # Lowest price since entry (for shorts)
    max_favorable_excursion: float = 0.0  # Best unrealized return seen
    max_adverse_excursion: float = 0.0  # Worst unrealized return seen
    initial_quantity: float | None = None  # Original size when opened
    context: dict = field(default_factory=dict)  # Strategy-provided context

    def __post_init__(self):
        # Initialize water marks to entry price
        if self.high_water_mark is None:
            self.high_water_mark = self.entry_price
        if self.low_water_mark is None:
            self.low_water_mark = self.entry_price
        if self.initial_quantity is None:
            self.initial_quantity = self.quantity

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.entry_price) * self.quantity

    def pnl_percent(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price

    def update_water_marks(self, current_price: float) -> None:
        """Update high/low water marks and excursion tracking."""
        # Update water marks
        if current_price > self.high_water_mark:
            self.high_water_mark = current_price
        if current_price < self.low_water_mark:
            self.low_water_mark = current_price

        # Update MFE/MAE
        current_return = self.pnl_percent(current_price)
        if current_return > self.max_favorable_excursion:
            self.max_favorable_excursion = current_return
        if current_return < self.max_adverse_excursion:
            self.max_adverse_excursion = current_return

    @property
    def side(self) -> str:
        """Return 'long' or 'short' based on quantity sign."""
        return "long" if self.quantity > 0 else "short"


@dataclass
class Fill:
    order_id: str
    asset: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Trade:
    """Completed round-trip trade."""

    asset: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    bars_held: int
    commission: float = 0.0
    slippage: float = 0.0
    entry_signals: dict[str, float] = field(default_factory=dict)
    exit_signals: dict[str, float] = field(default_factory=dict)
