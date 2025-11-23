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


class StopFillMode(Enum):
    """Stop/take-profit fill price mode.

    Different frameworks handle stop order fills differently:
    - STOP_PRICE: Fill at exact stop/target price (standard model, default)
                  Matches VectorBT Pro with OHLC and Backtrader behavior
    - CLOSE_PRICE: Fill at bar's close price when stop triggers
                   Matches VectorBT Pro with close-only data
    - BAR_EXTREME: Fill at bar's low (stop-loss) or high (take-profit)
                   Worst/best case model (conservative/optimistic)
    - NEXT_BAR_OPEN: Fill at next bar's open price when stop triggers
                     Matches Zipline behavior (strategy-level stops)
    """

    STOP_PRICE = "stop_price"  # Fill at exact stop/target price (default, VBT Pro OHLC, Backtrader)
    CLOSE_PRICE = "close_price"  # Fill at close price (VBT Pro close-only)
    BAR_EXTREME = "bar_extreme"  # Fill at bar's low/high (conservative/optimistic)
    NEXT_BAR_OPEN = "next_bar_open"  # Fill at next bar's open (Zipline)


class AssetClass(Enum):
    """Asset class for contract specification."""

    EQUITY = "equity"  # Stocks, ETFs (multiplier=1)
    FUTURE = "future"  # Futures contracts (multiplier varies)
    FOREX = "forex"  # FX pairs (pip value varies)


@dataclass
class ContractSpec:
    """Contract specification for an asset.

    Defines the characteristics that affect P&L calculation and margin:
    - Equities: multiplier=1, tick_size=0.01
    - Futures: multiplier varies (ES=$50, CL=$1000, etc.)
    - Forex: pip value varies by pair and account currency

    Example:
        # E-mini S&P 500 futures
        es_spec = ContractSpec(
            symbol="ES",
            asset_class=AssetClass.FUTURE,
            multiplier=50.0,      # $50 per point
            tick_size=0.25,       # Minimum move
            margin=15000.0,       # Initial margin per contract
        )

        # Apple stock
        aapl_spec = ContractSpec(
            symbol="AAPL",
            asset_class=AssetClass.EQUITY,
            # multiplier=1.0 (default)
            # tick_size=0.01 (default)
        )
    """

    symbol: str
    asset_class: AssetClass = AssetClass.EQUITY
    multiplier: float = 1.0  # Point value ($ per point move)
    tick_size: float = 0.01  # Minimum price increment
    margin: float | None = None  # Initial margin per contract (overrides account default)
    currency: str = "USD"


class StopLevelBasis(Enum):
    """Basis for calculating stop/take-profit levels.

    Different frameworks calculate stop levels from different reference prices:
    - FILL_PRICE: Calculate from actual entry fill price (ml4t default)
                  stop_level = fill_price * (1 - pct)
    - SIGNAL_PRICE: Calculate from signal close price at order time (Backtrader)
                    stop_level = signal_close * (1 - pct)

    In NEXT_BAR mode, fill_price is next bar's open while signal_price is
    current bar's close. This creates a small difference in stop levels.
    """

    FILL_PRICE = "fill_price"  # Use actual entry fill price (default)
    SIGNAL_PRICE = "signal_price"  # Use signal close price at order time (Backtrader)


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
    multiplier: float = 1.0  # Contract multiplier (for futures)

    def __post_init__(self):
        # Initialize water marks to entry price
        if self.high_water_mark is None:
            self.high_water_mark = self.entry_price
        if self.low_water_mark is None:
            self.low_water_mark = self.entry_price
        if self.initial_quantity is None:
            self.initial_quantity = self.quantity

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L including contract multiplier."""
        return (current_price - self.entry_price) * self.quantity * self.multiplier

    def pnl_percent(self, current_price: float) -> float:
        if self.entry_price == 0:
            return 0.0
        return (current_price - self.entry_price) / self.entry_price

    def notional_value(self, current_price: float) -> float:
        """Calculate notional value of position."""
        return abs(self.quantity) * current_price * self.multiplier

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
