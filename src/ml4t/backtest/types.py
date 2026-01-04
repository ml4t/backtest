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


class ExitReason(str, Enum):
    """Reason for trade exit - used for analysis and debugging.

    This enum is part of the cross-library API specification, designed to be
    identical across Python, Numba, and Rust implementations.
    """

    SIGNAL = "signal"  # Normal signal-based exit
    STOP_LOSS = "stop_loss"  # Stop-loss triggered
    TAKE_PROFIT = "take_profit"  # Take-profit triggered
    TRAILING_STOP = "trailing_stop"  # Trailing stop triggered
    TIME_STOP = "time_stop"  # Max hold time exceeded
    END_OF_DATA = "end_of_data"  # Backtest ended with open position


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
    rejection_reason: str | None = None  # Reason if order was rejected
    # Internal risk management fields (set by broker)
    _signal_price: float | None = None  # Close price at order creation time
    _risk_exit_reason: str | None = None  # Human-readable reason (legacy, for logging)
    _exit_reason: ExitReason | None = None  # Typed exit reason (preferred)
    _risk_fill_price: float | None = None  # Stop/target price for risk exits


@dataclass
class Position:
    """Unified position tracking for strategy and accounting.

    Supports both long and short positions with:
    - Weighted average cost basis tracking
    - Mark-to-market price tracking
    - Risk metrics (MFE/MAE, water marks)
    - Contract multipliers for futures

    Attributes:
        asset: Asset identifier (e.g., "AAPL", "ES")
        quantity: Position size (positive=long, negative=short)
        entry_price: Weighted average entry price (cost basis)
        entry_time: Timestamp when position was first opened
        current_price: Latest mark-to-market price (updated each bar)
        bars_held: Number of bars this position has been held

    Examples:
        Long position:
            Position("AAPL", 100, 150.0, datetime.now())
            -> quantity=100, unrealized_pnl depends on current_price

        Short position:
            Position("AAPL", -100, 150.0, datetime.now())
            -> quantity=-100, profit if price drops
    """

    asset: str
    quantity: float  # Positive for long, negative for short
    entry_price: float  # Weighted average cost basis
    entry_time: datetime
    current_price: float | None = None  # Mark-to-market price (set each bar)
    bars_held: int = 0
    # Risk tracking fields
    high_water_mark: float | None = None  # Highest price since entry (for longs)
    low_water_mark: float | None = None  # Lowest price since entry (for shorts)
    max_favorable_excursion: float = 0.0  # Best unrealized return seen
    max_adverse_excursion: float = 0.0  # Worst unrealized return seen
    initial_quantity: float | None = None  # Original size when opened
    context: dict = field(default_factory=dict)  # Strategy-provided context
    multiplier: float = 1.0  # Contract multiplier (for futures)
    entry_commission: float = 0.0  # Commission paid on entry (for Trade PnL)

    def __post_init__(self):
        # Initialize water marks to entry price
        if self.high_water_mark is None:
            self.high_water_mark = self.entry_price
        if self.low_water_mark is None:
            self.low_water_mark = self.entry_price
        if self.initial_quantity is None:
            self.initial_quantity = self.quantity
        if self.current_price is None:
            self.current_price = self.entry_price

    @property
    def avg_entry_price(self) -> float:
        """Alias for entry_price (accounting compatibility)."""
        return self.entry_price

    @property
    def market_value(self) -> float:
        """Current market value of the position.

        For long positions: positive value (asset on balance sheet)
        For short positions: negative value (liability on balance sheet)

        Returns:
            Market value = quantity × current_price
        """
        price = self.current_price if self.current_price is not None else self.entry_price
        return self.quantity * price * self.multiplier

    def unrealized_pnl(self, current_price: float | None = None) -> float:
        """Calculate unrealized P&L including contract multiplier.

        Args:
            current_price: Price to calculate P&L at. If None, uses self.current_price.

        Returns:
            Unrealized P&L = (current_price - entry_price) × quantity × multiplier
        """
        price = current_price if current_price is not None else self.current_price
        if price is None:
            price = self.entry_price
        return (price - self.entry_price) * self.quantity * self.multiplier

    def pnl_percent(self, current_price: float | None = None) -> float:
        """Calculate percentage return on position.

        Args:
            current_price: Price to calculate return at. If None, uses self.current_price.
        """
        price = current_price if current_price is not None else self.current_price
        if price is None:
            price = self.entry_price
        if self.entry_price == 0:
            return 0.0
        return (price - self.entry_price) / self.entry_price

    def notional_value(self, current_price: float | None = None) -> float:
        """Calculate notional value of position.

        Args:
            current_price: Price to calculate value at. If None, uses self.current_price.
        """
        price = current_price if current_price is not None else self.current_price
        if price is None:
            price = self.entry_price
        return abs(self.quantity) * price * self.multiplier

    def update_water_marks(
        self,
        current_price: float,
        bar_high: float | None = None,
        bar_low: float | None = None,
        use_high_for_hwm: bool = False,
    ) -> None:
        """Update high/low water marks and excursion tracking.

        Args:
            current_price: Current bar's close price
            bar_high: Bar's high price (used for HWM if use_high_for_hwm=True)
            bar_low: Bar's low price (used for LWM tracking if provided)
            use_high_for_hwm: If True, use bar_high for HWM (VBT Pro mode).
                              If False, use current_price (close) for HWM (default).
        """
        # Update current price
        self.current_price = current_price

        # Select HWM source based on configuration
        high_for_hwm = bar_high if use_high_for_hwm and bar_high is not None else current_price
        low_for_lwm = bar_low if bar_low is not None else current_price

        # Update water marks (guaranteed non-None after __post_init__)
        if self.high_water_mark is None or high_for_hwm > self.high_water_mark:
            self.high_water_mark = high_for_hwm
        if self.low_water_mark is None or low_for_lwm < self.low_water_mark:
            self.low_water_mark = low_for_lwm

        # Update MFE/MAE using bar extremes (more accurate than close only)
        # For longs: MFE from high, MAE from low
        # For shorts: MFE from low, MAE from high
        if self.quantity > 0:  # Long position
            mfe_return = self.pnl_percent(high_for_hwm)
            mae_return = self.pnl_percent(low_for_lwm)
        else:  # Short position
            mfe_return = self.pnl_percent(low_for_lwm)
            mae_return = self.pnl_percent(high_for_hwm)

        if mfe_return > self.max_favorable_excursion:
            self.max_favorable_excursion = mfe_return
        if mae_return < self.max_adverse_excursion:
            self.max_adverse_excursion = mae_return

    @property
    def side(self) -> str:
        """Return 'long' or 'short' based on quantity sign."""
        return "long" if self.quantity > 0 else "short"

    def __repr__(self) -> str:
        """String representation for debugging."""
        direction = "LONG" if self.quantity > 0 else "SHORT"
        price = self.current_price if self.current_price is not None else self.entry_price
        pnl = self.unrealized_pnl()
        return (
            f"Position({direction} {abs(self.quantity):.2f} {self.asset} "
            f"@ ${self.entry_price:.2f}, "
            f"current ${price:.2f}, "
            f"PnL ${pnl:+.2f})"
        )


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
    """Completed round-trip trade.

    This dataclass is part of the cross-library API specification, designed to
    produce identical Parquet output across Python, Numba, and Rust implementations.
    """

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
    # Exit reason for trade analysis (cross-library API field)
    exit_reason: str = "signal"  # ExitReason enum value as string
    # Deprecated: signals now handled via post-process join (enrich_trades_with_signals)
    # Kept for backward compatibility but not exported to DataFrame
    entry_signals: dict[str, float] = field(default_factory=dict)
    exit_signals: dict[str, float] = field(default_factory=dict)
    # MFE/MAE preserved from Position for trade analysis
    max_favorable_excursion: float = 0.0  # Best unrealized return during trade
    max_adverse_excursion: float = 0.0  # Worst unrealized return during trade

    @property
    def side(self) -> str:
        """Return 'long' or 'short' based on quantity sign."""
        return "long" if self.quantity > 0 else "short"
