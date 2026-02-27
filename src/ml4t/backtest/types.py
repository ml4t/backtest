"""Core types for backtesting engine."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

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


class ExecutionMode(str, Enum):
    """Order execution timing mode."""

    SAME_BAR = "same_bar"  # Orders fill at current bar's close (default)
    NEXT_BAR = "next_bar"  # Orders fill at next bar's open (like Backtrader)


class StopFillMode(str, Enum):
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


class StopLevelBasis(str, Enum):
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
        use_low_for_lwm: bool = False,
    ) -> None:
        """Update high/low water marks and excursion tracking.

        Args:
            current_price: Current bar's close price
            bar_high: Bar's high price (used for HWM if use_high_for_hwm=True)
            bar_low: Bar's low price (used for LWM if use_low_for_lwm=True)
            use_high_for_hwm: If True, use bar_high for HWM (VBT Pro OHLC mode).
                              If False, use current_price (close) for HWM (default).
            use_low_for_lwm: If True, use bar_low for LWM (VBT Pro OHLC mode).
                             If False, use current_price (close) for LWM (default).
        """
        # Update current price
        self.current_price = current_price

        # Select HWM source based on configuration
        high_for_hwm = bar_high if use_high_for_hwm and bar_high is not None else current_price
        low_for_lwm = bar_low if use_low_for_lwm and bar_low is not None else current_price

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
    """Round-trip trade (closed or open).

    This dataclass is part of the cross-library API specification, designed to
    produce identical Parquet output across Python, Numba, and Rust implementations.

    For open trades (status="open"), exit_time and exit_price represent
    mark-to-market values at the end of the backtest period.

    Schema Alignment (v0.1.0a6):
        - symbol: Asset identifier (was 'asset' in earlier versions)
        - fees: Total transaction fees (was 'commission')
        - mfe/mae: Max favorable/adverse excursion (was 'max_favorable_excursion'/'max_adverse_excursion')
        - direction: Derived property from quantity sign
    """

    symbol: str  # Asset identifier (aligned with ml4t-diagnostic TradeRecord)
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float  # Signed: positive=long, negative=short
    pnl: float
    pnl_percent: float
    bars_held: int
    fees: float = 0.0  # Total transaction fees (aligned with ml4t-diagnostic)
    slippage: float = 0.0
    # Exit reason for trade analysis (cross-library API field)
    exit_reason: str = "signal"  # ExitReason enum value as string
    # Trade status: "closed" (actually exited) or "open" (mark-to-market at end)
    status: str = "closed"
    # MFE/MAE preserved from Position for trade analysis (shorter field names)
    mfe: float = 0.0  # Max favorable excursion (best unrealized return)
    mae: float = 0.0  # Max adverse excursion (worst unrealized return)
    # Optional metadata extension point
    metadata: dict[str, Any] | None = None

    @property
    def direction(self) -> str:
        """Return 'long' or 'short' based on quantity sign."""
        return "long" if self.quantity > 0 else "short"

    @property
    def is_open(self) -> bool:
        """Return True if this is an open (mark-to-market) trade."""
        return self.status == "open"


@dataclass
class PartialExit:
    """Record of a partial position exit (scaling down).

    This dataclass tracks when a position is partially reduced, enabling
    strategies to access trade history during the backtest for stateful
    decision-making (e.g., adjusting position sizing based on recent wins/losses).

    Unlike Trade which represents a fully closed round-trip, PartialExit
    captures incremental reductions while the position remains open.
    """

    symbol: str  # Asset identifier
    timestamp: datetime  # When the partial exit occurred
    quantity: float  # Quantity exited (positive value)
    direction: str  # 'long' or 'short' (original position direction)
    entry_price: float  # Average entry price at time of exit
    exit_price: float  # Fill price of the exit
    pnl: float  # Realized P&L from this portion
    pnl_percent: float  # Return as decimal (0.05 = 5%)
    exit_reason: str = "partial_exit"  # Why the exit occurred
    fees: float = 0.0  # Transaction fees for this exit

    @property
    def is_win(self) -> bool:
        """Return True if this partial exit was profitable."""
        return self.pnl > 0


@dataclass
class AssetTradingStats:
    """Per-asset trading statistics for stateful decision-making.

    Tracks realized P&L events (both full closes and partial exits) to enable
    strategies to make decisions based on recent trading performance.

    This dataclass provides O(1) aggregate statistics plus O(N) recent history
    where N is the configurable window size.

    Example usage in strategy:
        def on_data(self, timestamp, data, context, broker):
            stats = broker.get_asset_stats("BTC")

            # Kelly criterion sizing based on recent win rate
            if stats.recent_win_rate > 0.6:
                size = 100  # Full size when winning
            elif stats.recent_win_rate < 0.4:
                size = 25   # Reduced size when losing
            else:
                size = 50   # Normal size

            # Stop trading after N consecutive losses
            if stats.total_trades > 0 and stats.recent_win_rate == 0:
                return  # Sit out until win streak improves
    """

    # All-time aggregates (O(1) memory)
    total_realized_pnl: float = 0.0
    total_trades: int = 0  # Count of P&L realizations (full close or partial exit)
    total_wins: int = 0

    # Recent window - last N P&L events (O(N) memory, N configurable)
    recent_pnls: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_wins: int = 0  # Count of wins in window

    # Current session (for intraday, reset at session boundary)
    session_pnl: float = 0.0
    session_trades: int = 0
    session_wins: int = 0
    session_id: int | None = None  # Current session identifier

    @property
    def win_rate(self) -> float:
        """All-time win rate as decimal (0.0 to 1.0).

        Returns 0.0 if no trades have been recorded.
        """
        if self.total_trades == 0:
            return 0.0
        return self.total_wins / self.total_trades

    @property
    def recent_win_rate(self) -> float:
        """Win rate for recent window as decimal (0.0 to 1.0).

        Returns 0.0 if no trades in recent window.
        """
        if len(self.recent_pnls) == 0:
            return 0.0
        return self.recent_wins / len(self.recent_pnls)

    @property
    def recent_expectancy(self) -> float:
        """Average P&L per trade in recent window.

        Returns 0.0 if no trades in recent window.
        """
        if len(self.recent_pnls) == 0:
            return 0.0
        return sum(self.recent_pnls) / len(self.recent_pnls)

    @property
    def recent_total_pnl(self) -> float:
        """Total P&L in recent window."""
        return sum(self.recent_pnls)

    @property
    def session_win_rate(self) -> float:
        """Win rate for current session as decimal (0.0 to 1.0).

        Returns 0.0 if no trades in current session.
        """
        if self.session_trades == 0:
            return 0.0
        return self.session_wins / self.session_trades

    @property
    def avg_pnl(self) -> float:
        """All-time average P&L per trade.

        Returns 0.0 if no trades have been recorded.
        """
        if self.total_trades == 0:
            return 0.0
        return self.total_realized_pnl / self.total_trades

    def record_pnl(self, pnl: float) -> None:
        """Record a P&L event (internal use by broker).

        Updates all-time aggregates and recent window. The recent window
        uses a circular buffer that automatically drops the oldest entry
        when full.

        Args:
            pnl: Realized P&L from a full close or partial exit
        """
        # Update all-time aggregates
        self.total_realized_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.total_wins += 1

        # Update recent window (circular buffer)
        # If buffer is full, the oldest entry will be dropped
        if len(self.recent_pnls) == self.recent_pnls.maxlen:
            # Oldest entry is about to be dropped - adjust recent_wins
            old_pnl = self.recent_pnls[0]
            if old_pnl > 0:
                self.recent_wins -= 1

        self.recent_pnls.append(pnl)
        if pnl > 0:
            self.recent_wins += 1

        # Update session stats
        self.session_pnl += pnl
        self.session_trades += 1
        if pnl > 0:
            self.session_wins += 1

    def reset_session(self, new_session_id: int | None = None) -> None:
        """Reset session statistics (called at session boundary).

        Args:
            new_session_id: Identifier for the new session (e.g., date ordinal)
        """
        self.session_pnl = 0.0
        self.session_trades = 0
        self.session_wins = 0
        self.session_id = new_session_id

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"AssetTradingStats("
            f"trades={self.total_trades}, "
            f"win_rate={self.win_rate:.1%}, "
            f"total_pnl=${self.total_realized_pnl:,.2f}, "
            f"recent_win_rate={self.recent_win_rate:.1%})"
        )
