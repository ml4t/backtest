"""Efficient trade tracking for QEngine backtests.

Tracks entry/exit fills and constructs complete trade records with minimal overhead.
Designed for high-performance backtesting with large numbers of trades.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import polars as pl

from qengine.core.event import FillEvent
from qengine.core.precision import PrecisionManager
from qengine.core.types import AssetId, OrderSide


@dataclass
class TradeRecord:
    """Single completed trade record (entry + exit)."""

    # Trade identification
    trade_id: int
    asset_id: AssetId

    # Entry details
    entry_dt: datetime
    entry_price: float
    entry_quantity: float
    entry_commission: float
    entry_slippage: float
    entry_order_id: str

    # Exit details
    exit_dt: datetime
    exit_price: float
    exit_quantity: float
    exit_commission: float
    exit_slippage: float
    exit_order_id: str

    # Trade metrics
    pnl: float
    return_pct: float
    duration_bars: int
    direction: str  # "long" or "short"

    # Optional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame construction."""
        return {
            "trade_id": self.trade_id,
            "asset_id": self.asset_id,
            "entry_dt": self.entry_dt,
            "entry_price": self.entry_price,
            "entry_quantity": self.entry_quantity,
            "entry_commission": self.entry_commission,
            "entry_slippage": self.entry_slippage,
            "entry_order_id": self.entry_order_id,
            "exit_dt": self.exit_dt,
            "exit_price": self.exit_price,
            "exit_quantity": self.exit_quantity,
            "exit_commission": self.exit_commission,
            "exit_slippage": self.exit_slippage,
            "exit_order_id": self.exit_order_id,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "duration_bars": self.duration_bars,
            "direction": self.direction,
        }


@dataclass
class OpenPosition:
    """Tracks an open position waiting for exit."""

    asset_id: AssetId
    entry_dt: datetime
    entry_price: float
    quantity: float
    entry_commission: float
    entry_slippage: float
    entry_order_id: str
    direction: str
    entry_bar_idx: int


class TradeTracker:
    """
    Efficient trade tracker that builds complete trade records from fills.

    Uses FIFO (First-In-First-Out) matching to pair entries with exits.
    Designed for minimal overhead during backtest execution.

    Performance characteristics:
    - O(1) fill processing
    - O(1) amortized trade completion
    - Minimal memory allocation (reuses structures)
    - Lazy DataFrame construction (only when requested)
    """

    def __init__(self, precision_manager: Optional[PrecisionManager] = None):
        """
        Initialize trade tracker.

        Args:
            precision_manager: PrecisionManager for cash rounding (USD precision for P&L)
        """
        self.precision_manager = precision_manager

        # Open positions by asset (FIFO queue)
        self._open_positions: dict[AssetId, list[OpenPosition]] = {}

        # Completed trades (raw records)
        self._trades: list[TradeRecord] = []

        # Trade ID counter
        self._next_trade_id = 0

        # Bar index for duration calculation
        self._current_bar_idx = 0

        # Performance stats
        self._total_fills_processed = 0
        self._total_trades_completed = 0

    def on_bar(self) -> None:
        """Increment bar counter for duration tracking."""
        self._current_bar_idx += 1

    def on_fill(self, fill_event: FillEvent) -> list[TradeRecord]:
        """
        Process a fill event and generate completed trades if applicable.

        Args:
            fill_event: Fill event from broker

        Returns:
            List of completed trade records (empty if position still open)
        """
        self._total_fills_processed += 1
        completed_trades = []

        asset_id = fill_event.asset_id
        quantity = fill_event.fill_quantity
        is_buy = fill_event.side == OrderSide.BUY

        # Get or create position queue for this asset
        if asset_id not in self._open_positions:
            self._open_positions[asset_id] = []

        position_queue = self._open_positions[asset_id]

        # Determine if this is opening or closing
        if not position_queue:
            # Opening new position
            self._open_new_position(fill_event)
        else:
            # Check if this closes existing positions
            existing_position = position_queue[0]
            is_long = existing_position.direction == "long"
            is_closing = (is_long and not is_buy) or (not is_long and is_buy)

            if is_closing:
                # Close position(s) with FIFO matching
                remaining = abs(quantity)
                while remaining > 0 and position_queue:
                    position = position_queue[0]
                    close_qty = min(remaining, position.quantity)

                    # Create completed trade
                    trade = self._close_position(position, fill_event, close_qty)
                    completed_trades.append(trade)

                    # Update position
                    position.quantity -= close_qty
                    remaining -= close_qty

                    # Check if fully closed using precision-aware check (Location 1/12)
                    is_closed = position.quantity <= 1e-9
                    if self.precision_manager:
                        is_closed = self.precision_manager.is_position_zero(position.quantity)
                    if is_closed:
                        position_queue.pop(0)

                # If still have quantity left, opening new reverse position
                # Use precision-aware check (Location 2/12)
                has_remaining = remaining > 1e-9
                if self.precision_manager:
                    has_remaining = not self.precision_manager.is_position_zero(remaining)
                if has_remaining:
                    # Create new fill event for remaining quantity
                    self._open_new_position(fill_event, remaining)
            else:
                # Adding to existing position
                self._open_new_position(fill_event)

        self._total_trades_completed += len(completed_trades)
        return completed_trades

    def _open_new_position(self, fill_event: FillEvent, quantity: float | None = None) -> None:
        """Open a new position from a fill."""
        asset_id = fill_event.asset_id
        qty = quantity if quantity is not None else abs(fill_event.fill_quantity)

        position = OpenPosition(
            asset_id=asset_id,
            entry_dt=fill_event.timestamp,
            entry_price=fill_event.fill_price,
            quantity=qty,
            entry_commission=fill_event.commission,
            entry_slippage=fill_event.slippage,
            entry_order_id=str(fill_event.order_id),
            direction="long" if fill_event.side == OrderSide.BUY else "short",
            entry_bar_idx=self._current_bar_idx,
        )

        self._open_positions[asset_id].append(position)

    def _close_position(
        self, position: OpenPosition, exit_fill: FillEvent, close_quantity: float
    ) -> TradeRecord:
        """Close a position and create trade record."""
        # Calculate P&L
        if position.direction == "long":
            gross_pnl = close_quantity * (exit_fill.fill_price - position.entry_price)
        else:
            gross_pnl = close_quantity * (position.entry_price - exit_fill.fill_price)

        # Round gross P&L to avoid float drift (Location 3/12)
        if self.precision_manager:
            gross_pnl = self.precision_manager.round_cash(gross_pnl)

        # Subtract costs (proportional to quantity closed)
        qty_fraction = close_quantity / position.quantity
        entry_costs = (position.entry_commission + position.entry_slippage) * qty_fraction
        # Round entry costs (Location 4/12)
        if self.precision_manager:
            entry_costs = self.precision_manager.round_cash(entry_costs)

        exit_costs = exit_fill.commission + exit_fill.slippage
        # Round exit costs (Location 5/12)
        if self.precision_manager:
            exit_costs = self.precision_manager.round_cash(exit_costs)

        net_pnl = gross_pnl - entry_costs - exit_costs
        # Round net P&L to avoid float drift (Location 6/12)
        if self.precision_manager:
            net_pnl = self.precision_manager.round_cash(net_pnl)

        # Calculate return percentage
        capital_at_risk = close_quantity * position.entry_price
        return_pct = (net_pnl / capital_at_risk * 100) if capital_at_risk > 0 else 0.0
        # Round return percentage (Location 7/12)
        if self.precision_manager:
            return_pct = self.precision_manager.round_cash(return_pct)

        # Duration
        duration_bars = self._current_bar_idx - position.entry_bar_idx

        # Calculate proportional costs for trade record
        proportional_entry_commission = position.entry_commission * qty_fraction
        proportional_entry_slippage = position.entry_slippage * qty_fraction
        # Round proportional costs (Locations 8-9/12)
        if self.precision_manager:
            proportional_entry_commission = self.precision_manager.round_cash(proportional_entry_commission)
            proportional_entry_slippage = self.precision_manager.round_cash(proportional_entry_slippage)

        # Create trade record
        trade = TradeRecord(
            trade_id=self._next_trade_id,
            asset_id=position.asset_id,
            entry_dt=position.entry_dt,
            entry_price=position.entry_price,
            entry_quantity=close_quantity,
            entry_commission=proportional_entry_commission,
            entry_slippage=proportional_entry_slippage,
            entry_order_id=position.entry_order_id,
            exit_dt=exit_fill.timestamp,
            exit_price=exit_fill.fill_price,
            exit_quantity=close_quantity,
            exit_commission=exit_fill.commission,
            exit_slippage=exit_fill.slippage,
            exit_order_id=str(exit_fill.order_id),
            pnl=net_pnl,
            return_pct=return_pct,
            duration_bars=duration_bars,
            direction=position.direction,
        )

        self._next_trade_id += 1
        self._trades.append(trade)
        return trade

    def get_trades_df(self) -> pl.DataFrame:
        """
        Get all completed trades as a Polars DataFrame.

        Column names follow snake_case convention (not PascalCase like VectorBT).

        Returns:
            Polars DataFrame with trade records
        """
        if not self._trades:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(
                schema={
                    "trade_id": pl.Int64,
                    "asset_id": pl.Utf8,
                    "entry_dt": pl.Datetime,
                    "entry_price": pl.Float64,
                    "entry_quantity": pl.Float64,
                    "entry_commission": pl.Float64,
                    "entry_slippage": pl.Float64,
                    "entry_order_id": pl.Utf8,
                    "exit_dt": pl.Datetime,
                    "exit_price": pl.Float64,
                    "exit_quantity": pl.Float64,
                    "exit_commission": pl.Float64,
                    "exit_slippage": pl.Float64,
                    "exit_order_id": pl.Utf8,
                    "pnl": pl.Float64,
                    "return_pct": pl.Float64,
                    "duration_bars": pl.Int64,
                    "direction": pl.Utf8,
                }
            )

        # Convert to dictionaries
        trade_dicts = [t.to_dict() for t in self._trades]

        # Create DataFrame (Polars is highly efficient at this)
        return pl.DataFrame(trade_dicts)

    def get_trade_count(self) -> int:
        """Get number of completed trades."""
        return len(self._trades)

    def get_open_position_count(self) -> int:
        """Get number of currently open positions."""
        return sum(len(positions) for positions in self._open_positions.values())

    def get_open_positions_as_trades(self, current_timestamp: datetime, current_price: float) -> list[TradeRecord]:
        """
        Convert currently open positions to trade records (for end-of-backtest reporting).

        This is useful for validation/comparison with other engines that report
        open positions as "trades" with the exit being the end of the backtest.

        Args:
            current_timestamp: Current/final timestamp to use as exit_dt
            current_price: Current/final price to use for exit_price

        Returns:
            List of trade records for open positions (exit = current timestamp/price)
        """
        open_trades = []

        for asset_id, positions in self._open_positions.items():
            for position in positions:
                # Calculate theoretical P&L if closed at current price
                if position.direction == "long":
                    gross_pnl = position.quantity * (current_price - position.entry_price)
                else:
                    gross_pnl = position.quantity * (position.entry_price - current_price)

                # Round gross P&L (Location 10/12)
                if self.precision_manager:
                    gross_pnl = self.precision_manager.round_cash(gross_pnl)

                # No exit costs since not actually closed
                net_pnl = gross_pnl - (position.entry_commission + position.entry_slippage)
                # Round net P&L (Location 11/12)
                if self.precision_manager:
                    net_pnl = self.precision_manager.round_cash(net_pnl)

                # Calculate return percentage
                capital_at_risk = position.quantity * position.entry_price
                return_pct = (net_pnl / capital_at_risk * 100) if capital_at_risk > 0 else 0.0
                # Round return percentage (Location 12/12)
                if self.precision_manager:
                    return_pct = self.precision_manager.round_cash(return_pct)

                # Duration
                duration_bars = self._current_bar_idx - position.entry_bar_idx

                # Create pseudo-trade record
                trade = TradeRecord(
                    trade_id=self._next_trade_id + len(open_trades),
                    asset_id=asset_id,
                    entry_dt=position.entry_dt,
                    entry_price=position.entry_price,
                    entry_quantity=position.quantity,
                    entry_commission=position.entry_commission,
                    entry_slippage=position.entry_slippage,
                    entry_order_id=position.entry_order_id,
                    exit_dt=current_timestamp,  # Current/final timestamp
                    exit_price=current_price,  # Current/final price
                    exit_quantity=position.quantity,
                    exit_commission=0.0,  # No exit commission (not actually closed)
                    exit_slippage=0.0,  # No exit slippage (not actually closed)
                    exit_order_id="OPEN",  # Mark as open position
                    pnl=net_pnl,
                    return_pct=return_pct,
                    duration_bars=duration_bars,
                    direction=position.direction,
                )

                open_trades.append(trade)

        return open_trades

    def get_stats(self) -> dict[str, Any]:
        """Get tracker statistics."""
        return {
            "total_fills_processed": self._total_fills_processed,
            "total_trades_completed": self._total_trades_completed,
            "open_positions": self.get_open_position_count(),
            "avg_fills_per_trade": (
                self._total_fills_processed / self._total_trades_completed
                if self._total_trades_completed > 0
                else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self._open_positions.clear()
        self._trades.clear()
        self._next_trade_id = 0
        self._current_bar_idx = 0
        self._total_fills_processed = 0
        self._total_trades_completed = 0
