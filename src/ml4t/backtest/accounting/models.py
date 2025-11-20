"""Data models for accounting system.

This module defines the core data structures used throughout the accounting system,
including the unified Position class that works for both cash and margin accounts.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """Unified position tracking for both long and short positions.

    This class replaces the simple Position class from engine.py and adds:
    - Weighted average cost basis tracking
    - Mark-to-market price tracking
    - Support for negative quantities (short positions)
    - Market value and unrealized P&L calculations

    Attributes:
        asset: Asset identifier (e.g., "AAPL", "BTC-USD")
        quantity: Position size (positive=long, negative=short)
        avg_entry_price: Weighted average entry price (cost basis)
        current_price: Latest mark-to-market price
        entry_time: Timestamp when position was first opened
        bars_held: Number of bars this position has been held

    Examples:
        Long position:
            Position("AAPL", 100, 150.0, 155.0, datetime.now(), 5)
            -> quantity=100, unrealized_pnl=+$500

        Short position:
            Position("AAPL", -100, 150.0, 145.0, datetime.now(), 3)
            -> quantity=-100, unrealized_pnl=+$500 (price dropped)
    """

    asset: str
    quantity: float  # Positive for long, negative for short
    avg_entry_price: float  # Weighted average cost basis
    current_price: float  # Last known market price for mark-to-market
    entry_time: datetime
    bars_held: int = 0

    @property
    def market_value(self) -> float:
        """Current market value of the position.

        For long positions: positive value (asset on balance sheet)
        For short positions: negative value (liability on balance sheet)

        Returns:
            Market value = quantity × current_price

        Examples:
            Long 100 @ $150: market_value = +$15,000
            Short 100 @ $150: market_value = -$15,000
        """
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss on the position.

        Calculated as the difference between current price and average entry
        price, multiplied by quantity. Works correctly for both long and short.

        Returns:
            Unrealized P&L = (current_price - avg_entry_price) × quantity

        Examples:
            Long 100, bought @ $150, now @ $155:
                unrealized_pnl = (155 - 150) × 100 = +$500

            Short 100, sold @ $150, now @ $145:
                unrealized_pnl = (145 - 150) × (-100) = +$500

            Short 100, sold @ $150, now @ $155:
                unrealized_pnl = (155 - 150) × (-100) = -$500
        """
        return (self.current_price - self.avg_entry_price) * self.quantity

    def __repr__(self) -> str:
        """String representation for debugging."""
        direction = "LONG" if self.quantity > 0 else "SHORT"
        return (
            f"Position({direction} {abs(self.quantity):.2f} {self.asset} "
            f"@ ${self.avg_entry_price:.2f}, "
            f"current ${self.current_price:.2f}, "
            f"PnL ${self.unrealized_pnl:+.2f})"
        )
