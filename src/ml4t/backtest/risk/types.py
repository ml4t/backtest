"""Core types for risk management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class ActionType(Enum):
    """Types of actions a position rule can return."""

    HOLD = auto()  # Do nothing
    EXIT_FULL = auto()  # Close entire position
    EXIT_PARTIAL = auto()  # Close portion of position
    ADJUST_STOP = auto()  # Move stop price


@dataclass
class PositionAction:
    """Action returned by a position rule.

    Attributes:
        action: Type of action to take
        pct: Percentage to exit (for EXIT_PARTIAL), 0-1
        stop_price: New stop price (for ADJUST_STOP)
        reason: Human-readable reason for action (for logging)
    """

    action: ActionType
    pct: float = 1.0
    stop_price: float | None = None
    reason: str = ""

    @classmethod
    def hold(cls) -> "PositionAction":
        """Convenience: return HOLD action."""
        return cls(ActionType.HOLD)

    @classmethod
    def exit_full(cls, reason: str = "") -> "PositionAction":
        """Convenience: return EXIT_FULL action."""
        return cls(ActionType.EXIT_FULL, reason=reason)

    @classmethod
    def exit_partial(cls, pct: float, reason: str = "") -> "PositionAction":
        """Convenience: return EXIT_PARTIAL action."""
        return cls(ActionType.EXIT_PARTIAL, pct=pct, reason=reason)

    @classmethod
    def adjust_stop(cls, price: float, reason: str = "") -> "PositionAction":
        """Convenience: return ADJUST_STOP action."""
        return cls(ActionType.ADJUST_STOP, stop_price=price, reason=reason)


@dataclass
class PositionState:
    """Current state of a position for rule evaluation.

    This dataclass provides all the information rules need to make decisions.
    All monetary values are in the position's currency.

    Attributes:
        asset: Asset symbol
        side: "long" or "short"
        entry_price: Average entry price
        current_price: Current market price
        quantity: Current position size (absolute)
        initial_quantity: Original position size when opened
        unrealized_pnl: Current unrealized P&L in currency
        unrealized_return: Current unrealized return as decimal (0.05 = 5%)
        bars_held: Number of bars since position opened
        high_water_mark: Highest price since entry (for longs)
        low_water_mark: Lowest price since entry (for shorts)
        max_favorable_excursion: Best unrealized return seen
        max_adverse_excursion: Worst unrealized return seen
        entry_time: When position was opened
        current_time: Current timestamp
        context: Optional strategy-provided context (signals, indicators, etc.)
    """

    asset: str
    side: str  # "long" or "short"
    entry_price: float
    current_price: float
    quantity: int
    initial_quantity: int
    unrealized_pnl: float
    unrealized_return: float
    bars_held: int
    high_water_mark: float
    low_water_mark: float
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    entry_time: datetime | None = None
    current_time: datetime | None = None
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def is_long(self) -> bool:
        """True if long position."""
        return self.side == "long"

    @property
    def is_short(self) -> bool:
        """True if short position."""
        return self.side == "short"

    @property
    def is_profitable(self) -> bool:
        """True if position is currently profitable."""
        return self.unrealized_return > 0

    @property
    def drawdown_from_peak(self) -> float:
        """Current drawdown from max favorable excursion."""
        if self.max_favorable_excursion <= 0:
            return 0.0
        return (self.max_favorable_excursion - self.unrealized_return) / (
            1 + self.max_favorable_excursion
        )
