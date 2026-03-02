"""Shared core helpers for broker decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..types import ExitReason, OrderSide

if TYPE_CHECKING:
    from ..types import Order, Position

# Floating-point tolerance for cash comparisons ($0.01 = 1 cent).
# Prevents order rejections due to rounding in equity/price arithmetic.
CASH_TOLERANCE: float = 0.01


@dataclass
class SubmitOrderOptions:
    """Internal options for submit_order behavior."""

    eligible_in_next_bar_mode: bool = False


def is_exit_order(order: Order, positions: dict[str, Position]) -> bool:
    """Check if an order reduces an existing position without reversing."""
    pos = positions.get(order.asset)
    if pos is None or pos.quantity == 0:
        return False

    signed_qty = order.quantity if order.side is OrderSide.BUY else -order.quantity

    if pos.quantity > 0 and signed_qty < 0:
        return pos.quantity + signed_qty >= 0
    if pos.quantity < 0 and signed_qty > 0:
        return pos.quantity + signed_qty <= 0
    return False


def reason_to_exit_reason(reason: str) -> ExitReason:
    """Map human-readable rule reason to typed ExitReason."""
    reason_lower = reason.lower()
    if "stop_loss" in reason_lower:
        return ExitReason.STOP_LOSS
    elif "take_profit" in reason_lower:
        return ExitReason.TAKE_PROFIT
    elif "trailing" in reason_lower:
        return ExitReason.TRAILING_STOP
    elif "time" in reason_lower:
        return ExitReason.TIME_STOP
    elif "end_of_data" in reason_lower:
        return ExitReason.END_OF_DATA
    else:
        return ExitReason.SIGNAL
