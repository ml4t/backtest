"""Shared core helpers for broker decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from math import fabs, isfinite, ulp
from typing import TYPE_CHECKING

from ..types import ExitReason, OrderSide

if TYPE_CHECKING:
    from ..types import Order, Position

# Floating-point tolerance for cash comparisons ($0.01 = 1 cent).
# Prevents order rejections due to rounding in equity/price arithmetic.
CASH_TOLERANCE: float = 0.01

# Quantity-zero tolerance.
#
# A fill updates a position with ``new_qty = old_qty + signed_qty``. When the
# close is economically exact the two operands cancel as real numbers, but they
# are floats produced by different routes, so the sum can land a few units in
# the last place away from zero instead of on it. The size of that residue is
# set by the spacing of float64 at the scale of the quantities being cancelled,
# and float64 spacing grows with magnitude.
#
# A single absolute epsilon cannot express that. 1e-12 is wider than one ULP
# while |q| < 8192 and narrower than one ULP from |q| = 8192 upward, so an
# absolute rule silently stops closing positions at exactly 2**13 units.
#
# The floor preserves the historical absolute behaviour wherever float64 spacing
# is finer than it, which is every magnitude below 8192.
QTY_ZERO_FLOOR: float = 1e-12
QTY_ZERO_ULPS: int = 16


def quantity_zero_tolerance(*operands: float) -> float:
    """Return the tolerance under which a residual quantity counts as zero.

    The tolerance is the larger of :data:`QTY_ZERO_FLOOR` and
    :data:`QTY_ZERO_ULPS` units in the last place of the largest finite operand.

    Pass the quantities that *produced* the residual — for a fill, the pre-fill
    position and the signed fill size. Do not pass the residual itself: it is
    approximately zero by construction, so it carries no scale and a tolerance
    derived from it could never fire.

    Non-finite operands are ignored, so the tolerance is always a finite
    positive number; if no operand is finite the floor applies. The tolerance is
    built from magnitudes, so it is symmetric for long and short quantities.

    Args:
        *operands: Quantities defining the scale of the operation.

    Returns:
        The tolerance, always >= ``QTY_ZERO_FLOOR``.
    """
    scale = 0.0
    for operand in operands:
        if isfinite(operand):
            magnitude = fabs(operand)
            if magnitude > scale:
                scale = magnitude
    if scale == 0.0:
        return QTY_ZERO_FLOOR
    return max(QTY_ZERO_FLOOR, QTY_ZERO_ULPS * ulp(scale))


@dataclass
class SubmitOrderOptions:
    """Internal options for submit_order behavior."""

    eligible_in_next_bar_mode: bool = False
    rebalance_id: str | None = None


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
    elif "risk_liquidation" in reason_lower or "liquidat" in reason_lower:
        return ExitReason.RISK_LIQUIDATION
    elif "end_of_data" in reason_lower:
        return ExitReason.END_OF_DATA
    else:
        return ExitReason.SIGNAL
