"""Shared core helpers for broker decomposition."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import copysign, fabs, isclose, isfinite, ulp
from typing import TYPE_CHECKING

from ..types import ExitReason, OrderSide

if TYPE_CHECKING:
    from ..types import Order, Position

# Floating-point tolerance for cash comparisons ($0.01 = 1 cent).
# Prevents order rejections due to rounding in equity/price arithmetic.
CASH_TOLERANCE: float = 0.01
QUANTITY_ZERO_FLOOR: float = 1e-12
QUANTITY_ZERO_ULPS: int = 16


def add_with_zero_cancellation(left: float, right: float) -> float:
    """Add values while collapsing tolerance-equivalent opposite amounts to zero."""
    if copysign(1.0, left) != copysign(1.0, right) and isclose(
        abs(left), abs(right), rel_tol=1e-9, abs_tol=1e-12
    ):
        return 0.0
    return left + right


def quantity_zero_tolerance(*operands: float) -> float:
    """Return the larger of the absolute floor and 16 ULP at the operand scale.

    Cancellation residue inherits the scale of the values being cancelled, so callers pass the
    pre-operation quantities rather than the near-zero result.
    """
    scale = max((fabs(value) for value in operands if isfinite(value)), default=0.0)
    if scale == 0.0:
        return QUANTITY_ZERO_FLOOR
    return max(QUANTITY_ZERO_FLOOR, QUANTITY_ZERO_ULPS * ulp(scale))


@dataclass
class SubmitOrderOptions:
    """Internal options for submit_order behavior."""

    eligible_in_next_bar_mode: bool = False
    rebalance_id: str | None = None
    risk_exit_reason: str | None = None
    exit_reason: ExitReason | None = None
    risk_fill_price: float | None = None
    target_intent_id: str | None = None
    child_intent_id: str | None = None
    intent_idempotency_key: str | None = None


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
    elif "risk_liquidation" in reason_lower or re.search(
        r"\b(liquidation|liquidate)\b", reason_lower
    ):
        return ExitReason.RISK_LIQUIDATION
    elif "end_of_data" in reason_lower:
        return ExitReason.END_OF_DATA
    else:
        return ExitReason.SIGNAL
