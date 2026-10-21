"""Explicit scalar sizing conventions, independent of order admission and fills."""

from math import isfinite
from numbers import Real
from typing import Literal

QuantityRounding = Literal["truncate", "near_integer"]


def _finite_real(value: float, name: str) -> None:
    try:
        valid = isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite real number")


def round_quantity(amount: float, *, rounding: QuantityRounding = "truncate") -> int:
    """Truncate signed whole shares, optionally snapping within 1e-4 of an integer."""
    _finite_real(amount, "amount")
    if rounding not in {"truncate", "near_integer"}:
        raise ValueError("rounding must be truncate or near_integer")
    if rounding == "near_integer" and abs(amount - round(amount)) <= 0.0001:
        return int(round(amount))
    return int(amount)


def quantity_for_value(value: float, price: float, multiplier: float = 1.0) -> float:
    """Convert signed notional to unrounded units at an explicitly supplied price."""
    for name, number in [("value", value), ("price", price), ("multiplier", multiplier)]:
        _finite_real(number, name)
    if price <= 0 or multiplier <= 0:
        raise ValueError("price and multiplier must be positive")
    unit_notional = price * multiplier
    if not isfinite(unit_notional):
        raise ValueError("price times multiplier must be finite")
    return value / unit_notional


def quantity_to_target(
    target: float, held: float, *, rounding: QuantityRounding = "truncate"
) -> int:
    """Round the signed delta after subtracting held units; pending orders are separate."""
    _finite_real(target, "target")
    _finite_real(held, "held")
    return round_quantity(target - held, rounding=rounding)
