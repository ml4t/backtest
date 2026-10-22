"""The variation-cash fill transition shared by execution and reservation projections."""

from __future__ import annotations


def variation_fill(
    old_quantity: float,
    old_basis: float | None,
    signed_quantity: float,
    new_quantity: float,
    price: float,
    multiplier: float,
    mode: str,
) -> tuple[float, float, float | None]:
    payout_quantity = 0.0
    if mode == "on_mark" and old_quantity:
        payout_quantity = old_quantity
    elif old_quantity * signed_quantity < 0:
        payout_quantity = min(abs(old_quantity), abs(signed_quantity)) * (
            1 if old_quantity > 0 else -1
        )
    payout = 0.0
    if payout_quantity:
        if old_basis is None:
            raise ValueError("An exposed futures fill requires its captured variation basis")
        payout = payout_quantity * (price - old_basis) * multiplier
    if not new_quantity:
        basis = None
    elif mode == "on_mark" or not old_quantity or old_quantity * new_quantity < 0:
        basis = price
    elif old_quantity * signed_quantity > 0:
        if old_basis is None:
            raise ValueError("An exposed futures fill requires its captured variation basis")
        basis = (old_quantity * old_basis + signed_quantity * price) / new_quantity
    else:
        basis = old_basis
    return payout, payout_quantity, basis
