"""Deterministic marketable IOC execution against displayed market-by-price depth."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from ml4t.specs import OrderSide


class DepthConsumption(StrEnum):
    """How simulated fills reduce later availability at the same level."""

    REUSE = "reuse"
    REFRESH_LEVEL = "refresh_level"
    REFRESH_SIZE_AT_MATCH = "refresh_size_at_match"
    CARRY_CONSUMPTION = "carry_consumption"


class DisplayedDepthBook:
    """Replay integer price ticks and size lots on one instrument.

    Apply full snapshots and contiguous sequence updates. Size zero deletes a
    level. Depth is historical and does not model the market's response to
    simulated trades. Only marketable IOC sweeps are supported.
    """

    def __init__(self, consumption: DepthConsumption = DepthConsumption.CARRY_CONSUMPTION):
        self._consumption = DepthConsumption(consumption)
        self._levels: dict[OrderSide, dict[int, int]] = {side: {} for side in OrderSide}
        self._consumed: dict[OrderSide, dict[int, int]] = {side: {} for side in OrderSide}
        self._matched_sizes: dict[OrderSide, dict[int, int]] = {side: {} for side in OrderSide}
        self._sequence: int | None = None

    def update(
        self,
        sequence: int,
        levels: Iterable[tuple[OrderSide, int, int]],
        *,
        snapshot: bool = False,
    ) -> None:
        """Apply one validated batch without mutating state on invalid input."""
        if type(sequence) is not int or sequence < 0:
            raise ValueError("Book sequence must be a nonnegative integer")
        if not snapshot and (self._sequence is None or sequence != self._sequence + 1):
            raise ValueError("Book update requires a snapshot and a contiguous sequence")
        edits = [(OrderSide(side), price, size) for side, price, size in levels]
        if any(
            type(price) is not int or type(size) is not int or size < 0 for _, price, size in edits
        ):
            raise ValueError("Book levels require integer price ticks and nonnegative size lots")
        if snapshot:
            for side in OrderSide:
                self._levels[side].clear()
                self._consumed[side].clear()
                self._matched_sizes[side].clear()
        for side, price, size in edits:
            book = self._levels[side]
            consumed = self._consumed[side]
            if size == 0:
                book.pop(price, None)
                if self._consumption is not DepthConsumption.REFRESH_SIZE_AT_MATCH:
                    consumed.pop(price, None)
                continue
            book[price] = size
            if self._consumption is DepthConsumption.REFRESH_LEVEL:
                consumed.pop(price, None)
            elif self._consumption is DepthConsumption.CARRY_CONSUMPTION and price in consumed:
                consumed[price] = min(consumed[price], size)
        self._sequence = sequence

    def best(self, side: OrderSide) -> int | None:
        """Return the best displayed bid or ask on the specified resting side."""
        side = OrderSide(side)
        prices = self._levels[side]
        return (max if side is OrderSide.BUY else min)(prices, default=None)

    def sweep(
        self, side: OrderSide, quantity: int, *, limit: int | None = None
    ) -> tuple[tuple[int, int], ...]:
        """Return ordered (price ticks, size lots) fills, cancelling any remainder."""
        side = OrderSide(side)
        if self._sequence is None:
            raise ValueError("IOC execution requires a book snapshot")
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("IOC quantity must be a positive integer")
        if limit is not None and type(limit) is not int:
            raise ValueError("IOC limit must be an integer price tick")
        buy = side is OrderSide.BUY
        resting_side = OrderSide.SELL if buy else OrderSide.BUY
        book = self._levels[resting_side]
        consumed = self._consumed[resting_side]
        fills = []
        remaining = quantity
        for price in sorted(book, reverse=not buy):
            if limit is not None and (price > limit if buy else price < limit):
                break
            if self._consumption is DepthConsumption.REFRESH_SIZE_AT_MATCH:
                matched = self._matched_sizes[resting_side]
                if matched.get(price) != book[price]:
                    matched[price] = book[price]
                    consumed.pop(price, None)
            taken = min(remaining, book[price] - consumed.get(price, 0))
            if taken:
                fills.append((price, taken))
                if self._consumption is not DepthConsumption.REUSE:
                    consumed[price] = consumed.get(price, 0) + taken
                remaining -= taken
            if remaining == 0:
                break
        return tuple(fills)
