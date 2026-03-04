"""Shared test strategy implementations for ml4t-backtest tests.

Provides reusable strategies for common test patterns instead of
duplicating simple strategies across test files.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ml4t.backtest import OrderSide, OrderType, Strategy


class NoopStrategy(Strategy):
    """Strategy that does nothing. Useful for testing broker/engine internals."""

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Any,
    ) -> None:
        pass


class BuyOnceStrategy(Strategy):
    """Buys a fixed quantity on the first bar, then holds forever.

    Args:
        asset: Asset to buy.
        qty: Quantity to buy.
    """

    def __init__(self, asset: str = "TEST", qty: float = 100.0):
        self.asset = asset
        self.qty = qty
        self._bought = False

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Any,
    ) -> None:
        if not self._bought and self.asset in data:
            broker.submit_order(self.asset, self.qty, OrderSide.BUY)
            self._bought = True


class RoundTripStrategy(Strategy):
    """Enters on bar `entry_bar` and exits on bar `exit_bar`.

    Parameterized by direction (long/short) for systematic direction testing.

    Args:
        asset: Asset symbol.
        qty: Unsigned quantity (direction handled internally).
        entry_bar: Bar index to enter (0-based).
        exit_bar: Bar index to exit (0-based).
        direction: "long" or "short".
    """

    def __init__(
        self,
        asset: str = "TEST",
        qty: float = 100.0,
        entry_bar: int = 0,
        exit_bar: int = 2,
        direction: str = "long",
    ):
        self.asset = asset
        self.qty = qty
        self.entry_bar = entry_bar
        self.exit_bar = exit_bar
        self.direction = direction
        self._bar_count = 0

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Any,
    ) -> None:
        if self._bar_count == self.entry_bar and self.asset in data:
            side = OrderSide.BUY if self.direction == "long" else OrderSide.SELL
            broker.submit_order(self.asset, self.qty, side)
        elif self._bar_count == self.exit_bar:
            pos = broker.get_position(self.asset)
            if pos is not None:
                broker.close_position(self.asset)
        self._bar_count += 1


class OrderTypeStrategy(Strategy):
    """Enters with specified order type, exits with market order.

    Args:
        asset: Asset symbol.
        qty: Unsigned quantity.
        direction: "long" or "short".
        order_type: OrderType for entry.
        limit_price: Limit price for LIMIT entries.
        stop_price: Stop price for STOP entries.
        entry_bar: Bar index to submit entry (0-based).
        exit_bar: Bar index to close position (0-based).
    """

    def __init__(
        self,
        asset: str = "TEST",
        qty: float = 100.0,
        direction: str = "long",
        order_type: OrderType = OrderType.LIMIT,
        limit_price: float | None = None,
        stop_price: float | None = None,
        entry_bar: int = 0,
        exit_bar: int = 3,
    ):
        self.asset = asset
        self.qty = qty
        self.direction = direction
        self.order_type = order_type
        self.limit_price = limit_price
        self.stop_price = stop_price
        self.entry_bar = entry_bar
        self.exit_bar = exit_bar
        self._bar_count = 0

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Any,
    ) -> None:
        if self._bar_count == self.entry_bar and self.asset in data:
            side = OrderSide.BUY if self.direction == "long" else OrderSide.SELL
            broker.submit_order(
                self.asset,
                self.qty,
                side,
                order_type=self.order_type,
                limit_price=self.limit_price,
                stop_price=self.stop_price,
            )
        elif self._bar_count == self.exit_bar:
            pos = broker.get_position(self.asset)
            if pos is not None:
                broker.close_position(self.asset)
        self._bar_count += 1


class SignalStrategy(Strategy):
    """Trades based on a named signal column in the data.

    Buys when signal > 0, sells when signal < 0, closes on zero.

    Args:
        asset: Asset symbol.
        qty: Unsigned quantity.
        signal_name: Name of the signal key in data[asset]["signals"].
    """

    def __init__(
        self,
        asset: str = "TEST",
        qty: float = 100.0,
        signal_name: str = "signal",
    ):
        self.asset = asset
        self.qty = qty
        self.signal_name = signal_name

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Any,
    ) -> None:
        if self.asset not in data:
            return

        bar = data[self.asset]
        signals = bar.get("signals", {})
        signal_val = signals.get(self.signal_name, 0)

        pos = broker.get_position(self.asset)

        if signal_val > 0 and pos is None:
            broker.submit_order(self.asset, self.qty, OrderSide.BUY)
        elif signal_val < 0 and pos is None:
            broker.submit_order(self.asset, self.qty, OrderSide.SELL)
        elif signal_val == 0 and pos is not None:
            broker.close_position(self.asset)
