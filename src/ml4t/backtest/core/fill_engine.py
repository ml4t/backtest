"""Fill checks and fill execution helpers extracted from Broker."""

from __future__ import annotations

from ..config import ShareType
from ..types import ExecutionMode, OrderSide, OrderType


class FillEngine:
    """Owns fill-price checks, quantity helpers, and fill execution delegation."""

    def __init__(self, broker):
        self.broker = broker

    def get_available_cash(self) -> float:
        broker = self.broker
        if broker.cash_buffer_pct > 0:
            return broker.account.cash * (1.0 - broker.cash_buffer_pct)
        return broker.account.cash

    def apply_share_rounding(self, order) -> None:
        if self.broker.share_type == ShareType.INTEGER:
            order.quantity = float(int(order.quantity))

    def try_partial_fill(self, order, fill_price: float) -> bool:
        broker = self.broker
        available = self.get_available_cash()
        commission_rate = 0.0
        test_commission = broker.commission_model.calculate(order.asset, 1.0, fill_price)
        if fill_price > 0:
            commission_rate = test_commission / fill_price

        max_value = available / (1.0 + commission_rate) if commission_rate > 0 else available
        max_shares = max_value / fill_price if fill_price > 0 else 0

        if broker.share_type == ShareType.INTEGER:
            max_shares = float(int(max_shares))

        if max_shares <= 0:
            return False

        order.quantity = max_shares
        return bool(self.execute_fill(order, fill_price))

    def get_fill_price_for_order(self, order, use_open: bool) -> float | None:
        broker = self.broker
        if use_open and broker.execution_mode == ExecutionMode.NEXT_BAR:
            return broker._current_opens.get(order.asset)
        return broker._current_prices.get(order.asset)

    def get_effective_quantity(self, order) -> float:
        remaining = self.broker._partial_orders.get(order.order_id)
        if remaining is not None:
            return remaining
        return order.quantity

    def update_partial_order(self, order) -> None:
        remaining = self.broker._partial_orders.get(order.order_id)
        if remaining is not None:
            order.quantity = remaining

    def check_gap_through(
        self, side: OrderSide, stop_price: float, bar_open: float
    ) -> float | None:
        if side == OrderSide.SELL and bar_open <= stop_price:
            return bar_open
        if side == OrderSide.BUY and bar_open >= stop_price:
            return bar_open
        return None

    def check_market_fill(self, order, price: float) -> float:
        broker = self.broker
        risk_fill_price = getattr(order, "_risk_fill_price", None)
        if risk_fill_price is None:
            return price

        fill_price = risk_fill_price
        if broker.stop_slippage_rate > 0:
            if order.side == OrderSide.SELL:
                fill_price = fill_price * (1 - broker.stop_slippage_rate)
            else:
                fill_price = fill_price * (1 + broker.stop_slippage_rate)
        return fill_price

    def check_limit_fill(self, order, high: float, low: float) -> float | None:
        if order.limit_price is None:
            return None
        if (
            order.side == OrderSide.BUY
            and low <= order.limit_price
            or order.side == OrderSide.SELL
            and high >= order.limit_price
        ):
            return order.limit_price
        return None

    def check_stop_fill(self, order, high: float, low: float, bar_open: float) -> float | None:
        if order.stop_price is None:
            return None

        triggered = False
        if (
            order.side == OrderSide.BUY
            and high >= order.stop_price
            or order.side == OrderSide.SELL
            and low <= order.stop_price
        ):
            triggered = True

        if not triggered:
            return None

        gap_price = self.check_gap_through(order.side, order.stop_price, bar_open)
        return gap_price if gap_price is not None else order.stop_price

    def update_and_check_trailing_stop(
        self, order, high: float, low: float, bar_open: float
    ) -> float | None:
        if order.trail_amount is None:
            return None

        if order.side == OrderSide.SELL:
            new_stop = high - order.trail_amount
            if order.stop_price is None or new_stop > order.stop_price:
                order.stop_price = new_stop
            if order.stop_price is None or low > order.stop_price:
                return None
        else:
            new_stop = low + order.trail_amount
            if order.stop_price is None or new_stop < order.stop_price:
                order.stop_price = new_stop
            if order.stop_price is None or high < order.stop_price:
                return None

        assert order.stop_price is not None
        gap_price = self.check_gap_through(order.side, order.stop_price, bar_open)
        return gap_price if gap_price is not None else order.stop_price

    def check_fill(self, order, price: float) -> float | None:
        broker = self.broker
        high = broker._current_highs.get(order.asset, price)
        low = broker._current_lows.get(order.asset, price)
        bar_open = broker._current_opens.get(order.asset, price)

        if order.order_type == OrderType.MARKET:
            return self.check_market_fill(order, price)
        if order.order_type == OrderType.LIMIT:
            return self.check_limit_fill(order, high, low)
        if order.order_type == OrderType.STOP:
            return self.check_stop_fill(order, high, low, bar_open)
        if order.order_type == OrderType.TRAILING_STOP:
            return self.update_and_check_trailing_stop(order, high, low, bar_open)
        return None

    def execute_fill(self, order, base_price: float) -> bool:
        return self.broker._fill_executor.execute(order, base_price)
