"""Fill checks and fill execution helpers extracted from Broker."""

from __future__ import annotations

from ..config import ExecutionPrice, ShareType
from ..types import ExecutionMode, OrderSide, OrderType


class FillEngine:
    """Owns fill-price checks, quantity helpers, and fill execution delegation."""

    def __init__(self, broker):
        self.broker = broker

    def get_available_cash(self) -> float:
        broker = self.broker
        spendable = broker.account.policy.get_spendable_cash(
            broker.account.cash, broker.account.positions
        )
        if broker.cash_buffer_pct > 0:
            return spendable * (1.0 - broker.cash_buffer_pct)
        return spendable

    def apply_share_rounding(self, order) -> None:
        if self.broker.share_type == ShareType.INTEGER:
            order.quantity = float(int(order.quantity))

    def _commission_rate(self, order, fill_price: float) -> float:
        if fill_price <= 0:
            return 0.0
        test_commission = self.broker.commission_model.calculate(order.asset, 1.0, fill_price)
        return test_commission / fill_price

    def get_max_affordable_quantity(self, order, fill_price: float) -> float:
        """Return max fillable quantity under current cash constraints."""
        broker = self.broker
        commission_rate = self._commission_rate(order, fill_price)
        gross_per_share = fill_price * (1.0 + commission_rate)
        if gross_per_share <= 0:
            return 0.0

        available = self.get_available_cash()
        current_qty = broker.account.get_position_quantity(order.asset)
        short_policy = getattr(broker.short_cash_policy, "value", "")

        if order.side == OrderSide.BUY and current_qty < 0 and short_policy == "lock_notional":
            # VectorBT lock_cash-style cap for covering/reversing short positions.
            position = broker.account.get_position(order.asset)
            if position is None:
                return max(0.0, available / gross_per_share)

            debt = abs(position.quantity) * position.entry_price * position.multiplier
            cover_req_cash = abs(position.quantity) * gross_per_share
            cover_free_cash = available + 2.0 * debt - cover_req_cash

            if cover_free_cash > 0:
                cash_limit = available + 2.0 * debt
            elif cover_free_cash < 0:
                avg_entry_price = position.entry_price * position.multiplier
                denom = gross_per_share - 2.0 * avg_entry_price
                if denom <= 0:
                    cash_limit = 0.0
                else:
                    max_short_size = available / denom
                    cash_limit = max(0.0, max_short_size * gross_per_share)
            else:
                cash_limit = broker.account.cash

            return max(0.0, cash_limit / gross_per_share)

        if order.side == OrderSide.SELL and short_policy == "lock_notional":
            # VectorBT lock_cash-style cap for short selling with locked free cash.
            long_qty = max(current_qty, 0.0)
            long_cash = long_qty * fill_price * max(0.0, 1.0 - commission_rate)
            total_free_cash = available + long_cash

            if total_free_cash <= 0:
                return max(0.0, long_qty)

            max_short_qty = total_free_cash / gross_per_share
            return max(0.0, long_qty + max_short_qty)

        return max(0.0, available / gross_per_share)

    def try_partial_fill(self, order, fill_price: float) -> bool:
        max_shares = self.get_max_affordable_quantity(order, fill_price)

        if self.broker.share_type == ShareType.INTEGER:
            max_shares = float(int(max_shares))

        if max_shares <= 0:
            return False

        order.quantity = max_shares
        return bool(self.execute_fill(order, fill_price))

    def get_fill_price_for_order(self, order, use_open: bool) -> float | None:
        broker = self.broker
        if use_open and broker.execution_mode == ExecutionMode.NEXT_BAR:
            return broker._current_opens.get(order.asset)
        # Dispatch on execution_price setting
        ep = broker.execution_price
        if ep == ExecutionPrice.OPEN:
            return broker._current_opens.get(order.asset)
        if ep == ExecutionPrice.MID:
            h = broker._current_highs.get(order.asset)
            lo = broker._current_lows.get(order.asset)
            if h is not None and lo is not None:
                return (h + lo) / 2.0
            return broker._current_prices.get(order.asset)
        if ep == ExecutionPrice.VWAP:
            # VWAP not available at bar level; fall back to close
            return broker._current_prices.get(order.asset)
        # Default: CLOSE
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
