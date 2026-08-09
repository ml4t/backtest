"""Fill checks and fill execution helpers extracted from Broker."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..config import ExecutionPrice, ShareType
from ..types import OrderSide, OrderType
from .shared import CASH_TOLERANCE
from .state import MarketState, OrderState

if TYPE_CHECKING:
    from ..broker import Broker
    from ..execution.fill_executor import FillExecutor


class FillEngine:
    """Owns fill-price checks, quantity helpers, and fill execution delegation."""

    def __init__(
        self,
        broker: Broker,
        *,
        market: MarketState,
        orders: OrderState,
        executor: FillExecutor,
    ) -> None:
        self.broker = broker
        self.market = market
        self.orders = orders
        self.executor = executor

    def get_available_cash(self) -> float:
        broker = self.broker
        if (
            broker.short_cash_policy.value == "lock_notional"
            and not broker.account.policy.allow_leverage
        ):
            spendable = broker.account._lock_notional_free_cash
        else:
            spendable = broker.account.policy.get_spendable_cash(
                broker.account.cash, broker.account.positions
            )
        if broker.cash_buffer_pct > 0:
            return spendable * (1.0 - broker.cash_buffer_pct)
        return spendable

    def apply_share_rounding(self, order) -> None:
        if self.broker.share_type == ShareType.INTEGER:
            order.quantity = float(int(order.quantity))

    def _can_afford_quantity(self, order, fill_price: float, quantity: float) -> bool:
        if quantity <= 0:
            return True
        candidate = replace(order, quantity=quantity)
        valid, _reason = self.broker.gatekeeper.validate_order(candidate, fill_price)
        return valid

    def get_max_affordable_quantity(self, order, fill_price: float) -> float:
        """Return max fillable quantity under current cash constraints."""
        if fill_price <= 0 or order.quantity <= 0:
            return 0.0

        if self.broker.share_type == ShareType.INTEGER:
            high_int = int(order.quantity)
            if high_int <= 0:
                return 0.0
            if self._can_afford_quantity(order, fill_price, float(high_int)):
                return float(high_int)

            low_int = 0
            while low_int < high_int:
                mid_int = (low_int + high_int + 1) // 2
                if self._can_afford_quantity(order, fill_price, float(mid_int)):
                    low_int = mid_int
                else:
                    high_int = mid_int - 1
            return float(low_int)

        high = order.quantity
        if self._can_afford_quantity(order, fill_price, high):
            return high

        low = 0.0
        for _ in range(64):
            mid = (low + high) / 2.0
            if self._can_afford_quantity(order, fill_price, mid):
                low = mid
            else:
                high = mid
        cash_tolerance = CASH_TOLERANCE
        if (
            self.broker.short_cash_policy.value == "lock_notional"
            and not self.broker.account.policy.allow_leverage
        ):
            cash_tolerance = 0.0
        return max(0.0, low - cash_tolerance / fill_price)

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
        if order.child_intent_id is not None and use_open:
            return broker.get_price_for_source(
                ExecutionPrice.OPEN,
                order.asset,
                side=order.side,
                use_open=True,
            )
        if order.order_type is OrderType.MOC:
            return broker.get_price_for_source(
                ExecutionPrice.CLOSE,
                order.asset,
                side=order.side,
                use_open=False,
            )
        return broker.get_price_for_source(
            broker.execution_price,
            order.asset,
            side=order.side,
            use_open=use_open,
        )

    def get_effective_quantity(self, order) -> float:
        remaining = self.orders.partial_quantities.get(order.order_id)
        if remaining is not None:
            return remaining
        return order.quantity

    def update_partial_order(self, order) -> None:
        remaining = self.orders.partial_quantities.get(order.order_id)
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
        high = self.market.highs.get(order.asset, price)
        low = self.market.lows.get(order.asset, price)
        bar_open = self.market.opens.get(order.asset, price)

        if order.order_type in {OrderType.MARKET, OrderType.MOC}:
            return self.check_market_fill(order, price)
        if order.order_type == OrderType.LIMIT:
            return self.check_limit_fill(order, high, low)
        if order.order_type == OrderType.STOP:
            return self.check_stop_fill(order, high, low, bar_open)
        if order.order_type == OrderType.TRAILING_STOP:
            return self.update_and_check_trailing_stop(order, high, low, bar_open)
        return None

    def execute_fill(self, order, base_price: float) -> bool:
        return self.executor.execute(order, base_price)
