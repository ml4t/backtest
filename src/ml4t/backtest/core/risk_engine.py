"""Risk-rule orchestration extracted from Broker."""

from __future__ import annotations

from ..types import OrderSide, OrderType
from ..risk.types import ActionType
from .shared import SubmitOrderOptions, reason_to_exit_reason


class RiskEngine:
    """Evaluates position rules and manages deferred exits."""

    def __init__(self, broker):
        self.broker = broker

    def evaluate_position_rules(self):
        broker = self.broker
        exit_orders = []

        for asset, pos in list(broker.positions.items()):
            rules = broker._get_position_rules(asset)
            if rules is None:
                continue

            price = broker._current_prices.get(asset)
            if price is None:
                continue

            state = broker._build_position_state(pos, price)
            action = rules.evaluate(state)

            if action.action == ActionType.EXIT_FULL:
                if action.defer_fill:
                    broker._pending_exits[asset] = {
                        "reason": action.reason,
                        "pct": 1.0,
                        "quantity": pos.quantity,
                        "fill_price": action.fill_price,
                    }
                else:
                    order = broker.submit_order(
                        asset,
                        -pos.quantity,
                        order_type=OrderType.MARKET,
                        _options=SubmitOrderOptions(eligible_in_next_bar_mode=True),
                    )
                    if order:
                        order._risk_exit_reason = action.reason
                        order._exit_reason = reason_to_exit_reason(action.reason)
                        order._risk_fill_price = action.fill_price
                        exit_orders.append(order)
                        broker._stop_exits_this_bar.add(asset)

            elif action.action == ActionType.EXIT_PARTIAL:
                if action.defer_fill:
                    exit_qty = abs(pos.quantity) * action.pct
                    if exit_qty > 0:
                        broker._pending_exits[asset] = {
                            "reason": action.reason,
                            "pct": action.pct,
                            "quantity": exit_qty if pos.quantity > 0 else -exit_qty,
                            "fill_price": action.fill_price,
                        }
                else:
                    exit_qty = abs(pos.quantity) * action.pct
                    if exit_qty > 0:
                        actual_qty = -exit_qty if pos.quantity > 0 else exit_qty
                        order = broker.submit_order(
                            asset,
                            actual_qty,
                            order_type=OrderType.MARKET,
                            _options=SubmitOrderOptions(eligible_in_next_bar_mode=True),
                        )
                        if order:
                            order._risk_exit_reason = action.reason
                            order._exit_reason = reason_to_exit_reason(action.reason)
                            order._risk_fill_price = action.fill_price
                            exit_orders.append(order)

        return exit_orders

    def process_pending_exits(self):
        broker = self.broker
        exit_orders = []

        for asset, pending in list(broker._pending_exits.items()):
            pos = broker.positions.get(asset)
            if pos is None:
                del broker._pending_exits[asset]
                continue

            open_price = broker._current_opens.get(asset)
            if open_price is None:
                continue

            stored_fill_price = pending.get("fill_price")
            if broker.stop_fill_mode.value == "stop_price" and stored_fill_price is not None:
                exit_side = OrderSide.SELL if pending["quantity"] > 0 else OrderSide.BUY
                gap_price = broker._check_gap_through(exit_side, stored_fill_price, open_price)
                fill_price = gap_price if gap_price is not None else stored_fill_price
            else:
                fill_price = open_price

            exit_qty = pending["quantity"]
            order = broker.submit_order(
                asset,
                -exit_qty,
                order_type=OrderType.MARKET,
                _options=SubmitOrderOptions(eligible_in_next_bar_mode=True),
            )
            if order:
                order._risk_exit_reason = pending["reason"]
                order._exit_reason = reason_to_exit_reason(pending["reason"])
                order._risk_fill_price = fill_price
                exit_orders.append(order)

            del broker._pending_exits[asset]

        return exit_orders
