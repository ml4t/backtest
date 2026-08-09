"""Risk-rule orchestration extracted from Broker."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING

from ml4t.specs import BarPathPolicy

from ..config import TrailStopTiming
from ..preopen import AmbiguousBarPathError
from ..risk.types import ActionType, PositionAction, PositionState
from ..types import OrderSide, OrderType
from .shared import SubmitOrderOptions, reason_to_exit_reason
from .state import MarketState, RiskState

if TYPE_CHECKING:
    from ..accounting import AccountState
    from ..broker import Broker
    from .fill_engine import FillEngine


class RiskEngine:
    """Evaluates position rules and manages deferred exits."""

    def __init__(
        self,
        broker: Broker,
        *,
        account: AccountState,
        market: MarketState,
        risk: RiskState,
        fill_engine: FillEngine,
    ) -> None:
        self.broker = broker
        self.account = account
        self.market = market
        self.risk = risk
        self.fill_engine = fill_engine

    def evaluate_position_rules(self):
        broker = self.broker
        exit_orders = []

        for asset, pos in list(self.account.positions.items()):
            rules = self._get_position_rules(asset)
            if rules is None:
                continue

            price = self.market.prices.get(asset)
            if price is None:
                continue

            state = self._build_position_state(pos, price)
            action = self._evaluate_rules(rules, state, pos)

            if action.action == ActionType.EXIT_FULL:
                if action.defer_fill:
                    self.risk.pending_exits[asset] = {
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
                        _options=SubmitOrderOptions(
                            eligible_in_next_bar_mode=True,
                            risk_exit_reason=action.reason,
                            exit_reason=reason_to_exit_reason(action.reason),
                            risk_fill_price=action.fill_price,
                        ),
                    )
                    if order:
                        exit_orders.append(order)
                        self.risk.stop_exits_this_bar.add(asset)

            elif action.action == ActionType.EXIT_PARTIAL:
                if action.defer_fill:
                    exit_qty = abs(pos.quantity) * action.pct
                    if exit_qty > 0:
                        self.risk.pending_exits[asset] = {
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
                            _options=SubmitOrderOptions(
                                eligible_in_next_bar_mode=True,
                                risk_exit_reason=action.reason,
                                exit_reason=reason_to_exit_reason(action.reason),
                                risk_fill_price=action.fill_price,
                            ),
                        )
                        if order:
                            exit_orders.append(order)

        return exit_orders

    def _evaluate_rules(self, rules, state: PositionState, position) -> PositionAction:
        activation_bar = position.context.get("rule_activation_bar_index")
        policy = position.context.get("bar_path_policy")
        if activation_bar != self.market.bar_index or policy is None:
            return rules.evaluate(state)
        policy = policy if isinstance(policy, BarPathPolicy) else BarPathPolicy(policy)
        if policy is BarPathPolicy.OPEN_HIGH_LOW_CLOSE:
            return self._evaluate_path(rules, state, high_first=True)
        if policy is BarPathPolicy.OPEN_LOW_HIGH_CLOSE:
            return self._evaluate_path(rules, state, high_first=False)

        high_first_action = self._evaluate_path(copy.deepcopy(rules), state, high_first=True)
        low_first_action = self._evaluate_path(copy.deepcopy(rules), state, high_first=False)
        if policy is BarPathPolicy.REJECT_AMBIGUOUS:
            if high_first_action != low_first_action:
                raise AmbiguousBarPathError(
                    f"post-open rule outcome for {state.asset} depends on daily high-low order"
                )
            return self._evaluate_path(rules, state, high_first=True)

        high_first = self._more_conservative_path(state, high_first_action, low_first_action)
        return self._evaluate_path(rules, state, high_first=high_first)

    @staticmethod
    def _evaluate_path(rules, state: PositionState, *, high_first: bool) -> PositionAction:
        if state.bar_open is None or state.bar_high is None or state.bar_low is None:
            return rules.evaluate(state)
        points = (
            (state.bar_high, state.bar_low, state.current_price)
            if high_first
            else (state.bar_low, state.bar_high, state.current_price)
        )
        previous = state.bar_open
        high_water_mark = state.high_water_mark
        low_water_mark = state.low_water_mark
        max_favorable_excursion = state.max_favorable_excursion
        max_adverse_excursion = state.max_adverse_excursion
        for point in points:
            context = dict(state.context)
            context["trail_stop_timing"] = TrailStopTiming.LAGGED
            raw_return = (point - state.entry_price) / state.entry_price
            unrealized_return = raw_return if state.is_long else -raw_return
            max_favorable_excursion = max(max_favorable_excursion, unrealized_return)
            max_adverse_excursion = min(max_adverse_excursion, unrealized_return)
            phase_state = replace(
                state,
                current_price=point,
                unrealized_pnl=(point - state.entry_price)
                * state.quantity
                * state.multiplier
                * (1 if state.is_long else -1),
                unrealized_return=unrealized_return,
                high_water_mark=high_water_mark,
                low_water_mark=low_water_mark,
                max_favorable_excursion=max_favorable_excursion,
                max_adverse_excursion=max_adverse_excursion,
                bar_open=previous,
                bar_high=max(previous, point),
                bar_low=min(previous, point),
                context=context,
            )
            action = rules.evaluate(phase_state)
            if action.action is not ActionType.HOLD:
                return action
            high_water_mark = max(high_water_mark, point)
            low_water_mark = min(low_water_mark, point)
            previous = point
        return PositionAction.hold()

    @staticmethod
    def _more_conservative_path(
        state: PositionState,
        high_first: PositionAction,
        low_first: PositionAction,
    ) -> bool:
        if high_first.action is ActionType.HOLD:
            return low_first.action is ActionType.HOLD
        if low_first.action is ActionType.HOLD:
            return True
        high_price = high_first.fill_price
        low_price = low_first.fill_price
        if high_price is None or low_price is None or high_price == low_price:
            return True
        return high_price < low_price if state.is_long else high_price > low_price

    def _get_position_rules(self, asset: str):
        if asset in self.risk.position_rules_by_asset:
            return self.risk.position_rules_by_asset[asset]
        return self.risk.position_rules

    def _build_position_state(self, pos, current_price: float):
        asset = pos.asset
        context = pos.context
        initial_qty = pos.initial_quantity if pos.initial_quantity is not None else pos.quantity

        return PositionState(
            asset=asset,
            side=pos.side,
            entry_price=pos.entry_price,
            current_price=current_price,
            quantity=abs(pos.quantity),
            initial_quantity=abs(initial_qty),
            unrealized_pnl=pos.unrealized_pnl(current_price),
            unrealized_return=pos.pnl_percent(current_price),
            bars_held=pos.bars_held,
            high_water_mark=pos.high_water_mark
            if pos.high_water_mark is not None
            else pos.entry_price,
            low_water_mark=pos.low_water_mark
            if pos.low_water_mark is not None
            else pos.entry_price,
            bar_open=self.market.opens.get(asset),
            bar_high=self.market.highs.get(asset),
            bar_low=self.market.lows.get(asset),
            max_favorable_excursion=pos.max_favorable_excursion,
            max_adverse_excursion=pos.max_adverse_excursion,
            entry_time=pos.entry_time,
            current_time=self.market.time,
            context=context,
            multiplier=pos.multiplier,
        )

    def process_pending_exits(self):
        broker = self.broker
        exit_orders = []

        for asset, pending in list(self.risk.pending_exits.items()):
            pos = self.account.positions.get(asset)
            if pos is None:
                del self.risk.pending_exits[asset]
                continue

            open_price = self.market.opens.get(asset)
            if open_price is None:
                continue

            stored_fill_price = pending.get("fill_price")
            if broker.stop_fill_mode.value == "stop_price" and stored_fill_price is not None:
                exit_side = OrderSide.SELL if pending["quantity"] > 0 else OrderSide.BUY
                gap_price = self.fill_engine.check_gap_through(
                    exit_side, stored_fill_price, open_price
                )
                fill_price = gap_price if gap_price is not None else stored_fill_price
            else:
                fill_price = open_price

            exit_qty = pending["quantity"]
            order = broker.submit_order(
                asset,
                -exit_qty,
                order_type=OrderType.MARKET,
                _options=SubmitOrderOptions(
                    eligible_in_next_bar_mode=True,
                    risk_exit_reason=pending["reason"],
                    exit_reason=reason_to_exit_reason(pending["reason"]),
                    risk_fill_price=fill_price,
                ),
            )
            if order:
                exit_orders.append(order)

            del self.risk.pending_exits[asset]

        return exit_orders
