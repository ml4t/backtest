"""Unit tests for composite rules (RuleChain, AllOf, AnyOf)."""

from datetime import datetime

import pytest

from ml4t.backtest.risk.position.composite import AllOf, AnyOf, RuleChain
from ml4t.backtest.risk.position.static import StopLoss, TakeProfit, TimeExit
from ml4t.backtest.risk.types import ActionType, PositionState


def make_position_state(
    side: str = "long",
    entry_price: float = 100.0,
    current_price: float = 100.0,
    bars_held: int = 0,
    unrealized_return: float = 0.0,
) -> PositionState:
    """Helper to create PositionState for testing."""
    return PositionState(
        asset="AAPL",
        side=side,
        entry_price=entry_price,
        current_price=current_price,
        quantity=100,
        initial_quantity=100,
        unrealized_pnl=(current_price - entry_price) * 100,
        unrealized_return=unrealized_return,
        bars_held=bars_held,
        high_water_mark=max(entry_price, current_price),
        low_water_mark=min(entry_price, current_price),
        entry_time=datetime.now(),
        current_time=datetime.now(),
        context={},
    )


class TestRuleChain:
    """Tests for RuleChain (first non-HOLD wins)."""

    def test_first_rule_triggers(self):
        """First rule in chain that returns non-HOLD wins."""
        chain = RuleChain(
            [
                StopLoss(pct=0.05),  # Triggers at -5%
                TakeProfit(pct=0.10),  # Triggers at +10%
            ]
        )

        # Stop loss triggers first
        state = make_position_state(entry_price=100.0, current_price=94.0)
        action = chain.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "stop_loss" in action.reason

    def test_second_rule_triggers_when_first_holds(self):
        """Second rule triggers when first holds."""
        chain = RuleChain(
            [
                StopLoss(pct=0.05),  # Doesn't trigger
                TakeProfit(pct=0.10),  # Triggers at +10%
            ]
        )

        # Take profit triggers (stop didn't)
        state = make_position_state(entry_price=100.0, current_price=111.0)
        action = chain.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "take_profit" in action.reason

    def test_all_hold_returns_hold(self):
        """Returns HOLD when all rules hold."""
        chain = RuleChain(
            [
                StopLoss(pct=0.05),  # Doesn't trigger
                TakeProfit(pct=0.10),  # Doesn't trigger
            ]
        )

        # Neither triggers
        state = make_position_state(entry_price=100.0, current_price=102.0)
        action = chain.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_empty_chain_holds(self):
        """Empty chain returns HOLD."""
        chain = RuleChain([])
        state = make_position_state()
        action = chain.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_priority_order_respected(self):
        """Rules are evaluated in order, first wins."""
        # Both could trigger, but stop has priority
        chain = RuleChain(
            [
                StopLoss(pct=0.15),  # Triggers at -15%
                TakeProfit(pct=0.05),  # Would also trigger at -5% (loss > -5%)
            ]
        )

        # Price at 84 (-16%)
        state = make_position_state(entry_price=100.0, current_price=84.0)
        action = chain.evaluate(state)
        assert "stop_loss" in action.reason  # Stop has priority


class TestAllOf:
    """Tests for AllOf (all rules must trigger)."""

    def test_all_trigger(self):
        """Action returned when all rules trigger."""
        rule = AllOf(
            [
                StopLoss(pct=0.05),  # Triggers below 95
                TimeExit(max_bars=5),  # Triggers after 5 bars
            ]
        )

        # Both conditions met
        state = make_position_state(
            entry_price=100.0,
            current_price=90.0,  # Stop triggered
            bars_held=10,  # Time triggered
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert " AND " in action.reason

    def test_one_holds_returns_hold(self):
        """Returns HOLD if any rule holds."""
        rule = AllOf(
            [
                StopLoss(pct=0.05),  # Triggers
                TimeExit(max_bars=20),  # Doesn't trigger
            ]
        )

        # Stop triggered but time not reached
        state = make_position_state(
            entry_price=100.0,
            current_price=90.0,
            bars_held=5,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_all_hold_returns_hold(self):
        """Returns HOLD when all rules hold."""
        rule = AllOf(
            [
                StopLoss(pct=0.05),
                TimeExit(max_bars=20),
            ]
        )

        # Neither triggers
        state = make_position_state(
            entry_price=100.0,
            current_price=98.0,
            bars_held=5,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_empty_allof_raises(self):
        """Empty AllOf raises IndexError (edge case - don't use empty AllOf)."""
        rule = AllOf([])
        state = make_position_state()
        # Empty AllOf tries to access first action which doesn't exist
        with pytest.raises(IndexError):
            rule.evaluate(state)


class TestAnyOf:
    """Tests for AnyOf (first non-HOLD wins, same as RuleChain)."""

    def test_first_trigger_wins(self):
        """First rule to trigger wins."""
        rule = AnyOf(
            [
                StopLoss(pct=0.05),
                TakeProfit(pct=0.10),
            ]
        )

        # Stop triggers
        state = make_position_state(entry_price=100.0, current_price=94.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "stop_loss" in action.reason

    def test_second_triggers_when_first_holds(self):
        """Second rule triggers if first holds."""
        rule = AnyOf(
            [
                StopLoss(pct=0.05),
                TakeProfit(pct=0.10),
            ]
        )

        # Profit triggers
        state = make_position_state(entry_price=100.0, current_price=111.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "take_profit" in action.reason

    def test_all_hold_returns_hold(self):
        """Returns HOLD when all rules hold."""
        rule = AnyOf(
            [
                StopLoss(pct=0.05),
                TakeProfit(pct=0.10),
            ]
        )

        state = make_position_state(entry_price=100.0, current_price=102.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_empty_anyof_holds(self):
        """Empty AnyOf returns HOLD."""
        rule = AnyOf([])
        state = make_position_state()
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD


class TestNestedComposites:
    """Tests for nested composite rules."""

    def test_chain_of_allofs(self):
        """Can chain AllOf rules."""
        # Exit if (stop AND held 5+ bars) OR (profit AND held 10+ bars)
        rule = RuleChain(
            [
                AllOf(
                    [
                        StopLoss(pct=0.05),
                        TimeExit(max_bars=5),
                    ]
                ),
                AllOf(
                    [
                        TakeProfit(pct=0.10),
                        TimeExit(max_bars=10),
                    ]
                ),
            ]
        )

        # Stop + time met
        state1 = make_position_state(
            entry_price=100.0,
            current_price=90.0,
            bars_held=6,
        )
        action1 = rule.evaluate(state1)
        assert action1.action == ActionType.EXIT_FULL

        # Profit + time met
        state2 = make_position_state(
            entry_price=100.0,
            current_price=112.0,
            bars_held=12,
        )
        action2 = rule.evaluate(state2)
        assert action2.action == ActionType.EXIT_FULL

        # Neither combination met
        state3 = make_position_state(
            entry_price=100.0,
            current_price=112.0,  # Profit but not enough bars
            bars_held=5,
        )
        action3 = rule.evaluate(state3)
        assert action3.action == ActionType.HOLD
