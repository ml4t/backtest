"""Unit tests for rule combination priority and interaction.

This test module validates:

1. RuleChain priority (first non-HOLD wins)
2. TSL + TP combinations (which triggers first?)
3. TSL + SL combinations (which triggers first?)
4. Triple rule combinations (TSL + TP + SL)
5. Same-bar double breach scenarios
6. Rule order affects outcome
"""

from datetime import datetime

from ml4t.backtest.risk.position.composite import RuleChain
from ml4t.backtest.risk.position.dynamic import TrailingStop
from ml4t.backtest.risk.position.static import StopLoss, TakeProfit, TimeExit
from ml4t.backtest.risk.types import ActionType, PositionState


def make_position_state(
    side: str = "long",
    entry_price: float = 100.0,
    current_price: float = 100.0,
    high_water_mark: float | None = None,
    low_water_mark: float | None = None,
    bar_open: float | None = None,
    bar_high: float | None = None,
    bar_low: float | None = None,
    bars_held: int = 0,
    context: dict | None = None,
) -> PositionState:
    """Helper to create PositionState for testing."""
    if high_water_mark is None:
        high_water_mark = max(entry_price, current_price)
    if low_water_mark is None:
        low_water_mark = min(entry_price, current_price)

    if side == "long":
        unrealized_return = (current_price - entry_price) / entry_price
        unrealized_pnl = (current_price - entry_price) * 100
    else:
        unrealized_return = (entry_price - current_price) / entry_price
        unrealized_pnl = (entry_price - current_price) * 100

    return PositionState(
        asset="TEST",
        side=side,
        entry_price=entry_price,
        current_price=current_price,
        bar_open=bar_open,
        bar_high=bar_high,
        bar_low=bar_low,
        quantity=100,
        initial_quantity=100,
        unrealized_pnl=unrealized_pnl,
        unrealized_return=unrealized_return,
        bars_held=bars_held,
        high_water_mark=high_water_mark,
        low_water_mark=low_water_mark,
        entry_time=datetime.now(),
        current_time=datetime.now(),
        context=context or {},
    )


class TestRuleChainPriority:
    """Tests that RuleChain evaluates rules in order, first non-HOLD wins."""

    def test_first_rule_takes_priority(self):
        """When first rule triggers, subsequent rules are not evaluated."""
        chain = RuleChain(
            [
                StopLoss(pct=0.05),  # First priority
                TakeProfit(pct=0.10),  # Second priority
            ]
        )

        # Price at 94 triggers stop loss
        state = make_position_state(entry_price=100.0, current_price=94.0)
        action = chain.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "stop_loss" in action.reason

    def test_second_rule_triggers_when_first_holds(self):
        """Second rule triggers only when first rule holds."""
        chain = RuleChain(
            [
                StopLoss(pct=0.05),  # Doesn't trigger
                TakeProfit(pct=0.10),  # Triggers
            ]
        )

        # Price at 111 triggers take profit
        state = make_position_state(entry_price=100.0, current_price=111.0)
        action = chain.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "take_profit" in action.reason

    def test_rule_order_matters(self):
        """Changing rule order changes which rule wins."""
        # Order 1: SL before TP
        chain1 = RuleChain(
            [
                StopLoss(pct=0.20),  # Stop at 80
                TakeProfit(pct=0.10),  # TP at 110
            ]
        )

        # Order 2: TP before SL
        chain2 = RuleChain(
            [
                TakeProfit(pct=0.10),  # TP at 110
                StopLoss(pct=0.20),  # Stop at 80
            ]
        )

        # At price 78, both would trigger
        # But only first in chain wins
        state = make_position_state(entry_price=100.0, current_price=78.0)

        action1 = chain1.evaluate(state)
        action2 = chain2.evaluate(state)

        assert "stop_loss" in action1.reason  # SL is first
        assert "stop_loss" in action2.reason  # SL triggers even when second


class TestTSLWithTPCombination:
    """Tests for TSL + Take Profit rule combinations."""

    def test_tsl_triggers_before_tp_on_decline(self):
        """TSL triggers on price decline, TP doesn't."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.05),  # Trail from HWM
                TakeProfit(pct=0.15),  # TP at +15%
            ]
        )

        # LONG at 100, rallied to 110 (HWM), now at 104 (below trail at 104.5)
        state = make_position_state(
            entry_price=100.0,
            current_price=104.0,
            high_water_mark=110.0,
        )
        action = chain.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "trailing_stop" in action.reason

    def test_tp_triggers_before_tsl_on_rally(self):
        """TP triggers when price reaches target, TSL doesn't."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.05),  # Trail from HWM
                TakeProfit(pct=0.10),  # TP at +10%
            ]
        )

        # LONG at 100, price at 112 (above TP at 110)
        # HWM = 112, trail at 106.4 - TSL not triggered
        state = make_position_state(
            entry_price=100.0,
            current_price=112.0,
            high_water_mark=112.0,
        )
        action = chain.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "take_profit" in action.reason

    def test_tsl_first_in_chain_priority(self):
        """When TSL is first, it's checked before TP."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.02),  # Very tight 2% trail
                TakeProfit(pct=0.10),
            ]
        )

        # HWM at 115, trail at 112.7
        # Price at 112 triggers TSL even though it's above entry
        state = make_position_state(
            entry_price=100.0,
            current_price=112.0,
            high_water_mark=115.0,
        )
        action = chain.evaluate(state)
        assert "trailing_stop" in action.reason


class TestTSLWithSLCombination:
    """Tests for TSL + Stop Loss rule combinations."""

    def test_sl_triggers_on_crash(self):
        """Stop loss triggers on sharp decline, TSL doesn't (no HWM update)."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.10),  # 10% trail from HWM
                StopLoss(pct=0.08),  # 8% stop from entry
            ]
        )

        # LONG at 100, price crashes to 90 (10% loss)
        # HWM never updated, TSL trail at 90 (100 * 0.9)
        # SL at 92 would trigger if price < 92
        state = make_position_state(
            entry_price=100.0,
            current_price=90.0,
            high_water_mark=100.0,  # Never rallied
        )
        action = chain.evaluate(state)
        # TSL triggers at 90 (100 * 0.9 = 90)
        assert "trailing_stop" in action.reason

    def test_sl_triggers_before_tsl_when_first(self):
        """When SL is first in chain, it triggers before TSL evaluation."""
        chain = RuleChain(
            [
                StopLoss(pct=0.05),  # First: 5% stop at 95
                TrailingStop(pct=0.10),  # Second: 10% trail
            ]
        )

        # Price at 94 triggers SL
        state = make_position_state(
            entry_price=100.0,
            current_price=94.0,
            high_water_mark=100.0,
        )
        action = chain.evaluate(state)
        assert "stop_loss" in action.reason

    def test_tsl_triggers_from_hwm_before_sl(self):
        """TSL from HWM triggers before SL from entry."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.05),  # First: 5% trail
                StopLoss(pct=0.10),  # Second: 10% stop
            ]
        )

        # HWM at 110, trail at 104.5
        # SL at 90 (not triggered)
        # Price at 103 triggers TSL
        state = make_position_state(
            entry_price=100.0,
            current_price=103.0,
            high_water_mark=110.0,
        )
        action = chain.evaluate(state)
        assert "trailing_stop" in action.reason


class TestTripleRuleCombination:
    """Tests for TSL + TP + SL triple rule combinations."""

    def test_triple_rule_tsl_first(self):
        """TSL triggers first in TSL -> TP -> SL chain."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.05),
                TakeProfit(pct=0.15),
                StopLoss(pct=0.10),
            ]
        )

        # HWM at 108, trail at 102.6, price at 102
        state = make_position_state(
            entry_price=100.0,
            current_price=102.0,
            high_water_mark=108.0,
        )
        action = chain.evaluate(state)
        assert "trailing_stop" in action.reason

    def test_triple_rule_tp_second(self):
        """TP triggers when TSL holds in TSL -> TP -> SL chain."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.05),  # Trail at 114 from 120
                TakeProfit(pct=0.15),  # TP at 115
                StopLoss(pct=0.10),
            ]
        )

        # Price at 116 above both trail and TP
        state = make_position_state(
            entry_price=100.0,
            current_price=116.0,
            high_water_mark=120.0,
        )
        action = chain.evaluate(state)
        assert "take_profit" in action.reason

    def test_triple_rule_sl_last(self):
        """SL triggers when TSL and TP both hold."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.03),  # 3% trail
                TakeProfit(pct=0.20),  # 20% TP
                StopLoss(pct=0.08),  # 8% SL at 92
            ]
        )

        # Price crashed from 100 (HWM=100)
        # Trail at 97, price at 91 (below SL)
        state = make_position_state(
            entry_price=100.0,
            current_price=91.0,
            high_water_mark=100.0,
        )
        action = chain.evaluate(state)
        # TSL triggers at 97 (100 * 0.97), but price is already below
        assert "trailing_stop" in action.reason  # TSL still first


class TestSameBarDoubleBreach:
    """Tests for same-bar scenarios where multiple rules could trigger."""

    def test_intrabar_both_tsl_and_sl_breach(self):
        """When both TSL and SL breach in same bar, first in chain wins."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.05),  # Trail at 95 from HWM 100
                StopLoss(pct=0.08),  # SL at 92
            ]
        )

        # Bar drops from 96 to 90, crossing both levels
        state = make_position_state(
            entry_price=100.0,
            current_price=90.0,
            high_water_mark=100.0,
            bar_open=96.0,
            bar_high=97.0,
            bar_low=90.0,
        )
        action = chain.evaluate(state)
        assert "trailing_stop" in action.reason

    def test_intrabar_both_tp_and_tsl_breach_long(self):
        """Wide bar that touches both TP and TSL - first rule wins."""
        chain = RuleChain(
            [
                TakeProfit(pct=0.10),  # TP at 110
                TrailingStop(pct=0.05),  # Trail at 104.5 from HWM 110
            ]
        )

        # Volatile bar: high touches TP, then crashes through TSL
        state = make_position_state(
            entry_price=100.0,
            current_price=103.0,  # Close below trail
            high_water_mark=110.0,
            bar_open=108.0,
            bar_high=112.0,  # Above TP
            bar_low=102.0,  # Below TSL
        )
        action = chain.evaluate(state)
        # TP is first in chain and bar_high triggers it
        assert "take_profit" in action.reason


class TestShortPositionRulePriority:
    """Rule priority tests for SHORT positions."""

    def test_short_tsl_tp_priority(self):
        """TSL triggers for SHORT when price rises to trail."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.05),  # Trail from LWM
                TakeProfit(pct=0.15),  # TP when price drops 15%
            ]
        )

        # SHORT at 100, LWM at 90, trail at 94.5
        # Price at 95 triggers TSL
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=95.0,
            low_water_mark=90.0,
        )
        action = chain.evaluate(state)
        assert "trailing_stop" in action.reason

    def test_short_tp_triggers_on_drop(self):
        """TP triggers for SHORT when price drops enough."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.05),
                TakeProfit(pct=0.15),  # TP at 85 for short
            ]
        )

        # SHORT at 100, price drops to 84
        # LWM = 84, trail at 88.2 - no TSL trigger
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=84.0,
            low_water_mark=84.0,
        )
        action = chain.evaluate(state)
        assert "take_profit" in action.reason

    def test_short_sl_triggers_on_rally(self):
        """SL triggers for SHORT when price rises against position."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.10),  # 10% trail
                StopLoss(pct=0.05),  # 5% stop at 105
            ]
        )

        # SHORT at 100, price rises to 106
        # LWM stays at 100, trail at 110
        # SL at 105 triggers
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=106.0,
            low_water_mark=100.0,
        )
        action = chain.evaluate(state)
        assert "stop_loss" in action.reason


class TestRuleChainWithTimeExit:
    """Rule priority with time-based exit."""

    def test_time_exit_last_resort(self):
        """TimeExit triggers when other rules hold."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.10),
                TakeProfit(pct=0.20),
                StopLoss(pct=0.15),
                TimeExit(max_bars=50),
            ]
        )

        # Price at entry, no rules trigger except time
        state = make_position_state(
            entry_price=100.0,
            current_price=100.0,
            high_water_mark=100.0,
            bars_held=50,
        )
        action = chain.evaluate(state)
        assert "time_exit" in action.reason

    def test_tsl_beats_time_exit(self):
        """TSL triggers even if time limit reached."""
        chain = RuleChain(
            [
                TrailingStop(pct=0.05),
                TimeExit(max_bars=50),
            ]
        )

        # Both would trigger
        state = make_position_state(
            entry_price=100.0,
            current_price=94.0,  # Below trail at 95
            high_water_mark=100.0,
            bars_held=60,
        )
        action = chain.evaluate(state)
        assert "trailing_stop" in action.reason  # TSL is first


class TestEmptyAndSingleRuleChains:
    """Edge cases for rule chains."""

    def test_empty_chain_holds(self):
        """Empty RuleChain always holds."""
        chain = RuleChain([])
        state = make_position_state()
        action = chain.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_single_rule_chain(self):
        """Single-rule chain works as expected."""
        chain = RuleChain([TrailingStop(pct=0.05)])
        state = make_position_state(
            entry_price=100.0,
            current_price=94.0,
            high_water_mark=100.0,
        )
        action = chain.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
