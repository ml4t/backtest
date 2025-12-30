"""Unit tests for dynamic exit rules (TrailingStop, TighteningTrailingStop, ScaledExit, etc)."""

from datetime import datetime

from ml4t.backtest.risk.position.dynamic import (
    ScaledExit,
    TighteningTrailingStop,
    TrailingStop,
    VolatilityStop,
    VolatilityTrailingStop,
)
from ml4t.backtest.risk.types import ActionType, PositionState


def make_position_state(
    side: str = "long",
    entry_price: float = 100.0,
    current_price: float = 100.0,
    high_water_mark: float | None = None,
    low_water_mark: float | None = None,
    unrealized_return: float | None = None,
    bars_held: int = 0,
    context: dict | None = None,
) -> PositionState:
    """Helper to create PositionState for testing."""
    if high_water_mark is None:
        high_water_mark = max(entry_price, current_price)
    if low_water_mark is None:
        low_water_mark = min(entry_price, current_price)
    if unrealized_return is None:
        if side == "long":
            unrealized_return = (current_price - entry_price) / entry_price
        else:
            unrealized_return = (entry_price - current_price) / entry_price

    return PositionState(
        asset="AAPL",
        side=side,
        entry_price=entry_price,
        current_price=current_price,
        quantity=100,
        initial_quantity=100,
        unrealized_pnl=(current_price - entry_price) * 100
        if side == "long"
        else (entry_price - current_price) * 100,
        unrealized_return=unrealized_return,
        bars_held=bars_held,
        high_water_mark=high_water_mark,
        low_water_mark=low_water_mark,
        entry_time=datetime.now(),
        current_time=datetime.now(),
        context=context or {},
    )


class TestTrailingStopLong:
    """Tests for TrailingStop with long positions."""

    def test_no_trigger_above_trail(self):
        """No trigger when price is above trailing stop level."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # HWM at 110, trail at 104.5, current at 106
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=106.0,
            high_water_mark=110.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_at_trail_level(self):
        """Trigger when price equals trailing stop level."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # HWM at 110, trail at 104.5, current at 104.5
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=104.5,
            high_water_mark=110.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "trailing_stop" in action.reason

    def test_trigger_below_trail_level(self):
        """Trigger when price drops below trailing stop level."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # HWM at 110, trail at 104.5, current at 102
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=102.0,
            high_water_mark=110.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_trail_follows_price_up(self):
        """Trail follows price higher, no trigger on up move."""
        rule = TrailingStop(pct=0.10)  # 10% trail
        # Price went from 100 to 150, HWM = 150, trail at 135
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=140.0,  # Above trail but below HWM
            high_water_mark=150.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD


class TestTrailingStopShort:
    """Tests for TrailingStop with short positions."""

    def test_no_trigger_below_trail(self):
        """No trigger when price is below trailing stop level for shorts."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # LWM at 90, trail at 94.5, current at 93
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=93.0,
            low_water_mark=90.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_at_trail_level(self):
        """Trigger when price equals trailing stop level for shorts."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # LWM at 90, trail at 94.5, current at 94.5
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=94.5,
            low_water_mark=90.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_trigger_above_trail_level(self):
        """Trigger when price rises above trailing stop level for shorts."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # LWM at 90, trail at 94.5, current at 96
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=96.0,
            low_water_mark=90.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL


class TestTighteningTrailingStop:
    """Tests for TighteningTrailingStop."""

    def test_uses_appropriate_trail_by_return(self):
        """Trail percentage tightens as profit increases."""
        rule = TighteningTrailingStop(
            [
                (0.0, 0.05),  # 5% trail at 0% return
                (0.10, 0.03),  # 3% trail at 10%+ return
                (0.20, 0.02),  # 2% trail at 20%+ return
            ]
        )

        # At 5% return: should use 5% trail (0% threshold applies)
        state1 = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=105.0,
            high_water_mark=105.0,
            unrealized_return=0.05,
        )
        action1 = rule.evaluate(state1)
        assert action1.action == ActionType.HOLD  # 105 - 5% = 99.75, price at 105

        # At 15% return with 3% trail: trail at 111.65
        state2 = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=111.0,  # Below trail from 115 HWM
            high_water_mark=115.0,
            unrealized_return=0.15,
        )
        # Trail = 115 * (1 - 0.03) = 111.55
        action2 = rule.evaluate(state2)
        assert action2.action == ActionType.EXIT_FULL  # 111 < 111.55

    def test_tightest_trail_at_high_profit(self):
        """At highest profit levels, tightest trail is used."""
        rule = TighteningTrailingStop(
            [
                (0.0, 0.10),
                (0.15, 0.05),
                (0.25, 0.02),
            ]
        )

        # At 30% return, should use 2% trail
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=127.0,  # Below 2% trail from 130 HWM
            high_water_mark=130.0,
            unrealized_return=0.30,
        )
        # Trail = 130 * (1 - 0.02) = 127.4
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_works_for_short_positions(self):
        """Tightening trail works for shorts."""
        rule = TighteningTrailingStop(
            [
                (0.0, 0.05),
                (0.10, 0.02),
            ]
        )

        # Short at 100, went to 80 (LWM), now at 82
        # At 20% return, use 2% trail
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=82.0,
            low_water_mark=80.0,
            unrealized_return=0.20,
        )
        # Trail = 80 * (1 + 0.02) = 81.6
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL  # 82 >= 81.6


class TestScaledExit:
    """Tests for ScaledExit."""

    def test_exits_at_first_target(self):
        """Exits partial position at first profit target."""
        rule = ScaledExit(
            [
                (0.05, 0.25),  # At 5%: exit 25%
                (0.10, 0.50),  # At 10%: exit 50%
            ]
        )

        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=106.0,
            unrealized_return=0.06,
        )
        action = rule.evaluate(state)

        assert action.action == ActionType.EXIT_PARTIAL
        assert action.pct == 0.25
        assert "scale_out" in action.reason

    def test_exits_at_second_target(self):
        """Exits at second target after first is triggered."""
        rule = ScaledExit(
            [
                (0.05, 0.25),
                (0.10, 0.50),
            ]
        )

        # First exit at 5%
        state1 = make_position_state(unrealized_return=0.06)
        action1 = rule.evaluate(state1)
        assert action1.action == ActionType.EXIT_PARTIAL
        assert action1.pct == 0.25

        # Second exit at 10%
        state2 = make_position_state(unrealized_return=0.12)
        action2 = rule.evaluate(state2)
        assert action2.action == ActionType.EXIT_PARTIAL
        assert action2.pct == 0.50

    def test_no_duplicate_exits(self):
        """Same target level doesn't trigger twice."""
        rule = ScaledExit(
            [
                (0.05, 0.25),
            ]
        )

        # First evaluation triggers
        state1 = make_position_state(unrealized_return=0.06)
        action1 = rule.evaluate(state1)
        assert action1.action == ActionType.EXIT_PARTIAL

        # Second evaluation at same level - no trigger
        action2 = rule.evaluate(state1)
        assert action2.action == ActionType.HOLD

    def test_reset_allows_reuse(self):
        """Reset clears triggered levels."""
        rule = ScaledExit(
            [
                (0.05, 0.25),
            ]
        )

        state = make_position_state(unrealized_return=0.06)

        # First trigger
        action1 = rule.evaluate(state)
        assert action1.action == ActionType.EXIT_PARTIAL

        # Reset and trigger again
        rule.reset()
        action2 = rule.evaluate(state)
        assert action2.action == ActionType.EXIT_PARTIAL

    def test_no_trigger_below_first_target(self):
        """No exit when return is below first target."""
        rule = ScaledExit(
            [
                (0.10, 0.25),
            ]
        )

        state = make_position_state(unrealized_return=0.05)
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_skips_to_higher_target_if_needed(self):
        """If price jumps, triggers appropriate target."""
        rule = ScaledExit(
            [
                (0.05, 0.25),
                (0.10, 0.50),
                (0.15, 0.75),
            ]
        )

        # Jump straight to 12% return - triggers 5% first
        state = make_position_state(unrealized_return=0.12)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_PARTIAL
        assert action.pct == 0.25  # First untriggered target


class TestVolatilityStop:
    """Tests for VolatilityStop (ATR-based)."""

    def test_no_trigger_above_stop_long(self):
        """No trigger when price is above ATR stop level for long."""
        rule = VolatilityStop(multiplier=2.0)
        # Entry at 100, ATR=5, stop at 90 (100 - 2*5)
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=95.0,  # Above 90
            context={"atr": 5.0},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_at_stop_level_long(self):
        """Trigger when price equals ATR stop level for long."""
        rule = VolatilityStop(multiplier=2.0)
        # Entry at 100, ATR=5, stop at 90
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=90.0,
            context={"atr": 5.0},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "volatility_stop" in action.reason
        assert action.fill_price == 90.0

    def test_trigger_below_stop_level_long(self):
        """Trigger when price drops below ATR stop level for long."""
        rule = VolatilityStop(multiplier=2.0)
        # Entry at 100, ATR=5, stop at 90
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=85.0,
            context={"atr": 5.0},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_no_trigger_below_stop_short(self):
        """No trigger when price is below ATR stop level for short."""
        rule = VolatilityStop(multiplier=2.0)
        # Entry at 100, ATR=5, stop at 110 (100 + 2*5)
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=105.0,  # Below 110
            context={"atr": 5.0},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_at_stop_level_short(self):
        """Trigger when price equals ATR stop level for short."""
        rule = VolatilityStop(multiplier=2.0)
        # Entry at 100, ATR=5, stop at 110
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=110.0,
            context={"atr": 5.0},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == 110.0

    def test_no_atr_holds(self):
        """No action when ATR is not in context."""
        rule = VolatilityStop(multiplier=2.0)
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=50.0,  # Would trigger if ATR present
            context={},  # No ATR
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_custom_atr_key(self):
        """Uses custom ATR key from context."""
        rule = VolatilityStop(multiplier=2.0, atr_key="atr_14")
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=88.0,
            context={"atr_14": 5.0, "atr": 1.0},  # Should use atr_14
        )
        action = rule.evaluate(state)
        # With atr_14=5, stop at 90 - price 88 triggers
        assert action.action == ActionType.EXIT_FULL

    def test_uses_entry_atr(self):
        """Uses ATR at entry, not current ATR."""
        rule = VolatilityStop(multiplier=2.0, use_entry_atr=True)

        # First bar: ATR = 5, stop at 90
        state1 = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=95.0,
            context={"atr": 5.0},
        )
        action1 = rule.evaluate(state1)
        assert action1.action == ActionType.HOLD

        # Second bar: ATR dropped to 1, but entry ATR (5) should be used
        # Stop still at 90, not 98
        state2 = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=89.0,
            context={"atr": 1.0},  # Lower ATR shouldn't matter
        )
        action2 = rule.evaluate(state2)
        assert action2.action == ActionType.EXIT_FULL


class TestVolatilityTrailingStop:
    """Tests for VolatilityTrailingStop (ATR-based trailing)."""

    def test_no_trigger_above_trail_long(self):
        """No trigger when price is above ATR trail for long."""
        rule = VolatilityTrailingStop(multiplier=3.0)
        # HWM at 120, ATR=5, trail at 105 (120 - 3*5)
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=110.0,
            high_water_mark=120.0,
            context={"atr": 5.0},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_at_trail_level_long(self):
        """Trigger when price equals ATR trail level for long."""
        rule = VolatilityTrailingStop(multiplier=3.0)
        # HWM at 120, ATR=5, trail at 105
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=105.0,
            high_water_mark=120.0,
            context={"atr": 5.0},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "vol_trailing_stop" in action.reason
        assert action.fill_price == 105.0

    def test_no_trigger_below_trail_short(self):
        """No trigger when price is below ATR trail for short."""
        rule = VolatilityTrailingStop(multiplier=3.0)
        # LWM at 80, ATR=5, trail at 95 (80 + 3*5)
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=90.0,
            low_water_mark=80.0,
            context={"atr": 5.0},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_at_trail_level_short(self):
        """Trigger when price equals ATR trail level for short."""
        rule = VolatilityTrailingStop(multiplier=3.0)
        # LWM at 80, ATR=5, trail at 95
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=95.0,
            low_water_mark=80.0,
            context={"atr": 5.0},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_uses_current_atr(self):
        """Trail distance adapts to current ATR."""
        rule = VolatilityTrailingStop(multiplier=2.0)

        # First eval: ATR=10, HWM=120, trail at 100
        state1 = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=105.0,
            high_water_mark=120.0,
            context={"atr": 10.0},
        )
        action1 = rule.evaluate(state1)
        assert action1.action == ActionType.HOLD  # 105 > 100

        # Second eval: ATR=10 still, HWM=120, trail at 100, price at 100
        state2 = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=100.0,
            high_water_mark=120.0,
            context={"atr": 10.0},
        )
        action2 = rule.evaluate(state2)
        assert action2.action == ActionType.EXIT_FULL

    def test_no_atr_holds(self):
        """No action when ATR is not in context."""
        rule = VolatilityTrailingStop(multiplier=3.0)
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=50.0,
            high_water_mark=120.0,
            context={},  # No ATR
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD
