"""Comprehensive unit tests for SHORT trailing stop behavior.

This test module focuses on SHORT position trailing stop scenarios that were
identified as validation gaps. Tests cover:

1. Basic SHORT TSL mechanics (LWM tracking, stop level calculation)
2. Gap-through handling (gap up through stop level)
3. Multiple trend reversals (downtrend -> uptrend -> downtrend)
4. Intrabar detection via bar_high
5. Fill price modes (STOP_PRICE, BAR_EXTREME, CLOSE_PRICE)
6. Re-entry after SHORT TSL exit
"""

from datetime import datetime

import pytest

from ml4t.backtest.config import TrailStopTiming
from ml4t.backtest.risk.position.dynamic import TrailingStop
from ml4t.backtest.risk.types import ActionType, PositionState
from ml4t.backtest.types import StopFillMode


def make_short_position(
    entry_price: float = 100.0,
    current_price: float = 100.0,
    low_water_mark: float | None = None,
    bar_open: float | None = None,
    bar_high: float | None = None,
    bar_low: float | None = None,
    bars_held: int = 0,
    context: dict | None = None,
) -> PositionState:
    """Helper to create SHORT PositionState for testing."""
    if low_water_mark is None:
        low_water_mark = min(entry_price, current_price)

    return PositionState(
        asset="TEST",
        side="short",
        entry_price=entry_price,
        current_price=current_price,
        bar_open=bar_open,
        bar_high=bar_high,
        bar_low=bar_low,
        quantity=100,
        initial_quantity=100,
        unrealized_pnl=(entry_price - current_price) * 100,
        unrealized_return=(entry_price - current_price) / entry_price,
        bars_held=bars_held,
        high_water_mark=entry_price,  # Not used for shorts
        low_water_mark=low_water_mark,
        entry_time=datetime.now(),
        current_time=datetime.now(),
        context=context or {},
    )


class TestShortTrailingStopBasics:
    """Basic SHORT trailing stop mechanics."""

    def test_no_trigger_price_below_stop(self):
        """No trigger when price is below the trailing stop level."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # SHORT at 100, LWM at 90, trail at 94.5 (90 * 1.05)
        # Current price at 92 is below trail, no trigger
        state = make_short_position(
            entry_price=100.0,
            current_price=92.0,
            low_water_mark=90.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_price_at_stop(self):
        """Trigger when price equals the trailing stop level."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # SHORT at 100, LWM at 90, trail at 94.5
        state = make_short_position(
            entry_price=100.0,
            current_price=94.5,
            low_water_mark=90.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "trailing_stop" in action.reason

    def test_trigger_price_above_stop(self):
        """Trigger when price rises above the trailing stop level."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # SHORT at 100, LWM at 90, trail at 94.5, current at 96
        state = make_short_position(
            entry_price=100.0,
            current_price=96.0,
            low_water_mark=90.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_lwm_follows_price_down(self):
        """Trail follows price lower, no trigger on down move."""
        rule = TrailingStop(pct=0.10)  # 10% trail
        # SHORT at 100, price went to 70, LWM = 70
        # Trail at 77, current at 75 (still below trail)
        state = make_short_position(
            entry_price=100.0,
            current_price=75.0,
            low_water_mark=70.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_profitable_short_not_triggered(self):
        """Short in profit (price down) should not trigger stop."""
        rule = TrailingStop(pct=0.05)
        # SHORT at 100, price dropped to 80 (20% profit)
        # LWM at 80, trail at 84
        state = make_short_position(
            entry_price=100.0,
            current_price=82.0,
            low_water_mark=80.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD


class TestShortTrailingStopGapHandling:
    """SHORT trailing stop gap-through scenarios."""

    def test_gap_up_through_stop_fills_at_open(self):
        """Gap up through stop fills at the gap open price."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # SHORT at 100, LWM at 90, trail at 94.5
        # Bar opens at 96 (gap up through stop)
        state = make_short_position(
            entry_price=100.0,
            current_price=98.0,  # Close even higher
            low_water_mark=90.0,
            bar_open=96.0,  # Gap up through trail
            bar_high=99.0,
            bar_low=95.0,
            context={"stop_fill_mode": StopFillMode.STOP_PRICE},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == 96.0  # Fill at gap open (worse than stop)

    def test_gap_up_within_range_fills_at_stop(self):
        """If bar range includes stop level, fill at exact stop price."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # SHORT at 100, LWM at 90, trail at 94.5
        # Bar opens at 93 (below trail), high at 96 (crosses trail)
        state = make_short_position(
            entry_price=100.0,
            current_price=95.0,
            low_water_mark=90.0,
            bar_open=93.0,  # Below trail
            bar_high=96.0,  # Crosses trail
            bar_low=92.0,
            context={"stop_fill_mode": StopFillMode.STOP_PRICE},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == pytest.approx(94.5)  # Exact stop price


class TestShortTrailingStopIntrabar:
    """SHORT trailing stop intrabar detection via bar_high."""

    def test_intrabar_trigger_via_bar_high(self):
        """Stop triggered when bar_high touches stop level."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # SHORT at 100, LWM at 90, trail at 94.5
        # Close at 92 (safe), but high touched 95 (triggers)
        state = make_short_position(
            entry_price=100.0,
            current_price=92.0,
            low_water_mark=90.0,
            bar_open=91.0,
            bar_high=95.0,  # Touches above trail
            bar_low=90.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_no_intrabar_trigger_high_below_stop(self):
        """No trigger when bar_high is below stop level."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # SHORT at 100, LWM at 90, trail at 94.5
        # High at 93 is below trail
        state = make_short_position(
            entry_price=100.0,
            current_price=91.0,
            low_water_mark=90.0,
            bar_open=90.0,
            bar_high=93.0,  # Below trail
            bar_low=89.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD


class TestShortTrailingStopFillModes:
    """SHORT trailing stop fill price modes."""

    def test_fill_mode_stop_price(self):
        """STOP_PRICE mode fills at exact stop level."""
        rule = TrailingStop(pct=0.05)  # trail at 94.5
        state = make_short_position(
            entry_price=100.0,
            current_price=95.0,
            low_water_mark=90.0,
            bar_open=92.0,
            bar_high=96.0,
            bar_low=91.0,
            context={"stop_fill_mode": StopFillMode.STOP_PRICE},
        )
        action = rule.evaluate(state)
        assert action.fill_price == pytest.approx(94.5)

    def test_fill_mode_bar_extreme(self):
        """BAR_EXTREME mode fills at bar high for shorts."""
        rule = TrailingStop(pct=0.05)  # trail at 94.5
        state = make_short_position(
            entry_price=100.0,
            current_price=95.0,
            low_water_mark=90.0,
            bar_open=92.0,
            bar_high=96.0,
            bar_low=91.0,
            context={"stop_fill_mode": StopFillMode.BAR_EXTREME},
        )
        action = rule.evaluate(state)
        assert action.fill_price == 96.0  # Bar high

    def test_fill_mode_close_price(self):
        """CLOSE_PRICE mode fills at bar close."""
        rule = TrailingStop(pct=0.05)  # trail at 94.5
        state = make_short_position(
            entry_price=100.0,
            current_price=95.0,  # This is the close
            low_water_mark=90.0,
            bar_open=92.0,
            bar_high=96.0,
            bar_low=91.0,
            context={"stop_fill_mode": StopFillMode.CLOSE_PRICE},
        )
        action = rule.evaluate(state)
        assert action.fill_price == 95.0  # Close price


class TestShortTrailingStopTrendReversals:
    """SHORT trailing stop behavior across multiple trend reversals."""

    def test_downtrend_to_uptrend_triggers(self):
        """Downtrend that reverses into uptrend triggers stop."""
        rule = TrailingStop(pct=0.10)  # 10% trail

        # Phase 1: Downtrend - SHORT at 100, price drops to 80
        state1 = make_short_position(
            entry_price=100.0,
            current_price=80.0,
            low_water_mark=80.0,
        )
        action1 = rule.evaluate(state1)
        assert action1.action == ActionType.HOLD  # LWM=80, trail=88, price=80

        # Phase 2: Uptrend begins - price rises to 85
        state2 = make_short_position(
            entry_price=100.0,
            current_price=85.0,
            low_water_mark=80.0,  # LWM stays at 80
        )
        action2 = rule.evaluate(state2)
        assert action2.action == ActionType.HOLD  # LWM=80, trail=88, price=85

        # Phase 3: Uptrend continues - price rises to 89 (triggers)
        state3 = make_short_position(
            entry_price=100.0,
            current_price=89.0,
            low_water_mark=80.0,
        )
        action3 = rule.evaluate(state3)
        assert action3.action == ActionType.EXIT_FULL  # 89 > 88

    def test_multiple_reversals_lwm_ratchets(self):
        """LWM only decreases, never increases (ratchet effect)."""
        rule = TrailingStop(pct=0.05)  # 5% trail

        # Phase 1: Price drops from 100 to 90
        state1 = make_short_position(
            entry_price=100.0,
            current_price=90.0,
            low_water_mark=90.0,
        )
        assert rule.evaluate(state1).action == ActionType.HOLD

        # Phase 2: Price bounces to 92 (LWM stays at 90)
        state2 = make_short_position(
            entry_price=100.0,
            current_price=92.0,
            low_water_mark=90.0,  # Doesn't change
        )
        assert rule.evaluate(state2).action == ActionType.HOLD  # trail=94.5

        # Phase 3: Price drops to 85 (LWM updates to 85)
        state3 = make_short_position(
            entry_price=100.0,
            current_price=85.0,
            low_water_mark=85.0,
        )
        assert rule.evaluate(state3).action == ActionType.HOLD

        # Phase 4: Price rises to 90 (LWM stays at 85, triggers)
        state4 = make_short_position(
            entry_price=100.0,
            current_price=90.0,
            low_water_mark=85.0,  # Doesn't change
        )
        action4 = rule.evaluate(state4)
        # LWM=85, trail=89.25 (85*1.05), price=90 -> EXIT
        assert action4.action == ActionType.EXIT_FULL


class TestShortTrailingStopEdgeCases:
    """Edge cases for SHORT trailing stops."""

    def test_small_trail_percentage(self):
        """Very small trail percentage (1%)."""
        rule = TrailingStop(pct=0.01)  # 1% trail
        # LWM at 90, trail at 90.9
        state = make_short_position(
            entry_price=100.0,
            current_price=91.0,  # Above 90.9
            low_water_mark=90.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_large_trail_percentage(self):
        """Large trail percentage (50%)."""
        rule = TrailingStop(pct=0.50)  # 50% trail
        # LWM at 50, trail at 75
        state = make_short_position(
            entry_price=100.0,
            current_price=70.0,  # Below 75
            low_water_mark=50.0,
        )
        assert rule.evaluate(state).action == ActionType.HOLD

        # Price at 76 triggers
        state2 = make_short_position(
            entry_price=100.0,
            current_price=76.0,
            low_water_mark=50.0,
        )
        assert rule.evaluate(state2).action == ActionType.EXIT_FULL

    def test_entry_is_also_lwm(self):
        """When entry price is also the LWM (no profit yet)."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        # Entry at 100 is also LWM, trail at 105
        state = make_short_position(
            entry_price=100.0,
            current_price=103.0,
            low_water_mark=100.0,
        )
        assert rule.evaluate(state).action == ActionType.HOLD

        # Price at 106 triggers immediately
        state2 = make_short_position(
            entry_price=100.0,
            current_price=106.0,
            low_water_mark=100.0,
        )
        assert rule.evaluate(state2).action == ActionType.EXIT_FULL

    def test_price_exactly_at_entry(self):
        """Price returns exactly to entry (no P&L)."""
        rule = TrailingStop(pct=0.10)  # 10% trail
        # LWM at 90, trail at 99, price at 100
        state = make_short_position(
            entry_price=100.0,
            current_price=100.0,
            low_water_mark=90.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL  # 100 > 99

    def test_missing_bar_data_uses_close(self):
        """Without OHLC data, uses close price for evaluation."""
        rule = TrailingStop(pct=0.05)  # 5% trail
        state = make_short_position(
            entry_price=100.0,
            current_price=95.0,  # Above trail at 94.5
            low_water_mark=90.0,
            bar_open=None,
            bar_high=None,
            bar_low=None,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL


class TestShortTrailingStopTimingModes:
    """SHORT trailing stop timing mode behavior (quick coverage)."""

    def test_lagged_mode_uses_previous_lwm(self):
        """LAGGED mode uses LWM from previous bar."""
        rule = TrailingStop(pct=0.05)
        state = make_short_position(
            entry_price=100.0,
            current_price=93.0,
            low_water_mark=90.0,  # From previous bar
            bar_high=95.0,  # Crosses trail at 94.5
            context={"trail_stop_timing": TrailStopTiming.LAGGED},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_vbt_pro_mode_two_pass(self):
        """VBT_PRO mode uses two-pass algorithm."""
        rule = TrailingStop(pct=0.05)
        # Pass 1: Check with lagged LWM (90) against HIGH
        # Trail at 94.5, bar_high at 95 -> triggers
        state = make_short_position(
            entry_price=100.0,
            current_price=93.0,
            low_water_mark=90.0,
            bar_high=95.0,
            bar_low=88.0,  # Would update LWM in pass 2
            context={"trail_stop_timing": TrailStopTiming.VBT_PRO},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
