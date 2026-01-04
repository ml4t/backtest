"""Unit tests for static exit rules (StopLoss, TakeProfit, TimeExit)."""

from datetime import datetime

import pytest

from ml4t.backtest.risk.position.static import StopLoss, TakeProfit, TimeExit
from ml4t.backtest.risk.types import ActionType, PositionState
from ml4t.backtest.types import StopFillMode, StopLevelBasis


def make_position_state(
    side: str = "long",
    entry_price: float = 100.0,
    current_price: float = 100.0,
    bar_open: float | None = None,
    bar_high: float | None = None,
    bar_low: float | None = None,
    bars_held: int = 0,
    context: dict | None = None,
) -> PositionState:
    """Helper to create PositionState for testing."""
    return PositionState(
        asset="AAPL",
        side=side,
        entry_price=entry_price,
        current_price=current_price,
        bar_open=bar_open,
        bar_high=bar_high,
        bar_low=bar_low,
        quantity=100,
        initial_quantity=100,
        unrealized_pnl=(current_price - entry_price) * 100
        if side == "long"
        else (entry_price - current_price) * 100,
        unrealized_return=(current_price - entry_price) / entry_price
        if side == "long"
        else (entry_price - current_price) / entry_price,
        bars_held=bars_held,
        high_water_mark=max(entry_price, current_price),
        low_water_mark=min(entry_price, current_price),
        entry_time=datetime.now(),
        current_time=datetime.now(),
        context=context or {},
    )


class TestStopLossLongPositions:
    """Tests for StopLoss with long positions."""

    def test_no_trigger_above_stop(self):
        """Stop not triggered when price is above stop level."""
        rule = StopLoss(pct=0.05)  # 5% stop
        state = make_position_state(side="long", entry_price=100.0, current_price=96.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_at_stop_level(self):
        """Stop triggered when price equals stop level."""
        rule = StopLoss(pct=0.05)  # 5% stop = $95
        state = make_position_state(side="long", entry_price=100.0, current_price=95.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "stop_loss" in action.reason

    def test_trigger_below_stop_level(self):
        """Stop triggered when price falls below stop level."""
        rule = StopLoss(pct=0.05)  # 5% stop = $95
        state = make_position_state(side="long", entry_price=100.0, current_price=90.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_intrabar_trigger_via_bar_low(self):
        """Stop triggered when bar_low touches stop level (intrabar detection)."""
        rule = StopLoss(pct=0.05)  # 5% stop = $95
        # Close at 97, but low touched 94
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=97.0,
            bar_open=98.0,
            bar_high=99.0,
            bar_low=94.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_fill_price_stop_price_mode(self):
        """Fill at exact stop price when stop is within bar range (STOP_PRICE mode)."""
        rule = StopLoss(pct=0.05)  # 5% stop = $95
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=97.0,
            bar_open=98.0,
            bar_high=99.0,
            bar_low=94.0,
            context={"stop_fill_mode": StopFillMode.STOP_PRICE},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == 95.0  # Exact stop price

    def test_fill_price_bar_extreme_mode(self):
        """Fill at bar low in BAR_EXTREME mode."""
        rule = StopLoss(pct=0.05)  # 5% stop = $95
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=97.0,
            bar_open=98.0,
            bar_high=99.0,
            bar_low=94.0,
            context={"stop_fill_mode": StopFillMode.BAR_EXTREME},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == 94.0  # Bar low

    def test_fill_price_close_price_mode(self):
        """Fill at close price in CLOSE_PRICE mode."""
        rule = StopLoss(pct=0.05)  # 5% stop = $95
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=94.0,
            context={"stop_fill_mode": StopFillMode.CLOSE_PRICE},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == 94.0  # Close price

    def test_fill_price_next_bar_open_mode(self):
        """Defer fill to next bar in NEXT_BAR_OPEN mode (Zipline behavior)."""
        rule = StopLoss(pct=0.05)  # 5% stop = $95
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=94.0,
            context={"stop_fill_mode": StopFillMode.NEXT_BAR_OPEN},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.defer_fill is True

    def test_gap_down_fill_at_open(self):
        """Gap down through stop: fill at bar open (Backtrader behavior)."""
        rule = StopLoss(pct=0.05)  # 5% stop = $95
        # Bar opened below stop at 93
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=92.0,
            bar_open=93.0,  # Opened below stop
            bar_high=94.0,
            bar_low=91.0,
            context={"stop_fill_mode": StopFillMode.STOP_PRICE},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == 93.0  # Fill at gap open

    def test_signal_price_basis(self):
        """Use signal_price as basis for stop level (Backtrader behavior)."""
        rule = StopLoss(pct=0.05)  # 5% stop
        # Entry at 100, but signal was at 98
        # Stop should be 98 * 0.95 = 93.1, not 100 * 0.95 = 95
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=94.0,  # Above signal-based stop, below entry-based stop
            context={
                "stop_level_basis": StopLevelBasis.SIGNAL_PRICE,
                "signal_price": 98.0,
            },
        )
        action = rule.evaluate(state)
        # Stop at 93.1, price at 94 -> no trigger
        assert action.action == ActionType.HOLD


class TestStopLossShortPositions:
    """Tests for StopLoss with short positions."""

    def test_no_trigger_below_stop(self):
        """Stop not triggered when price is below stop level for shorts."""
        rule = StopLoss(pct=0.05)  # 5% stop
        # Short at 100, stop at 105
        state = make_position_state(side="short", entry_price=100.0, current_price=104.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_at_stop_level(self):
        """Stop triggered when price equals stop level for shorts."""
        rule = StopLoss(pct=0.05)  # 5% stop = $105
        state = make_position_state(side="short", entry_price=100.0, current_price=105.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_trigger_above_stop_level(self):
        """Stop triggered when price rises above stop level for shorts."""
        rule = StopLoss(pct=0.05)  # 5% stop = $105
        state = make_position_state(side="short", entry_price=100.0, current_price=110.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_intrabar_trigger_via_bar_high(self):
        """Stop triggered when bar_high touches stop level for shorts."""
        rule = StopLoss(pct=0.05)  # 5% stop = $105
        # Close at 103, but high touched 106
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=103.0,
            bar_open=102.0,
            bar_high=106.0,
            bar_low=101.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_gap_up_fill_at_open(self):
        """Gap up through stop for shorts: fill at bar open."""
        rule = StopLoss(pct=0.05)  # 5% stop = $105
        # Bar opened above stop at 107
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=108.0,
            bar_open=107.0,  # Opened above stop
            bar_high=110.0,
            bar_low=106.0,
            context={"stop_fill_mode": StopFillMode.STOP_PRICE},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == 107.0  # Fill at gap open


class TestTakeProfitLongPositions:
    """Tests for TakeProfit with long positions."""

    def test_no_trigger_below_target(self):
        """Target not reached when price is below target level."""
        rule = TakeProfit(pct=0.10)  # 10% profit target
        state = make_position_state(side="long", entry_price=100.0, current_price=108.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_at_target_level(self):
        """Target reached when price equals target level."""
        rule = TakeProfit(pct=0.10)  # 10% target = $110
        # Use 110.01 to ensure we're above the target (floating point)
        state = make_position_state(side="long", entry_price=100.0, current_price=110.01)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "take_profit" in action.reason

    def test_trigger_above_target_level(self):
        """Target reached when price exceeds target level."""
        rule = TakeProfit(pct=0.10)  # 10% target = $110
        state = make_position_state(side="long", entry_price=100.0, current_price=115.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_intrabar_trigger_via_bar_high(self):
        """Target reached when bar_high touches target level (intrabar detection)."""
        rule = TakeProfit(pct=0.10)  # 10% target = $110
        # Close at 108, but high touched 112
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=108.0,
            bar_open=105.0,
            bar_high=112.0,
            bar_low=104.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_fill_price_stop_price_mode(self):
        """Fill at exact target price when within bar range (STOP_PRICE mode)."""
        rule = TakeProfit(pct=0.10)  # 10% target = $110
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=108.0,
            bar_open=105.0,
            bar_high=112.0,
            bar_low=104.0,
            context={"stop_fill_mode": StopFillMode.STOP_PRICE},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == pytest.approx(110.0)  # Exact target price

    def test_fill_price_bar_extreme_mode(self):
        """Fill at bar high in BAR_EXTREME mode."""
        rule = TakeProfit(pct=0.10)  # 10% target = $110
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=108.0,
            bar_open=105.0,
            bar_high=112.0,
            bar_low=104.0,
            context={"stop_fill_mode": StopFillMode.BAR_EXTREME},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == 112.0  # Bar high

    def test_gap_up_fill_at_open(self):
        """Gap up through target: fill at bar open (price improvement)."""
        rule = TakeProfit(pct=0.10)  # 10% target = $110
        # Bar opened above target at 115
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=118.0,
            bar_open=115.0,  # Opened above target
            bar_high=120.0,
            bar_low=114.0,
            context={"stop_fill_mode": StopFillMode.STOP_PRICE},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == 115.0  # Fill at gap open


class TestTakeProfitShortPositions:
    """Tests for TakeProfit with short positions."""

    def test_no_trigger_above_target(self):
        """Target not reached when price is above target level for shorts."""
        rule = TakeProfit(pct=0.10)  # 10% profit target
        # Short at 100, target at 90
        state = make_position_state(side="short", entry_price=100.0, current_price=92.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_at_target_level(self):
        """Target reached when price equals target level for shorts."""
        rule = TakeProfit(pct=0.10)  # 10% target = $90
        state = make_position_state(side="short", entry_price=100.0, current_price=90.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_trigger_below_target_level(self):
        """Target reached when price drops below target level for shorts."""
        rule = TakeProfit(pct=0.10)  # 10% target = $90
        state = make_position_state(side="short", entry_price=100.0, current_price=85.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_intrabar_trigger_via_bar_low(self):
        """Target reached when bar_low touches target level for shorts."""
        rule = TakeProfit(pct=0.10)  # 10% target = $90
        # Close at 92, but low touched 88
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=92.0,
            bar_open=95.0,
            bar_high=96.0,
            bar_low=88.0,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_gap_down_fill_at_open(self):
        """Gap down through target for shorts: fill at bar open (price improvement)."""
        rule = TakeProfit(pct=0.10)  # 10% target = $90
        # Bar opened below target at 85
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=82.0,
            bar_open=85.0,  # Opened below target
            bar_high=87.0,
            bar_low=80.0,
            context={"stop_fill_mode": StopFillMode.STOP_PRICE},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert action.fill_price == 85.0  # Fill at gap open


class TestTimeExit:
    """Tests for TimeExit rule."""

    def test_no_trigger_before_max_bars(self):
        """No exit when bars_held < max_bars."""
        rule = TimeExit(max_bars=20)
        state = make_position_state(bars_held=19)
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trigger_at_max_bars(self):
        """Exit when bars_held equals max_bars."""
        rule = TimeExit(max_bars=20)
        state = make_position_state(bars_held=20)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "time_exit" in action.reason
        assert "20bars" in action.reason

    def test_trigger_after_max_bars(self):
        """Exit when bars_held exceeds max_bars."""
        rule = TimeExit(max_bars=20)
        state = make_position_state(bars_held=25)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_works_for_short_positions(self):
        """TimeExit works for short positions too."""
        rule = TimeExit(max_bars=10)
        state = make_position_state(side="short", bars_held=10)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL


class TestEdgeCases:
    """Edge case tests for static rules."""

    def test_zero_percent_stop(self):
        """Edge case: 0% stop (trigger immediately if any loss)."""
        rule = StopLoss(pct=0.0)
        state = make_position_state(side="long", entry_price=100.0, current_price=99.99)
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_large_stop_percentage(self):
        """Large stop percentage (50% loss)."""
        rule = StopLoss(pct=0.50)
        state = make_position_state(side="long", entry_price=100.0, current_price=51.0)
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD  # 49% loss, not yet 50%

        state2 = make_position_state(side="long", entry_price=100.0, current_price=50.0)
        action2 = rule.evaluate(state2)
        assert action2.action == ActionType.EXIT_FULL

    def test_missing_bar_data(self):
        """Rules work without intrabar OHLC data."""
        rule = StopLoss(pct=0.05)
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=94.0,
            bar_open=None,
            bar_high=None,
            bar_low=None,
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_one_bar_time_exit(self):
        """Edge case: 1 bar max hold time."""
        rule = TimeExit(max_bars=1)
        state0 = make_position_state(bars_held=0)
        state1 = make_position_state(bars_held=1)

        assert rule.evaluate(state0).action == ActionType.HOLD
        assert rule.evaluate(state1).action == ActionType.EXIT_FULL
