"""Unit tests for signal-based exit rules (SignalExit, VolatilityTrailingStop)."""

from datetime import datetime

from ml4t.backtest.risk.position.signal import SignalExit, VolatilityTrailingStop
from ml4t.backtest.risk.types import ActionType, PositionState


def make_position_state(
    side: str = "long",
    entry_price: float = 100.0,
    current_price: float = 100.0,
    high_water_mark: float | None = None,
    low_water_mark: float | None = None,
    context: dict | None = None,
) -> PositionState:
    """Helper to create PositionState for testing."""
    if high_water_mark is None:
        high_water_mark = max(entry_price, current_price)
    if low_water_mark is None:
        low_water_mark = min(entry_price, current_price)

    return PositionState(
        asset="AAPL",
        side=side,
        entry_price=entry_price,
        current_price=current_price,
        quantity=100,
        initial_quantity=100,
        unrealized_pnl=0.0,
        unrealized_return=0.0,
        bars_held=0,
        high_water_mark=high_water_mark,
        low_water_mark=low_water_mark,
        entry_time=datetime.now(),
        current_time=datetime.now(),
        context=context or {},
    )


class TestSignalExitLong:
    """Tests for SignalExit with long positions."""

    def test_no_signal_holds(self):
        """No exit when signal is not present in context."""
        rule = SignalExit(signal_name="exit_signal")
        state = make_position_state(side="long", context={})
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_positive_signal_holds(self):
        """Long position holds on positive signal."""
        rule = SignalExit(signal_name="exit_signal", threshold=0.0)
        state = make_position_state(side="long", context={"exit_signal": 0.5})
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_negative_signal_exits(self):
        """Long position exits on negative signal."""
        rule = SignalExit(signal_name="exit_signal", threshold=0.0)
        state = make_position_state(side="long", context={"exit_signal": -0.5})
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "signal_exit" in action.reason

    def test_threshold_respected(self):
        """Signal must exceed threshold to trigger."""
        rule = SignalExit(signal_name="exit_signal", threshold=0.3)

        # Signal at -0.2 (below threshold magnitude)
        state1 = make_position_state(side="long", context={"exit_signal": -0.2})
        action1 = rule.evaluate(state1)
        assert action1.action == ActionType.HOLD

        # Signal at -0.5 (exceeds threshold)
        state2 = make_position_state(side="long", context={"exit_signal": -0.5})
        action2 = rule.evaluate(state2)
        assert action2.action == ActionType.EXIT_FULL

    def test_custom_signal_name(self):
        """Can use custom signal name."""
        rule = SignalExit(signal_name="my_signal", threshold=0.0)
        state = make_position_state(side="long", context={"my_signal": -1.0})
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL


class TestSignalExitShort:
    """Tests for SignalExit with short positions."""

    def test_negative_signal_holds(self):
        """Short position holds on negative signal."""
        rule = SignalExit(signal_name="exit_signal", threshold=0.0)
        state = make_position_state(side="short", context={"exit_signal": -0.5})
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_positive_signal_exits(self):
        """Short position exits on positive signal."""
        rule = SignalExit(signal_name="exit_signal", threshold=0.0)
        state = make_position_state(side="short", context={"exit_signal": 0.5})
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_threshold_respected(self):
        """Signal must exceed threshold for shorts too."""
        rule = SignalExit(signal_name="exit_signal", threshold=0.3)

        # Signal at +0.2 (below threshold)
        state1 = make_position_state(side="short", context={"exit_signal": 0.2})
        action1 = rule.evaluate(state1)
        assert action1.action == ActionType.HOLD

        # Signal at +0.5 (exceeds threshold)
        state2 = make_position_state(side="short", context={"exit_signal": 0.5})
        action2 = rule.evaluate(state2)
        assert action2.action == ActionType.EXIT_FULL


class TestVolatilityTrailingStopLong:
    """Tests for VolatilityTrailingStop with long positions."""

    def test_no_volatility_holds(self):
        """No exit when volatility is not in context."""
        rule = VolatilityTrailingStop(volatility_key="atr", multiplier=2.0)
        state = make_position_state(side="long", context={})
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_zero_volatility_holds(self):
        """No exit when volatility is zero."""
        rule = VolatilityTrailingStop(volatility_key="atr", multiplier=2.0)
        state = make_position_state(side="long", context={"atr": 0.0})
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_negative_volatility_holds(self):
        """No exit when volatility is negative (invalid)."""
        rule = VolatilityTrailingStop(volatility_key="atr", multiplier=2.0)
        state = make_position_state(side="long", context={"atr": -1.0})
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_trail_based_on_volatility(self):
        """Trail distance calculated from volatility."""
        rule = VolatilityTrailingStop(volatility_key="atr", multiplier=2.0)
        # ATR = 2.50, multiplier = 2.0, trail = 5.0
        # HWM = 110, stop at 105
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=104.0,  # Below stop
            high_water_mark=110.0,
            context={"atr": 2.50},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL
        assert "vol_trail" in action.reason

    def test_above_trail_holds(self):
        """No exit when price is above volatility trail."""
        rule = VolatilityTrailingStop(volatility_key="atr", multiplier=2.0)
        # ATR = 2.50, multiplier = 2.0, trail = 5.0
        # HWM = 110, stop at 105
        state = make_position_state(
            side="long",
            entry_price=100.0,
            current_price=106.0,  # Above stop
            high_water_mark=110.0,
            context={"atr": 2.50},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD

    def test_custom_volatility_key(self):
        """Can use custom volatility key."""
        rule = VolatilityTrailingStop(volatility_key="my_vol", multiplier=3.0)
        # vol = 1.0, multiplier = 3.0, trail = 3.0
        # HWM = 100, stop at 97
        state = make_position_state(
            side="long",
            current_price=96.0,  # Below stop
            high_water_mark=100.0,
            context={"my_vol": 1.0},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL


class TestVolatilityTrailingStopShort:
    """Tests for VolatilityTrailingStop with short positions."""

    def test_trail_for_shorts(self):
        """Trail calculated from low water mark for shorts."""
        rule = VolatilityTrailingStop(volatility_key="atr", multiplier=2.0)
        # ATR = 2.50, multiplier = 2.0, trail = 5.0
        # LWM = 90, stop at 95
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=96.0,  # Above stop
            low_water_mark=90.0,
            context={"atr": 2.50},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.EXIT_FULL

    def test_below_trail_holds(self):
        """No exit when price is below volatility trail for shorts."""
        rule = VolatilityTrailingStop(volatility_key="atr", multiplier=2.0)
        # ATR = 2.50, multiplier = 2.0, trail = 5.0
        # LWM = 90, stop at 95
        state = make_position_state(
            side="short",
            entry_price=100.0,
            current_price=93.0,  # Below stop
            low_water_mark=90.0,
            context={"atr": 2.50},
        )
        action = rule.evaluate(state)
        assert action.action == ActionType.HOLD
