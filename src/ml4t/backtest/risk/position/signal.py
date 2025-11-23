"""Signal-based exit rules - exits triggered by strategy signals."""

from dataclasses import dataclass

from ..types import PositionAction, PositionState


@dataclass
class SignalExit:
    """Exit when strategy provides an exit signal.

    Looks for a signal in the position's context dict. For long positions,
    exits on negative signals; for short positions, exits on positive signals.

    Args:
        signal_name: Key to look for in state.context
        threshold: Signal must exceed this threshold to trigger
                   (in absolute value terms)

    Example:
        # In strategy.on_data():
        broker.update_position_context("AAPL", {"exit_signal": -0.5})

        # Rule definition:
        rule = SignalExit(signal_name="exit_signal", threshold=0.3)
        # Long exits if exit_signal < -0.3
        # Short exits if exit_signal > 0.3
    """

    signal_name: str = "exit_signal"
    threshold: float = 0.0

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if signal exceeds threshold."""
        signal = state.context.get(self.signal_name)
        if signal is None:
            return PositionAction.hold()

        if state.is_long:
            # Long position: exit on negative signal
            if signal < -self.threshold:
                return PositionAction.exit_full(f"signal_exit_{signal:.2f}")
        else:
            # Short position: exit on positive signal
            if signal > self.threshold:
                return PositionAction.exit_full(f"signal_exit_{signal:.2f}")

        return PositionAction.hold()


@dataclass
class VolatilityTrailingStop:
    """Trailing stop based on volatility (e.g., ATR).

    Trail distance is dynamically adjusted based on a volatility measure
    provided in the position context.

    Args:
        volatility_key: Key in context containing current volatility value
        multiplier: Trail = volatility * multiplier

    Example:
        # In strategy.on_data():
        broker.update_position_context("AAPL", {"atr": 2.50})

        # Rule definition:
        rule = VolatilityTrailingStop(volatility_key="atr", multiplier=2.0)
        # Trail distance = 2.50 * 2.0 = 5.0 price units
    """

    volatility_key: str = "atr"
    multiplier: float = 2.0

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if price retraces beyond volatility-based trail."""
        vol = state.context.get(self.volatility_key)
        if vol is None or vol <= 0:
            return PositionAction.hold()

        trail_distance = vol * self.multiplier

        if state.is_long:
            stop_price = state.high_water_mark - trail_distance
            if state.current_price <= stop_price:
                return PositionAction.exit_full(f"vol_trail_{self.multiplier}x_{vol:.2f}")
        else:
            stop_price = state.low_water_mark + trail_distance
            if state.current_price >= stop_price:
                return PositionAction.exit_full(f"vol_trail_{self.multiplier}x_{vol:.2f}")

        return PositionAction.hold()
