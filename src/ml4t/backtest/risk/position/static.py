"""Static exit rules - fixed thresholds that don't change."""

from dataclasses import dataclass

from ..types import PositionAction, PositionState


def _get_stop_fill_mode(context: dict):
    """Get StopFillMode from context, defaulting to STOP_PRICE."""
    from ml4t.backtest.types import StopFillMode

    return context.get("stop_fill_mode", StopFillMode.STOP_PRICE)


def _get_stop_base_price(state, context: dict) -> float:
    """Get the base price for stop level calculation.

    If stop_level_basis is SIGNAL_PRICE and signal_price is available,
    use signal_price (Backtrader behavior). Otherwise use entry_price.
    """
    from ml4t.backtest.types import StopLevelBasis

    basis = context.get("stop_level_basis", StopLevelBasis.FILL_PRICE)
    if basis == StopLevelBasis.SIGNAL_PRICE:
        signal_price = context.get("signal_price")
        if signal_price is not None:
            return signal_price
    return state.entry_price


@dataclass
class StopLoss:
    """Exit when stop price is breached during the bar.

    Stop orders trigger when the bar's price range touches the stop level.
    Fill price depends on StopFillMode configuration:
    - STOP_PRICE: Fill at exact stop price (standard model, default)
    - BAR_EXTREME: Fill at bar's low (matches VectorBT Pro behavior)

    For long positions: stop triggers if bar_low <= stop_price
    For short positions: stop triggers if bar_high >= stop_price

    Args:
        pct: Maximum loss as decimal (0.05 = 5% loss triggers exit)

    Example:
        rule = StopLoss(pct=0.05)  # Exit at -5%
    """

    pct: float

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if stop price was breached during the bar."""
        from ml4t.backtest.types import StopFillMode

        # Get base price for stop level calculation (entry_price or signal_price)
        base_price = _get_stop_base_price(state, state.context)

        # Calculate stop price from base
        if state.is_long:
            stop_price = base_price * (1 - self.pct)
            # Check if stop was triggered during bar (low touched stop level)
            triggered = (
                state.bar_low is not None and state.bar_low <= stop_price
            ) or state.current_price <= stop_price
        else:  # short
            stop_price = base_price * (1 + self.pct)
            # Check if stop was triggered during bar (high touched stop level)
            triggered = (
                state.bar_high is not None and state.bar_high >= stop_price
            ) or state.current_price >= stop_price

        if triggered:
            # Determine fill price based on mode
            fill_mode = _get_stop_fill_mode(state.context)
            if fill_mode == StopFillMode.NEXT_BAR_OPEN:
                # Zipline model: defer exit to next bar, fill at open
                return PositionAction.exit_full(
                    reason=f"stop_loss_{self.pct:.1%}",
                    defer_fill=True,  # Broker will fill at next bar's open
                )
            elif fill_mode == StopFillMode.CLOSE_PRICE:
                # VectorBT Pro close-only model: always fill at close price
                fill_price = state.current_price
            elif fill_mode == StopFillMode.BAR_EXTREME:
                # Conservative model: fill at bar's extreme (worst case)
                if state.is_long:
                    fill_price = state.bar_low if state.bar_low is not None else stop_price
                else:
                    fill_price = state.bar_high if state.bar_high is not None else stop_price
            else:
                # Standard model (STOP_PRICE): fill at exact stop price if within bar range
                # If bar gaps through stop, fill at open (gap behavior)
                if state.is_long:
                    # For long stops: check if bar opened below stop (gap down)
                    # or if stop is within bar range
                    if state.bar_open is not None and state.bar_open <= stop_price:
                        # Bar opened below stop - fill at open (Backtrader gap behavior)
                        fill_price = state.bar_open
                    elif (
                        state.bar_low is not None
                        and state.bar_high is not None
                        and state.bar_low <= stop_price <= state.bar_high
                    ):
                        # Stop within bar range - fill at exact stop
                        fill_price = stop_price
                    else:
                        # Gap through (VBT behavior) - fill at close
                        fill_price = state.current_price
                else:
                    # For short stops: check if bar opened above stop (gap up)
                    if state.bar_open is not None and state.bar_open >= stop_price:
                        # Bar opened above stop - fill at open (gap behavior)
                        fill_price = state.bar_open
                    elif (
                        state.bar_low is not None
                        and state.bar_high is not None
                        and state.bar_low <= stop_price <= state.bar_high
                    ):
                        # Stop within bar range - fill at exact stop
                        fill_price = stop_price
                    else:
                        # Gap through (VBT behavior) - fill at close
                        fill_price = state.current_price

            return PositionAction.exit_full(
                reason=f"stop_loss_{self.pct:.1%}",
                fill_price=fill_price,
            )
        return PositionAction.hold()


@dataclass
class TakeProfit:
    """Exit when target price is reached during the bar.

    Take-profit orders trigger when the bar's price range touches the target.
    Fill price depends on StopFillMode configuration:
    - STOP_PRICE: Fill at exact target price (standard model, default)
    - BAR_EXTREME: Fill at bar's high (matches VectorBT Pro behavior)

    For long positions: triggers if bar_high >= target_price
    For short positions: triggers if bar_low <= target_price

    Args:
        pct: Target profit as decimal (0.10 = 10% profit triggers exit)

    Example:
        rule = TakeProfit(pct=0.10)  # Exit at +10%
    """

    pct: float

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if target price was reached during the bar."""
        from ml4t.backtest.types import StopFillMode

        # Get base price for target level calculation (entry_price or signal_price)
        base_price = _get_stop_base_price(state, state.context)

        # Calculate target price from base
        if state.is_long:
            target_price = base_price * (1 + self.pct)
            # Check if target was reached during bar (high touched target)
            triggered = (
                state.bar_high is not None and state.bar_high >= target_price
            ) or state.current_price >= target_price
        else:  # short
            target_price = base_price * (1 - self.pct)
            # Check if target was reached during bar (low touched target)
            triggered = (
                state.bar_low is not None and state.bar_low <= target_price
            ) or state.current_price <= target_price

        if triggered:
            # Determine fill price based on mode
            fill_mode = _get_stop_fill_mode(state.context)
            if fill_mode == StopFillMode.NEXT_BAR_OPEN:
                # Zipline model: defer exit to next bar, fill at open
                return PositionAction.exit_full(
                    reason=f"take_profit_{self.pct:.1%}",
                    defer_fill=True,  # Broker will fill at next bar's open
                )
            elif fill_mode == StopFillMode.CLOSE_PRICE:
                # VectorBT Pro close-only model: always fill at close price
                fill_price = state.current_price
            elif fill_mode == StopFillMode.BAR_EXTREME:
                # Optimistic model: fill at bar's extreme (best case for profits)
                if state.is_long:
                    fill_price = state.bar_high if state.bar_high is not None else target_price
                else:
                    fill_price = state.bar_low if state.bar_low is not None else target_price
            else:
                # Standard model (STOP_PRICE): fill at exact target price if within bar range
                # If bar gaps through target, fill at open/close (gap behavior)
                if state.is_long:
                    # For long targets: check if bar opened above target (price improvement)
                    # or if target is within bar range
                    if state.bar_open is not None and state.bar_open >= target_price:
                        # Bar opened above target - fill at open (Backtrader behavior)
                        fill_price = state.bar_open
                    elif (
                        state.bar_low is not None
                        and state.bar_high is not None
                        and state.bar_low <= target_price <= state.bar_high
                    ):
                        # Target within bar range - fill at exact target
                        fill_price = target_price
                    else:
                        # Gap through - fill at close
                        fill_price = state.current_price
                else:
                    # For short targets: check if bar opened below target (price improvement)
                    if state.bar_open is not None and state.bar_open <= target_price:
                        # Bar opened below target - fill at open (price improvement)
                        fill_price = state.bar_open
                    elif (
                        state.bar_low is not None
                        and state.bar_high is not None
                        and state.bar_low <= target_price <= state.bar_high
                    ):
                        # Target within bar range - fill at exact target
                        fill_price = target_price
                    else:
                        # Gap through - fill at close
                        fill_price = state.current_price

            return PositionAction.exit_full(
                reason=f"take_profit_{self.pct:.1%}",
                fill_price=fill_price,
            )
        return PositionAction.hold()


@dataclass
class TimeExit:
    """Exit after holding for a specified number of bars.

    Args:
        max_bars: Maximum bars to hold position

    Example:
        rule = TimeExit(max_bars=20)  # Exit after 20 bars
    """

    max_bars: int

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if held too long."""
        if state.bars_held >= self.max_bars:
            # Time exits fill at current close price
            return PositionAction.exit_full(f"time_exit_{self.max_bars}bars")
        return PositionAction.hold()
