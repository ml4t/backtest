"""Dynamic exit rules - thresholds that change based on position state."""

from dataclasses import dataclass, field

from ..types import PositionAction, PositionState


def _get_stop_fill_mode_for_trail(context: dict):
    """Get StopFillMode from context, defaulting to STOP_PRICE."""
    from ml4t.backtest.types import StopFillMode

    return context.get("stop_fill_mode", StopFillMode.STOP_PRICE)


@dataclass
class TrailingStop:
    """Exit when price retraces from high water mark.

    For longs: Exit if price drops X% from highest price since entry
    For shorts: Exit if price rises X% from lowest price since entry

    Fill price depends on StopFillMode configuration:
    - STOP_PRICE: Fill at exact trail level (default)
    - CLOSE_PRICE: Fill at bar's close price (VBT Pro behavior)

    Args:
        pct: Trail percentage as decimal (0.05 = 5% trail)

    Example:
        rule = TrailingStop(pct=0.05)  # 5% trailing stop
    """

    pct: float

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if price retraces beyond trail.

        Uses bar_low/bar_high for intrabar trigger detection.
        Handles gap-through: if bar opens beyond stop level, fill at open.

        Fill price depends on StopFillMode configuration:
        - STOP_PRICE: Fill at exact trail level (default, VBT Pro behavior)
        - CLOSE_PRICE: Fill at bar's close price
        - BAR_EXTREME: Fill at bar's low (long) or high (short)

        Gap-through handling: When bar opens beyond the stop level (gap down for
        longs, gap up for shorts), the fill is at the open price regardless of
        StopFillMode. This matches VBT Pro behavior.
        """
        fill_mode = _get_stop_fill_mode_for_trail(state.context)

        if state.is_long:
            # Long: stop triggers when bar's low drops below (hwm - trail%)
            # Also triggers on gap-through: when bar opens below trail level
            stop_price = state.high_water_mark * (1 - self.pct)
            bar_low = state.bar_low if state.bar_low is not None else state.current_price
            bar_open = state.bar_open if state.bar_open is not None else state.current_price
            # Check both low touch AND gap-through (open below trail)
            if bar_low <= stop_price or bar_open < stop_price:
                # Determine fill price based on StopFillMode and gap-through
                fill_price = self._get_fill_price_long(
                    stop_price, state.current_price, bar_low, bar_open, fill_mode
                )
                return PositionAction.exit_full(
                    f"trailing_stop_{self.pct:.1%}",
                    fill_price=fill_price,
                )
        else:
            # Short: stop triggers when bar's high rises above (lwm + trail%)
            # Also triggers on gap-through: when bar opens above trail level
            stop_price = state.low_water_mark * (1 + self.pct)
            bar_high = state.bar_high if state.bar_high is not None else state.current_price
            bar_open = state.bar_open if state.bar_open is not None else state.current_price
            # Check both high touch AND gap-through (open above trail)
            if bar_high >= stop_price or bar_open > stop_price:
                # Determine fill price based on StopFillMode and gap-through
                fill_price = self._get_fill_price_short(
                    stop_price, state.current_price, bar_high, bar_open, fill_mode
                )
                return PositionAction.exit_full(
                    f"trailing_stop_{self.pct:.1%}",
                    fill_price=fill_price,
                )

        return PositionAction.hold()

    def _get_fill_price_long(
        self, stop_price: float, close: float, bar_low: float, bar_open: float, fill_mode
    ) -> float:
        """Get fill price for long position exit based on StopFillMode.

        Handles gap-through: if bar opens below stop level, fill at open.
        This matches VBT Pro behavior for gap downs through the stop.
        """
        from ml4t.backtest.types import StopFillMode

        # Gap-through: if bar opens below stop, fill at open (worse price)
        if bar_open < stop_price:
            return bar_open

        if fill_mode == StopFillMode.CLOSE_PRICE:
            return close
        elif fill_mode == StopFillMode.BAR_EXTREME:
            return bar_low
        else:  # STOP_PRICE (default) or NEXT_BAR_OPEN
            return stop_price

    def _get_fill_price_short(
        self, stop_price: float, close: float, bar_high: float, bar_open: float, fill_mode
    ) -> float:
        """Get fill price for short position exit based on StopFillMode.

        Handles gap-through: if bar opens above stop level, fill at open.
        This matches VBT Pro behavior for gap ups through the stop.
        """
        from ml4t.backtest.types import StopFillMode

        # Gap-through: if bar opens above stop, fill at open (worse price)
        if bar_open > stop_price:
            return bar_open

        if fill_mode == StopFillMode.CLOSE_PRICE:
            return close
        elif fill_mode == StopFillMode.BAR_EXTREME:
            return bar_high
        else:  # STOP_PRICE (default) or NEXT_BAR_OPEN
            return stop_price


@dataclass
class TighteningTrailingStop:
    """Trailing stop that tightens as profit increases.

    The trail percentage decreases at higher profit levels, locking in
    more gains as the position becomes more profitable.

    Args:
        schedule: List of (return_threshold, trail_pct) tuples.
                  Must be sorted by return_threshold ascending.

    Example:
        rule = TighteningTrailingStop([
            (0.0, 0.05),   # At 0% return: 5% trail
            (0.10, 0.03),  # At 10%+ return: 3% trail
            (0.20, 0.02),  # At 20%+ return: 2% trail
        ])
    """

    schedule: list[tuple[float, float]]

    def __post_init__(self):
        # Sort by return threshold descending for efficient lookup
        self._schedule = sorted(self.schedule, key=lambda x: x[0], reverse=True)

    def _get_trail_pct(self, unrealized_return: float) -> float:
        """Get applicable trail percentage based on current return."""
        for threshold, trail_pct in self._schedule:
            if unrealized_return >= threshold:
                return trail_pct
        # Default to last (loosest) trail if no threshold met
        return self._schedule[-1][1] if self._schedule else 0.05

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if price retraces beyond dynamic trail."""
        trail_pct = self._get_trail_pct(state.unrealized_return)

        if state.is_long:
            stop_price = state.high_water_mark * (1 - trail_pct)
            if state.current_price <= stop_price:
                return PositionAction.exit_full(
                    f"tightening_trail_{trail_pct:.1%}_at_{state.unrealized_return:.1%}"
                )
        else:
            stop_price = state.low_water_mark * (1 + trail_pct)
            if state.current_price >= stop_price:
                return PositionAction.exit_full(
                    f"tightening_trail_{trail_pct:.1%}_at_{state.unrealized_return:.1%}"
                )

        return PositionAction.hold()


@dataclass
class ScaledExit:
    """Exit portions of position at profit targets.

    Allows scaling out of positions by exiting a percentage at each
    profit level. Tracks which levels have been triggered to avoid
    duplicate exits.

    Args:
        targets: List of (return_threshold, exit_pct) tuples.
                 exit_pct is relative to CURRENT position size.

    Example:
        rule = ScaledExit([
            (0.05, 0.25),  # At +5%: exit 25% of position
            (0.10, 0.33),  # At +10%: exit 33% of remaining
            (0.15, 0.50),  # At +15%: exit 50% of remaining
        ])

    Note:
        This rule tracks triggered levels internally. For proper operation
        in backtesting, create a new instance per position.
    """

    targets: list[tuple[float, float]]
    _triggered: set[float] = field(default_factory=set, repr=False)

    def __post_init__(self):
        # Sort by return threshold ascending
        self._targets = sorted(self.targets, key=lambda x: x[0])

    def reset(self):
        """Reset triggered levels (call when position closes)."""
        self._triggered = set()

    def evaluate(self, state: PositionState) -> PositionAction:
        """Check if any untriggered profit target is hit."""
        for threshold, exit_pct in self._targets:
            if threshold not in self._triggered and state.unrealized_return >= threshold:
                self._triggered.add(threshold)
                return PositionAction.exit_partial(
                    exit_pct, f"scale_out_{threshold:.0%}_{exit_pct:.0%}"
                )
        return PositionAction.hold()


@dataclass
class VolatilityStop:
    """Exit when price moves beyond ATR-based stop distance.

    Uses Average True Range (ATR) from context to set adaptive stop distance.
    Stop tightens/widens automatically based on market volatility.

    Args:
        multiplier: ATR multiplier for stop distance (e.g., 2.0 = 2×ATR)
        atr_key: Key to look up ATR value in state.context (default: "atr")
        use_entry_atr: If True, use ATR at entry (from context). If False,
                       recalculate stop each bar using current ATR.

    Example:
        # 2x ATR stop using ATR from context
        rule = VolatilityStop(multiplier=2.0)

        # Strategy must provide ATR in context:
        broker.set_context({"atr": current_atr_value})

    Note:
        Requires strategy to compute ATR and pass via context dict.
        If ATR not found in context, rule does nothing (returns HOLD).
    """

    multiplier: float = 2.0
    atr_key: str = "atr"
    use_entry_atr: bool = True
    _entry_atr: float | None = field(default=None, repr=False)

    def reset(self):
        """Reset entry ATR (call when position closes)."""
        self._entry_atr = None

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if price moves beyond ATR-based stop."""
        # Get ATR from context
        atr = state.context.get(self.atr_key)
        if atr is None or atr <= 0:
            # No ATR available, cannot evaluate
            return PositionAction.hold()

        # Capture entry ATR on first evaluation
        if self.use_entry_atr:
            if self._entry_atr is None:
                self._entry_atr = atr
            atr = self._entry_atr

        stop_distance = atr * self.multiplier

        if state.is_long:
            # Long: stop is entry - (ATR × multiplier)
            stop_price = state.entry_price - stop_distance
            if state.current_price <= stop_price:
                return PositionAction.exit_full(
                    f"volatility_stop_{self.multiplier:.1f}x_atr",
                    fill_price=stop_price,
                )
        else:
            # Short: stop is entry + (ATR × multiplier)
            stop_price = state.entry_price + stop_distance
            if state.current_price >= stop_price:
                return PositionAction.exit_full(
                    f"volatility_stop_{self.multiplier:.1f}x_atr",
                    fill_price=stop_price,
                )

        return PositionAction.hold()


@dataclass
class VolatilityTrailingStop:
    """Trailing stop with ATR-based distance.

    Combines trailing stop behavior with volatility-adjusted distance.
    The trail distance adapts to market volatility via ATR.

    Args:
        multiplier: ATR multiplier for trail distance (e.g., 3.0 = 3×ATR)
        atr_key: Key to look up ATR value in state.context

    Example:
        rule = VolatilityTrailingStop(multiplier=3.0)

    Note:
        Trail distance uses current ATR, so adapts to changing volatility.
    """

    multiplier: float = 3.0
    atr_key: str = "atr"

    def evaluate(self, state: PositionState) -> PositionAction:
        """Exit if price retraces beyond ATR-based trail."""
        atr = state.context.get(self.atr_key)
        if atr is None or atr <= 0:
            return PositionAction.hold()

        trail_distance = atr * self.multiplier

        if state.is_long:
            # Long: trail from high water mark
            stop_price = state.high_water_mark - trail_distance
            if state.current_price <= stop_price:
                return PositionAction.exit_full(
                    f"vol_trailing_stop_{self.multiplier:.1f}x_atr",
                    fill_price=stop_price,
                )
        else:
            # Short: trail from low water mark
            stop_price = state.low_water_mark + trail_distance
            if state.current_price >= stop_price:
                return PositionAction.exit_full(
                    f"vol_trailing_stop_{self.multiplier:.1f}x_atr",
                    fill_price=stop_price,
                )

        return PositionAction.hold()
