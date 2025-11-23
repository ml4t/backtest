"""Volume participation limits for realistic execution."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .result import ExecutionResult


class ExecutionLimits(ABC):
    """Base class for execution limits.

    Execution limits determine how much of an order can be filled
    in a single bar based on volume constraints.
    """

    @abstractmethod
    def calculate(
        self,
        order_quantity: float,
        bar_volume: float | None,
        price: float,
    ) -> ExecutionResult:
        """Calculate executable quantity given volume constraints.

        Args:
            order_quantity: Total order quantity (positive)
            bar_volume: Bar's trading volume (None if unavailable)
            price: Current price for the asset

        Returns:
            ExecutionResult with fillable and remaining quantities
        """
        pass


@dataclass
class NoLimits(ExecutionLimits):
    """No volume limits - fill entire order immediately.

    This is the default behavior for simple backtests where
    market impact and liquidity are not a concern.
    """

    def calculate(
        self,
        order_quantity: float,
        bar_volume: float | None,
        price: float,
    ) -> ExecutionResult:
        """Fill entire order with no restrictions."""
        return ExecutionResult(
            fillable_quantity=order_quantity,
            remaining_quantity=0.0,
            adjusted_price=price,
            impact_cost=0.0,
            participation_rate=0.0
            if bar_volume is None or bar_volume == 0
            else order_quantity / bar_volume,
        )


@dataclass
class VolumeParticipationLimit(ExecutionLimits):
    """Limit order fill to a percentage of bar volume.

    Common institutional constraint to avoid excessive market impact.
    Typical values: 5-20% of average daily volume.

    Args:
        max_participation: Maximum fraction of bar volume to fill (0.0-1.0)
                          Default 0.10 = 10% of bar volume
        min_volume: Minimum bar volume required to execute (default 0)
                   Orders won't fill on bars with volume below this

    Example:
        limit = VolumeParticipationLimit(max_participation=0.10)
        # If bar volume is 10,000 shares, max fill is 1,000 shares
        # Order for 2,500 shares fills 1,000, queues 1,500
    """

    max_participation: float = 0.10
    min_volume: float = 0.0

    def calculate(
        self,
        order_quantity: float,
        bar_volume: float | None,
        price: float,
    ) -> ExecutionResult:
        """Calculate fillable quantity based on participation rate."""
        # No volume data - can't apply limit, fill entire order
        if bar_volume is None:
            return ExecutionResult(
                fillable_quantity=order_quantity,
                remaining_quantity=0.0,
                adjusted_price=price,
                participation_rate=0.0,
            )

        # Volume below minimum - no fill this bar
        if bar_volume < self.min_volume:
            return ExecutionResult(
                fillable_quantity=0.0,
                remaining_quantity=order_quantity,
                adjusted_price=price,
                participation_rate=0.0,
            )

        # Calculate max fillable based on participation limit
        max_fillable = bar_volume * self.max_participation
        fillable = min(order_quantity, max_fillable)
        remaining = order_quantity - fillable

        participation_rate = fillable / bar_volume if bar_volume > 0 else 0.0

        return ExecutionResult(
            fillable_quantity=fillable,
            remaining_quantity=remaining,
            adjusted_price=price,
            participation_rate=participation_rate,
        )


@dataclass
class AdaptiveParticipationLimit(ExecutionLimits):
    """Participation rate that adapts based on volatility/spread.

    More aggressive when conditions are favorable (tight spread, low vol),
    more conservative when conditions are unfavorable.

    Args:
        base_participation: Base participation rate (default 0.10)
        volatility_key: Key in context for current volatility
        spread_key: Key in context for current bid-ask spread
        max_participation: Upper bound on participation (default 0.25)
        min_participation: Lower bound on participation (default 0.02)

    Example:
        limit = AdaptiveParticipationLimit(base_participation=0.10)
        # Participation increases when volatility is low
        # Participation decreases when spread is wide
    """

    base_participation: float = 0.10
    volatility_factor: float = 0.5  # Reduce participation by this * normalized_vol
    max_participation: float = 0.25
    min_participation: float = 0.02
    avg_volatility: float = 0.02  # Baseline volatility for normalization

    def calculate(
        self,
        order_quantity: float,
        bar_volume: float | None,
        price: float,
        volatility: float | None = None,
    ) -> ExecutionResult:
        """Calculate fillable with adaptive participation."""
        if bar_volume is None:
            return ExecutionResult(
                fillable_quantity=order_quantity,
                remaining_quantity=0.0,
                adjusted_price=price,
            )

        # Adjust participation based on volatility
        participation = self.base_participation

        if volatility is not None and self.avg_volatility > 0:
            # Higher volatility = lower participation
            vol_ratio = volatility / self.avg_volatility
            adjustment = self.volatility_factor * (vol_ratio - 1.0)
            participation = participation * (1.0 - adjustment)

        # Clamp to bounds
        participation = max(self.min_participation, min(self.max_participation, participation))

        max_fillable = bar_volume * participation
        fillable = min(order_quantity, max_fillable)
        remaining = order_quantity - fillable

        return ExecutionResult(
            fillable_quantity=fillable,
            remaining_quantity=remaining,
            adjusted_price=price,
            participation_rate=fillable / bar_volume if bar_volume > 0 else 0.0,
        )
