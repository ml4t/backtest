"""Market impact models for realistic execution costs."""

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

PERMANENT_FRACTION_DEFAULT = 0.5


class MarketImpactModel(ABC):
    """Base class for market impact models.

    Market impact models estimate how order execution affects price.
    Larger orders relative to volume cause more adverse price movement.
    """

    @abstractmethod
    def calculate(
        self,
        quantity: float,
        price: float,
        volume: float | None,
        is_buy: bool,
    ) -> float:
        """Calculate price impact.

        Args:
            quantity: Order quantity (positive)
            price: Current market price
            volume: Bar volume (None if unavailable)
            is_buy: True for buy orders, False for sell

        Returns:
            Adverse impact in price units. Values must be finite and non-negative
            for buys, or finite and non-positive for sells. Models that represent
            price improvement must do so through a separate execution-price model.
        """
        pass


@dataclass
class NoImpact(MarketImpactModel):
    """No market impact - fill at quoted price.

    Default for simple backtests. Appropriate for small orders
    relative to market volume.
    """

    def calculate(
        self,
        quantity: float,
        price: float,
        volume: float | None,
        is_buy: bool,
    ) -> float:
        """No impact - return 0."""
        return 0.0


@dataclass
class LinearImpact(MarketImpactModel):
    """Linear market impact model.

    Impact = coefficient * (quantity / volume) * price

    Simple model where impact scales linearly with participation rate.
    Appropriate for liquid markets with moderate order sizes.

    Like every model here it is a single-order concession model: `calculate` sees one
    order, and the impact it returns is charged to that order's fill price only. Nothing
    is carried into the price for later orders, so a parent order worked in slices is
    charged the same concession on every slice. A caller who needs permanent impact -
    the part of the move that does not revert and is paid again by every later slice -
    accumulates it outside the model.

    Args:
        coefficient: Impact scaling factor (default 0.1)
                    Higher values = more impact per unit participation
        permanent_fraction: Deprecated and inert; scheduled for removal in 0.2.0.
                    `calculate` has never read it, so every value charges the same
                    price. Setting it to anything but its default warns.

    Example:
        model = LinearImpact(coefficient=0.1)
        # 10% participation at $100 price = $1.00 impact
    """

    coefficient: float = 0.1
    permanent_fraction: float = PERMANENT_FRACTION_DEFAULT

    def __setattr__(self, name: str, value: Any) -> None:
        """Warn once per assignment that sets the inert persistence parameter.

        The warning goes here rather than in `__post_init__` because the field is
        settable after construction as well as through it, and a model assembled and
        then adjusted is the case that most looks like it is configuring something.
        A non-frozen dataclass routes `__init__` through `__setattr__` too, so one
        guard covers both and fires once each time a value is actually set.

        The default is the one value that does not warn, because a dataclass cannot
        tell a caller who passed 0.5 from one who passed nothing. Leaving it alone is
        also the case that loses nothing when the field goes: the model charges the
        same price either way.
        """
        if name == "permanent_fraction" and value != PERMANENT_FRACTION_DEFAULT:
            warnings.warn(
                "LinearImpact.permanent_fraction is inert and will be removed in "
                "ml4t-backtest 0.2.0. calculate() has never read it, so this model "
                "charges the same impact at every value; the engine applies an impact "
                "model to one order at a time and has no state in which a permanent "
                "component could persist into later fills. Accumulate permanent impact "
                "in the caller instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__setattr__(name, value)

    def calculate(
        self,
        quantity: float,
        price: float,
        volume: float | None,
        is_buy: bool,
    ) -> float:
        """Calculate linear impact."""
        if volume is None or volume == 0:
            return 0.0

        participation = quantity / volume
        impact = self.coefficient * participation * price

        # Apply direction (buys push price up, sells push price down)
        return impact if is_buy else -impact


@dataclass
class SquareRootImpact(MarketImpactModel):
    """Square root market impact model (Almgren-Chriss style).

    Impact = coefficient * sigma * sqrt(quantity / ADV) * price

    Based on academic market microstructure research. Impact scales
    with the square root of order size, which matches empirical observations.

    Args:
        coefficient: Scaling factor (default 0.5, typical range 0.1-1.0)
        volatility: Daily volatility (sigma, default 0.02 = 2%)
        adv_factor: Average daily volume as multiple of bar volume
                   (default 1.0 for daily bars, 390 for minute bars)

    Example:
        model = SquareRootImpact(coefficient=0.5, volatility=0.02)
        # For order = 1% of ADV at 2% vol, $100 price:
        # Impact = 0.5 * 0.02 * sqrt(0.01) * 100 = $0.10
    """

    coefficient: float = 0.5
    volatility: float = 0.02
    adv_factor: float = 1.0

    def calculate(
        self,
        quantity: float,
        price: float,
        volume: float | None,
        is_buy: bool,
    ) -> float:
        """Calculate square root impact."""
        if volume is None or volume == 0:
            return 0.0

        adv = volume * self.adv_factor
        participation = quantity / adv

        # Square root impact
        impact = self.coefficient * self.volatility * math.sqrt(participation) * price

        return impact if is_buy else -impact


@dataclass
class PowerLawImpact(MarketImpactModel):
    """Generalized power law impact model.

    Impact = coefficient * (quantity / volume)^exponent * price

    Flexible model that can represent various impact regimes.
    - exponent = 1.0: Linear (like LinearImpact)
    - exponent = 0.5: Square root (like SquareRootImpact)
    - exponent < 0.5: Concave (impact flattens for large orders)
    - exponent > 1.0: Convex (impact accelerates for large orders)

    Args:
        coefficient: Scaling factor (default 0.1)
        exponent: Power law exponent (default 0.5)
        min_impact: Minimum impact per trade (fixed cost, default 0)

    Example:
        model = PowerLawImpact(coefficient=0.1, exponent=0.6)
    """

    coefficient: float = 0.1
    exponent: float = 0.5
    min_impact: float = 0.0

    def calculate(
        self,
        quantity: float,
        price: float,
        volume: float | None,
        is_buy: bool,
    ) -> float:
        """Calculate power law impact."""
        if volume is None or volume == 0:
            return self.min_impact if is_buy else -self.min_impact

        participation = quantity / volume

        # Power law impact
        impact = self.coefficient * (participation**self.exponent) * price
        impact = max(impact, self.min_impact)

        return impact if is_buy else -impact
