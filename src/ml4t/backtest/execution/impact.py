"""Market impact models for realistic execution costs."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


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
            Impact in price units (positive = adverse, negative = favorable)
            For buys: price increases; for sells: price decreases
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

    Args:
        coefficient: Impact scaling factor (default 0.1)
                    Higher values = more impact per unit participation
        permanent_fraction: Fraction of impact that is permanent (0-1)
                           Remainder is temporary and reverts

    Example:
        model = LinearImpact(coefficient=0.1)
        # 10% participation at $100 price = $1.00 impact
    """

    coefficient: float = 0.1
    permanent_fraction: float = 0.5

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
