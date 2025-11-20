"""Slippage models for ml4t.backtest."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ml4t.backtest.core.types import OrderSide, OrderType, Price, Quantity
from ml4t.backtest.execution.order import Order

if TYPE_CHECKING:
    from ml4t.backtest.core.event import MarketEvent


class SlippageModel(ABC):
    """Abstract base class for slippage models.

    Slippage models determine the actual fill price based on order characteristics
    and market conditions.
    """

    @abstractmethod
    def calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: MarketEvent | None = None,
    ) -> Price:
        """Calculate the fill price with slippage.

        Args:
            order: The order being filled
            market_price: Current market price
            market_event: Optional MarketEvent with additional data (bid/ask, volume, etc.)

        Returns:
            The fill price including slippage
        """

    @abstractmethod
    def calculate_slippage_cost(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
    ) -> float:
        """Calculate the slippage cost in currency terms.

        Args:
            order: The order being filled
            fill_quantity: Quantity being filled
            market_price: Current market price
            fill_price: Actual fill price

        Returns:
            Slippage cost in currency terms
        """


class NoSlippage(SlippageModel):
    """No slippage - all orders fill at market price.

    Primarily used for testing or ideal conditions.
    """

    def calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: MarketEvent | None = None,
    ) -> Price:
        """Fill at market price."""
        return market_price

    def calculate_slippage_cost(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
    ) -> float:
        """No slippage cost."""
        return 0.0


class FixedSlippage(SlippageModel):
    """Fixed spread slippage model.

    Assumes a fixed spread for all orders.
    Buy orders fill at ask (market + spread/2)
    Sell orders fill at bid (market - spread/2)

    Args:
        spread: Fixed spread amount (default 0.01)
    """

    def __init__(self, spread: float = 0.01):
        """Initialize with fixed spread."""
        if spread < 0:
            raise ValueError("Spread must be non-negative")
        self.spread = spread

    def calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: MarketEvent | None = None,
    ) -> Price:
        """Calculate fill price with fixed spread."""
        half_spread = self.spread / 2

        if order.is_buy:
            # Buy at ask (worse price)
            return market_price + half_spread
        # Sell at bid (worse price)
        return market_price - half_spread

    def calculate_slippage_cost(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
    ) -> float:
        """Calculate slippage cost from spread."""
        # Cost is the absolute difference times quantity
        return abs(fill_price - market_price) * fill_quantity


class PercentageSlippage(SlippageModel):
    """Percentage-based slippage model.

    Slippage is a percentage of the market price.

    Args:
        slippage_pct: Slippage percentage (default 0.1%)
        min_slippage: Minimum slippage amount (default 0.001)
    """

    def __init__(self, slippage_pct: float = 0.001, min_slippage: float = 0.001):
        """Initialize with percentage parameters."""
        if slippage_pct < 0:
            raise ValueError("Slippage percentage must be non-negative")
        if min_slippage < 0:
            raise ValueError("Minimum slippage must be non-negative")

        self.slippage_pct = slippage_pct
        self.min_slippage = min_slippage

    def calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: MarketEvent | None = None,
    ) -> Price:
        """Calculate fill price with percentage slippage."""
        # Calculate slippage amount
        slippage_amount = max(market_price * self.slippage_pct, self.min_slippage)

        if order.is_buy:
            # Buy at higher price
            return market_price + slippage_amount
        # Sell at lower price
        return market_price - slippage_amount

    def calculate_slippage_cost(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
    ) -> float:
        """Calculate slippage cost."""
        return abs(fill_price - market_price) * fill_quantity


class LinearImpactSlippage(SlippageModel):
    """Linear market impact slippage model.

    Slippage increases linearly with order size.

    Args:
        base_slippage: Base slippage for minimal orders (default 0.0001)
        impact_coefficient: Impact per unit of order size (default 0.00001)
    """

    def __init__(
        self,
        base_slippage: float = 0.0001,
        impact_coefficient: float = 0.00001,
    ):
        """Initialize with impact parameters."""
        if base_slippage < 0:
            raise ValueError("Base slippage must be non-negative")
        if impact_coefficient < 0:
            raise ValueError("Impact coefficient must be non-negative")

        self.base_slippage = base_slippage
        self.impact_coefficient = impact_coefficient

    def calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: MarketEvent | None = None,
    ) -> Price:
        """Calculate fill price with linear impact."""
        # Linear impact based on order size
        impact = self.base_slippage + self.impact_coefficient * order.quantity
        slippage_amount = market_price * impact

        if order.is_buy:
            return market_price + slippage_amount
        return market_price - slippage_amount

    def calculate_slippage_cost(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
    ) -> float:
        """Calculate slippage cost."""
        return abs(fill_price - market_price) * fill_quantity


class SquareRootImpactSlippage(SlippageModel):
    """Square root market impact model (Almgren-Chriss style).

    Slippage increases with the square root of order size, modeling
    non-linear market impact for large orders.

    Args:
        temporary_impact: Temporary impact coefficient (default 0.1)
        permanent_impact: Permanent impact coefficient (default 0.05)
    """

    def __init__(
        self,
        temporary_impact: float = 0.1,
        permanent_impact: float = 0.05,
    ):
        """Initialize with impact parameters."""
        if temporary_impact < 0:
            raise ValueError("Temporary impact must be non-negative")
        if permanent_impact < 0:
            raise ValueError("Permanent impact must be non-negative")

        self.temporary_impact = temporary_impact
        self.permanent_impact = permanent_impact

    def calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: MarketEvent | None = None,
    ) -> Price:
        """Calculate fill price with square root impact."""
        import math

        # Square root impact model
        order_size_impact = math.sqrt(order.quantity / 1000.0)  # Normalize by 1000 shares

        # Combine temporary and permanent impact
        total_impact = (
            self.temporary_impact * order_size_impact
            + self.permanent_impact * order_size_impact / 2
        )

        # Convert to price impact
        slippage_amount = market_price * total_impact * 0.01  # Convert to percentage

        if order.is_buy:
            return market_price + slippage_amount
        return market_price - slippage_amount

    def calculate_slippage_cost(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
    ) -> float:
        """Calculate slippage cost."""
        return abs(fill_price - market_price) * fill_quantity


class VolumeShareSlippage(SlippageModel):
    """Volume-based slippage model.

    Slippage is based on the percentage of daily volume being traded.
    Larger orders relative to volume have more impact.

    Args:
        volume_limit: Maximum percentage of volume per bar (default 0.025 = 2.5%)
        price_impact: Price impact coefficient (default 0.1)
    """

    def __init__(
        self,
        volume_limit: float = 0.025,
        price_impact: float = 0.1,
    ):
        """Initialize with volume parameters."""
        if not 0 < volume_limit <= 1:
            raise ValueError("Volume limit must be between 0 and 1")
        if price_impact < 0:
            raise ValueError("Price impact must be non-negative")

        self.volume_limit = volume_limit
        self.price_impact = price_impact
        self._daily_volume: float | None = None

    def set_daily_volume(self, volume: float) -> None:
        """Set the daily volume for impact calculation.

        Args:
            volume: Daily volume
        """
        self._daily_volume = volume

    def calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: MarketEvent | None = None,
    ) -> Price:
        """Calculate fill price based on volume impact."""
        if self._daily_volume is None or self._daily_volume == 0:
            # No volume data, use minimal slippage
            slippage_amount = market_price * 0.0001
        else:
            # Calculate volume share
            volume_share = min(order.quantity / self._daily_volume, self.volume_limit)

            # Quadratic impact model (like Zipline)
            impact = volume_share**2 * self.price_impact
            slippage_amount = market_price * impact

        if order.is_buy:
            return market_price + slippage_amount
        return market_price - slippage_amount

    def calculate_slippage_cost(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
    ) -> float:
        """Calculate slippage cost."""
        return abs(fill_price - market_price) * fill_quantity


class AssetClassSlippage(SlippageModel):
    """Asset class specific slippage model.

    Different slippage rates for different asset classes.

    Args:
        equity_slippage: Slippage for equities (default 0.01%)
        future_slippage: Slippage for futures (default 0.02%)
        option_slippage: Slippage for options (default 0.05%)
        fx_slippage: Slippage for forex (default 0.005%)
        crypto_slippage: Slippage for crypto (default 0.1%)
    """

    def __init__(
        self,
        equity_slippage: float = 0.0001,
        future_slippage: float = 0.0002,
        option_slippage: float = 0.0005,
        fx_slippage: float = 0.00005,
        crypto_slippage: float = 0.001,
    ):
        """Initialize with asset class specific rates."""
        self.slippage_rates = {
            "equity": equity_slippage,
            "future": future_slippage,
            "option": option_slippage,
            "forex": fx_slippage,
            "fx": fx_slippage,  # Alias
            "crypto": crypto_slippage,
        }
        self.default_slippage = equity_slippage

    def calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: MarketEvent | None = None,
    ) -> Price:
        """Calculate fill price based on asset class."""
        # Get asset class from order metadata or default
        asset_class = order.metadata.get("asset_class", "equity")
        slippage_rate = self.slippage_rates.get(asset_class, self.default_slippage)

        slippage_amount = market_price * slippage_rate

        if order.is_buy:
            return market_price + slippage_amount
        return market_price - slippage_amount

    def calculate_slippage_cost(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
    ) -> float:
        """Calculate slippage cost."""
        return abs(fill_price - market_price) * fill_quantity


class SpreadAwareSlippage(SlippageModel):
    """Slippage model that uses bid/ask spread from MarketEvent.

    Fills at mid-price ± k × spread, where k is a configurable factor (default 0.5).
    If bid/ask data is unavailable, falls back to PercentageSlippage.

    Args:
        spread_factor: Fraction of spread to pay (default 0.5 = fill at mid)
        fallback_slippage_pct: Fallback percentage when spread unavailable (default 0.001)

    Example:
        If bid=99.98, ask=100.02 (spread=0.04), mid=100.00:
        - Buy with k=0.5: Fill at 100.00 + 0.5×0.04/2 = 100.01
        - Sell with k=0.5: Fill at 100.00 - 0.5×0.04/2 = 99.99
    """

    def __init__(self, spread_factor: float = 0.5, fallback_slippage_pct: float = 0.001):
        """Initialize spread-aware slippage model."""
        if spread_factor < 0:
            raise ValueError("Spread factor must be non-negative")
        if fallback_slippage_pct < 0:
            raise ValueError("Fallback slippage percentage must be non-negative")

        self.spread_factor = spread_factor
        self.fallback_slippage_pct = fallback_slippage_pct

    def calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: MarketEvent | None = None,
    ) -> Price:
        """Calculate fill price using bid/ask spread if available."""
        # Try to use bid/ask from MarketEvent
        if (
            market_event is not None
            and market_event.bid_price is not None
            and market_event.ask_price is not None
            and market_event.bid_price > 0
            and market_event.ask_price > 0
        ):
            bid = market_event.bid_price
            ask = market_event.ask_price
            spread = ask - bid
            mid = (bid + ask) / 2

            # Fill at mid ± k × half_spread
            half_spread = spread / 2
            slippage = self.spread_factor * half_spread

            if order.is_buy:
                return mid + slippage
            else:
                return mid - slippage

        # Fallback to percentage slippage
        slippage_amount = market_price * self.fallback_slippage_pct
        if order.is_buy:
            return market_price + slippage_amount
        else:
            return market_price - slippage_amount

    def calculate_slippage_cost(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
    ) -> float:
        """Calculate slippage cost."""
        return abs(fill_price - market_price) * fill_quantity


class VolumeAwareSlippage(SlippageModel):
    """Slippage model that scales with order size relative to volume.

    Calculates slippage as f(order_size / volume), where f can be:
    - Linear: impact = base + linear_coeff × participation_rate
    - Square-root: impact = base + sqrt_coeff × sqrt(participation_rate)

    If volume data is unavailable, falls back to PercentageSlippage.

    Args:
        base_slippage_pct: Base slippage percentage (default 0.0001)
        linear_impact_coeff: Linear impact coefficient (default 0.01)
        sqrt_impact_coeff: Square-root impact coefficient (default 0.0)
        fallback_slippage_pct: Fallback when volume unavailable (default 0.001)
        max_participation_rate: Maximum participation rate (default 0.1 = 10%)

    Example:
        Order size = 1000, Volume = 100000 (participation = 0.01 = 1%)
        Linear model: impact = 0.0001 + 0.01 × 0.01 = 0.0002 = 0.02%
    """

    def __init__(
        self,
        base_slippage_pct: float = 0.0001,
        linear_impact_coeff: float = 0.01,
        sqrt_impact_coeff: float = 0.0,
        fallback_slippage_pct: float = 0.001,
        max_participation_rate: float = 0.1,
    ):
        """Initialize volume-aware slippage model."""
        if base_slippage_pct < 0:
            raise ValueError("Base slippage must be non-negative")
        if linear_impact_coeff < 0:
            raise ValueError("Linear impact coefficient must be non-negative")
        if sqrt_impact_coeff < 0:
            raise ValueError("Square-root impact coefficient must be non-negative")
        if fallback_slippage_pct < 0:
            raise ValueError("Fallback slippage must be non-negative")
        if max_participation_rate <= 0 or max_participation_rate > 1:
            raise ValueError("Max participation rate must be in (0, 1]")

        self.base_slippage_pct = base_slippage_pct
        self.linear_impact_coeff = linear_impact_coeff
        self.sqrt_impact_coeff = sqrt_impact_coeff
        self.fallback_slippage_pct = fallback_slippage_pct
        self.max_participation_rate = max_participation_rate

    def calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: MarketEvent | None = None,
    ) -> Price:
        """Calculate fill price using volume impact if available."""
        # Try to use volume from MarketEvent
        if market_event is not None and market_event.volume is not None and market_event.volume > 0:
            volume = market_event.volume
            participation_rate = min(order.quantity / volume, self.max_participation_rate)

            # Calculate impact: base + linear × rate + sqrt × sqrt(rate)
            import math

            impact_pct = (
                self.base_slippage_pct
                + self.linear_impact_coeff * participation_rate
                + self.sqrt_impact_coeff * math.sqrt(participation_rate)
            )

            slippage_amount = market_price * impact_pct
        else:
            # Fallback to percentage slippage
            slippage_amount = market_price * self.fallback_slippage_pct

        if order.is_buy:
            return market_price + slippage_amount
        else:
            return market_price - slippage_amount

    def calculate_slippage_cost(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
    ) -> float:
        """Calculate slippage cost."""
        return abs(fill_price - market_price) * fill_quantity


class OrderTypeDependentSlippage(SlippageModel):
    """Slippage model with different rates per order type.

    Market orders pay more slippage (immediate execution, worse price).
    Limit orders pay less (patient, better price).
    Stop orders pay medium slippage (conditional execution).

    Args:
        market_slippage_pct: Slippage for MARKET orders (default 0.001)
        limit_slippage_pct: Slippage for LIMIT orders (default 0.0001)
        stop_slippage_pct: Slippage for STOP/STOP_LIMIT orders (default 0.0005)
        default_slippage_pct: Slippage for other order types (default 0.001)

    Example:
        Market order: 0.10% slippage (aggressive execution)
        Limit order: 0.01% slippage (patient execution)
        Stop order: 0.05% slippage (conditional execution)
    """

    def __init__(
        self,
        market_slippage_pct: float = 0.001,
        limit_slippage_pct: float = 0.0001,
        stop_slippage_pct: float = 0.0005,
        default_slippage_pct: float = 0.001,
    ):
        """Initialize order-type-dependent slippage model."""
        if market_slippage_pct < 0:
            raise ValueError("Market slippage must be non-negative")
        if limit_slippage_pct < 0:
            raise ValueError("Limit slippage must be non-negative")
        if stop_slippage_pct < 0:
            raise ValueError("Stop slippage must be non-negative")
        if default_slippage_pct < 0:
            raise ValueError("Default slippage must be non-negative")

        self.slippage_rates = {
            OrderType.MARKET: market_slippage_pct,
            OrderType.LIMIT: limit_slippage_pct,
            OrderType.STOP: stop_slippage_pct,
            OrderType.STOP_LIMIT: stop_slippage_pct,  # Same as STOP
            OrderType.TRAILING_STOP: stop_slippage_pct,  # Same as STOP
        }
        self.default_slippage_pct = default_slippage_pct

    def calculate_fill_price(
        self,
        order: Order,
        market_price: Price,
        market_event: MarketEvent | None = None,
    ) -> Price:
        """Calculate fill price based on order type."""
        # Get slippage rate for this order type
        slippage_pct = self.slippage_rates.get(order.order_type, self.default_slippage_pct)
        slippage_amount = market_price * slippage_pct

        if order.is_buy:
            return market_price + slippage_amount
        else:
            return market_price - slippage_amount

    def calculate_slippage_cost(
        self,
        order: Order,
        fill_quantity: Quantity,
        market_price: Price,
        fill_price: Price,
    ) -> float:
        """Calculate slippage cost."""
        return abs(fill_price - market_price) * fill_quantity
