"""Slippage models for QEngine."""

from abc import ABC, abstractmethod

from qengine.core.types import Price, Quantity
from qengine.execution.order import Order


class SlippageModel(ABC):
    """Abstract base class for slippage models.

    Slippage models determine the actual fill price based on order characteristics
    and market conditions.
    """

    @abstractmethod
    def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
        """Calculate the fill price with slippage.

        Args:
            order: The order being filled
            market_price: Current market price

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

    def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
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

    def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
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

    def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
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

    def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
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

    def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
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

    def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
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

    def calculate_fill_price(self, order: Order, market_price: Price) -> Price:
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
