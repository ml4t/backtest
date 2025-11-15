"""Market impact models for realistic price simulation.

Market impact differs from slippage in that it represents the actual change
in market prices due to trading activity, affecting all subsequent orders.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml4t.backtest.core.types import AssetId, Price, Quantity
    from ml4t.backtest.execution.order import Order


@dataclass
class ImpactState:
    """Tracks market impact state for an asset."""

    permanent_impact: float = 0.0  # Permanent price shift
    temporary_impact: float = 0.0  # Temporary price displacement
    last_update: datetime | None = None
    volume_traded: float = 0.0  # Recent volume for impact calculation

    def get_total_impact(self) -> float:
        """Get total current impact."""
        return self.permanent_impact + self.temporary_impact

    def decay_temporary_impact(self, decay_rate: float, time_elapsed: float) -> None:
        """Decay temporary impact over time."""
        if time_elapsed > 0:
            # Exponential decay
            self.temporary_impact *= math.exp(-decay_rate * time_elapsed)
            # Clean up near-zero values
            if abs(self.temporary_impact) < 1e-10:
                self.temporary_impact = 0.0


class MarketImpactModel(ABC):
    """Abstract base class for market impact models."""

    def __init__(self):
        """Initialize impact model."""
        # Track impact state per asset
        self.impact_states: dict[AssetId, ImpactState] = {}

    @abstractmethod
    def calculate_impact(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        market_price: "Price",
        timestamp: datetime,
    ) -> tuple[float, float]:
        """Calculate permanent and temporary market impact.

        Args:
            order: The order being filled
            fill_quantity: Quantity being filled
            market_price: Current market price
            timestamp: Time of the fill

        Returns:
            Tuple of (permanent_impact, temporary_impact) as price changes
        """

    def update_market_state(
        self,
        asset_id: "AssetId",
        permanent_impact: float,
        temporary_impact: float,
        timestamp: datetime,
    ) -> None:
        """Update the market state with new impact.

        Args:
            asset_id: Asset identifier
            permanent_impact: Permanent price change
            temporary_impact: Temporary price displacement
            timestamp: Time of the update
        """
        if asset_id not in self.impact_states:
            self.impact_states[asset_id] = ImpactState()

        state = self.impact_states[asset_id]

        # Apply time decay to existing temporary impact
        if state.last_update is not None:
            time_elapsed = (timestamp - state.last_update).total_seconds()
            self.apply_decay(asset_id, time_elapsed)

        # Add new impacts
        state.permanent_impact += permanent_impact
        state.temporary_impact += temporary_impact
        state.last_update = timestamp

    def apply_decay(self, asset_id: "AssetId", time_elapsed: float) -> None:
        """Apply time decay to temporary impact.

        Args:
            asset_id: Asset identifier
            time_elapsed: Time elapsed in seconds
        """
        if asset_id in self.impact_states:
            # Default decay rate (can be overridden)
            decay_rate = getattr(self, "decay_rate", 0.1)
            self.impact_states[asset_id].decay_temporary_impact(decay_rate, time_elapsed)

    def get_current_impact(
        self,
        asset_id: "AssetId",
        timestamp: datetime | None = None,
    ) -> float:
        """Get current total market impact for an asset.

        Args:
            asset_id: Asset identifier
            timestamp: Current time for decay calculation

        Returns:
            Total price impact (permanent + temporary)
        """
        if asset_id not in self.impact_states:
            return 0.0

        state = self.impact_states[asset_id]

        # Apply decay if timestamp provided
        if timestamp and state.last_update:
            time_elapsed = (timestamp - state.last_update).total_seconds()
            self.apply_decay(asset_id, time_elapsed)

        return state.get_total_impact()

    def reset(self) -> None:
        """Reset all impact states."""
        self.impact_states.clear()


class NoMarketImpact(MarketImpactModel):
    """No market impact model for testing."""

    def calculate_impact(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        market_price: "Price",
        timestamp: datetime,
    ) -> tuple[float, float]:
        """Calculate zero market impact."""
        return 0.0, 0.0


class LinearMarketImpact(MarketImpactModel):
    """Linear market impact model.

    Impact is proportional to order size relative to average daily volume.
    """

    def __init__(
        self,
        permanent_impact_factor: float = 0.1,
        temporary_impact_factor: float = 0.5,
        avg_daily_volume: float = 1_000_000,
        decay_rate: float = 0.1,
    ):
        """Initialize linear impact model.

        Args:
            permanent_impact_factor: Permanent impact per unit of volume fraction
            temporary_impact_factor: Temporary impact per unit of volume fraction
            avg_daily_volume: Average daily volume for normalization
            decay_rate: Decay rate for temporary impact (per second)
        """
        super().__init__()
        self.permanent_impact_factor = permanent_impact_factor
        self.temporary_impact_factor = temporary_impact_factor
        self.avg_daily_volume = avg_daily_volume
        self.decay_rate = decay_rate

    def calculate_impact(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        market_price: "Price",
        timestamp: datetime,
    ) -> tuple[float, float]:
        """Calculate linear market impact."""
        # Volume fraction (what percentage of ADV is this trade?)
        volume_fraction = fill_quantity / self.avg_daily_volume

        # Linear impact proportional to volume fraction
        permanent_impact = market_price * self.permanent_impact_factor * volume_fraction
        temporary_impact = market_price * self.temporary_impact_factor * volume_fraction

        # Buy orders push price up, sell orders push price down
        from ml4t.backtest.execution.order import OrderSide

        if order.side == OrderSide.SELL:
            permanent_impact = -permanent_impact
            temporary_impact = -temporary_impact

        return permanent_impact, temporary_impact


class AlmgrenChrissImpact(MarketImpactModel):
    """Almgren-Chriss market impact model.

    Sophisticated model with square-root permanent impact and linear temporary impact.
    Based on "Optimal Execution of Portfolio Transactions" (2001).
    """

    def __init__(
        self,
        permanent_impact_const: float = 0.01,
        temporary_impact_const: float = 0.1,
        daily_volatility: float = 0.02,
        avg_daily_volume: float = 1_000_000,
        decay_rate: float = 0.05,
    ):
        """Initialize Almgren-Chriss model.

        Args:
            permanent_impact_const: Permanent impact constant (gamma)
            temporary_impact_const: Temporary impact constant (eta)
            daily_volatility: Daily return volatility
            avg_daily_volume: Average daily volume
            decay_rate: Decay rate for temporary impact
        """
        super().__init__()
        self.permanent_impact_const = permanent_impact_const
        self.temporary_impact_const = temporary_impact_const
        self.daily_volatility = daily_volatility
        self.avg_daily_volume = avg_daily_volume
        self.decay_rate = decay_rate

    def calculate_impact(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        market_price: "Price",
        timestamp: datetime,
    ) -> tuple[float, float]:
        """Calculate Almgren-Chriss market impact."""
        # Normalized volume (fraction of ADV)
        volume_fraction = fill_quantity / self.avg_daily_volume

        # Permanent impact: square-root of volume fraction
        # g(v) = gamma * sign(v) * |v|^0.5
        permanent_impact = (
            self.permanent_impact_const
            * self.daily_volatility
            * market_price
            * math.sqrt(volume_fraction)
        )

        # Temporary impact: linear in trading rate
        # h(v) = eta * v
        temporary_impact = (
            self.temporary_impact_const * self.daily_volatility * market_price * volume_fraction
        )

        # Adjust sign based on order side
        from ml4t.backtest.execution.order import OrderSide

        if order.side == OrderSide.SELL:
            permanent_impact = -permanent_impact
            temporary_impact = -temporary_impact

        return permanent_impact, temporary_impact


class PropagatorImpact(MarketImpactModel):
    """Propagator model for market impact.

    Based on Bouchaud et al. model where impact propagates and decays
    according to a power law kernel.
    """

    def __init__(
        self,
        impact_coefficient: float = 0.1,
        propagator_exponent: float = 0.5,
        decay_exponent: float = 0.7,
        avg_daily_volume: float = 1_000_000,
    ):
        """Initialize propagator model.

        Args:
            impact_coefficient: Base impact coefficient
            propagator_exponent: Exponent for volume impact (typically 0.5)
            decay_exponent: Exponent for time decay (typically 0.5-0.7)
            avg_daily_volume: Average daily volume
        """
        super().__init__()
        self.impact_coefficient = impact_coefficient
        self.propagator_exponent = propagator_exponent
        self.decay_exponent = decay_exponent
        self.avg_daily_volume = avg_daily_volume

        # Track order history for propagation
        self.order_history: list[tuple[datetime, float, float]] = []

    def calculate_impact(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        market_price: "Price",
        timestamp: datetime,
    ) -> tuple[float, float]:
        """Calculate propagator market impact."""
        # Normalized volume
        volume_fraction = fill_quantity / self.avg_daily_volume

        # Instantaneous impact: power law in volume
        instant_impact = (
            self.impact_coefficient * market_price * (volume_fraction**self.propagator_exponent)
        )

        # Calculate propagated impact from historical orders
        propagated_impact = 0.0
        cutoff_time = timestamp - timedelta(hours=1)  # Only consider recent history

        for hist_time, hist_volume, hist_price in self.order_history[-100:]:  # Limit history
            if hist_time < cutoff_time:
                continue

            time_diff = (timestamp - hist_time).total_seconds()
            if time_diff > 0:
                # Power law decay
                decay_factor = (1 + time_diff) ** (-self.decay_exponent)
                propagated_impact += (
                    self.impact_coefficient
                    * hist_price
                    * (abs(hist_volume) / self.avg_daily_volume) ** self.propagator_exponent
                    * decay_factor
                    * (1 if hist_volume > 0 else -1)
                )

        # Store this order for future propagation
        from ml4t.backtest.execution.order import OrderSide

        signed_volume = fill_quantity if order.side == OrderSide.BUY else -fill_quantity
        self.order_history.append((timestamp, signed_volume, market_price))

        # Clean old history
        if len(self.order_history) > 1000:
            self.order_history = self.order_history[-500:]

        # Adjust sign
        if order.side == OrderSide.SELL:
            instant_impact = -instant_impact

        # Split into permanent and temporary
        # Propagator model typically has mostly temporary impact
        permanent_impact = instant_impact * 0.2
        temporary_impact = instant_impact * 0.8 + propagated_impact

        return permanent_impact, temporary_impact

    def reset(self) -> None:
        """Reset impact states and history."""
        super().reset()
        self.order_history.clear()


class IntraDayMomentum(MarketImpactModel):
    """Intraday momentum impact model.

    Models the tendency for large trades to create momentum that
    attracts further trading in the same direction.
    """

    def __init__(
        self,
        base_impact: float = 0.05,
        momentum_factor: float = 0.3,
        momentum_decay: float = 0.2,
        avg_daily_volume: float = 1_000_000,
    ):
        """Initialize momentum impact model.

        Args:
            base_impact: Base impact coefficient
            momentum_factor: How much momentum affects impact
            momentum_decay: Decay rate for momentum
            avg_daily_volume: Average daily volume
        """
        super().__init__()
        self.base_impact = base_impact
        self.momentum_factor = momentum_factor
        self.momentum_decay = momentum_decay
        self.avg_daily_volume = avg_daily_volume

        # Track momentum state per asset
        self.momentum_states: dict[AssetId, float] = {}

    def calculate_impact(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        market_price: "Price",
        timestamp: datetime,
    ) -> tuple[float, float]:
        """Calculate momentum-based impact."""
        asset_id = order.asset_id
        volume_fraction = fill_quantity / self.avg_daily_volume

        # Get current momentum
        momentum = self.momentum_states.get(asset_id, 0.0)

        # Base impact
        base_impact_value = self.base_impact * market_price * volume_fraction

        # Momentum enhancement (same-direction trades have larger impact)
        from ml4t.backtest.execution.order import OrderSide

        trade_direction = 1.0 if order.side == OrderSide.BUY else -1.0

        momentum_enhancement = 1.0 + self.momentum_factor * abs(momentum)
        if momentum * trade_direction > 0:  # Same direction as momentum
            impact = base_impact_value * momentum_enhancement
        else:  # Against momentum
            impact = base_impact_value / momentum_enhancement

        # Update momentum (exponential moving average)
        new_momentum = (
            momentum * (1 - self.momentum_decay)
            + trade_direction * volume_fraction * self.momentum_decay
        )
        self.momentum_states[asset_id] = new_momentum

        # Apply direction
        if order.side == OrderSide.SELL:
            impact = -impact

        # Split impact (momentum creates more temporary impact)
        permanent_impact = impact * 0.3
        temporary_impact = impact * 0.7

        return permanent_impact, temporary_impact

    def reset(self) -> None:
        """Reset all states."""
        super().reset()
        self.momentum_states.clear()


class ObizhaevWangImpact(MarketImpactModel):
    """Obizhaev-Wang market impact model.

    Models impact based on order book dynamics and trade informativeness.
    """

    def __init__(
        self,
        price_impact_const: float = 0.1,
        information_share: float = 0.3,
        book_depth: float = 100_000,
        resilience_rate: float = 0.5,
    ):
        """Initialize Obizhaev-Wang model.

        Args:
            price_impact_const: Price impact constant (lambda)
            information_share: Share of informed trading (alpha)
            book_depth: Typical order book depth
            resilience_rate: Rate of order book resilience
        """
        super().__init__()
        self.price_impact_const = price_impact_const
        self.information_share = information_share
        self.book_depth = book_depth
        self.resilience_rate = resilience_rate
        self.decay_rate = resilience_rate  # For base class decay

    def calculate_impact(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        market_price: "Price",
        timestamp: datetime,
    ) -> tuple[float, float]:
        """Calculate Obizhaev-Wang impact."""
        # Normalized order size relative to book depth
        size_ratio = fill_quantity / self.book_depth

        # Information-based permanent impact
        permanent_impact = (
            self.information_share * self.price_impact_const * market_price * size_ratio
        )

        # Mechanical temporary impact from eating through book
        temporary_impact = (
            (1 - self.information_share) * self.price_impact_const * market_price * size_ratio
        )

        # Adjust for order side
        from ml4t.backtest.execution.order import OrderSide

        if order.side == OrderSide.SELL:
            permanent_impact = -permanent_impact
            temporary_impact = -temporary_impact

        return permanent_impact, temporary_impact
