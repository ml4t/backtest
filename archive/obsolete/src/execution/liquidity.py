"""Liquidity modeling for realistic order fills."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ml4t.backtest.core.types import AssetId, Price, Quantity
from ml4t.backtest.execution.order import Order


@dataclass
class LiquidityInfo:
    """Information about available liquidity for an asset."""

    asset_id: AssetId
    available_volume: Quantity
    impact_threshold: Quantity = 0.0  # Volume above which price impact occurs
    max_single_order: Optional[Quantity] = None  # Maximum single order size


class LiquidityModel(ABC):
    """Abstract base class for liquidity modeling."""

    @abstractmethod
    def get_available_volume(
        self,
        asset_id: AssetId,
        price: Price,
        side: str,  # 'buy' or 'sell'
    ) -> Quantity:
        """
        Get available volume for an asset at a given price and side.

        Args:
            asset_id: Asset identifier
            price: Price level
            side: Order side ('buy' or 'sell')

        Returns:
            Available volume that can be traded
        """

    @abstractmethod
    def update_volume(
        self,
        asset_id: AssetId,
        price: Price,
        side: str,
        volume_consumed: Quantity,
    ) -> None:
        """
        Update available volume after a fill.

        Args:
            asset_id: Asset identifier
            price: Fill price
            side: Order side
            volume_consumed: Volume that was consumed
        """

    def get_max_fill_quantity(
        self,
        order: Order,
        market_price: Price,
    ) -> Quantity:
        """
        Get maximum quantity that can be filled considering liquidity constraints.

        Args:
            order: Order to check
            market_price: Current market price

        Returns:
            Maximum fillable quantity
        """
        side = "buy" if order.is_buy else "sell"
        available = self.get_available_volume(order.asset_id, market_price, side)
        return min(order.remaining_quantity, available)


class ConstantLiquidityModel(LiquidityModel):
    """Simple liquidity model with constant available volume per asset."""

    def __init__(self, default_volume: Quantity = 1000000.0):
        """
        Initialize with constant liquidity.

        Args:
            default_volume: Default available volume for all assets
        """
        self.default_volume = default_volume
        self.liquidity_info: dict[AssetId, LiquidityInfo] = {}

    def set_liquidity(self, asset_id: AssetId, liquidity_info: LiquidityInfo) -> None:
        """Set specific liquidity parameters for an asset."""
        self.liquidity_info[asset_id] = liquidity_info

    def get_available_volume(
        self,
        asset_id: AssetId,
        price: Price,
        side: str,
    ) -> Quantity:
        """Get available volume (constant model)."""
        if asset_id in self.liquidity_info:
            return self.liquidity_info[asset_id].available_volume
        return self.default_volume

    def update_volume(
        self,
        asset_id: AssetId,
        price: Price,
        side: str,
        volume_consumed: Quantity,
    ) -> None:
        """Update available volume (no-op for constant model)."""
        # Constant model doesn't track consumption
        pass


class RealisticLiquidityModel(LiquidityModel):
    """More realistic liquidity model with volume depletion and regeneration."""

    def __init__(
        self,
        default_volume: Quantity = 100000.0,
        regeneration_rate: float = 0.1,  # 10% per time period
        price_impact_threshold: Quantity = 10000.0,
    ):
        """
        Initialize realistic liquidity model.

        Args:
            default_volume: Default available volume
            regeneration_rate: Rate at which liquidity regenerates
            price_impact_threshold: Volume threshold for price impact
        """
        self.default_volume = default_volume
        self.regeneration_rate = regeneration_rate
        self.price_impact_threshold = price_impact_threshold

        # Track current available liquidity
        self.current_liquidity: dict[AssetId, LiquidityInfo] = {}
        self.last_update: dict[AssetId, float] = {}  # Timestamp tracking

    def _get_or_create_liquidity(self, asset_id: AssetId) -> LiquidityInfo:
        """Get or create liquidity info for an asset."""
        if asset_id not in self.current_liquidity:
            self.current_liquidity[asset_id] = LiquidityInfo(
                asset_id=asset_id,
                available_volume=self.default_volume,
                impact_threshold=self.price_impact_threshold,
            )
        return self.current_liquidity[asset_id]

    def _regenerate_liquidity(self, asset_id: AssetId, current_time: float) -> None:
        """Regenerate liquidity over time."""
        if asset_id in self.last_update:
            time_elapsed = current_time - self.last_update[asset_id]
            if time_elapsed > 0:
                liquidity = self.current_liquidity[asset_id]
                regeneration = time_elapsed * self.regeneration_rate * self.default_volume
                liquidity.available_volume = min(
                    self.default_volume,
                    liquidity.available_volume + regeneration,
                )

        self.last_update[asset_id] = current_time

    def get_available_volume(
        self,
        asset_id: AssetId,
        price: Price,
        side: str,
    ) -> Quantity:
        """Get available volume considering depletion."""
        import time
        current_time = time.time()

        # Regenerate liquidity since last update
        self._regenerate_liquidity(asset_id, current_time)

        liquidity = self._get_or_create_liquidity(asset_id)
        return liquidity.available_volume

    def update_volume(
        self,
        asset_id: AssetId,
        price: Price,
        side: str,
        volume_consumed: Quantity,
    ) -> None:
        """Update available volume after consumption."""
        import time
        current_time = time.time()

        self._regenerate_liquidity(asset_id, current_time)
        liquidity = self._get_or_create_liquidity(asset_id)

        # Reduce available volume
        liquidity.available_volume = max(0, liquidity.available_volume - volume_consumed)

        self.last_update[asset_id] = current_time


class VolumeLimitedLiquidityModel(LiquidityModel):
    """Liquidity model with explicit volume limits per asset."""

    def __init__(self):
        """Initialize volume-limited model."""
        self.volume_limits: dict[AssetId, Quantity] = {}
        self.current_volumes: dict[AssetId, Quantity] = {}
        self.default_limit = 50000.0

    def set_volume_limit(self, asset_id: AssetId, volume_limit: Quantity) -> None:
        """Set volume limit for specific asset."""
        self.volume_limits[asset_id] = volume_limit
        if asset_id not in self.current_volumes:
            self.current_volumes[asset_id] = volume_limit

    def reset_volume(self, asset_id: AssetId) -> None:
        """Reset available volume to limit (e.g., daily reset)."""
        limit = self.volume_limits.get(asset_id, self.default_limit)
        self.current_volumes[asset_id] = limit

    def get_available_volume(
        self,
        asset_id: AssetId,
        price: Price,
        side: str,
    ) -> Quantity:
        """Get currently available volume."""
        if asset_id not in self.current_volumes:
            limit = self.volume_limits.get(asset_id, self.default_limit)
            self.current_volumes[asset_id] = limit

        return self.current_volumes[asset_id]

    def update_volume(
        self,
        asset_id: AssetId,
        price: Price,
        side: str,
        volume_consumed: Quantity,
    ) -> None:
        """Reduce available volume after fill."""
        if asset_id in self.current_volumes:
            self.current_volumes[asset_id] = max(
                0, self.current_volumes[asset_id] - volume_consumed
            )


__all__ = [
    "LiquidityModel",
    "LiquidityInfo",
    "ConstantLiquidityModel",
    "RealisticLiquidityModel",
    "VolumeLimitedLiquidityModel",
]
