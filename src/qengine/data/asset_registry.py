"""Asset registry for managing asset specifications.

Minimal implementation to unblock qengine imports.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AssetSpec:
    """Specification for a tradeable asset."""

    asset_id: str
    asset_type: str  # 'stock', 'future', 'crypto', etc.
    tick_size: float = 0.01
    lot_size: int = 1
    multiplier: float = 1.0
    margin_requirement: float = 1.0  # 1.0 = 100% (no leverage)

    # Trading hours (TODO: integrate with market calendars)
    tradeable_hours: Optional[tuple[int, int]] = None  # (start_hour, end_hour)


class AssetRegistry:
    """Registry of asset specifications."""

    def __init__(self):
        self._assets: Dict[str, AssetSpec] = {}

        # Default specifications for common asset types
        self._register_defaults()

    def _register_defaults(self):
        """Register default specifications for common assets."""
        # Default crypto spec
        self.register(AssetSpec(
            asset_id="BTC",
            asset_type="crypto",
            tick_size=0.01,
            lot_size=1,
            multiplier=1.0,
            margin_requirement=1.0
        ))

        # Default stock spec
        self.register(AssetSpec(
            asset_id="DEFAULT_STOCK",
            asset_type="stock",
            tick_size=0.01,
            lot_size=1,
            multiplier=1.0,
            margin_requirement=1.0
        ))

    def register(self, spec: AssetSpec) -> None:
        """Register an asset specification.

        Args:
            spec: Asset specification to register
        """
        self._assets[spec.asset_id] = spec

    def get(self, asset_id: str) -> AssetSpec:
        """Get asset specification by ID.

        Args:
            asset_id: Asset identifier

        Returns:
            Asset specification

        Raises:
            KeyError: If asset not found
        """
        if asset_id not in self._assets:
            # Return default spec for unknown assets
            return AssetSpec(
                asset_id=asset_id,
                asset_type="unknown",
                tick_size=0.01,
                lot_size=1,
                multiplier=1.0,
                margin_requirement=1.0
            )
        return self._assets[asset_id]

    def list_assets(self) -> list[str]:
        """List all registered asset IDs.

        Returns:
            List of asset IDs
        """
        return list(self._assets.keys())
