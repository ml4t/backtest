"""Margin management for derivatives and leveraged trading."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from qengine.core.assets import AssetRegistry, AssetSpec
from qengine.core.types import AssetId, Cash, Price, Quantity


@dataclass
class MarginRequirement:
    """Margin requirements for a position."""

    asset_id: AssetId
    initial_margin: Cash
    maintenance_margin: Cash
    current_margin: Cash
    excess_margin: Cash
    margin_call: bool = False
    liquidation_price: Optional[Price] = None


@dataclass
class MarginAccount:
    """
    Manages margin requirements for derivatives and leveraged trading.

    Handles:
    - Futures margin requirements
    - Options margin for sellers
    - FX leverage
    - Crypto perpetuals and leveraged trading
    - Portfolio margining
    """

    cash_balance: Cash
    initial_margin_requirement: Cash = 0.0
    maintenance_margin_requirement: Cash = 0.0
    margin_used: Cash = 0.0
    available_margin: Cash = 0.0
    positions: dict[AssetId, dict[str, Any]] = field(default_factory=dict)

    # Risk parameters
    margin_call_level: float = 1.0  # 100% of maintenance margin
    liquidation_level: float = 0.8  # 80% of maintenance margin

    def __init__(self, initial_cash: Cash, asset_registry: AssetRegistry):
        """Initialize margin account."""
        self.cash_balance = initial_cash
        self.available_margin = initial_cash
        self.asset_registry = asset_registry
        self.positions = {}
        self.margin_calls: list[MarginRequirement] = []

    def check_margin_requirement(
        self,
        asset_id: AssetId,
        quantity: Quantity,
        price: Price,
    ) -> tuple[bool, Cash]:
        """
        Check if there's sufficient margin for a new position.

        Args:
            asset_id: Asset to trade
            quantity: Quantity to trade
            price: Current price

        Returns:
            Tuple of (has_sufficient_margin, required_margin)
        """
        asset_spec = self.asset_registry.get(asset_id)
        if not asset_spec:
            # Default to equity-like behavior
            required = abs(quantity) * float(price)
            return self.available_margin >= required, required

        required_margin = asset_spec.get_margin_requirement(quantity, price)
        has_margin = self.available_margin >= required_margin

        return has_margin, required_margin

    def open_position(
        self,
        asset_id: AssetId,
        quantity: Quantity,
        price: Price,
        timestamp: datetime,
    ) -> bool:
        """
        Open or modify a position with margin.

        Args:
            asset_id: Asset to trade
            quantity: Quantity to trade (positive for long, negative for short)
            price: Entry price
            timestamp: Transaction time

        Returns:
            Success status
        """
        has_margin, required_margin = self.check_margin_requirement(asset_id, quantity, price)

        if not has_margin:
            return False

        asset_spec = self.asset_registry.get(asset_id)

        if asset_id in self.positions:
            # Modify existing position
            pos = self.positions[asset_id]
            old_margin = pos["margin_used"]

            # Update position
            pos["quantity"] += quantity
            pos["avg_price"] = (
                (pos["avg_price"] * abs(pos["quantity"] - quantity) + float(price) * abs(quantity))
                / abs(pos["quantity"])
                if pos["quantity"] != 0
                else 0
            )
            pos["last_price"] = float(price)
            pos["timestamp"] = timestamp

            # Recalculate margin
            if asset_spec:
                new_margin = asset_spec.get_margin_requirement(pos["quantity"], price)
                pos["margin_used"] = new_margin
                margin_change = new_margin - old_margin
            else:
                pos["margin_used"] = abs(pos["quantity"]) * price
                margin_change = pos["margin_used"] - old_margin

            # Update account margins
            self.margin_used += margin_change
            self.available_margin -= margin_change

            # Remove position if closed
            if pos["quantity"] == 0:
                self.margin_used -= pos["margin_used"]
                self.available_margin += pos["margin_used"]
                del self.positions[asset_id]
        else:
            # Open new position
            self.positions[asset_id] = {
                "quantity": quantity,
                "avg_price": float(price),
                "last_price": float(price),
                "margin_used": required_margin,
                "timestamp": timestamp,
                "asset_spec": asset_spec,
            }

            self.margin_used = float(self.margin_used) + float(required_margin)
            self.available_margin = float(self.available_margin) - float(required_margin)

        # Update margin requirements
        self._update_margin_requirements()

        return True

    def update_prices(self, prices: dict[AssetId, Price]) -> list[MarginRequirement]:
        """
        Update positions with new prices and check margin requirements.

        Args:
            prices: Current market prices

        Returns:
            List of margin requirements/calls
        """
        margin_status = []

        for asset_id, price in prices.items():
            if asset_id not in self.positions:
                continue

            pos = self.positions[asset_id]
            pos["last_price"] = price

            # Calculate unrealized P&L
            pos["quantity"] * (price - pos["avg_price"])

            # Update margin for this position
            asset_spec = pos.get("asset_spec")
            if asset_spec and asset_spec.requires_margin:
                current_margin = asset_spec.get_margin_requirement(pos["quantity"], price)

                # Calculate liquidation price
                liquidation_price = self._calculate_liquidation_price(asset_id, pos, asset_spec)

                # Check margin status
                margin_req = MarginRequirement(
                    asset_id=asset_id,
                    initial_margin=asset_spec.initial_margin * abs(pos["quantity"]),
                    maintenance_margin=asset_spec.maintenance_margin * abs(pos["quantity"]),
                    current_margin=current_margin,
                    excess_margin=self.available_margin,
                    margin_call=current_margin
                    > float(self.available_margin) * self.margin_call_level,
                    liquidation_price=liquidation_price,
                )

                margin_status.append(margin_req)

                # Force liquidation if below threshold
                if current_margin > float(self.available_margin) * self.liquidation_level:
                    self._force_liquidation(asset_id, price)

        # Update total equity
        self._update_equity()

        return margin_status

    def _calculate_liquidation_price(
        self,
        _asset_id: AssetId,
        position: dict[str, Any],
        asset_spec: AssetSpec,
    ) -> Optional[Price]:
        """Calculate liquidation price for a position."""
        if not asset_spec.requires_margin:
            return None

        quantity = position["quantity"]
        avg_price = position["avg_price"]

        if asset_spec.asset_class.value == "future":
            # Futures liquidation when margin depleted
            maintenance_margin = asset_spec.maintenance_margin * abs(quantity)
            if quantity > 0:  # Long position
                return float(
                    avg_price
                    - (float(self.available_margin) - maintenance_margin)
                    / (quantity * asset_spec.contract_size)
                )
            # Short position
            return float(
                avg_price
                + (float(self.available_margin) - maintenance_margin)
                / (abs(quantity) * asset_spec.contract_size)
            )
        if asset_spec.asset_class.value == "fx":
            # FX liquidation based on leverage
            margin_used = position["margin_used"]
            if quantity > 0:
                return float(
                    avg_price
                    * (1 - self.liquidation_level * margin_used / (abs(quantity) * avg_price))
                )
            return float(
                avg_price * (1 + self.liquidation_level * margin_used / (abs(quantity) * avg_price))
            )

        return None

    def _force_liquidation(self, asset_id: AssetId, price: Price) -> None:
        """Force liquidate a position due to margin call."""
        if asset_id in self.positions:
            pos = self.positions[asset_id]

            # Return margin to available
            self.margin_used -= pos["margin_used"]
            self.available_margin += pos["margin_used"]

            # Calculate and apply loss
            loss = pos["quantity"] * (price - pos["avg_price"])
            self.cash_balance += loss  # Loss reduces cash

            # Remove position
            del self.positions[asset_id]

    def _update_margin_requirements(self) -> None:
        """Update total margin requirements."""
        self.initial_margin_requirement = 0.0
        self.maintenance_margin_requirement = 0.0

        for _asset_id, pos in self.positions.items():
            asset_spec = pos.get("asset_spec")
            if asset_spec and asset_spec.requires_margin:
                self.initial_margin_requirement += asset_spec.initial_margin * abs(pos["quantity"])
                self.maintenance_margin_requirement += asset_spec.maintenance_margin * abs(
                    pos["quantity"],
                )

    def _update_equity(self) -> None:
        """Update total account equity."""
        total_unrealized = 0.0

        for pos in self.positions.values():
            unrealized = pos["quantity"] * (pos["last_price"] - pos["avg_price"])

            # Apply contract multiplier for futures
            asset_spec = pos.get("asset_spec")
            if asset_spec and asset_spec.asset_class.value == "future":
                unrealized *= asset_spec.contract_size

            total_unrealized += unrealized

        # Update available margin with unrealized P&L
        self.available_margin = (
            float(self.cash_balance) + total_unrealized - float(self.margin_used)
        )

    def get_margin_status(self) -> dict[str, Any]:
        """Get current margin account status."""
        total_unrealized = sum(
            pos["quantity"] * (pos["last_price"] - pos["avg_price"])
            for pos in self.positions.values()
        )

        return {
            "cash_balance": self.cash_balance,
            "margin_used": self.margin_used,
            "available_margin": self.available_margin,
            "initial_requirement": self.initial_margin_requirement,
            "maintenance_requirement": self.maintenance_margin_requirement,
            "unrealized_pnl": total_unrealized,
            "total_equity": self.cash_balance + total_unrealized,
            "margin_utilization": float(self.margin_used) / float(self.available_margin)
            if self.available_margin > 0
            else 0,
            "num_positions": len(self.positions),
            "has_margin_call": any(mc.margin_call for mc in self.margin_calls),
        }
