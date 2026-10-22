"""Explicit dated-contract boundaries on the native simulation clock."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from ml4t.specs import LifecyclePhase

from .controls import OrderControlContext, TradingControlViolation
from .execution.schedule import RebalanceSchedule
from .types import AssetClass

if TYPE_CHECKING:
    from .broker import Broker
    from .engine import Engine
    from .runtime import RuntimeServices


@dataclass(frozen=True)
class FuturesLifecyclePolicy:
    """Close or roll at a declared physical notice/expiry boundary.

    Choose the applicable boundary upstream, including its calendar convention.
    Dates and unknown terms are not executable instants. Orders use the next
    actual quote and ordinary execution controls. Missing quotes leave pending
    orders; no synthetic expiry settlement or terminal liquidation is performed.
    """

    event_id: str
    asset: str
    effective_at: datetime
    available_at: datetime
    reason: Literal["first_notice", "last_trade", "auto_close", "expiration"]
    source: str
    revision_id: str
    replacement_asset: str | None = None
    new_quantity: float | None = None

    def __post_init__(self) -> None:
        for name in ("event_id", "asset", "source", "revision_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Lifecycle {name} must be a nonempty string")
        if any(
            not isinstance(t, datetime) or t != t for t in (self.effective_at, self.available_at)
        ):
            raise TypeError("Lifecycle clocks require known physical datetime instants")
        if (self.effective_at.tzinfo is None) != (self.available_at.tzinfo is None):
            raise ValueError("Lifecycle clocks require compatible timezone awareness")
        if self.available_at > self.effective_at:
            raise ValueError("Lifecycle boundary cannot precede declared availability")
        if self.reason not in {"first_notice", "last_trade", "auto_close", "expiration"}:
            raise ValueError("Unknown lifecycle boundary reason")
        if self.replacement_asset is not None and (
            not isinstance(self.replacement_asset, str)
            or not self.replacement_asset.strip()
            or self.replacement_asset == self.asset
        ):
            raise ValueError("Lifecycle roll requires a distinct replacement asset")
        if self.new_quantity is not None:
            from math import isfinite

            if (
                self.replacement_asset is None
                or isinstance(self.new_quantity, bool)
                or not isinstance(self.new_quantity, (int, float))
                or not isfinite(self.new_quantity)
                or self.new_quantity == 0
            ):
                raise ValueError("Declare a finite nonzero new_quantity only for a roll")

    def _register(self, runtime: RuntimeServices) -> None:
        runtime.scheduler.register(
            self._execute,
            phase=LifecyclePhase.INTRABAR,
            at="clock",
            schedule=RebalanceSchedule.explicit_timestamps([self.effective_at]),
        )

    def _check_engine(self, engine: Engine) -> None:
        for asset in (self.asset, self.replacement_asset):
            if asset is None:
                continue
            spec = engine.broker.get_contract_spec(asset)
            if spec is None or spec.asset_class is not AssetClass.FUTURE:
                raise ValueError("Lifecycle policies require declared FUTURE ContractSpecs")
        timestamp = engine.feed.timestamps[0]
        if (timestamp.tzinfo is None) != (self.effective_at.tzinfo is None):
            raise ValueError("Lifecycle and feed clocks require compatible timezone awareness")

    def validate(self, request: OrderControlContext) -> None:
        """After the boundary allow reductions only, including existing pending legs."""
        if (
            request.asset != self.asset
            or request.timestamp is None
            or request.timestamp < self.effective_at
        ):
            return
        resulting = request.position_quantity + request.pending_quantity + request.quantity
        if (
            request.quantity * request.position_quantity > 0
            or abs(resulting) > abs(request.position_quantity)
            or resulting * request.position_quantity < 0
        ):
            raise TradingControlViolation(
                "Order increases exposure after the futures lifecycle boundary"
            )

    def _execute(self, timestamp: datetime, data: dict, context: dict, broker: Broker) -> None:
        for order in broker.get_pending_orders(self.asset):
            broker.cancel_order(order.order_id)
        if self.replacement_asset is not None:
            broker.roll_position(
                self.asset,
                self.replacement_asset,
                roll_id=self.event_id,
                decision_at=timestamp,
                available_at=self.available_at,
                new_quantity=self.new_quantity,
                selection={
                    "reason": self.reason,
                    "source": self.source,
                    "revision_id": self.revision_id,
                    "effective_at": self.effective_at.isoformat(),
                },
            )
        else:
            position = broker.get_position(self.asset)
            if position is not None and position.quantity:
                broker.submit_order(self.asset, -position.quantity)
