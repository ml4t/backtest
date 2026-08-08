"""Causal strategy callback dispatch for the backtest engine."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ml4t.specs import LIFECYCLE_V1, LifecycleContract, LifecyclePhase


@dataclass(frozen=True, slots=True)
class LifecycleInvocation:
    """One strategy callback invocation retained for parity checks."""

    phase: LifecyclePhase
    callback: str
    event_time: datetime | None


class _BrokerTransaction:
    """Capture mutable broker state only when a callback first mutates it."""

    _DEEP_ATTRIBUTES = (
        "account",
        "positions",
        "_position_rules",
        "_position_rules_by_asset",
        "_pending_exits",
        "_partial_orders",
        "_filled_this_bar",
        "_stop_exits_this_bar",
        "_positions_created_this_bar",
        "_asset_stats",
        "_stats_config",
    )
    _SCALAR_ATTRIBUTES = (
        "_order_counter",
        "_rebalance_counter",
        "_session_config",
        "_last_session_id",
    )

    def __init__(self, broker: Any) -> None:
        self.broker = broker
        self.state: dict[str, Any] | None = None

    def capture(self) -> None:
        """Capture a bounded state snapshot once, before the first mutation."""
        if self.state is not None:
            return
        pending_orders = tuple(self.broker.pending_orders)
        self.state = {
            "deep": {
                name: copy.deepcopy(getattr(self.broker, name))
                for name in self._DEEP_ATTRIBUTES
                if hasattr(self.broker, name)
            },
            "scalars": {
                name: getattr(self.broker, name)
                for name in self._SCALAR_ATTRIBUTES
                if hasattr(self.broker, name)
            },
            "orders": tuple(self.broker.orders),
            "pending_orders": pending_orders,
            "pending_order_state": {
                id(order): copy.deepcopy(vars(order)) for order in pending_orders
            },
            "fills_length": len(self.broker.fills),
            "trades_length": len(self.broker.trades),
            "orders_this_bar": tuple(self.broker._orders_this_bar),
            "orders_this_bar_ids": set(self.broker._orders_this_bar_ids),
        }

    def rollback(self) -> None:
        """Restore captured state; a read-only callback has nothing to restore."""
        if self.state is None:
            return
        for name, value in self.state["deep"].items():
            setattr(self.broker, name, value)
        for name, value in self.state["scalars"].items():
            setattr(self.broker, name, value)
        for order in self.state["pending_orders"]:
            order.__dict__.clear()
            order.__dict__.update(copy.deepcopy(self.state["pending_order_state"][id(order)]))
        self.broker.orders[:] = self.state["orders"]
        self.broker.pending_orders[:] = self.state["pending_orders"]
        del self.broker.fills[self.state["fills_length"] :]
        del self.broker.trades[self.state["trades_length"] :]
        self.broker._orders_this_bar[:] = self.state["orders_this_bar"]
        self.broker._orders_this_bar_ids = self.state["orders_this_bar_ids"]
        self.broker.gatekeeper.account = self.broker.account


class LifecycleDispatcher:
    """Invoke strategy callbacks under one versioned contract."""

    def __init__(self, strategy: Any, contract: LifecycleContract = LIFECYCLE_V1) -> None:
        self.strategy = strategy
        self.contract = contract
        self.invocations: list[LifecycleInvocation] = []
        self._counts = dict.fromkeys(LifecyclePhase, 0)

    @property
    def callback_counts(self) -> dict[LifecyclePhase, int]:
        """Return a copy of successful and failed callback invocation counts."""
        return dict(self._counts)

    def dispatch(
        self,
        phase: LifecyclePhase,
        broker: Any,
        *args: Any,
        event_time: datetime | None = None,
    ) -> Any:
        """Invoke one callback and roll back broker mutation if it raises."""
        specification = self.contract.phase_spec(phase)
        callback = getattr(self.strategy, specification.callback)
        transaction = _BrokerTransaction(broker)
        if broker._lifecycle_transaction is not None:
            raise RuntimeError("nested lifecycle dispatch is not supported")
        broker._lifecycle_transaction = transaction
        self._counts[phase] += 1
        self.invocations.append(LifecycleInvocation(phase, specification.callback, event_time))
        try:
            return callback(*args)
        except BaseException:
            transaction.rollback()
            raise
        finally:
            broker._lifecycle_transaction = None

    def validate_completed_run(self, market_event_count: int) -> None:
        """Validate exactly-once boundaries and ordinary event callback counts."""
        for phase in (
            LifecyclePhase.CAUSAL_INITIALIZATION,
            LifecyclePhase.RUN_START,
            LifecyclePhase.RUN_END,
        ):
            self.contract.phase_spec(phase).validate_count(self._counts[phase])
        self.contract.phase_spec(LifecyclePhase.MARKET_EVENT).validate_count(
            self._counts[LifecyclePhase.MARKET_EVENT],
            event_count=market_event_count,
        )


def callback_trace(
    invocations: Sequence[LifecycleInvocation],
) -> tuple[tuple[str, str, datetime | None], ...]:
    """Return a stable callback trace for cross-engine comparison."""
    return tuple(
        (invocation.phase.value, invocation.callback, invocation.event_time)
        for invocation in invocations
    )


__all__ = ["LifecycleDispatcher", "LifecycleInvocation", "callback_trace"]
