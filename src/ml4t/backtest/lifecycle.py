"""Causal strategy callback dispatch for the backtest engine."""

from __future__ import annotations

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

    def __init__(self, broker: Any) -> None:
        self.broker = broker
        self.state: dict[str, Any] | None = None

    def capture(self) -> None:
        """Capture a bounded state snapshot once, before the first mutation."""
        if self.state is not None:
            return
        self.state = self.broker._snapshot_lifecycle_state()

    def rollback(self) -> None:
        """Restore captured state; a read-only callback has nothing to restore."""
        if self.state is None:
            return
        self.broker._restore_lifecycle_state(self.state)


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
        previous_phase = broker._begin_lifecycle_dispatch(phase, transaction)
        self._counts[phase] += 1
        self.invocations.append(LifecycleInvocation(phase, specification.callback, event_time))
        try:
            return callback(*args)
        except BaseException:
            transaction.rollback()
            raise
        finally:
            broker._end_lifecycle_dispatch(previous_phase)

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
