"""Base strategy class for backtesting."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Any


class Strategy(ABC):
    """Base strategy class."""

    def on_before_risk(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Any,
    ) -> None:
        """Run strategy logic immediately before current-bar position risk.

        The broker has registered the current bar's prices before this callback.
        In ``NEXT_BAR`` mode, a priced, policy-valid market entry submitted from a
        flat position by this callback on a prior bar can fill at the current open
        before the callback runs. Partial fills are visible while the remainder stays
        pending.
        Newly opened positions start risk evaluation on the following bar, preserving
        next-bar timing. Untriggered limit or stop orders remain pending, so guarded
        entries must check both ``broker.get_position(asset)`` and
        ``broker.get_pending_orders(asset)``. Ordinary orders submitted here remain
        pending until the next bar. In ``SAME_BAR`` mode, the callback runs before
        regular pending orders are processed; a market order is visible to current-bar
        risk only when ``immediate_fill=True``.

        Strategies can pyramid explicitly by submitting additional orders without
        the flat-position and pending-order guard.
        """
        return None

    @abstractmethod
    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],
        context: dict[str, Any],
        broker: Any,  # Avoid circular import, use Any for broker type
    ) -> None:
        """Called for each timestamp with all available data."""
        pass

    def on_start(self, broker: Any) -> None:  # noqa: B027
        """Called before backtest starts."""
        pass

    def on_prepare(
        self,
        broker: Any,
        timestamps: Sequence[datetime],
        config: Any | None = None,
    ) -> None:
        """Called before on_start with access to the full feed timestamp universe."""
        return None

    def on_end(self, broker: Any) -> None:  # noqa: B027
        """Called after backtest ends."""
        pass
