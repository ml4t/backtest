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
        In ``NEXT_BAR`` mode, orders from prior bars and deferred exits have also
        filled at the current open, so position and pending-order queries expose
        post-open state. Ordinary orders submitted here remain pending until the
        next bar. In ``SAME_BAR`` mode, the callback runs before regular pending
        orders are processed; a market order is visible to current-bar risk only
        when ``immediate_fill=True``.

        A position guard therefore prevents duplicate next-bar entries. Strategies
        can pyramid explicitly by submitting additional orders without that guard,
        and can inspect ``broker.get_pending_orders(asset)`` when pending intent is
        relevant to their sizing rule.
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
