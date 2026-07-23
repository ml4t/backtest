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
        """Called before position rules are evaluated for the current bar."""
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
