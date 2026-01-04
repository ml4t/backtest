"""Base strategy class for backtesting."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class Strategy(ABC):
    """Base strategy class."""

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

    def on_end(self, broker: Any) -> None:  # noqa: B027
        """Called after backtest ends."""
        pass
