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
        """Initialize strategy state before the first bar.

        The broker is configured, but no market bar has been registered. Use
        this callback for position rules and state that does not require prices.
        """
        pass

    def on_prepare(
        self,
        broker: Any,
        config: Any | None = None,
    ) -> None:
        """Called after on_start with causal configuration and no future feed data."""
        return None

    def on_end(self, broker: Any) -> None:  # noqa: B027
        """Finalize strategy state after the final bar.

        Open positions remain marked in the returned result; the engine does not
        submit automatic end-of-data closing orders before this callback.
        """
        pass

    def _validate_completed_run(self) -> None:
        """Validate internal strategy state after a successful run."""
        return None
