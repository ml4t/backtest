"""Context management for market-wide data in ml4t.backtest.

Context provides access to market-wide data (VIX, SPY, regime indicators) that:
- Is shared across all assets in a multi-asset backtest
- Changes only on bar boundaries (timestamp-based)
- Saves 50x memory vs embedding in every MarketEvent

Design:
- Context is a timestamped dictionary of market-wide values
- Engine creates one Context per timestamp
- All strategies receive the same Context instance (memory efficient)
- Context is immutable after creation (no strategy can modify it)

Example:
    # Engine creates context
    context = Context(
        timestamp=datetime(2024, 1, 15),
        data={'VIX': 18.5, 'SPY': 485.0, 'regime': 'bull'}
    )

    # Strategy uses context
    def on_market_data(self, event: MarketEvent, context: dict):
        if context.get('VIX', 0) > 30:
            return  # Don't trade in high volatility
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Context:
    """
    Market-wide context data for a specific timestamp.

    This class provides immutable access to market-wide data that is:
    - Shared across all assets (VIX, market indices, regime indicators)
    - Constant within a timestamp (changes only on bar boundaries)
    - Memory efficient (one instance per timestamp, not per asset)

    Attributes:
        timestamp: The timestamp this context applies to
        data: Dictionary of market-wide values (VIX, SPY, etc.)

    Example:
        context = Context(
            timestamp=datetime(2024, 1, 15, 9, 30),
            data={
                'VIX': 18.5,
                'SPY': 485.0,
                'regime': 'bull',
                'trend_strength': 0.85
            }
        )

        # Access in strategy
        vix = context['VIX']
        if vix > 30:
            # High volatility - reduce position sizing
    """

    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """
        Get context value by key.

        Args:
            key: Context key (e.g., 'VIX', 'SPY')

        Returns:
            Context value

        Raises:
            KeyError: If key not found in context
        """
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get context value with default fallback.

        Args:
            key: Context key (e.g., 'VIX', 'SPY')
            default: Default value if key not found

        Returns:
            Context value or default
        """
        return self.data.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in context."""
        return key in self.data

    def keys(self):
        """Return context keys."""
        return self.data.keys()

    def values(self):
        """Return context values."""
        return self.data.values()

    def items(self):
        """Return context items."""
        return self.data.items()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert context to plain dictionary for serialization.

        Returns:
            Dictionary with timestamp and data
        """
        return {
            'timestamp': self.timestamp.isoformat(),
            'data': self.data.copy()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Context':
        """
        Create Context from dictionary.

        Args:
            data: Dictionary with 'timestamp' and 'data' keys

        Returns:
            Context instance
        """
        timestamp = datetime.fromisoformat(data['timestamp'])
        return cls(timestamp=timestamp, data=data.get('data', {}))

    def __repr__(self) -> str:
        keys = ', '.join(self.data.keys())
        return f"Context(timestamp={self.timestamp.isoformat()}, keys=[{keys}])"


class ContextCache:
    """
    Cache for Context objects to avoid recreating identical contexts.

    For multi-asset backtests, many assets may share the same timestamp.
    This cache ensures we create only one Context instance per timestamp,
    saving memory and improving performance.

    Example:
        cache = ContextCache()

        # First call creates context
        ctx1 = cache.get_or_create(
            timestamp=ts,
            data={'VIX': 18.5, 'SPY': 485.0}
        )

        # Second call returns cached instance (same timestamp)
        ctx2 = cache.get_or_create(
            timestamp=ts,
            data={'VIX': 18.5, 'SPY': 485.0}
        )

        assert ctx1 is ctx2  # Same object instance
    """

    def __init__(self):
        """Initialize empty cache."""
        self._cache: dict[datetime, Context] = {}

    def get_or_create(self, timestamp: datetime, data: dict[str, Any]) -> Context:
        """
        Get cached context or create new one.

        Args:
            timestamp: Timestamp for context
            data: Market-wide data dictionary

        Returns:
            Context instance (cached or newly created)
        """
        if timestamp not in self._cache:
            self._cache[timestamp] = Context(timestamp=timestamp, data=data)
        return self._cache[timestamp]

    def get(self, timestamp: datetime) -> Context | None:
        """
        Get cached context for timestamp.

        Args:
            timestamp: Timestamp to lookup

        Returns:
            Cached Context or None if not found
        """
        return self._cache.get(timestamp)

    def clear(self):
        """Clear all cached contexts."""
        self._cache.clear()

    def clear_before(self, timestamp: datetime):
        """
        Clear cached contexts before a timestamp.

        Useful for memory management in long backtests.

        Args:
            timestamp: Clear contexts before this timestamp
        """
        to_remove = [ts for ts in self._cache.keys() if ts < timestamp]
        for ts in to_remove:
            del self._cache[ts]

    def size(self) -> int:
        """
        Get number of cached contexts.

        Returns:
            Cache size
        """
        return len(self._cache)

    def __repr__(self) -> str:
        return f"ContextCache(size={len(self._cache)})"


__all__ = ["Context", "ContextCache"]
