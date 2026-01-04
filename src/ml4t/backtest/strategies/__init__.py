"""Pre-built strategy templates for common trading patterns.

This module provides reusable strategy base classes that encapsulate
common trading patterns. Extend these templates instead of writing
strategies from scratch.

Available Templates:
    SignalFollowingStrategy: Follow pre-computed signals (ML predictions, indicators)
    MomentumStrategy: Trend-following based on lookback returns
    MeanReversionStrategy: Buy oversold, sell at mean reversion
    LongShortStrategy: Long winners, short losers based on ranking

Example:
    >>> from ml4t.backtest import Engine, Mode, DataFeed
    >>> from ml4t.backtest.strategies import SignalFollowingStrategy
    >>>
    >>> class MyMLStrategy(SignalFollowingStrategy):
    ...     signal_column = "prediction"
    ...     position_size = 0.05
    ...
    ...     def should_enter_long(self, signal):
    ...         return signal > 0.7
    ...
    ...     def should_exit(self, signal):
    ...         return signal < 0.3
    >>>
    >>> engine = Engine.from_mode(feed, MyMLStrategy(), Mode.REALISTIC)
    >>> result = engine.run()
"""

from .templates import (
    LongShortStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    SignalFollowingStrategy,
)

__all__ = [
    "SignalFollowingStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "LongShortStrategy",
]
