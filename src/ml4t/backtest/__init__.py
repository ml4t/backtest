"""ML4T Backtest - Event-driven backtesting engine.

A minimal, event-driven backtesting engine with:
- Point-in-time correctness
- Exit-first order processing
- Cash and margin account policies
- VectorBT-validated results

Example:
    >>> from ml4t.backtest import Engine, Strategy, BacktestConfig
    >>> 
    >>> class MyStrategy(Strategy):
    ...     def on_bar(self, bar):
    ...         if self.position == 0:
    ...             self.buy(size=100)
    >>> 
    >>> result = Engine(data, MyStrategy()).run()
    >>> print(result.metrics)
"""

from ml4t.backtest.types import (
    Order,
    OrderType,
    OrderSide,
    Fill,
    Position,
    Trade,
)
from ml4t.backtest.config import BacktestConfig
from ml4t.backtest.strategy import Strategy
from ml4t.backtest.engine import Engine
from ml4t.backtest.broker import Broker
from ml4t.backtest.datafeed import DataFeed
from ml4t.backtest.result import BacktestResult

try:
    from ml4t.backtest._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    # Core types
    "Order",
    "OrderType",
    "OrderSide",
    "Fill",
    "Position",
    "Trade",
    # Main classes
    "BacktestConfig",
    "Strategy",
    "Engine",
    "Broker",
    "DataFeed",
    "BacktestResult",
]


def list_order_types() -> list[str]:
    """List available order types."""
    return [t.name for t in OrderType]


def list_presets() -> list[str]:
    """List available configuration presets."""
    from ml4t.backtest import presets
    return [name for name in dir(presets) if not name.startswith('_')]
