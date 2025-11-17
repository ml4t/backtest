"""ml4t.backtest - A state-of-the-art event-driven backtesting engine.

ml4t.backtest is designed for high-performance backtesting of machine learning-driven
trading strategies with a focus on preventing data leakage and providing
realistic market simulation.
"""

__version__ = "0.1.0"

from ml4t.backtest.config import BacktestConfig
from ml4t.backtest.core import Clock, Event
from ml4t.backtest.data import DataFeed
from ml4t.backtest.engine import BacktestEngine, BacktestResults
from ml4t.backtest.strategy import Strategy

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResults",
    "Clock",
    "DataFeed",
    "Event",
    "Strategy",
]
