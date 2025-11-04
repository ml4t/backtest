"""QEngine - A state-of-the-art event-driven backtesting engine.

QEngine is designed for high-performance backtesting of machine learning-driven
trading strategies with a focus on preventing data leakage and providing
realistic market simulation.
"""

__version__ = "0.1.0"

from qengine.core import Clock, Event, EventBus
from qengine.data import DataFeed
from qengine.engine import BacktestEngine, BacktestResults
from qengine.strategy import Strategy

__all__ = [
    "BacktestEngine",
    "BacktestResults",
    "Clock",
    "DataFeed",
    "Event",
    "EventBus",
    "Strategy",
]
