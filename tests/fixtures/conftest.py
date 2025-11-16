"""Pytest configuration for ML signal fixtures.

This file makes the fixtures in ml_signal_data.py available to all tests
without requiring explicit imports.

The fixtures are automatically discovered by pytest when this package
is in the tests/ directory.
"""

# Import all fixtures to make them discoverable
from .ml_signal_data import (
    bear_market_data,
    bull_market_data,
    context_data,
    high_volatility_data,
    low_volatility_data,
    mean_reverting_data,
    ml_data_scenario,
    ml_signal_data,
    trending_market_data,
)

__all__ = [
    "ml_signal_data",
    "context_data",
    "bull_market_data",
    "bear_market_data",
    "high_volatility_data",
    "low_volatility_data",
    "trending_market_data",
    "mean_reverting_data",
    "ml_data_scenario",
]
