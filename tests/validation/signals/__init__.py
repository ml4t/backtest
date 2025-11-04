"""Platform-independent signal generators."""
from .base import Signal, SignalGenerator
from .ma_crossover import MACrossoverSignals
from .mean_reversion import MeanReversionSignals
from .random_signals import RandomSignals

__all__ = [
    'Signal',
    'SignalGenerator',
    'MACrossoverSignals',
    'MeanReversionSignals',
    'RandomSignals',
]
