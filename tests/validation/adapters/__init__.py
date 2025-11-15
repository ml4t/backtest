"""Platform adapters for cross-platform validation."""
from .backtrader_adapter import BacktraderAdapter
from .base import BacktestResult, PlatformAdapter, Trade
from .ml4t.backtest_adapter import ml4t.backtestAdapter
from .vectorbt_adapter import VectorBTFreeAdapter, VectorBTProAdapter
from .zipline_adapter import ZiplineAdapter

__all__ = [
    'PlatformAdapter',
    'Trade',
    'BacktestResult',
    'ml4t.backtestAdapter',
    'VectorBTProAdapter',
    'VectorBTFreeAdapter',
    'ZiplineAdapter',
    'BacktraderAdapter',
]
