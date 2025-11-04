"""Platform adapters for cross-platform validation."""
from .backtrader_adapter import BacktraderAdapter
from .base import BacktestResult, PlatformAdapter, Trade
from .qengine_adapter import QEngineAdapter
from .vectorbt_adapter import VectorBTFreeAdapter, VectorBTProAdapter
from .zipline_adapter import ZiplineAdapter

__all__ = [
    'PlatformAdapter',
    'Trade',
    'BacktestResult',
    'QEngineAdapter',
    'VectorBTProAdapter',
    'VectorBTFreeAdapter',
    'ZiplineAdapter',
    'BacktraderAdapter',
]
