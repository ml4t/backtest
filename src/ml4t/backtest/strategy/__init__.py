"""Strategy framework for ml4t.backtest."""

from ml4t.backtest.strategy.adapters import (
    DataFrameAdapter,
    ExternalStrategyInterface,
    PITData,
    StrategyAdapter,
    StrategySignal,
)
from ml4t.backtest.strategy.base import Strategy
from ml4t.backtest.strategy.crypto_basis_adapter import (
    CryptoBasisAdapter,
    CryptoBasisExternalStrategy,
    create_crypto_basis_strategy,
)

__all__ = [
    "CryptoBasisAdapter",
    "CryptoBasisExternalStrategy",
    "DataFrameAdapter",
    "ExternalStrategyInterface",
    "PITData",
    "Strategy",
    "StrategyAdapter",
    "StrategySignal",
    "create_crypto_basis_strategy",
]
