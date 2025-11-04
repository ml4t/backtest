"""Platform-independent signal generator interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import polars as pl


@dataclass
class Signal:
    """Platform-agnostic trading signal.

    Attributes:
        timestamp: When signal was generated
        symbol: Asset symbol
        action: Trading action (BUY, SELL, CLOSE)
        quantity: Number of shares/contracts (None = close all)
        signal_id: Unique identifier for tracking
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        trailing_stop_pct: Optional trailing stop percentage
    """
    timestamp: datetime
    symbol: str
    action: Literal['BUY', 'SELL', 'CLOSE']
    quantity: float | None = None
    signal_id: str = ""
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop_pct: float | None = None

    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = f"{self.timestamp.isoformat()}_{self.symbol}_{self.action}"


class SignalGenerator(ABC):
    """Base class for platform-independent signal generation.

    All signal generators must:
    1. Be deterministic (same data -> same signals)
    2. Use only point-in-time data
    3. Output standardized Signal objects
    4. Be completely independent of any backtesting platform
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pl.DataFrame) -> list[Signal]:
        """Generate trading signals from OHLCV data.

        Args:
            data: Polars DataFrame with columns:
                  - timestamp: datetime
                  - symbol: str
                  - open, high, low, close: float
                  - volume: float (optional)

        Returns:
            List of Signal objects sorted by timestamp
        """
        pass

    def validate_data(self, data: pl.DataFrame) -> None:
        """Validate input data has required columns."""
        required = {'timestamp', 'symbol', 'open', 'high', 'low', 'close'}
        actual = set(data.columns)

        if not required.issubset(actual):
            missing = required - actual
            raise ValueError(f"Missing required columns: {missing}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
