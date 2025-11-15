"""Base adapter interface for platform-specific backtesting."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import polars as pl


@dataclass
class Trade:
    """Standardized trade representation across all platforms.

    Attributes:
        entry_time: When position was entered
        entry_price: Entry fill price
        exit_time: When position was exited (None if still open)
        exit_price: Exit fill price (None if still open)
        symbol: Asset symbol
        quantity: Position size (positive = long, negative = short)
        pnl: Realized P&L (None if still open)
        commission: Total commission paid (entry + exit)
        signal_id: ID of originating signal
        stop_loss_hit: Whether trade was closed by stop loss
        take_profit_hit: Whether trade was closed by take profit
        trailing_stop_hit: Whether trade was closed by trailing stop
    """
    entry_time: datetime
    entry_price: float
    symbol: str
    quantity: float
    signal_id: str
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl: float | None = None
    commission: float = 0.0
    stop_loss_hit: bool = False
    take_profit_hit: bool = False
    trailing_stop_hit: bool = False

    def is_closed(self) -> bool:
        """Check if trade is closed."""
        return self.exit_time is not None

    def trade_duration(self) -> float | None:
        """Calculate trade duration in days."""
        if not self.is_closed():
            return None
        return (self.exit_time - self.entry_time).total_seconds() / 86400


@dataclass
class BacktestResult:
    """Standardized backtest results across all platforms.

    Attributes:
        platform: Platform name (ml4t.backtest, vectorbt_pro, zipline, etc.)
        trades: List of all trades (open and closed)
        equity_curve: Time series of portfolio value
        metrics: Performance metrics dictionary
        execution_time: Backtest execution time in seconds
        config: Configuration used for the backtest
        metadata: Additional platform-specific information
    """
    platform: str
    trades: list[Trade]
    equity_curve: pl.DataFrame  # columns: timestamp, equity
    metrics: dict[str, float]
    execution_time: float
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_closed_trades(self) -> list[Trade]:
        """Get only closed trades."""
        return [t for t in self.trades if t.is_closed()]

    def get_open_trades(self) -> list[Trade]:
        """Get only open trades."""
        return [t for t in self.trades if not t.is_closed()]

    def total_pnl(self) -> float:
        """Calculate total realized P&L."""
        return sum(t.pnl for t in self.get_closed_trades() if t.pnl is not None)

    def win_rate(self) -> float:
        """Calculate win rate (%)."""
        closed = self.get_closed_trades()
        if not closed:
            return 0.0
        winning = sum(1 for t in closed if t.pnl and t.pnl > 0)
        return (winning / len(closed)) * 100

    def avg_win(self) -> float:
        """Calculate average winning trade P&L."""
        wins = [t.pnl for t in self.get_closed_trades() if t.pnl and t.pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0

    def avg_loss(self) -> float:
        """Calculate average losing trade P&L."""
        losses = [t.pnl for t in self.get_closed_trades() if t.pnl and t.pnl < 0]
        return sum(losses) / len(losses) if losses else 0.0


class PlatformAdapter(ABC):
    """Base class for platform-specific backtest adapters.

    Each adapter translates platform-independent signals into
    platform-specific backtest execution and standardized results.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run_backtest(
        self,
        signals: list,  # List of Signal objects
        data: Any,  # Platform-specific data format
        initial_capital: float = 100_000,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0,  # No slippage by default
        **kwargs
    ) -> BacktestResult:
        """Run backtest with given signals.

        Args:
            signals: List of platform-independent Signal objects
            data: Market data in platform-specific format
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage in basis points
            **kwargs: Platform-specific configuration

        Returns:
            BacktestResult with standardized trade and performance data
        """
        pass

    @abstractmethod
    def supports_stop_loss(self) -> bool:
        """Whether platform supports stop loss orders."""
        pass

    @abstractmethod
    def supports_take_profit(self) -> bool:
        """Whether platform supports take profit orders."""
        pass

    @abstractmethod
    def supports_trailing_stop(self) -> bool:
        """Whether platform supports trailing stop orders."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
