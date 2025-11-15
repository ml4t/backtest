"""
Base classes for cross-framework validation.

Defines common interfaces and data structures for validating ml4t.backtest
against other backtesting frameworks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypedDict

import pandas as pd


class Signal(TypedDict):
    """Pre-computed trading signal."""

    timestamp: datetime
    asset_id: str
    action: Literal["BUY", "SELL"]
    quantity: float


@dataclass
class TradeRecord:
    """Standardized trade record for cross-framework comparison."""

    timestamp: datetime
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    value: float
    commission: float = 0.0

    def __str__(self):
        ts_str = self.timestamp.strftime('%Y-%m-%d') if self.timestamp else 'N/A'
        return f"{ts_str} {self.action} {self.quantity:.2f} @ ${self.price:.2f}"


@dataclass
class ValidationResult:
    """Standardized result structure for cross-framework validation."""

    framework: str
    strategy: str
    initial_capital: float = 10000.0
    final_value: float = 0.0
    total_return: float = 0.0  # Percentage
    num_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    execution_time: float = 0.0
    memory_usage: float = 0.0  # MB
    trades: list[TradeRecord] = field(default_factory=list)
    daily_returns: pd.Series | None = None
    equity_curve: pd.Series | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def add_error(self, error: str):
        self.errors.append(error)

    def summary_dict(self) -> dict[str, Any]:
        """Return key metrics as dictionary for comparison tables."""
        return {
            "Framework": self.framework,
            "Final Value ($)": f"{self.final_value:,.2f}",
            "Return (%)": f"{self.total_return:.2f}",
            "Trades": self.num_trades,
            "Win Rate": f"{self.win_rate:.2%}",
            "Sharpe": f"{self.sharpe_ratio:.2f}",
            "Max DD (%)": f"{self.max_drawdown:.2f}",
            "Time (s)": f"{self.execution_time:.3f}",
            "Memory (MB)": f"{self.memory_usage:.1f}",
            "Status": "✓" if not self.has_errors else "✗",
        }


class BaseFrameworkAdapter(ABC):
    """Abstract base class for framework adapters."""

    def __init__(self, framework_name: str):
        self.framework_name = framework_name

    @abstractmethod
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy_params: dict[str, Any],
        initial_capital: float = 10000,
    ) -> ValidationResult:
        """
        Run backtest with given data and strategy parameters.

        Args:
            data: OHLCV data with DatetimeIndex
            strategy_params: Strategy configuration (windows, thresholds, etc.)
            initial_capital: Starting capital for backtest

        Returns:
            ValidationResult with performance metrics and trade records
        """

    def run_with_signals(
        self,
        data: pd.DataFrame,
        signals: list[Signal],
        initial_capital: float = 10000,
    ) -> ValidationResult:
        """
        Run backtest with pre-computed signals (eliminates calculation variance).

        This method provides pure execution validation by accepting pre-computed
        entry/exit signals. This eliminates variance from:
        - Different indicator calculations (MA, RSI, etc.)
        - Floating point rounding differences
        - Data source differences

        Args:
            data: OHLCV data with DatetimeIndex (for price lookup)
            signals: List of pre-computed trading signals with timestamps,
                actions (BUY/SELL), and quantities
            initial_capital: Starting capital for backtest

        Returns:
            ValidationResult with performance metrics and trade records

        Raises:
            NotImplementedError: If framework doesn't support signal-based execution
        """
        raise NotImplementedError(
            f"{self.framework_name} adapter does not yet support signal-based execution. "
            "Implement run_with_signals() to enable pure execution validation."
        )

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        required_cols = ["open", "high", "low", "close", "volume"]

        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        return True


class BaseStrategy(ABC):
    """Abstract base class for strategy implementations."""

    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate buy/sell signals for given data.

        Args:
            data: OHLCV data

        Returns:
            DataFrame with 'entries' and 'exits' boolean columns
        """

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Get strategy parameters for reproducibility."""


class MomentumStrategy(BaseStrategy):
    """Moving average crossover momentum strategy."""

    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(
            "MovingAverageCrossover",
            short_window=short_window,
            long_window=long_window,
        )
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MA crossover signals."""
        # Calculate moving averages
        ma_short = data["close"].rolling(window=self.short_window).mean()
        ma_long = data["close"].rolling(window=self.long_window).mean()

        # Generate signals
        entries = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
        exits = (ma_short <= ma_long) & (ma_short.shift(1) > ma_long.shift(1))

        # Remove NaN values
        valid_mask = ~(ma_short.isna() | ma_long.isna())
        entries = entries & valid_mask
        exits = exits & valid_mask

        return pd.DataFrame(
            {"entries": entries, "exits": exits, "ma_short": ma_short, "ma_long": ma_long},
            index=data.index,
        )

    def get_parameters(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "short_window": self.short_window,
            "long_window": self.long_window,
        }
