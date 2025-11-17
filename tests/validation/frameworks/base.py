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
class FrameworkConfig:
    """
    Unified configuration for cross-framework validation.

    This allows exact replication of any framework's behavior by setting
    appropriate parameters. All frameworks receive identical configuration
    to ensure apples-to-apples comparison.

    Example Presets:
        # Realistic (default) - Next-bar open fill, typical costs
        FrameworkConfig()

        # Backtrader compatible - Matches Backtrader defaults
        FrameworkConfig(fill_timing="next_open", backtrader_coo=False, backtrader_coc=False)

        # VectorBT compatible - Same-bar close fill (look-ahead bias!)
        FrameworkConfig(fill_timing="same_close", vectorbt_accumulate=False)
    """

    # Transaction costs
    commission_pct: float = 0.001       # 0.1% commission (percentage of trade value)
    commission_fixed: float = 0.0       # Fixed commission per trade ($)
    slippage_pct: float = 0.0005        # 0.05% slippage (percentage)
    slippage_fixed: float = 0.0         # Fixed slippage per share ($)

    # Execution timing (when fills occur)
    fill_timing: Literal["next_open", "next_close", "same_close"] = "next_open"

    # Capital and sizing
    initial_capital: float = 100000.0   # Starting cash
    fractional_shares: bool = False     # Allow fractional shares

    # Position management
    close_final_position: bool = False  # Auto-close open positions at backtest end
                                        # Default False: Framework should NOT make trading decisions
                                        # Set True only for comparison with frameworks that auto-close

    # Framework-specific behaviors
    backtrader_coo: bool = False        # Cheat-On-Open (dangerous!)
    backtrader_coc: bool = False        # Cheat-On-Close (dangerous!)
    vectorbt_accumulate: bool = False   # Allow same-bar re-entry

    def __post_init__(self):
        """Validate configuration."""
        if self.commission_pct < 0:
            raise ValueError("commission_pct must be non-negative")
        if self.slippage_pct < 0:
            raise ValueError("slippage_pct must be non-negative")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        # Warn about look-ahead bias configurations
        if self.fill_timing == "same_close":
            import warnings
            warnings.warn(
                "fill_timing='same_close' introduces look-ahead bias. "
                "Use 'next_open' for realistic backtesting.",
                UserWarning
            )
        if self.backtrader_coo or self.backtrader_coc:
            import warnings
            warnings.warn(
                "Backtrader COO/COC flags introduce look-ahead bias. "
                "Set to False for realistic backtesting.",
                UserWarning
            )

    @classmethod
    def backtrader_compatible(cls) -> "FrameworkConfig":
        """Preset matching Backtrader default behavior."""
        return cls(
            fill_timing="next_open",
            backtrader_coo=False,
            backtrader_coc=False,
        )

    @classmethod
    def vectorbt_compatible(cls) -> "FrameworkConfig":
        """Preset matching VectorBT default behavior (WARNING: look-ahead bias!)."""
        import warnings
        warnings.warn(
            "VectorBT default uses same-bar close fills (look-ahead bias). "
            "Consider using fill_timing='next_open' for realistic backtesting.",
            UserWarning
        )
        return cls(
            fill_timing="same_close",
            vectorbt_accumulate=False,
            fractional_shares=True,  # VectorBT allows fractional shares
        )

    @classmethod
    def realistic(cls) -> "FrameworkConfig":
        """Preset for realistic backtesting without look-ahead bias."""
        return cls(
            fill_timing="next_open",
            commission_pct=0.001,   # 0.1% commission
            slippage_pct=0.0005,    # 0.05% slippage
            backtrader_coo=False,
            backtrader_coc=False,
            vectorbt_accumulate=False,
            fractional_shares=False,
        )

    @classmethod
    def for_matching(cls) -> "FrameworkConfig":
        """
        Preset for cross-framework alignment testing.

        Uses same-bar close execution across all frameworks to eliminate
        timing-related divergence. WARNING: Introduces look-ahead bias.
        Use only for validation, not realistic backtesting.

        Configuration:
        - Same-bar close fills (all frameworks)
        - Minimal fees for cleaner comparison
        - Backtrader COC enabled (same-bar execution)
        - VectorBT accumulation disabled (flat-only trading)
        - Fractional shares enabled (prevents quantity mismatches)
        - No auto-close of final positions (framework doesn't make trading decisions)
        """
        import warnings
        warnings.warn(
            "for_matching() uses same-bar close fills (look-ahead bias). "
            "This is for validation only, NOT realistic backtesting.",
            UserWarning
        )
        return cls(
            fill_timing="same_close",
            commission_pct=0.0,      # No fees for cleaner comparison
            slippage_pct=0.0,
            backtrader_coo=False,
            backtrader_coc=True,     # Enable same-bar close execution
            vectorbt_accumulate=False,  # Flat-only trading
            fractional_shares=True,  # Prevent quantity mismatches
            close_final_position=False,  # Don't auto-close (frameworks shouldn't decide)
        )

    @classmethod
    def for_zipline_matching(cls) -> "FrameworkConfig":
        """
        Preset for matching Zipline's execution model across all frameworks.

        Zipline inherently fills orders on T+1 (next trading day), so this config
        makes all other frameworks delay execution by 1 day to achieve 100% alignment.

        Configuration:
        - Next-open fills (VectorBT, Zipline execute at next bar's open)
        - Backtrader COC=True (fills at signal bar's close - API limitation)
        - ml4t.backtest execution_delay=False (same-bar close fills to match Backtrader)
        - No fees for cleaner comparison
        - Fractional shares enabled (prevents quantity mismatches)
        - No auto-close of final positions

        IMPORTANT - Backtrader Limitation:
        Backtrader's sizing API (AllInSizer, order_target_value) calculates size
        at order placement using current bar's close price. With COC=False (next-bar
        fills), the price can gap and cause order rejections due to insufficient funds.

        To achieve 100% capital utilization with Backtrader, we MUST use COC=True
        (same-bar close fills), which introduces look-ahead bias. This is a documented
        Backtrader API limitation, not a recommended production configuration.

        ml4t.backtest is configured with execution_delay=False to match Backtrader's
        same-bar fills. VectorBT and Zipline use next-open fills (realistic).

        Source: backtrader/brokers/bbroker.py:893-903, sizers/percents_sizer.py:43
        """
        return cls(
            fill_timing="same_close",    # Backtrader-compatible (COC=True)
            commission_pct=0.0,          # No fees for cleaner comparison
            slippage_pct=0.0,
            backtrader_coo=False,
            backtrader_coc=True,         # Enable COC for Backtrader (API limitation)
            vectorbt_accumulate=False,   # Flat-only trading
            fractional_shares=True,      # Prevent quantity mismatches
            close_final_position=False,  # Don't auto-close
        )


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

    @abstractmethod
    def run_with_signals(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        config: FrameworkConfig | None = None,
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
            signals: DataFrame with DatetimeIndex and boolean columns:
                - 'entry': True when should enter position
                - 'exit': True when should exit position
            config: FrameworkConfig for execution parameters (costs, timing, etc.).
                   If None, uses FrameworkConfig.realistic() defaults.

        Returns:
            ValidationResult with performance metrics and trade records

        Note:
            All frameworks receive IDENTICAL boolean signals and configuration
            to test execution fidelity only. No framework should calculate indicators.
        """

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format."""
        required_cols = ["open", "high", "low", "close", "volume"]

        if not all(col in data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in data.columns]
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")

        return True

    def validate_signals(self, signals: pd.DataFrame, data: pd.DataFrame) -> bool:
        """
        Validate signal format and alignment with data.

        Args:
            signals: Signal DataFrame with 'entry' and 'exit' boolean columns
            data: OHLCV data DataFrame

        Returns:
            True if valid

        Raises:
            ValueError: If signals are invalid
        """
        # Check required columns
        if not all(col in signals.columns for col in ["entry", "exit"]):
            raise ValueError("Signals must have 'entry' and 'exit' boolean columns")

        # Check index type
        if not isinstance(signals.index, pd.DatetimeIndex):
            raise ValueError("Signals must have DatetimeIndex")

        # Check data types
        if signals["entry"].dtype != bool or signals["exit"].dtype != bool:
            raise ValueError("Signal 'entry' and 'exit' columns must be boolean")

        # Check alignment - all signal timestamps must exist in data
        missing_timestamps = signals.index.difference(data.index)
        if len(missing_timestamps) > 0:
            raise ValueError(
                f"Signal timestamps not found in data: {missing_timestamps[:5]} "
                f"(showing first 5 of {len(missing_timestamps)})"
            )

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
