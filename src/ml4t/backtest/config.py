"""
Backtest Configuration

Centralized configuration for all backtesting behavior. This allows:
1. Consistent behavior across all backtests
2. Easy replication of other frameworks (Backtrader, VectorBT, Zipline)
3. Clear documentation of all configurable behaviors
4. No code changes needed - just swap configuration files

Usage:
    from ml4t.backtest import BacktestConfig

    # Load default config
    config = BacktestConfig()

    # Load preset (e.g., backtrader-compatible)
    config = BacktestConfig.from_preset("backtrader")

    # Load from file
    config = BacktestConfig.from_yaml("my_config.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class FillTiming(str, Enum):
    """When orders are filled relative to signal generation."""

    SAME_BAR = "same_bar"  # Fill on same bar as signal (look-ahead bias risk)
    NEXT_BAR_OPEN = "next_bar_open"  # Fill at next bar's open (most realistic)
    NEXT_BAR_CLOSE = "next_bar_close"  # Fill at next bar's close


class ExecutionPrice(str, Enum):
    """Price used for order execution."""

    CLOSE = "close"  # Use bar's close price
    OPEN = "open"  # Use bar's open price
    VWAP = "vwap"  # Volume-weighted average price (requires volume data)
    MID = "mid"  # (high + low) / 2


class ShareType(str, Enum):
    """Type of share quantities allowed."""

    FRACTIONAL = "fractional"  # Allow fractional shares (0.5, 1.234, etc.)
    INTEGER = "integer"  # Round down to whole shares (like most real brokers)


class SizingMethod(str, Enum):
    """How position size is calculated."""

    PERCENT_OF_PORTFOLIO = "percent_of_portfolio"  # % of total portfolio value
    PERCENT_OF_CASH = "percent_of_cash"  # % of available cash only
    FIXED_VALUE = "fixed_value"  # Fixed dollar amount per position
    FIXED_SHARES = "fixed_shares"  # Fixed number of shares


class SignalProcessing(str, Enum):
    """How signals are processed relative to existing positions."""

    CHECK_POSITION = "check_position"  # Only act if no existing position (event-driven)
    PROCESS_ALL = "process_all"  # Process all signals regardless (vectorized)


class CommissionModel(str, Enum):
    """Commission calculation method."""

    NONE = "none"  # No commission
    PERCENTAGE = "percentage"  # % of trade value
    PER_SHARE = "per_share"  # Fixed amount per share
    PER_TRADE = "per_trade"  # Fixed amount per trade
    TIERED = "tiered"  # Volume-based tiers


class SlippageModel(str, Enum):
    """Slippage calculation method."""

    NONE = "none"  # No slippage
    PERCENTAGE = "percentage"  # % of price
    FIXED = "fixed"  # Fixed dollar amount
    VOLUME_BASED = "volume_based"  # Based on trade size vs volume


class DataFrequency(str, Enum):
    """Data frequency for the backtest."""

    DAILY = "daily"  # Daily bars (EOD)
    MINUTE_1 = "1m"  # 1-minute bars
    MINUTE_5 = "5m"  # 5-minute bars
    MINUTE_15 = "15m"  # 15-minute bars
    MINUTE_30 = "30m"  # 30-minute bars
    HOURLY = "1h"  # Hourly bars
    IRREGULAR = "irregular"  # Trade bars, tick aggregations (no fixed frequency)


class TrailHwmSource(str, Enum):
    """Source for trailing stop high-water mark UPDATE calculation.

    Controls which price is used to update HWM on each bar AFTER entry:
    - CLOSE: Use close prices for HWM updates (default, matches backtest-nb/rs)
    - HIGH: Use high prices for HWM updates (VBT Pro with OHLC data)

    Note: Initial HWM on entry bar is controlled by InitialHwmSource.
    """

    CLOSE = "close"  # Use close prices for HWM updates (default)
    HIGH = "high"  # Use high prices for HWM updates (VBT Pro with OHLC)


class InitialHwmSource(str, Enum):
    """Source for initial high-water mark on position entry.

    Controls what price is used for HWM when a new position is created:
    - FILL_PRICE: Use the actual fill price including slippage (default)
    - BAR_CLOSE: Use the bar's close price
    - BAR_HIGH: Use the bar's high price (VBT Pro with OHLC data)

    VBT Pro with OHLC data uses BAR_HIGH for initial HWM. This is because
    VBT Pro updates HWM from bar highs vectorially, including the entry bar.
    Most event-driven frameworks use the actual fill price.
    """

    FILL_PRICE = "fill_price"  # Use fill price (default, most frameworks)
    BAR_CLOSE = "bar_close"  # Use bar's close
    BAR_HIGH = "bar_high"  # Use bar's high (VBT Pro with OHLC)


@dataclass
class BacktestConfig:
    """
    Complete configuration for backtesting behavior.

    All behavioral differences between frameworks are captured here.
    Load presets to match specific frameworks exactly.
    """

    # === Execution Timing ===
    fill_timing: FillTiming = FillTiming.NEXT_BAR_OPEN
    execution_price: ExecutionPrice = ExecutionPrice.CLOSE

    # === Position Sizing ===
    share_type: ShareType = ShareType.FRACTIONAL
    sizing_method: SizingMethod = SizingMethod.PERCENT_OF_PORTFOLIO
    default_position_pct: float = 0.10  # 10% of portfolio per position

    # === Signal Processing ===
    signal_processing: SignalProcessing = SignalProcessing.CHECK_POSITION
    accumulate_positions: bool = False  # Allow adding to existing positions

    # === Commission ===
    commission_model: CommissionModel = CommissionModel.PERCENTAGE
    commission_rate: float = 0.001  # 0.1% per trade
    commission_per_share: float = 0.0  # $ per share (if per_share model)
    commission_per_trade: float = 0.0  # $ per trade (if per_trade model)
    commission_minimum: float = 0.0  # Minimum commission per trade

    # === Slippage ===
    slippage_model: SlippageModel = SlippageModel.PERCENTAGE
    slippage_rate: float = 0.001  # 0.1%
    slippage_fixed: float = 0.0  # $ per share (if fixed model)
    stop_slippage_rate: float = 0.0  # Additional slippage for stop/risk exits (on top of normal)

    # === Cash Management ===
    initial_cash: float = 100000.0
    allow_negative_cash: bool = False
    cash_buffer_pct: float = 0.0  # Reserve this % of cash (0 = use all)

    # === Order Handling ===
    reject_on_insufficient_cash: bool = True
    partial_fills_allowed: bool = False

    # === Account Type ===
    account_type: str = "cash"  # "cash" or "margin"
    margin_requirement: float = 0.5  # 50% margin requirement

    # === Calendar & Timezone ===
    calendar: str | None = None  # Exchange calendar (e.g., "NYSE", "CME_Equity", "LSE")
    timezone: str = "UTC"  # Default timezone for naive datetimes
    data_frequency: DataFrequency = DataFrequency.DAILY  # Data frequency
    enforce_sessions: bool = False  # Skip bars outside trading sessions (requires calendar)

    # === Trailing Stop Configuration ===
    trail_hwm_source: TrailHwmSource = TrailHwmSource.CLOSE  # HWM source for trailing stop

    # === Metadata ===
    preset_name: str | None = None  # Name of preset this was loaded from

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "execution": {
                "fill_timing": self.fill_timing.value,
                "execution_price": self.execution_price.value,
            },
            "position_sizing": {
                "share_type": self.share_type.value,
                "sizing_method": self.sizing_method.value,
                "default_position_pct": self.default_position_pct,
            },
            "signals": {
                "signal_processing": self.signal_processing.value,
                "accumulate_positions": self.accumulate_positions,
            },
            "commission": {
                "model": self.commission_model.value,
                "rate": self.commission_rate,
                "per_share": self.commission_per_share,
                "per_trade": self.commission_per_trade,
                "minimum": self.commission_minimum,
            },
            "slippage": {
                "model": self.slippage_model.value,
                "rate": self.slippage_rate,
                "fixed": self.slippage_fixed,
                "stop_rate": self.stop_slippage_rate,
            },
            "cash": {
                "initial": self.initial_cash,
                "allow_negative": self.allow_negative_cash,
                "buffer_pct": self.cash_buffer_pct,
            },
            "orders": {
                "reject_on_insufficient_cash": self.reject_on_insufficient_cash,
                "partial_fills_allowed": self.partial_fills_allowed,
            },
            "account": {
                "type": self.account_type,
                "margin_requirement": self.margin_requirement,
            },
        }

    @classmethod
    def from_dict(cls, data: dict, preset_name: str | None = None) -> BacktestConfig:
        """Create config from dictionary."""
        exec_cfg = data.get("execution", {})
        sizing_cfg = data.get("position_sizing", {})
        signal_cfg = data.get("signals", {})
        comm_cfg = data.get("commission", {})
        slip_cfg = data.get("slippage", {})
        cash_cfg = data.get("cash", {})
        order_cfg = data.get("orders", {})
        acct_cfg = data.get("account", {})

        return cls(
            # Execution
            fill_timing=FillTiming(exec_cfg.get("fill_timing", "next_bar_open")),
            execution_price=ExecutionPrice(exec_cfg.get("execution_price", "close")),
            # Sizing
            share_type=ShareType(sizing_cfg.get("share_type", "fractional")),
            sizing_method=SizingMethod(sizing_cfg.get("sizing_method", "percent_of_portfolio")),
            default_position_pct=sizing_cfg.get("default_position_pct", 0.10),
            # Signals
            signal_processing=SignalProcessing(
                signal_cfg.get("signal_processing", "check_position")
            ),
            accumulate_positions=signal_cfg.get("accumulate_positions", False),
            # Commission
            commission_model=CommissionModel(comm_cfg.get("model", "percentage")),
            commission_rate=comm_cfg.get("rate", 0.001),
            commission_per_share=comm_cfg.get("per_share", 0.0),
            commission_per_trade=comm_cfg.get("per_trade", 0.0),
            commission_minimum=comm_cfg.get("minimum", 0.0),
            # Slippage
            slippage_model=SlippageModel(slip_cfg.get("model", "percentage")),
            slippage_rate=slip_cfg.get("rate", 0.001),
            slippage_fixed=slip_cfg.get("fixed", 0.0),
            stop_slippage_rate=slip_cfg.get("stop_rate", 0.0),
            # Cash
            initial_cash=cash_cfg.get("initial", 100000.0),
            allow_negative_cash=cash_cfg.get("allow_negative", False),
            cash_buffer_pct=cash_cfg.get("buffer_pct", 0.0),
            # Orders
            reject_on_insufficient_cash=order_cfg.get("reject_on_insufficient_cash", True),
            partial_fills_allowed=order_cfg.get("partial_fills_allowed", False),
            # Account
            account_type=acct_cfg.get("type", "cash"),
            margin_requirement=acct_cfg.get("margin_requirement", 0.5),
            # Metadata
            preset_name=preset_name,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BacktestConfig:
        """Load config from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data, preset_name=path.stem)

    @classmethod
    def from_preset(cls, preset: str) -> BacktestConfig:
        """
        Load a predefined configuration preset.

        Available presets:
        - "default": Sensible defaults for general use
        - "backtrader": Match Backtrader's default behavior
        - "vectorbt": Match VectorBT's default behavior
        - "zipline": Match Zipline's default behavior
        - "realistic": Conservative settings for realistic simulation
        """
        presets = {
            "default": cls._default_preset(),
            "backtrader": cls._backtrader_preset(),
            "vectorbt": cls._vectorbt_preset(),
            "zipline": cls._zipline_preset(),
            "realistic": cls._realistic_preset(),
        }

        if preset not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

        config = presets[preset]
        config.preset_name = preset
        return config

    @classmethod
    def _default_preset(cls) -> BacktestConfig:
        """Default configuration - balanced between realism and ease of use."""
        return cls(
            fill_timing=FillTiming.NEXT_BAR_OPEN,
            execution_price=ExecutionPrice.OPEN,
            share_type=ShareType.FRACTIONAL,
            sizing_method=SizingMethod.PERCENT_OF_PORTFOLIO,
            default_position_pct=0.10,
            signal_processing=SignalProcessing.CHECK_POSITION,
            accumulate_positions=False,
            commission_model=CommissionModel.PERCENTAGE,
            commission_rate=0.001,
            slippage_model=SlippageModel.PERCENTAGE,
            slippage_rate=0.001,
            initial_cash=100000.0,
            allow_negative_cash=False,
            cash_buffer_pct=0.0,
            reject_on_insufficient_cash=True,
            partial_fills_allowed=False,
        )

    @classmethod
    def _backtrader_preset(cls) -> BacktestConfig:
        """
        Match Backtrader's default behavior.

        Key characteristics:
        - INTEGER shares (rounds down to whole shares)
        - Next-bar execution (COO disabled by default)
        - Check position state before acting
        - Percentage commission
        """
        return cls(
            fill_timing=FillTiming.NEXT_BAR_OPEN,
            execution_price=ExecutionPrice.OPEN,
            share_type=ShareType.INTEGER,  # Key difference!
            sizing_method=SizingMethod.PERCENT_OF_PORTFOLIO,
            default_position_pct=0.10,
            signal_processing=SignalProcessing.CHECK_POSITION,
            accumulate_positions=False,
            commission_model=CommissionModel.PERCENTAGE,
            commission_rate=0.001,
            slippage_model=SlippageModel.PERCENTAGE,
            slippage_rate=0.001,
            initial_cash=100000.0,
            allow_negative_cash=False,
            cash_buffer_pct=0.0,
            reject_on_insufficient_cash=True,
            partial_fills_allowed=False,
        )

    @classmethod
    def _vectorbt_preset(cls) -> BacktestConfig:
        """
        Match VectorBT's default behavior.

        Key characteristics:
        - FRACTIONAL shares
        - Same-bar execution (vectorized)
        - Process ALL signals (no position state check)
        - Percentage fees
        """
        return cls(
            fill_timing=FillTiming.SAME_BAR,  # Vectorized = same bar
            execution_price=ExecutionPrice.CLOSE,
            share_type=ShareType.FRACTIONAL,
            sizing_method=SizingMethod.PERCENT_OF_PORTFOLIO,
            default_position_pct=0.10,
            signal_processing=SignalProcessing.PROCESS_ALL,  # Key difference!
            accumulate_positions=False,  # Use accumulate=False
            commission_model=CommissionModel.PERCENTAGE,
            commission_rate=0.001,
            slippage_model=SlippageModel.PERCENTAGE,
            slippage_rate=0.001,
            initial_cash=100000.0,
            allow_negative_cash=False,
            cash_buffer_pct=0.0,
            reject_on_insufficient_cash=False,  # VectorBT is more permissive
            partial_fills_allowed=True,
        )

    @classmethod
    def _zipline_preset(cls) -> BacktestConfig:
        """
        Match Zipline's default behavior.

        Key characteristics:
        - Next-bar execution (order on bar N, fill on bar N+1)
        - Integer shares
        - Per-share commission (IB-style)
        - Volume-based slippage
        """
        return cls(
            fill_timing=FillTiming.NEXT_BAR_OPEN,
            execution_price=ExecutionPrice.OPEN,
            share_type=ShareType.INTEGER,
            sizing_method=SizingMethod.PERCENT_OF_PORTFOLIO,
            default_position_pct=0.10,
            signal_processing=SignalProcessing.CHECK_POSITION,
            accumulate_positions=False,
            commission_model=CommissionModel.PER_SHARE,  # Zipline uses per-share
            commission_rate=0.0,
            commission_per_share=0.005,  # $0.005 per share (IB-style)
            commission_minimum=1.0,  # $1 minimum
            slippage_model=SlippageModel.VOLUME_BASED,  # Key difference!
            slippage_rate=0.1,  # 10% of bar volume
            initial_cash=100000.0,
            allow_negative_cash=False,
            cash_buffer_pct=0.0,
            reject_on_insufficient_cash=True,
            partial_fills_allowed=True,  # Volume-based = partial fills
        )

    @classmethod
    def _realistic_preset(cls) -> BacktestConfig:
        """
        Conservative settings for realistic simulation.

        Key characteristics:
        - Integer shares (like real brokers)
        - Next-bar execution (no look-ahead)
        - Higher costs (more conservative)
        - Additional stop slippage (gaps hurt in fast markets)
        - Cash buffer (margin of safety)
        """
        return cls(
            fill_timing=FillTiming.NEXT_BAR_OPEN,
            execution_price=ExecutionPrice.OPEN,
            share_type=ShareType.INTEGER,
            sizing_method=SizingMethod.PERCENT_OF_PORTFOLIO,
            default_position_pct=0.05,  # Smaller positions
            signal_processing=SignalProcessing.CHECK_POSITION,
            accumulate_positions=False,
            commission_model=CommissionModel.PERCENTAGE,
            commission_rate=0.002,  # Higher commission
            slippage_model=SlippageModel.PERCENTAGE,
            slippage_rate=0.002,  # Higher slippage
            stop_slippage_rate=0.001,  # Extra 0.1% slippage for stop fills
            initial_cash=100000.0,
            allow_negative_cash=False,
            cash_buffer_pct=0.02,  # 2% cash buffer
            reject_on_insufficient_cash=True,
            partial_fills_allowed=False,
        )

    def describe(self) -> str:
        """Return human-readable description of configuration."""
        lines = [
            f"BacktestConfig (preset: {self.preset_name or 'custom'})",
            "=" * 50,
            "",
            "Execution:",
            f"  Fill timing: {self.fill_timing.value}",
            f"  Execution price: {self.execution_price.value}",
            "",
            "Position Sizing:",
            f"  Share type: {self.share_type.value}",
            f"  Sizing method: {self.sizing_method.value}",
            f"  Default position: {self.default_position_pct:.1%}",
            "",
            "Signal Processing:",
            f"  Processing: {self.signal_processing.value}",
            f"  Accumulate: {self.accumulate_positions}",
            "",
            "Costs:",
            f"  Commission: {self.commission_model.value} @ {self.commission_rate:.2%}",
            f"  Slippage: {self.slippage_model.value} @ {self.slippage_rate:.2%}",
            f"  Stop slippage: +{self.stop_slippage_rate:.2%}"
            if self.stop_slippage_rate > 0
            else "",
            "",
            "Cash:",
            f"  Initial: ${self.initial_cash:,.0f}",
            f"  Buffer: {self.cash_buffer_pct:.1%}",
            f"  Reject insufficient: {self.reject_on_insufficient_cash}",
        ]
        return "\n".join(lines)


# Export presets directory path for users who want to load custom YAML files
PRESETS_DIR = Path(__file__).parent / "presets"


class Mode(str, Enum):
    """Simplified mode selection for Engine initialization.

    A Mode is a convenient shorthand for BacktestConfig presets.
    Use this when you want sensible defaults without configuring every detail.

    Example:
        >>> from ml4t.backtest import Engine, Mode
        >>> engine = Engine.from_mode(feed, strategy, mode=Mode.REALISTIC)

    Available modes:
        DEFAULT: Balanced defaults (fractional shares, 0.1% costs)
        REALISTIC: Conservative for production use (integer shares, 0.2% costs)
        FAST: Minimal friction for quick prototyping (no costs)
        BACKTRADER: Match Backtrader behavior exactly
        VECTORBT: Match VectorBT behavior exactly
        ZIPLINE: Match Zipline behavior exactly
    """

    DEFAULT = "default"
    REALISTIC = "realistic"
    FAST = "fast"
    BACKTRADER = "backtrader"
    VECTORBT = "vectorbt"
    ZIPLINE = "zipline"

    def to_config(self) -> BacktestConfig:
        """Convert mode to a BacktestConfig instance."""
        if self == Mode.FAST:
            # Special case: fast mode minimizes friction
            return BacktestConfig(
                fill_timing=FillTiming.SAME_BAR,
                execution_price=ExecutionPrice.CLOSE,
                share_type=ShareType.FRACTIONAL,
                commission_model=CommissionModel.NONE,
                slippage_model=SlippageModel.NONE,
                preset_name="fast",
            )
        # All other modes map directly to presets
        return BacktestConfig.from_preset(self.value)
