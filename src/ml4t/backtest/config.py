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
    """How position size is calculated.

    .. deprecated::
        This enum is not consumed by any runtime code. Position sizing is
        always determined by strategy code. Retained for serialization
        backward compatibility only.
    """

    PERCENT_OF_PORTFOLIO = "percent_of_portfolio"  # % of total portfolio value
    PERCENT_OF_CASH = "percent_of_cash"  # % of available cash only
    FIXED_VALUE = "fixed_value"  # Fixed dollar amount per position
    FIXED_SHARES = "fixed_shares"  # Fixed number of shares


class FillOrdering(str, Enum):
    """Order processing sequence within a single bar.

    Controls how pending orders are sequenced during fill processing:

    EXIT_FIRST (default):
        All exits → mark-to-market → all entries (with gatekeeper validation).
        Capital-efficient: exits free cash before entries need it.
        Matches VectorBT ``call_seq='auto'`` behavior.

    FIFO:
        Orders process in submission order with sequential cash updates.
        Each order's gatekeeper check sees cash from all prior fills.
        Matches Backtrader's submission-order processing.
    """

    EXIT_FIRST = "exit_first"
    FIFO = "fifo"


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


class WaterMarkSource(str, Enum):
    """Source for water mark updates in trailing stops.

    Controls which price is used to update water marks on each bar AFTER entry.
    Works for both LONG (High Water Mark) and SHORT (Low Water Mark) positions:

    - CLOSE: Use close prices for water mark updates (default, simpler behavior)
    - BAR_EXTREME: Use HIGH for HWM (longs), LOW for LWM (shorts) - VBT Pro OHLC mode

    This is direction-agnostic and works identically for long-only, short-only,
    or combined long-short strategies.

    Note: Initial water mark on entry bar is controlled by InitialHwmSource.
    """

    CLOSE = "close"  # Use close prices for water mark updates (default)
    BAR_EXTREME = "bar_extreme"  # Use HIGH for HWM, LOW for LWM (VBT Pro with OHLC)

    # Deprecated alias for backward compatibility
    HIGH = "bar_extreme"  # Deprecated: use BAR_EXTREME instead


# Backward compatibility alias
TrailHwmSource = WaterMarkSource


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


class TrailStopTiming(str, Enum):
    """Timing of water mark update relative to trailing stop check.

    Controls when water marks are updated for trailing stop calculation:

    LAGGED mode (formerly END_OF_BAR):
        1. Check stop using HWM/LWM from PREVIOUS bar
        2. Update HWM/LWM at end of current bar
        ⚠️ This causes 1-bar delay in stop triggers vs VBT Pro

    INTRABAR mode:
        1. Compute live water mark: max/min(previous, current_bar_extreme)
        2. Check stop using live water mark against HIGH/LOW
        3. If triggered, fill per StopFillMode configuration
        4. Update water mark at end of bar
        ⚠️ Too aggressive - triggers on HIGH when VBT Pro only checks CLOSE

    VBT_PRO mode (true VBT Pro compatible, two-pass algorithm):
        1. First pass: Check stop using LAGGED water mark against HIGH/LOW
        2. If first pass doesn't trigger, update water mark from current bar extreme
        3. Second pass: Check stop using UPDATED water mark against CLOSE only
        4. This matches VBT Pro's exact algorithm where the second pass can only
           use CLOSE (can_use_ohlc=False in VBT Pro source code)

    VBT_PRO mode for SHORT positions:
        Pass 1: LWM from previous bar, check if HIGH >= stop
        Pass 2: Update LWM from bar_low, check if CLOSE >= stop (not HIGH!)

    VBT_PRO mode for LONG positions:
        Pass 1: HWM from previous bar, check if LOW <= stop
        Pass 2: Update HWM from bar_high, check if CLOSE <= stop (not LOW!)
    """

    LAGGED = "lagged"  # Use previous bar's water mark (1-bar lag)
    INTRABAR = "intrabar"  # Update water mark before check, triggers on HIGH/LOW
    VBT_PRO = "vbt_pro"  # Two-pass: LAGGED check, then INTRABAR check using CLOSE only


class ExecutionMode(str, Enum):
    """Order execution timing mode.

    Controls when orders are eligible for execution relative to signal generation.
    """

    SAME_BAR = "same_bar"  # Fill on same bar as order submission
    NEXT_BAR = "next_bar"  # Fill on next bar after order submission


@dataclass
class StatsConfig:
    """Configuration for per-asset trading statistics tracking.

    Controls how AssetTradingStats are computed and managed during backtesting.

    Attributes:
        recent_window_size: Number of recent trades to track (default 50).
            Larger windows provide more stable statistics but slower response
            to regime changes. Recommended: 3x average holding period in bars.
        track_session_stats: Whether to track per-session statistics.
            Requires session configuration to detect session boundaries.
        enabled: Whether stats tracking is enabled. Disable for maximum
            performance when stats are not needed.

    Example:
        # Configure stats for a strategy with ~3 day average holding period
        config = StatsConfig(
            recent_window_size=100,  # ~1 month of trades
            track_session_stats=True,
        )
        broker.configure_stats(config)
    """

    recent_window_size: int = 50
    track_session_stats: bool = True
    enabled: bool = True


class StopFillMode(str, Enum):
    """Price used for stop order fills.

    Controls what price is used when a stop order triggers.
    """

    STOP_PRICE = "stop_price"  # Fill at stop price (if not gapped)
    CLOSE_PRICE = "close_price"  # Fill at bar close (conservative)
    NEXT_BAR_OPEN = "next_bar_open"  # Fill at next bar's open


class StopLevelBasis(str, Enum):
    """Reference price for calculating stop levels.

    Controls what price the stop percentage/amount is applied to.
    """

    FILL_PRICE = "fill_price"  # Use actual fill price (most accurate)
    SIGNAL_PRICE = "signal_price"  # Use price at signal time (Backtrader style)


@dataclass
class BacktestConfig:
    """
    Complete configuration for backtesting behavior.

    All behavioral differences between frameworks are captured here.
    Load presets to match specific frameworks exactly.

    This is the single source of truth for all backtest settings.
    Broker and Engine are configured entirely from this dataclass.
    """

    # === Account Type (replaces class hierarchy) ===
    allow_short_selling: bool = False  # True for margin/crypto
    allow_leverage: bool = False  # True for margin only
    initial_margin: float = 0.5  # Only used if allow_leverage=True (Reg T = 0.5)
    long_maintenance_margin: float = 0.25  # Reg T standard for longs
    short_maintenance_margin: float = 0.30  # Reg T standard for shorts (higher!)
    fixed_margin_schedule: dict[str, tuple[float, float]] | None = None  # For futures

    # === Execution Timing ===
    fill_timing: FillTiming = FillTiming.NEXT_BAR_OPEN
    execution_price: ExecutionPrice = ExecutionPrice.CLOSE
    execution_mode: ExecutionMode = ExecutionMode.SAME_BAR  # Order execution timing

    # === Stop Configuration ===
    stop_fill_mode: StopFillMode = StopFillMode.STOP_PRICE
    stop_level_basis: StopLevelBasis = StopLevelBasis.FILL_PRICE
    trail_hwm_source: WaterMarkSource = WaterMarkSource.CLOSE
    initial_hwm_source: InitialHwmSource = InitialHwmSource.FILL_PRICE
    trail_stop_timing: TrailStopTiming = TrailStopTiming.LAGGED

    def validate(self, warn: bool = True) -> list[str]:
        """Validate configuration and return warnings for edge cases.

        Checks for configurations that may produce unexpected results or
        indicate potential issues. Returns a list of warning messages.

        Args:
            warn: If True, emit warnings via warnings.warn(). Default True.

        Returns:
            List of warning message strings (empty if no issues found).

        Example:
            config = BacktestConfig(fill_timing=FillTiming.SAME_BAR)
            warnings = config.validate()
            # ["SAME_BAR execution has look-ahead bias risk..."]
        """
        import warnings as _warnings

        issues: list[str] = []

        # Look-ahead bias warning
        if self.fill_timing == FillTiming.SAME_BAR:
            issues.append(
                "SAME_BAR execution has look-ahead bias risk. "
                "Use NEXT_BAR_OPEN for realistic backtesting."
            )

        # Zero cost warning
        if (
            self.commission_model == CommissionModel.NONE
            and self.slippage_model == SlippageModel.NONE
        ):
            issues.append(
                "Both commission and slippage are disabled. "
                "Results may be overly optimistic. Consider using Mode.REALISTIC."
            )

        # High position concentration
        if self.default_position_pct > 0.25:
            issues.append(
                f"Default position size ({self.default_position_pct:.0%}) exceeds 25%. "
                "High concentration increases single-stock risk."
            )

        # Volume-based slippage without partial fills
        if self.slippage_model == SlippageModel.VOLUME_BASED and not self.partial_fills_allowed:
            issues.append(
                "Volume-based slippage without partial_fills_allowed may cause "
                "orders to be rejected in low-volume conditions."
            )

        # High slippage + high commission
        total_cost = self.slippage_rate + self.commission_rate
        if total_cost > 0.01:  # > 1% round-trip
            issues.append(
                f"Total transaction cost ({total_cost:.2%}) is high. "
                "Verify this matches your broker's actual costs."
            )

        # Fractional shares warning for production
        if self.share_type == ShareType.FRACTIONAL and self.preset_name == "realistic":
            issues.append(
                "REALISTIC preset with fractional shares may not match all brokers. "
                "Set share_type=INTEGER for most accurate simulation."
            )

        # Margin parameter validation
        if self.allow_leverage:
            if not 0.0 < self.initial_margin <= 1.0:
                issues.append(f"initial_margin ({self.initial_margin}) must be in (0.0, 1.0]")
            if not 0.0 < self.long_maintenance_margin <= 1.0:
                issues.append(
                    f"long_maintenance_margin ({self.long_maintenance_margin}) must be in (0.0, 1.0]"
                )
            if not 0.0 < self.short_maintenance_margin <= 1.0:
                issues.append(
                    f"short_maintenance_margin ({self.short_maintenance_margin}) must be in (0.0, 1.0]"
                )
            if self.long_maintenance_margin >= self.initial_margin:
                issues.append(
                    f"long_maintenance_margin ({self.long_maintenance_margin}) must be < "
                    f"initial_margin ({self.initial_margin})"
                )
            if self.short_maintenance_margin >= self.initial_margin:
                issues.append(
                    f"short_maintenance_margin ({self.short_maintenance_margin}) must be < "
                    f"initial_margin ({self.initial_margin})"
                )

        # Emit warnings if requested
        if warn and issues:
            for msg in issues:
                _warnings.warn(msg, UserWarning, stacklevel=2)

        return issues

    def get_effective_account_settings(self) -> tuple[bool, bool]:
        """Get account settings as a tuple.

        Returns:
            Tuple of (allow_short_selling, allow_leverage)
        """
        return self.allow_short_selling, self.allow_leverage

    def get_effective_account_type(self) -> str:
        """Get account type string based on current settings.

        Returns:
            "cash", "crypto", or "margin" based on flags.
        """
        if self.allow_leverage:
            return "margin"
        elif self.allow_short_selling:
            return "crypto"
        else:
            return "cash"

    # === Position Sizing ===
    share_type: ShareType = ShareType.FRACTIONAL
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
    cash_buffer_pct: float = 0.0  # Reserve this % of cash (0 = use all)

    # === Order Handling ===
    reject_on_insufficient_cash: bool = True
    partial_fills_allowed: bool = False
    fill_ordering: FillOrdering = FillOrdering.EXIT_FIRST

    # === Calendar & Timezone ===
    calendar: str | None = None  # Exchange calendar (e.g., "NYSE", "CME_Equity", "LSE")
    timezone: str = "UTC"  # Default timezone for naive datetimes
    data_frequency: DataFrequency = DataFrequency.DAILY  # Data frequency
    enforce_sessions: bool = False  # Skip bars outside trading sessions (requires calendar)

    # === Metadata ===
    preset_name: str | None = None  # Name of preset this was loaded from

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "account": {
                "allow_short_selling": self.allow_short_selling,
                "allow_leverage": self.allow_leverage,
                "initial_margin": self.initial_margin,
                "long_maintenance_margin": self.long_maintenance_margin,
                "short_maintenance_margin": self.short_maintenance_margin,
                "fixed_margin_schedule": self.fixed_margin_schedule,
            },
            "execution": {
                "fill_timing": self.fill_timing.value,
                "execution_price": self.execution_price.value,
                "execution_mode": self.execution_mode.value,
            },
            "stops": {
                "stop_fill_mode": self.stop_fill_mode.value,
                "stop_level_basis": self.stop_level_basis.value,
                "trail_hwm_source": self.trail_hwm_source.value,
                "initial_hwm_source": self.initial_hwm_source.value,
                "trail_stop_timing": self.trail_stop_timing.value,
            },
            "position_sizing": {
                "share_type": self.share_type.value,
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
                "buffer_pct": self.cash_buffer_pct,
            },
            "orders": {
                "reject_on_insufficient_cash": self.reject_on_insufficient_cash,
                "partial_fills_allowed": self.partial_fills_allowed,
                "fill_ordering": self.fill_ordering.value,
            },
        }

    @classmethod
    def from_dict(cls, data: dict, preset_name: str | None = None) -> BacktestConfig:
        """Create config from dictionary."""
        acct_cfg = data.get("account", {})
        exec_cfg = data.get("execution", {})
        stops_cfg = data.get("stops", {})
        sizing_cfg = data.get("position_sizing", {})
        signal_cfg = data.get("signals", {})
        comm_cfg = data.get("commission", {})
        slip_cfg = data.get("slippage", {})
        cash_cfg = data.get("cash", {})
        order_cfg = data.get("orders", {})

        # Handle legacy account type for migration
        legacy_type = acct_cfg.get("type")
        legacy_margin_req = acct_cfg.get("margin_requirement")

        # Determine account settings from new or legacy fields
        if "allow_short_selling" in acct_cfg:
            # New format
            allow_short_selling = acct_cfg.get("allow_short_selling", False)
            allow_leverage = acct_cfg.get("allow_leverage", False)
        elif legacy_type is not None:
            # Convert legacy format to new flags
            if legacy_type == "cash":
                allow_short_selling, allow_leverage = False, False
            elif legacy_type == "crypto":
                allow_short_selling, allow_leverage = True, False
            elif legacy_type == "margin":
                allow_short_selling, allow_leverage = True, True
            else:
                raise ValueError(f"Unknown account type: '{legacy_type}'")
        else:
            # Default
            allow_short_selling = False
            allow_leverage = False

        return cls(
            # Account
            allow_short_selling=allow_short_selling,
            allow_leverage=allow_leverage,
            initial_margin=acct_cfg.get("initial_margin", legacy_margin_req or 0.5),
            long_maintenance_margin=acct_cfg.get("long_maintenance_margin", 0.25),
            short_maintenance_margin=acct_cfg.get("short_maintenance_margin", 0.30),
            fixed_margin_schedule=acct_cfg.get("fixed_margin_schedule"),
            # Execution
            fill_timing=FillTiming(exec_cfg.get("fill_timing", "next_bar_open")),
            execution_price=ExecutionPrice(exec_cfg.get("execution_price", "close")),
            execution_mode=ExecutionMode(exec_cfg.get("execution_mode", "same_bar")),
            # Stops
            stop_fill_mode=StopFillMode(stops_cfg.get("stop_fill_mode", "stop_price")),
            stop_level_basis=StopLevelBasis(stops_cfg.get("stop_level_basis", "fill_price")),
            trail_hwm_source=WaterMarkSource(stops_cfg.get("trail_hwm_source", "close")),
            initial_hwm_source=InitialHwmSource(stops_cfg.get("initial_hwm_source", "fill_price")),
            trail_stop_timing=TrailStopTiming(stops_cfg.get("trail_stop_timing", "lagged")),
            # Sizing
            share_type=ShareType(sizing_cfg.get("share_type", "fractional")),
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
            cash_buffer_pct=cash_cfg.get("buffer_pct", 0.0),
            # Orders
            reject_on_insufficient_cash=order_cfg.get("reject_on_insufficient_cash", True),
            partial_fills_allowed=order_cfg.get("partial_fills_allowed", False),
            fill_ordering=FillOrdering(order_cfg.get("fill_ordering", "exit_first")),
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
            # Account: cash-like (no shorts, no leverage)
            allow_short_selling=False,
            allow_leverage=False,
            # Execution
            fill_timing=FillTiming.NEXT_BAR_OPEN,
            execution_price=ExecutionPrice.OPEN,
            execution_mode=ExecutionMode.NEXT_BAR,
            # Stops
            stop_fill_mode=StopFillMode.STOP_PRICE,
            stop_level_basis=StopLevelBasis.FILL_PRICE,
            trail_hwm_source=WaterMarkSource.CLOSE,
            trail_stop_timing=TrailStopTiming.LAGGED,
            # Sizing
            share_type=ShareType.FRACTIONAL,
            default_position_pct=0.10,
            signal_processing=SignalProcessing.CHECK_POSITION,
            accumulate_positions=False,
            # Costs
            commission_model=CommissionModel.PERCENTAGE,
            commission_rate=0.001,
            slippage_model=SlippageModel.PERCENTAGE,
            slippage_rate=0.001,
            # Cash
            initial_cash=100000.0,
            cash_buffer_pct=0.0,
            reject_on_insufficient_cash=True,
            partial_fills_allowed=False,
            fill_ordering=FillOrdering.EXIT_FIRST,
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
        - Margin account (shorts and leverage allowed)
        - Stop level from signal price (not fill price)
        """
        return cls(
            # Account: margin (backtrader allows shorts and leverage)
            allow_short_selling=True,
            allow_leverage=True,
            initial_margin=0.5,
            long_maintenance_margin=0.25,
            short_maintenance_margin=0.30,
            # Execution
            fill_timing=FillTiming.NEXT_BAR_OPEN,
            execution_price=ExecutionPrice.OPEN,
            execution_mode=ExecutionMode.NEXT_BAR,
            # Stops: Backtrader calculates stops from signal price
            stop_fill_mode=StopFillMode.STOP_PRICE,
            stop_level_basis=StopLevelBasis.SIGNAL_PRICE,  # Key Backtrader behavior!
            trail_hwm_source=WaterMarkSource.CLOSE,
            trail_stop_timing=TrailStopTiming.LAGGED,
            # Sizing
            share_type=ShareType.INTEGER,  # Key difference!
            default_position_pct=0.10,
            signal_processing=SignalProcessing.CHECK_POSITION,
            accumulate_positions=False,
            # Costs
            commission_model=CommissionModel.PERCENTAGE,
            commission_rate=0.001,
            slippage_model=SlippageModel.PERCENTAGE,
            slippage_rate=0.001,
            # Cash
            initial_cash=100000.0,
            cash_buffer_pct=0.0,
            reject_on_insufficient_cash=True,
            partial_fills_allowed=False,
            fill_ordering=FillOrdering.FIFO,  # Backtrader processes in submission order
        )

    @classmethod
    def _vectorbt_preset(cls) -> BacktestConfig:
        """
        Match VectorBT Pro's default behavior.

        Key characteristics:
        - FRACTIONAL shares
        - Same-bar execution (vectorized)
        - Process ALL signals (no position state check)
        - Percentage fees
        - Shorts allowed (crypto-like), no leverage
        - Intrabar trailing stop timing (live HWM updates)
        - HWM from bar high (not close)
        """
        return cls(
            # Account: crypto-like (shorts OK, no leverage)
            allow_short_selling=True,
            allow_leverage=False,
            # Execution
            fill_timing=FillTiming.SAME_BAR,  # Vectorized = same bar
            execution_price=ExecutionPrice.CLOSE,
            execution_mode=ExecutionMode.SAME_BAR,
            # Stops: VBT Pro uses INTRABAR timing with HIGH for HWM
            stop_fill_mode=StopFillMode.STOP_PRICE,
            stop_level_basis=StopLevelBasis.FILL_PRICE,
            trail_hwm_source=WaterMarkSource.BAR_EXTREME,  # VBT Pro with OHLC!
            initial_hwm_source=InitialHwmSource.BAR_HIGH,  # VBT Pro uses bar high
            trail_stop_timing=TrailStopTiming.INTRABAR,  # Live HWM updates!
            # Sizing
            share_type=ShareType.FRACTIONAL,
            default_position_pct=0.10,
            signal_processing=SignalProcessing.PROCESS_ALL,  # Key difference!
            accumulate_positions=False,
            # Costs: often zero for quick prototyping
            commission_model=CommissionModel.NONE,
            commission_rate=0.0,
            slippage_model=SlippageModel.NONE,
            slippage_rate=0.0,
            # Cash
            initial_cash=100000.0,
            cash_buffer_pct=0.0,
            reject_on_insufficient_cash=False,  # VectorBT is more permissive
            partial_fills_allowed=True,
            fill_ordering=FillOrdering.EXIT_FIRST,  # VBT call_seq='auto'
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
        - Cash account (no shorts by default)
        """
        return cls(
            # Account: cash (Zipline is conservative by default)
            allow_short_selling=False,
            allow_leverage=False,
            # Execution
            fill_timing=FillTiming.NEXT_BAR_OPEN,
            execution_price=ExecutionPrice.OPEN,
            execution_mode=ExecutionMode.NEXT_BAR,
            # Stops
            stop_fill_mode=StopFillMode.STOP_PRICE,
            stop_level_basis=StopLevelBasis.FILL_PRICE,
            trail_hwm_source=WaterMarkSource.CLOSE,
            trail_stop_timing=TrailStopTiming.LAGGED,
            # Sizing
            share_type=ShareType.INTEGER,
            default_position_pct=0.10,
            signal_processing=SignalProcessing.CHECK_POSITION,
            accumulate_positions=False,
            # Costs: IB-style
            commission_model=CommissionModel.PER_SHARE,  # Zipline uses per-share
            commission_rate=0.0,
            commission_per_share=0.005,  # $0.005 per share (IB-style)
            commission_minimum=1.0,  # $1 minimum
            slippage_model=SlippageModel.VOLUME_BASED,  # Key difference!
            slippage_rate=0.1,  # 10% of bar volume
            # Cash
            initial_cash=100000.0,
            cash_buffer_pct=0.0,
            reject_on_insufficient_cash=True,
            partial_fills_allowed=True,  # Volume-based = partial fills
            fill_ordering=FillOrdering.EXIT_FIRST,
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
        - Cash account (most conservative)
        """
        return cls(
            # Account: cash (most conservative)
            allow_short_selling=False,
            allow_leverage=False,
            # Execution
            fill_timing=FillTiming.NEXT_BAR_OPEN,
            execution_price=ExecutionPrice.OPEN,
            execution_mode=ExecutionMode.NEXT_BAR,
            # Stops: realistic
            stop_fill_mode=StopFillMode.NEXT_BAR_OPEN,  # Conservative: fill at open
            stop_level_basis=StopLevelBasis.FILL_PRICE,
            trail_hwm_source=WaterMarkSource.CLOSE,
            trail_stop_timing=TrailStopTiming.LAGGED,
            # Sizing
            share_type=ShareType.INTEGER,
            default_position_pct=0.05,  # Smaller positions
            signal_processing=SignalProcessing.CHECK_POSITION,
            accumulate_positions=False,
            # Costs: higher for realism
            commission_model=CommissionModel.PERCENTAGE,
            commission_rate=0.002,  # Higher commission
            slippage_model=SlippageModel.PERCENTAGE,
            slippage_rate=0.002,  # Higher slippage
            stop_slippage_rate=0.001,  # Extra 0.1% slippage for stop fills
            # Cash
            initial_cash=100000.0,
            cash_buffer_pct=0.02,  # 2% cash buffer
            reject_on_insufficient_cash=True,
            partial_fills_allowed=False,
            fill_ordering=FillOrdering.EXIT_FIRST,
        )

    def describe(self) -> str:
        """Return human-readable description of configuration."""
        allow_shorts, allow_leverage = self.get_effective_account_settings()
        account_str = self.get_effective_account_type()

        lines = [
            f"BacktestConfig (preset: {self.preset_name or 'custom'})",
            "=" * 50,
            "",
            "Account:",
            f"  Type: {account_str}",
            f"  Short selling: {'allowed' if allow_shorts else 'disabled'}",
            f"  Leverage: {'enabled' if allow_leverage else 'disabled'}",
        ]

        if allow_leverage:
            lines.extend(
                [
                    f"  Initial margin: {self.initial_margin:.0%}",
                    f"  Long maintenance: {self.long_maintenance_margin:.0%}",
                    f"  Short maintenance: {self.short_maintenance_margin:.0%}",
                ]
            )

        lines.extend(
            [
                "",
                "Execution:",
                f"  Fill timing: {self.fill_timing.value}",
                f"  Execution mode: {self.execution_mode.value}",
                f"  Execution price: {self.execution_price.value}",
                "",
                "Stops:",
                f"  Fill mode: {self.stop_fill_mode.value}",
                f"  Level basis: {self.stop_level_basis.value}",
                f"  Trail HWM source: {self.trail_hwm_source.value}",
                f"  Trail timing: {self.trail_stop_timing.value}",
                "",
                "Position Sizing:",
                f"  Share type: {self.share_type.value}",
                f"  Default position: {self.default_position_pct:.1%}",
                "",
                "Signal Processing:",
                f"  Processing: {self.signal_processing.value}",
                f"  Accumulate: {self.accumulate_positions}",
                "",
                "Costs:",
                f"  Commission: {self.commission_model.value} @ {self.commission_rate:.2%}",
                f"  Slippage: {self.slippage_model.value} @ {self.slippage_rate:.2%}",
            ]
        )

        if self.stop_slippage_rate > 0:
            lines.append(f"  Stop slippage: +{self.stop_slippage_rate:.2%}")

        lines.extend(
            [
                "",
                "Orders:",
                f"  Fill ordering: {self.fill_ordering.value}",
                f"  Reject insufficient: {self.reject_on_insufficient_cash}",
                f"  Partial fills: {self.partial_fills_allowed}",
                "",
                "Cash:",
                f"  Initial: ${self.initial_cash:,.0f}",
                f"  Buffer: {self.cash_buffer_pct:.1%}",
            ]
        )

        return "\n".join(line for line in lines if line is not None)


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
            # Special case: fast mode minimizes friction for rapid prototyping
            return BacktestConfig(
                # Account: permissive (shorts allowed for flexibility)
                allow_short_selling=True,
                allow_leverage=False,
                # Execution: same-bar for speed
                fill_timing=FillTiming.SAME_BAR,
                execution_price=ExecutionPrice.CLOSE,
                execution_mode=ExecutionMode.SAME_BAR,
                # Sizing
                share_type=ShareType.FRACTIONAL,
                # Costs: none for frictionless testing
                commission_model=CommissionModel.NONE,
                slippage_model=SlippageModel.NONE,
                preset_name="fast",
            )
        # All other modes map directly to presets
        return BacktestConfig.from_preset(self.value)
