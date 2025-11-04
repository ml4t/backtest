"""
Strategy Specifications for Cross-Framework Validation

These are GENERIC strategy specifications that define trading logic
in a framework-agnostic way. Each framework adapter translates these
specifications into its native implementation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SignalType(Enum):
    """Types of trading signals."""

    LONG_ENTRY = "long_entry"
    LONG_EXIT = "long_exit"
    SHORT_ENTRY = "short_entry"
    SHORT_EXIT = "short_exit"


@dataclass
class IndicatorSpec:
    """Specification for a technical indicator."""

    name: str
    params: dict[str, Any]


@dataclass
class RuleSpec:
    """Specification for a trading rule."""

    signal_type: SignalType
    condition: str  # e.g., "rsi < 30", "ma_fast > ma_slow"
    required_indicators: list[str]


@dataclass
class StrategySpec:
    """
    Generic strategy specification that can be translated to any framework.

    This defines WHAT the strategy does, not HOW it's implemented.
    """

    name: str
    description: str
    indicators: list[IndicatorSpec]
    rules: list[RuleSpec]
    position_sizing: str  # "all_in", "fixed", "kelly", etc.
    risk_management: dict[str, Any]  # stop loss, take profit, etc.


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

RSI_MEAN_REVERSION = StrategySpec(
    name="RSI Mean Reversion",
    description="Buy oversold (RSI < 30), sell overbought (RSI > 70)",
    indicators=[IndicatorSpec(name="rsi", params={"period": 14})],
    rules=[
        RuleSpec(
            signal_type=SignalType.LONG_ENTRY,
            condition="rsi < 30",
            required_indicators=["rsi"],
        ),
        RuleSpec(
            signal_type=SignalType.LONG_EXIT,
            condition="rsi > 70",
            required_indicators=["rsi"],
        ),
    ],
    position_sizing="all_in",
    risk_management={"max_holding_period": 20, "stop_loss": None, "take_profit": None},
)


DUAL_MA_CROSSOVER = StrategySpec(
    name="Dual Moving Average Crossover",
    description="Classic trend following with MA crossovers",
    indicators=[
        IndicatorSpec(name="sma", params={"period": 20, "label": "fast"}),
        IndicatorSpec(name="sma", params={"period": 50, "label": "slow"}),
    ],
    rules=[
        RuleSpec(
            signal_type=SignalType.LONG_ENTRY,
            condition="sma_fast crosses_above sma_slow",
            required_indicators=["sma_fast", "sma_slow"],
        ),
        RuleSpec(
            signal_type=SignalType.LONG_EXIT,
            condition="sma_fast crosses_below sma_slow",
            required_indicators=["sma_fast", "sma_slow"],
        ),
    ],
    position_sizing="all_in",
    risk_management={"max_holding_period": None, "stop_loss": None, "take_profit": None},
)


BOLLINGER_BREAKOUT = StrategySpec(
    name="Bollinger Band Breakout",
    description="Breakout above upper band with volume confirmation",
    indicators=[
        IndicatorSpec(name="bbands", params={"period": 20, "std": 2.0}),
        IndicatorSpec(name="volume_sma", params={"period": 20}),
    ],
    rules=[
        RuleSpec(
            signal_type=SignalType.LONG_ENTRY,
            condition="(close > bb_upper) AND (volume > volume_sma * 1.5)",
            required_indicators=["bb_upper", "volume_sma", "volume", "close"],
        ),
        RuleSpec(
            signal_type=SignalType.LONG_EXIT,
            condition="close < bb_middle",
            required_indicators=["bb_middle", "close"],
        ),
    ],
    position_sizing="all_in",
    risk_management={
        "max_holding_period": 10,
        "stop_loss": 0.02,  # 2% stop loss
        "take_profit": None,
    },
)


ML_MOMENTUM_STRATEGY = StrategySpec(
    name="ML Momentum Strategy",
    description="Use ML predictions combined with momentum indicators",
    indicators=[
        IndicatorSpec(name="rsi", params={"period": 14}),
        IndicatorSpec(name="macd", params={"fast": 12, "slow": 26, "signal": 9}),
        IndicatorSpec(name="atr", params={"period": 14}),
        IndicatorSpec(name="ml_signal", params={"model": "xgboost", "threshold": 0.6}),
    ],
    rules=[
        RuleSpec(
            signal_type=SignalType.LONG_ENTRY,
            condition="(ml_signal > 0.6) AND (rsi > 50) AND (macd > macd_signal)",
            required_indicators=["ml_signal", "rsi", "macd", "macd_signal"],
        ),
        RuleSpec(
            signal_type=SignalType.LONG_EXIT,
            condition="(ml_signal < 0.4) OR (rsi < 30)",
            required_indicators=["ml_signal", "rsi"],
        ),
    ],
    position_sizing="risk_based",  # Position size based on ATR
    risk_management={
        "max_holding_period": 5,
        "stop_loss": "2 * atr",  # Dynamic stop based on ATR
        "take_profit": "3 * atr",  # Dynamic target based on ATR
    },
)


HIGH_FREQUENCY_SCALPING = StrategySpec(
    name="High Frequency Scalping",
    description="Rapid trades on small price movements",
    indicators=[
        IndicatorSpec(name="ema", params={"period": 5, "label": "fast"}),
        IndicatorSpec(name="ema", params={"period": 15, "label": "slow"}),
        IndicatorSpec(name="adx", params={"period": 14}),
        IndicatorSpec(name="volume_ratio", params={"period": 10}),
    ],
    rules=[
        RuleSpec(
            signal_type=SignalType.LONG_ENTRY,
            condition="(ema_fast crosses_above ema_slow) AND (adx > 25) AND (volume_ratio > 1.2)",
            required_indicators=["ema_fast", "ema_slow", "adx", "volume_ratio"],
        ),
        RuleSpec(
            signal_type=SignalType.LONG_EXIT,
            condition="(ema_fast crosses_below ema_slow) OR (profit_pct > 0.01) OR (loss_pct > 0.005)",
            required_indicators=["ema_fast", "ema_slow", "profit_pct", "loss_pct"],
        ),
    ],
    position_sizing="all_in",
    risk_management={
        "max_holding_period": 3,  # Exit after 3 bars max
        "stop_loss": 0.005,  # 0.5% stop
        "take_profit": 0.01,  # 1% target
    },
)


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

STRATEGY_REGISTRY = {
    "rsi_mean_reversion": RSI_MEAN_REVERSION,
    "dual_ma_crossover": DUAL_MA_CROSSOVER,
    "bollinger_breakout": BOLLINGER_BREAKOUT,
    "ml_momentum": ML_MOMENTUM_STRATEGY,
    "high_freq_scalping": HIGH_FREQUENCY_SCALPING,
}


def get_strategy_spec(name: str) -> StrategySpec:
    """Get strategy specification by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """List all available strategies."""
    return list(STRATEGY_REGISTRY.keys())
