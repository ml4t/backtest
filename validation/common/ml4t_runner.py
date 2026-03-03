"""ml4t.backtest runner for validation scenarios.

Converts scenario config into ml4t Engine execution.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path for ml4t.backtest imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.types import FrameworkResult, ScenarioConfig  # noqa: E402


def run_ml4t(
    scenario: ScenarioConfig,
    prices_df: pd.DataFrame,
    entries: np.ndarray,
    exits: np.ndarray | None = None,
    framework: str = "default",
) -> FrameworkResult:
    """Run ml4t.backtest for a scenario, configured to match a target framework.

    Args:
        scenario: Scenario configuration.
        prices_df: OHLCV DataFrame with DatetimeIndex.
        entries: Boolean array of entry signals.
        exits: Boolean array of exit signals (None for risk-rule-only exits).
        framework: Target framework name (determines execution mode overrides).

    Returns:
        FrameworkResult with trade data.
    """
    import polars as pl

    from ml4t.backtest._validation_imports import (
        BacktestConfig,
        DataFeed,
        Engine,
        ExecutionMode,
        Strategy,
    )
    from ml4t.backtest.config import CommissionType, ExecutionPrice, SlippageType

    asset = _get_asset_name(scenario, framework)

    # Build prices polars DataFrame
    timestamps = _extract_timestamps(prices_df)
    prices_pl = pl.DataFrame({
        "timestamp": timestamps,
        "asset": [asset] * len(prices_df),
        "open": prices_df["open"].tolist(),
        "high": prices_df["high"].tolist(),
        "low": prices_df["low"].tolist(),
        "close": prices_df["close"].tolist(),
        "volume": prices_df["volume"].astype(float).tolist(),
    })

    # Build signals polars DataFrame
    signals_dict: dict[str, list] = {
        "timestamp": timestamps,
        "asset": [asset] * len(prices_df),
    }
    for col in scenario.signal_columns:
        if col == "entry":
            signals_dict["entry"] = entries.tolist()
        elif col == "exit" and exits is not None:
            signals_dict["exit"] = exits.tolist()
        elif col in ("short_entry", "long_entry", "short_exit", "long_exit"):
            # For directional signals, map from entries/exits arrays
            if "entry" in col:
                signals_dict[col] = entries.tolist()
            else:
                signals_dict[col] = exits.tolist() if exits is not None else [False] * len(entries)

    signals_pl = pl.DataFrame(signals_dict) if len(signals_dict) > 2 else None

    # Build strategy
    strategy = _build_strategy(scenario, asset)

    # Build config
    config = _build_config(scenario, framework)

    feed = DataFeed(prices_df=prices_pl, signals_df=signals_pl)
    engine = Engine(feed, strategy, config)
    results = engine.run()

    # Extract results
    trade_list = []
    for t in results["trades"]:
        trade_dict: dict[str, Any] = {
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "pnl": t.pnl,
        }
        if hasattr(t, "quantity"):
            trade_dict["size"] = abs(t.quantity)
            trade_dict["direction"] = "Short" if t.quantity < 0 else "Long"
        if hasattr(t, "commission"):
            trade_dict["commission"] = t.commission
        trade_list.append(trade_dict)

    extra = {}
    if "commission" in scenario.extra_checks:
        extra["total_commission"] = sum(f.commission for f in results["fills"])

    return FrameworkResult(
        framework="ml4t.backtest",
        final_value=results["final_value"],
        total_pnl=results["final_value"] - scenario.initial_cash,
        num_trades=results["num_trades"],
        trades=trade_list,
        extra=extra,
    )


def _extract_timestamps(prices_df: pd.DataFrame) -> list:
    """Extract timestamps handling timezone-aware and naive DatetimeIndex."""
    if prices_df.index.tz is not None:
        return [ts.to_pydatetime().replace(tzinfo=None) for ts in prices_df.index]
    return prices_df.index.to_pydatetime().tolist()


def _get_asset_name(scenario: ScenarioConfig, framework: str) -> str:
    """Get asset name based on framework (Zipline uses 'TEST', others use 'AAPL')."""
    if framework == "zipline":
        return "TEST"
    # VBT Pro scenario_01 uses AAPL, most others use TEST
    return scenario.constants.get("asset", "TEST")


def _build_strategy(scenario: ScenarioConfig, asset: str):
    """Build the ml4t Strategy for this scenario."""
    from ml4t.backtest._validation_imports import OrderSide, Strategy
    from ml4t.backtest.risk import RuleChain

    risk_rules = _build_risk_rules(scenario)

    # Detect short direction from config and data generator name
    is_short = (
        scenario.ml4t_config.get("allow_short_selling", False)
        and "short" in scenario.data_generator.lower()
    )

    class ValidationStrategy(Strategy):
        def __init__(self):
            self._entered = False

        def on_start(self, broker):
            if risk_rules:
                if len(risk_rules) == 1:
                    broker.set_position_rules(risk_rules[0])
                else:
                    broker.set_position_rules(RuleChain(risk_rules))

        def on_data(self, timestamp, data, context, broker):
            if asset not in data:
                return
            signals = data[asset].get("signals", {})
            position = broker.get_position(asset)
            current_qty = position.quantity if position else 0

            if scenario.strategy_type == "long_signal":
                if signals.get("exit") and current_qty > 0:
                    broker.close_position(asset)
                elif signals.get("entry") and current_qty == 0:
                    broker.submit_order(asset, scenario.shares)

            elif scenario.strategy_type == "short_only":
                if signals.get("short_exit") and current_qty < 0:
                    broker.close_position(asset)
                elif signals.get("short_entry") and current_qty == 0:
                    broker.submit_order(asset, scenario.shares, OrderSide.SELL)

            elif scenario.strategy_type == "long_short":
                # Exit first
                if signals.get("long_exit") and current_qty > 0:
                    broker.close_position(asset)
                elif signals.get("short_exit") and current_qty < 0:
                    broker.close_position(asset)
                # Then entry
                if signals.get("long_entry") and current_qty == 0:
                    broker.submit_order(asset, scenario.shares)
                elif signals.get("short_entry") and current_qty == 0:
                    broker.submit_order(asset, scenario.shares, OrderSide.SELL)

            elif scenario.strategy_type == "risk_entry_only":
                # Entry only on first signal, exits handled by risk rules
                if signals.get("entry") and current_qty == 0:
                    if not self._entered or scenario.constants.get("allow_reentry", True):
                        side = OrderSide.SELL if is_short else OrderSide.BUY
                        broker.submit_order(asset, scenario.shares, side)
                        self._entered = True

            elif scenario.strategy_type == "single_entry":
                # One-time entry, exits handled by risk rules
                if not self._entered and current_qty == 0:
                    side = OrderSide.SELL if is_short else OrderSide.BUY
                    broker.submit_order(asset, scenario.shares, side)
                    self._entered = True

    return ValidationStrategy()


def _build_risk_rules(scenario: ScenarioConfig) -> list:
    """Build risk rules from scenario config."""
    from ml4t.backtest.risk import StopLoss, TakeProfit
    from ml4t.backtest.risk.position import TrailingStop

    rules = []
    for rule_cfg in scenario.risk_rules:
        rule_type = rule_cfg["type"]
        if rule_type == "StopLoss":
            rules.append(StopLoss(pct=rule_cfg["pct"]))
        elif rule_type == "TakeProfit":
            rules.append(TakeProfit(pct=rule_cfg["pct"]))
        elif rule_type == "TrailingStop":
            rules.append(TrailingStop(pct=rule_cfg["pct"]))
    return rules


def _build_config(scenario: ScenarioConfig, framework: str):
    """Build BacktestConfig with framework-specific overrides."""
    from ml4t.backtest._validation_imports import (
        BacktestConfig,
        ExecutionMode,
        StopFillMode,
        StopLevelBasis,
    )
    from ml4t.backtest.config import (
        CommissionType,
        ExecutionPrice,
        SlippageType,
        TrailStopTiming,
        WaterMarkSource,
    )

    # Start with scenario base config
    config_kwargs: dict[str, Any] = {
        "initial_cash": scenario.initial_cash,
        "allow_short_selling": False,
        "commission_type": CommissionType.NONE,
        "slippage_type": SlippageType.NONE,
    }

    # Apply scenario-level config
    config_kwargs.update(scenario.ml4t_config)

    # Apply framework-specific overrides
    if framework in ("vectorbt_pro", "vectorbt_oss"):
        config_kwargs.setdefault("execution_mode", ExecutionMode.SAME_BAR)
        config_kwargs.setdefault("execution_price", ExecutionPrice.CLOSE)
    elif framework == "backtrader":
        config_kwargs.setdefault("execution_mode", ExecutionMode.NEXT_BAR)
    elif framework == "zipline":
        config_kwargs.setdefault("execution_mode", ExecutionMode.NEXT_BAR)
    else:
        config_kwargs.setdefault("execution_mode", ExecutionMode.NEXT_BAR)

    # Apply per-framework overrides from scenario
    if framework in scenario.ml4t_overrides:
        config_kwargs.update(scenario.ml4t_overrides[framework])

    # Handle constants-based config
    constants = scenario.constants

    if "commission_rate" in constants:
        config_kwargs["commission_type"] = CommissionType.PERCENTAGE
        config_kwargs["commission_rate"] = constants["commission_rate"]
    if "per_share_rate" in constants:
        config_kwargs["commission_type"] = CommissionType.PER_SHARE
        config_kwargs["commission_per_share"] = constants["per_share_rate"]
    if "slippage_fixed" in constants:
        config_kwargs["slippage_type"] = SlippageType.FIXED
        config_kwargs["slippage_fixed"] = constants["slippage_fixed"]
    if "slippage_rate" in constants:
        config_kwargs["slippage_type"] = SlippageType.PERCENTAGE
        config_kwargs["slippage_rate"] = constants["slippage_rate"]

    # Convert string enum values to actual enums
    for key, val in list(config_kwargs.items()):
        if isinstance(val, str):
            if key == "execution_mode" and hasattr(ExecutionMode, val.upper()):
                config_kwargs[key] = ExecutionMode[val.upper()]
            elif key == "stop_fill_mode" and hasattr(StopFillMode, val.upper()):
                config_kwargs[key] = StopFillMode[val.upper()]
            elif key == "stop_level_basis" and hasattr(StopLevelBasis, val.upper()):
                config_kwargs[key] = StopLevelBasis[val.upper()]
            elif key == "trail_hwm_source" and hasattr(WaterMarkSource, val.upper()):
                config_kwargs[key] = WaterMarkSource[val.upper()]
            elif key == "trail_stop_timing" and hasattr(TrailStopTiming, val.upper()):
                config_kwargs[key] = TrailStopTiming[val.upper()]

    return BacktestConfig(**config_kwargs)
