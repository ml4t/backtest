"""VectorBT Pro framework driver for validation scenarios.

Consolidates 17 run_vectorbt_pro() functions into a single parameterized driver.

VectorBT Pro uses same-bar execution with close fills by default.
Licensed software; guarded by try/except ImportError.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.types import FrameworkResult, ScenarioConfig

SUPPORTED_VECTORBT_PRO_VERSION = "2026.6.27"


def _require_supported_version(actual_version: str) -> None:
    if actual_version != SUPPORTED_VECTORBT_PRO_VERSION:
        raise RuntimeError(
            "VectorBT Pro validation requires version "
            f"{SUPPORTED_VECTORBT_PRO_VERSION}, found {actual_version}."
        )


def _build_portfolio_kwargs(
    scenario: ScenarioConfig,
    prices_df: pd.DataFrame,
    entries: np.ndarray,
    exits: np.ndarray | None,
) -> dict[str, object]:
    constants = scenario.constants
    kwargs: dict[str, object] = {
        "open": prices_df["open"],
        "high": prices_df["high"],
        "low": prices_df["low"],
        "close": prices_df["close"],
        "entries": entries,
        "init_cash": scenario.initial_cash,
        "size": scenario.shares,
        "size_type": "amount",
        "accumulate": False,
        "freq": "D",
    }

    if exits is not None and scenario.strategy_type in ("long_signal", "long_short"):
        kwargs["exits"] = exits
    elif exits is not None and scenario.strategy_type == "short_only":
        kwargs["short_entries"] = entries
        kwargs["short_exits"] = exits
        del kwargs["entries"]

    if "commission_rate" in constants:
        kwargs["fees"] = constants["commission_rate"]
    elif "per_share_rate" in constants:
        kwargs["fees"] = 0.0
        kwargs["fixed_fees"] = constants["per_share_rate"] * scenario.shares
    else:
        kwargs["fees"] = 0.0

    if "slippage_rate" in constants:
        kwargs["slippage"] = constants["slippage_rate"]
    elif "slippage_fixed" in constants:
        kwargs["slippage"] = constants["slippage_fixed"] / prices_df["close"].mean()
    else:
        kwargs["slippage"] = 0.0

    risk_rules = {rule["type"]: rule["pct"] for rule in scenario.risk_rules}
    if "StopLoss" in risk_rules:
        kwargs["sl_stop"] = risk_rules["StopLoss"]
    if "TakeProfit" in risk_rules:
        kwargs["tp_stop"] = risk_rules["TakeProfit"]
    if "TrailingStop" in risk_rules:
        kwargs["tsl_stop"] = risk_rules["TrailingStop"]

    is_short = scenario.strategy_type == "short_only" or (
        scenario.ml4t_config.get("allow_short_selling", False)
        and "short" in scenario.data_generator.lower()
    )
    if is_short:
        if "entries" in kwargs:
            kwargs["short_entries"] = kwargs.pop("entries")
        if "exits" in kwargs:
            kwargs["short_exits"] = kwargs.pop("exits")
    return kwargs


def run(
    scenario: ScenarioConfig,
    prices_df: pd.DataFrame,
    entries: np.ndarray,
    exits: np.ndarray | None = None,
) -> FrameworkResult:
    """Run a validation scenario using VectorBT Pro.

    Args:
        scenario: Scenario configuration.
        prices_df: OHLCV DataFrame with DatetimeIndex.
        entries: Boolean entry signal array.
        exits: Boolean exit signal array (None for risk-rule-only exits).

    Returns:
        FrameworkResult with trade data.
    """
    try:
        import vectorbtpro as vbt
    except ImportError:
        raise ImportError(
            "VectorBT Pro not installed. Run in .venv-vectorbt-pro environment."
        ) from None
    _require_supported_version(importlib.metadata.version("vectorbtpro"))

    pf_kwargs = _build_portfolio_kwargs(scenario, prices_df, entries, exits)
    pf = vbt.Portfolio.from_signals(**pf_kwargs)

    # Extract results
    trades = pf.trades.records_readable
    final_value = float(pf.total_return * scenario.initial_cash + scenario.initial_cash)

    trade_list = trades.to_dict("records") if len(trades) > 0 else []
    order_list = pf.orders.records_readable.to_dict("records")
    normalized_trades = []
    for index, t in enumerate(trade_list):
        entry_order = order_list[index * 2] if index * 2 < len(order_list) else {}
        exit_order = order_list[index * 2 + 1] if index * 2 + 1 < len(order_list) else {}
        size = float(t.get("Size", scenario.shares))
        direction = t.get("Direction", "Long")
        entry_price = float(
            entry_order.get("Price", t.get("Avg Entry Price", t.get("Entry Price", 0)))
        )
        exit_price = float(exit_order.get("Price", t.get("Avg Exit Price", t.get("Exit Price", 0))))
        direction_sign = -1.0 if str(direction).lower() == "short" else 1.0
        fees = abs(float(entry_order.get("Fees", t.get("Entry Fees", 0)))) + abs(
            float(exit_order.get("Fees", t.get("Exit Fees", 0)))
        )
        normalized = {
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": (exit_price - entry_price) * size * direction_sign - fees,
            "size": size,
            "direction": direction,
        }
        normalized_trades.append(normalized)

    extra = {}
    if "commission" in scenario.extra_checks:
        fees = sum(abs(order.get("Fees", 0)) for order in order_list)
        extra["total_commission"] = fees
    if "exit_price" in scenario.extra_checks and normalized_trades:
        extra["exit_price"] = normalized_trades[0].get("exit_price")

    return FrameworkResult(
        framework="VectorBT Pro",
        final_value=final_value,
        total_pnl=final_value - scenario.initial_cash,
        num_trades=len(trades),
        trades=normalized_trades,
        extra=extra,
    )
