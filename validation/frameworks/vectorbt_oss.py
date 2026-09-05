"""VectorBT OSS framework driver for validation scenarios.

Provides one parameterized driver for the scenario matrix.

VectorBT OSS uses same-bar execution with close fills by default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.canonical_records import vectorbt_fills, vectorbt_trades
from common.capabilities import FRAMEWORK_CAPABILITIES
from common.types import FrameworkResult, ScenarioConfig


def run(
    scenario: ScenarioConfig,
    prices_df: pd.DataFrame,
    entries: np.ndarray,
    exits: np.ndarray | None = None,
) -> FrameworkResult:
    """Run a validation scenario using VectorBT OSS.

    Args:
        scenario: Scenario configuration.
        prices_df: OHLCV DataFrame with DatetimeIndex.
        entries: Boolean entry signal array.
        exits: Boolean exit signal array (None for risk-rule-only exits).

    Returns:
        FrameworkResult with trade data.
    """
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError(
            "VectorBT OSS not installed. Run in .venv-vectorbt-oss environment."
        ) from None

    constants = scenario.constants

    # Build portfolio kwargs — pass OHLC so VBT uses intrabar stop checks
    pf_kwargs = {
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

    # Handle exits
    if exits is not None and scenario.strategy_type in ("long_signal", "long_short"):
        pf_kwargs["exits"] = exits
    elif exits is not None and scenario.strategy_type == "short_only":
        pf_kwargs["exits"] = exits
        pf_kwargs["short_entries"] = entries
        pf_kwargs["short_exits"] = exits
        # Remove regular entries/exits for short-only
        del pf_kwargs["entries"]

    # Commission
    if "commission_rate" in constants:
        pf_kwargs["fees"] = constants["commission_rate"]
    elif "per_share_rate" in constants:
        pf_kwargs["fees"] = constants["per_share_rate"]
        pf_kwargs["fixed_fees"] = 0.0
    else:
        pf_kwargs["fees"] = 0.0

    # Slippage
    if "slippage_rate" in constants:
        pf_kwargs["slippage"] = constants["slippage_rate"]
    elif "slippage_fixed" in constants:
        pf_kwargs["slippage"] = constants["slippage_fixed"] / prices_df["close"].mean()
    else:
        pf_kwargs["slippage"] = 0.0

    # Risk rules
    if any(r["type"] == "StopLoss" for r in scenario.risk_rules):
        sl_pct = next(r["pct"] for r in scenario.risk_rules if r["type"] == "StopLoss")
        pf_kwargs["sl_stop"] = sl_pct

    if any(r["type"] == "TakeProfit" for r in scenario.risk_rules):
        tp_pct = next(r["pct"] for r in scenario.risk_rules if r["type"] == "TakeProfit")
        pf_kwargs["tp_stop"] = tp_pct

    if any(r["type"] == "TrailingStop" for r in scenario.risk_rules):
        trail_pct = next(r["pct"] for r in scenario.risk_rules if r["type"] == "TrailingStop")
        pf_kwargs["sl_stop"] = trail_pct
        pf_kwargs["sl_trail"] = True

    # Short direction detection
    is_short = scenario.strategy_type == "short_only" or (
        scenario.ml4t_config.get("allow_short_selling", False)
        and "short" in scenario.data_generator.lower()
    )
    if is_short:
        # VBT OSS uses short_entries/short_exits params
        if "entries" in pf_kwargs:
            pf_kwargs["short_entries"] = pf_kwargs.pop("entries")
        if "exits" in pf_kwargs:
            pf_kwargs["short_exits"] = pf_kwargs.pop("exits")

    # Run portfolio simulation
    pf = vbt.Portfolio.from_signals(**pf_kwargs)

    # Extract results
    trades = pf.trades.records_readable
    final_value = float(pf.final_value())

    trade_list = trades.to_dict("records") if len(trades) > 0 else []
    order_list = pf.orders.records_readable.to_dict("records")
    asset = str(scenario.constants.get("asset", "TEST"))
    normalized_trades = vectorbt_trades(trade_list, asset=asset)
    normalized_fills = vectorbt_fills(order_list, asset=asset)

    extra = {}
    if "commission" in scenario.extra_checks:
        # VBT OSS tracks entry/exit fees separately
        fees = sum(abs(order.get("Fees", 0)) for order in order_list)
        extra["total_commission"] = fees
    if "exit_price" in scenario.extra_checks and normalized_trades:
        extra["exit_price"] = normalized_trades[0].get("exit_price")

    return FrameworkResult(
        framework="VectorBT OSS",
        final_value=final_value,
        total_pnl=final_value - scenario.initial_cash,
        num_trades=len(trades),
        trades=normalized_trades,
        fills=normalized_fills,
        capabilities=FRAMEWORK_CAPABILITIES["vectorbt_oss"],
        extra=extra,
    )
