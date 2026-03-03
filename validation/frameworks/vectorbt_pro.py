"""VectorBT Pro framework driver for validation scenarios.

Consolidates 17 run_vectorbt_pro() functions into a single parameterized driver.

VectorBT Pro uses same-bar execution with close fills by default.
Licensed software — guarded by try/except ImportError.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.types import FrameworkResult, ScenarioConfig


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
        raise ImportError("VectorBT Pro not installed. Run in .venv-vectorbt-pro environment.")

    constants = scenario.constants

    # Build portfolio kwargs
    pf_kwargs = {
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
        pf_kwargs["short_entries"] = entries
        pf_kwargs["short_exits"] = exits
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

    # Short-only
    if scenario.strategy_type == "short_only":
        if "entries" in pf_kwargs:
            pf_kwargs["short_entries"] = pf_kwargs.pop("entries")
        if "exits" in pf_kwargs:
            pf_kwargs["short_exits"] = pf_kwargs.pop("exits")

    # Run portfolio simulation
    pf = vbt.Portfolio.from_signals(**pf_kwargs)

    # Extract results
    trades = pf.trades.records_readable
    final_value = float(pf.total_return * scenario.initial_cash + scenario.initial_cash)
    total_pnl = float(pf.total_profit)

    trade_list = trades.to_dict("records") if len(trades) > 0 else []
    normalized_trades = []
    for t in trade_list:
        normalized = {
            "entry_price": t.get("Avg Entry Price", t.get("Entry Price", 0)),
            "exit_price": t.get("Avg Exit Price", t.get("Exit Price", 0)),
            "pnl": t.get("PnL", t.get("P&L", 0)),
            "size": t.get("Size", scenario.shares),
            "direction": t.get("Direction", "Long"),
        }
        normalized_trades.append(normalized)

    extra = {}
    if "commission" in scenario.extra_checks:
        fees = sum(abs(t.get("Fees Paid", t.get("Fees", 0))) for t in trade_list)
        extra["total_commission"] = fees
    if "exit_price" in scenario.extra_checks and normalized_trades:
        extra["exit_price"] = normalized_trades[0].get("exit_price")

    return FrameworkResult(
        framework="VectorBT Pro",
        final_value=final_value,
        total_pnl=total_pnl,
        num_trades=len(trades),
        trades=normalized_trades,
        extra=extra,
    )
