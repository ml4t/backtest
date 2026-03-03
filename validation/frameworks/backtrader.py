"""Backtrader framework driver for validation scenarios.

Consolidates 16 run_backtrader() functions into a single parameterized driver.

Backtrader always uses next-bar execution (orders placed on bar N fill at bar N+1's open).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
    """Run a validation scenario using Backtrader.

    Args:
        scenario: Scenario configuration.
        prices_df: OHLCV DataFrame with DatetimeIndex.
        entries: Boolean entry signal array.
        exits: Boolean exit signal array (None for risk-rule-only exits).

    Returns:
        FrameworkResult with trade data.
    """
    try:
        import backtrader as bt
    except ImportError:
        raise ImportError("Backtrader not installed. Run in .venv-backtrader environment.")

    # Data feed adapter
    class PandasData(bt.feeds.PandasData):
        params = (
            ("datetime", None),
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("volume", "volume"),
            ("openinterest", -1),
        )

    # Build strategy class based on scenario type
    strategy_cls = _build_strategy(scenario, bt)

    # Setup cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(scenario.initial_cash)

    # Commission setup
    constants = scenario.constants
    if "commission_rate" in constants:
        cerebro.broker.setcommission(commission=constants["commission_rate"])
    elif "per_share_rate" in constants:
        cerebro.broker.setcommission(
            commission=constants["per_share_rate"],
            commtype=bt.CommInfoBase.COMM_FIXED,
        )
    else:
        cerebro.broker.setcommission(commission=0.0)

    # Slippage setup
    if "slippage_fixed" in constants:
        cerebro.broker.set_slippage_fixed(constants["slippage_fixed"])
    elif "slippage_rate" in constants:
        cerebro.broker.set_slippage_perc(constants["slippage_rate"])

    # Add data
    data = PandasData(dataname=prices_df)
    cerebro.adddata(data)

    # Add strategy with signals
    cerebro.addstrategy(strategy_cls, entries=entries, exits=exits, scenario=scenario)

    # Run
    results = cerebro.run()
    strategy = results[0]
    final_value = cerebro.broker.getvalue()

    # Extract results
    trade_list = []
    if hasattr(strategy, "trade_log"):
        trade_list = sorted(strategy.trade_log, key=lambda t: t.get("entry_time", 0))

    extra = {}
    if hasattr(strategy, "total_commission"):
        extra["total_commission"] = strategy.total_commission
    if trade_list and "exit_price" not in trade_list[0]:
        # Calculate exit price from pnl for risk-rule scenarios
        for t in trade_list:
            if "pnl" in t and "entry_price" in t and "size" in t:
                size = t["size"]
                if t.get("direction") == "Short":
                    t["exit_price"] = t["entry_price"] - t["pnl"] / abs(size)
                else:
                    t["exit_price"] = t["entry_price"] + t["pnl"] / abs(size)
    if scenario.extra_checks and "exit_price" in scenario.extra_checks and trade_list:
        extra["exit_price"] = trade_list[0].get("exit_price")

    return FrameworkResult(
        framework="Backtrader",
        final_value=final_value,
        total_pnl=final_value - scenario.initial_cash,
        num_trades=len(trade_list),
        trades=trade_list,
        extra=extra,
    )


def _build_strategy(scenario: ScenarioConfig, bt: Any) -> type:
    """Build a Backtrader Strategy class for this scenario."""

    if scenario.strategy_type in ("long_signal", "long_short"):
        return _signal_strategy(scenario, bt)
    elif scenario.strategy_type == "short_only":
        return _short_strategy(scenario, bt)
    elif scenario.strategy_type in ("risk_entry_only", "single_entry"):
        return _risk_entry_strategy(scenario, bt)
    else:
        raise ValueError(f"Unknown strategy type: {scenario.strategy_type}")


def _signal_strategy(scenario: ScenarioConfig, bt: Any) -> type:
    """Strategy with explicit entry/exit signals."""
    shares = scenario.shares

    class SignalStrategy(bt.Strategy):
        params = (("entries", None), ("exits", None), ("scenario", None))

        def __init__(self):
            self.bar_count = 0
            self.trade_log = []
            self.total_commission = 0.0
            self.pending_trade = None

        def next(self):
            idx = self.bar_count
            entries = self.params.entries
            exits = self.params.exits

            if idx >= len(entries):
                self.bar_count += 1
                return

            if exits is not None and idx < len(exits) and exits[idx] and self.position.size > 0:
                self.close()
            elif entries[idx] and self.position.size == 0:
                self.buy(size=shares)

            self.bar_count += 1

        def notify_trade(self, trade):
            if trade.justopened:
                self.pending_trade = {
                    "entry_time": bt.num2date(trade.dtopen),
                    "entry_price": trade.price,
                    "entry_size": trade.size,
                }
            elif trade.isclosed and self.pending_trade:
                entry_size = self.pending_trade["entry_size"]
                exit_price = self.pending_trade["entry_price"] + trade.pnl / abs(entry_size)

                self.trade_log.append({
                    "entry_time": self.pending_trade["entry_time"],
                    "exit_time": bt.num2date(trade.dtclose),
                    "entry_price": self.pending_trade["entry_price"],
                    "exit_price": exit_price,
                    "pnl": trade.pnl,
                    "pnlcomm": trade.pnlcomm,
                    "commission": trade.commission,
                    "size": abs(entry_size),
                    "direction": "Long",
                })
                self.pending_trade = None

        def notify_order(self, order):
            if order.status == order.Completed:
                self.total_commission += order.executed.comm

    return SignalStrategy


def _short_strategy(scenario: ScenarioConfig, bt: Any) -> type:
    """Short-only strategy."""
    shares = scenario.shares

    class ShortStrategy(bt.Strategy):
        params = (("entries", None), ("exits", None), ("scenario", None))

        def __init__(self):
            self.bar_count = 0
            self.trade_log = []
            self.pending_trade = None

        def next(self):
            idx = self.bar_count
            entries = self.params.entries
            exits = self.params.exits

            if idx >= len(entries):
                self.bar_count += 1
                return

            current_pos = self.position.size

            # Exit first (cover short)
            if exits is not None and idx < len(exits) and exits[idx] and current_pos < 0:
                self.close()
            # Then entry (open short)
            elif entries[idx] and current_pos == 0:
                self.sell(size=shares)

            self.bar_count += 1

        def notify_trade(self, trade):
            if trade.justopened:
                self.pending_trade = {
                    "entry_time": bt.num2date(trade.dtopen),
                    "entry_price": trade.price,
                    "entry_size": trade.size,
                }
            elif trade.isclosed and self.pending_trade:
                entry_size = self.pending_trade["entry_size"]
                if entry_size < 0:  # Short
                    exit_price = self.pending_trade["entry_price"] - trade.pnl / abs(entry_size)
                else:
                    exit_price = self.pending_trade["entry_price"] + trade.pnl / abs(entry_size)

                self.trade_log.append({
                    "entry_time": self.pending_trade["entry_time"],
                    "exit_time": bt.num2date(trade.dtclose),
                    "entry_price": self.pending_trade["entry_price"],
                    "exit_price": exit_price,
                    "size": abs(entry_size),
                    "pnl": trade.pnl,
                    "direction": "Short" if entry_size < 0 else "Long",
                })
                self.pending_trade = None

    return ShortStrategy


def _risk_entry_strategy(scenario: ScenarioConfig, bt: Any) -> type:
    """Entry-only strategy with risk-rule exits.

    Handles all rule combinations (TSL, SL, TP, TSL+TP, TSL+SL, SL+TP, TSL+SL+TP)
    using Backtrader's OCO (One-Cancels-Other) for automatic cancellation.
    """
    shares = scenario.shares
    single_entry = scenario.strategy_type == "single_entry"

    # Determine risk rule setup
    has_stop_loss = any(r["type"] == "StopLoss" for r in scenario.risk_rules)
    has_take_profit = any(r["type"] == "TakeProfit" for r in scenario.risk_rules)
    has_trailing_stop = any(r["type"] == "TrailingStop" for r in scenario.risk_rules)

    sl_pct = next((r["pct"] for r in scenario.risk_rules if r["type"] == "StopLoss"), None)
    tp_pct = next((r["pct"] for r in scenario.risk_rules if r["type"] == "TakeProfit"), None)
    trail_pct = next((r["pct"] for r in scenario.risk_rules if r["type"] == "TrailingStop"), None)

    is_short = (
        scenario.ml4t_config.get("allow_short_selling", False)
        and "short" in scenario.data_generator.lower()
    )

    class RiskEntryStrategy(bt.Strategy):
        params = (("entries", None), ("exits", None), ("scenario", None))

        def __init__(self):
            self.bar_count = 0
            self.trade_log = []
            self.entered_once = False
            self.entry_order = None
            self.exit_orders = []
            self.pending_trade = None
            self.needs_trail = False  # Defer trailing stop to notify_order

        def _submit_fixed_exits(self, ref_price):
            """Submit non-trailing exit orders at entry time, using signal close as ref.

            Returns list of submitted orders (for OCO linking with deferred trail).
            """
            orders = []
            if is_short:
                if has_stop_loss:
                    sl_price = ref_price * (1 + sl_pct)
                    orders.append(self.buy(
                        exectype=bt.Order.Stop, price=sl_price, size=shares,
                    ))
                if has_take_profit:
                    tp_price = ref_price * (1 - tp_pct)
                    orders.append(self.buy(
                        exectype=bt.Order.Limit, price=tp_price, size=shares,
                        oco=orders[0] if orders else None,
                    ))
            else:
                if has_stop_loss:
                    sl_price = ref_price * (1 - sl_pct)
                    orders.append(self.sell(
                        exectype=bt.Order.Stop, price=sl_price, size=shares,
                    ))
                if has_take_profit:
                    tp_price = ref_price * (1 + tp_pct)
                    orders.append(self.sell(
                        exectype=bt.Order.Limit, price=tp_price, size=shares,
                        oco=orders[0] if orders else None,
                    ))
            return orders

        def _submit_trail(self):
            """Submit trailing stop AFTER entry fills (deferred via notify_order).

            This ensures the trail initializes from the fill bar, not the signal bar.
            Links via OCO to any existing fixed exit orders.
            """
            first_existing = self.exit_orders[0] if self.exit_orders else None
            if is_short:
                trail = self.buy(
                    exectype=bt.Order.StopTrail, trailpercent=trail_pct,
                    size=shares, oco=first_existing,
                )
            else:
                trail = self.sell(
                    exectype=bt.Order.StopTrail, trailpercent=trail_pct,
                    size=shares, oco=first_existing,
                )
            self.exit_orders.append(trail)

        def next(self):
            idx = self.bar_count
            entries = self.params.entries

            if idx >= len(entries):
                self.bar_count += 1
                return

            current_pos = self.position.size

            if entries[idx] and current_pos == 0:
                if single_entry and self.entered_once:
                    self.bar_count += 1
                    return

                # Submit entry order
                if is_short:
                    self.entry_order = self.sell(size=shares)
                else:
                    self.entry_order = self.buy(size=shares)

                # Submit fixed exit orders now (SL/TP from signal close)
                ref_price = self.data.close[0]
                self.exit_orders = self._submit_fixed_exits(ref_price)

                # Defer trailing stop to after entry fills
                self.needs_trail = has_trailing_stop
                self.entered_once = True

            self.bar_count += 1

        def notify_trade(self, trade):
            if trade.justopened:
                self.pending_trade = {
                    "entry_time": bt.num2date(trade.dtopen),
                    "entry_price": trade.price,
                    "entry_size": trade.size,
                }
            elif trade.isclosed and self.pending_trade:
                entry_size = self.pending_trade["entry_size"]
                if entry_size < 0:
                    exit_price = self.pending_trade["entry_price"] - trade.pnl / abs(entry_size)
                else:
                    exit_price = self.pending_trade["entry_price"] + trade.pnl / abs(entry_size)

                self.trade_log.append({
                    "entry_time": self.pending_trade["entry_time"],
                    "exit_time": bt.num2date(trade.dtclose),
                    "entry_price": self.pending_trade["entry_price"],
                    "exit_price": exit_price,
                    "size": abs(entry_size),
                    "pnl": trade.pnl,
                    "direction": "Short" if entry_size < 0 else "Long",
                })
                self.pending_trade = None
                self.exit_orders = []

        def notify_order(self, order):
            if order.status == order.Completed and order == self.entry_order:
                # Entry filled — now submit deferred trailing stop
                self.entry_order = None
                if self.needs_trail:
                    self._submit_trail()
                    self.needs_trail = False

    return RiskEntryStrategy
