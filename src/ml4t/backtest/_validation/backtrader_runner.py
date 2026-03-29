"""Shared Backtrader helpers for internal validation workflows."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class BacktraderRunResult:
    """Standardized Backtrader run artifacts for benchmark consumers."""

    final_value: float
    num_trades: int
    trades_df: pd.DataFrame | None
    positions_df: pd.DataFrame | None
    transactions_df: pd.DataFrame | None


def load_backtrader_package() -> Any:
    """Import and return the Backtrader package."""
    return importlib.import_module("backtrader")


def transactions_to_trade_log(transactions: pd.DataFrame) -> pd.DataFrame | None:
    """Convert PyFolio transactions into completed round-trip trades."""
    if len(transactions) == 0 or "symbol" not in transactions.columns:
        return None

    trade_records: list[dict[str, object]] = []
    for symbol in transactions["symbol"].unique():
        symbol_txns = transactions[transactions["symbol"] == symbol].sort_index()

        running_pos = 0.0
        entry_time = None
        entry_price = None
        entry_size = 0.0

        for dt, row in symbol_txns.iterrows():
            amount = float(row["amount"])
            price = float(row["price"])
            prev_pos = running_pos
            running_pos += amount

            if prev_pos == 0 and running_pos != 0:
                entry_time = dt
                entry_price = price
                entry_size = amount
            elif prev_pos != 0 and running_pos == 0:
                assert entry_time is not None
                assert entry_price is not None
                pnl = (price - entry_price) * entry_size
                trade_records.append(
                    {
                        "entry_date": entry_time,
                        "exit_date": dt,
                        "asset": str(symbol),
                        "side": "long" if entry_size > 0 else "short",
                        "quantity": abs(entry_size),
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                    }
                )
                entry_time = None
                entry_price = None
            elif prev_pos != 0 and running_pos != 0 and (prev_pos > 0) != (running_pos > 0):
                assert entry_time is not None
                assert entry_price is not None
                pnl = (price - entry_price) * entry_size
                trade_records.append(
                    {
                        "entry_date": entry_time,
                        "exit_date": dt,
                        "asset": str(symbol),
                        "side": "long" if entry_size > 0 else "short",
                        "quantity": abs(entry_size),
                        "entry_price": entry_price,
                        "exit_price": price,
                        "pnl": pnl,
                    }
                )
                entry_time = dt
                entry_price = price
                entry_size = running_pos

    if not trade_records:
        return None
    return pd.DataFrame(trade_records).sort_values("entry_date").reset_index(drop=True)


def run_backtrader_target_shares(
    *,
    bt: Any,
    price_data: dict[str, pd.DataFrame],
    target_lookup: dict[str, dict[str, float]],
    initial_cash: float,
    commission_pct: float,
) -> BacktraderRunResult:
    """Run a canonical target-share strategy through Backtrader."""

    class TopBottomBTStrategy(bt.Strategy):
        def __init__(self):
            self.target_lookup = target_lookup
            self.data_by_name = {data._name: data for data in self.datas}

        def next(self):
            dt = self.datas[0].datetime.datetime(0)
            dt_key = dt.strftime("%Y-%m-%d")
            targets = self.target_lookup.get(dt_key, {})

            active_names = set(targets.keys())
            for data in self.datas:
                if self.getposition(data).size != 0:
                    active_names.add(data._name)

            for asset_name in sorted(active_names):
                data = self.data_by_name[asset_name]
                current_size = self.getposition(data).size
                target_size = targets.get(asset_name, 0.0)
                if current_size != target_size:
                    self.order_target_size(data=data, target=target_size)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(TopBottomBTStrategy)

    for asset_name, df in price_data.items():
        data = bt.feeds.PandasData(dataname=df, name=asset_name)
        cerebro.adddata(data)

    cerebro.broker.setcash(initial_cash)
    if commission_pct > 0:
        cerebro.broker.setcommission(commission=commission_pct)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")

    results = cerebro.run()
    strat = results[0]
    pyfolio_analyzer = strat.analyzers.getbyname("pyfolio")
    _returns, positions, transactions, _gross_lev = pyfolio_analyzer.get_pf_items()

    trades_df = transactions_to_trade_log(transactions)
    num_trades = len(trades_df) if trades_df is not None else 0

    return BacktraderRunResult(
        final_value=float(cerebro.broker.getvalue()),
        num_trades=num_trades,
        trades_df=trades_df,
        positions_df=positions,
        transactions_df=transactions,
    )
