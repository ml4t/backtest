"""Shared VectorBT helpers for internal validation workflows."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import pandas as pd
import polars as pl


def load_vectorbt_package(package_name: str = "vectorbt") -> Any:
    """Import and return a VectorBT package."""
    return importlib.import_module(package_name)


def prices_to_wide(prices: pl.DataFrame, value_col: str = "close") -> pd.DataFrame:
    """Convert long-format Polars prices to a wide pandas price matrix."""
    wide = (
        prices.select("timestamp", "symbol", value_col)
        .pivot(on="symbol", index="timestamp", values=value_col)
        .sort("timestamp")
        .to_pandas()
        .set_index("timestamp")
    )
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


def align_weights_to_prices(
    weights: pd.DataFrame, close: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align target weights to a price matrix."""
    common_assets = [column for column in close.columns if column in weights.columns]
    close = close[common_assets]
    weights_aligned = weights[common_assets].reindex(close.index)
    return close, weights_aligned


def run_vectorbt_orders(
    *,
    vbt: Any,
    close: pd.DataFrame,
    size: pd.DataFrame,
    size_type: str,
    init_cash: float,
    fees: float,
    slippage: float = 0.0,
    cash_sharing: bool = True,
    group_by: bool | None = None,
    lock_cash: bool | None = None,
):
    """Run `Portfolio.from_orders` with consistent defaults across harnesses."""
    kwargs: dict[str, object] = {
        "close": close,
        "size": size,
        "size_type": size_type,
        "init_cash": init_cash,
        "fees": fees,
        "slippage": slippage,
        "cash_sharing": cash_sharing,
    }
    if group_by is not None:
        kwargs["group_by"] = group_by
    if lock_cash is not None:
        kwargs["lock_cash"] = lock_cash
    return vbt.Portfolio.from_orders(**kwargs)


def get_equity_curve(portfolio: Any) -> pd.Series:
    """Extract a one-dimensional portfolio value series."""
    value_attr = portfolio.value
    equity = value_attr() if callable(value_attr) else value_attr
    if isinstance(equity, pd.DataFrame):
        equity = equity.iloc[:, 0] if equity.shape[1] == 1 else equity.sum(axis=1)
    equity = pd.Series(equity)
    equity.index = pd.DatetimeIndex(equity.index)
    equity.name = "portfolio_value"
    return equity


def extract_order_log(portfolio: Any) -> pd.DataFrame:
    """Extract standardized order records from a VectorBT portfolio."""
    orders = portfolio.orders.records_readable
    rows: list[dict[str, object]] = []
    if len(orders) > 0:
        for _, row in orders.iterrows():
            rows.append(
                {
                    "timestamp": row.get("Timestamp"),
                    "symbol": row.get("Column"),
                    "side": "buy" if row.get("Side") == "Buy" else "sell",
                    "quantity": float(row.get("Size", 0.0)),
                    "price": float(row.get("Price", 0.0)),
                    "commission": float(row.get("Fees", 0.0)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "side", "quantity", "price", "commission"]
        )
    return pd.DataFrame(rows)


def extract_trade_log(portfolio: Any) -> pd.DataFrame:
    """Extract standardized round-trip trade records from a VectorBT portfolio."""
    trades_readable = portfolio.trades.records_readable
    if len(trades_readable) == 0:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "asset",
                "side",
                "quantity",
                "entry_price",
                "exit_price",
                "pnl",
            ]
        )

    entry_col = "Entry Timestamp" if "Entry Timestamp" in trades_readable.columns else "Entry Index"
    trades_readable = trades_readable.sort_values(entry_col)
    rows: list[dict[str, object]] = []
    for _, row in trades_readable.iterrows():
        direction = str(row.get("Direction", "Long")).lower()
        rows.append(
            {
                "timestamp": row.get("Entry Timestamp", row.get("Entry Index")),
                "asset": row.get("Column", "unknown"),
                "side": "long" if direction == "long" else "short",
                "quantity": abs(float(row.get("Size", 0.0))),
                "entry_price": float(row.get("Avg Entry Price", row.get("Entry Price", 0.0))),
                "exit_price": float(row.get("Avg Exit Price", row.get("Exit Price", 0.0))),
                "pnl": float(row.get("PnL", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def summarize_order_run(
    equity: pd.Series, orders_df: pd.DataFrame, initial_cash: float
) -> Mapping[str, float]:
    """Compute lightweight summary metrics for order-based VectorBT runs."""
    total_return = float(equity.iloc[-1] / initial_cash - 1.0)
    daily_returns = equity.pct_change().dropna()
    sharpe = 0.0
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * (252**0.5))

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    total_commission = float(orders_df["commission"].sum()) if len(orders_df) > 0 else 0.0
    return {
        "initial_cash": initial_cash,
        "final_value": float(equity.iloc[-1]),
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) > 0 else 0.0,
        "num_trades": int(len(orders_df)),
        "total_commission": total_commission,
    }
