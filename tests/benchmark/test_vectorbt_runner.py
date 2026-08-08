"""Contracts for shared VectorBT validation normalization."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from ml4t.backtest._validation.vectorbt_runner import extract_order_log, extract_trade_log


def test_order_log_accepts_current_index_column_and_restores_event_order() -> None:
    records = pd.DataFrame(
        [
            {
                "Column": "MSFT",
                "Index": pd.Timestamp("2024-01-03"),
                "Size": 2.0,
                "Price": 102.0,
                "Fees": 0.2,
                "Side": "Sell",
            },
            {
                "Column": "AAPL",
                "Index": pd.Timestamp("2024-01-02"),
                "Size": 1.0,
                "Price": 100.0,
                "Fees": 0.1,
                "Side": "Buy",
            },
        ]
    )
    portfolio = SimpleNamespace(orders=SimpleNamespace(records_readable=records))

    normalized = extract_order_log(portfolio)

    assert normalized[["timestamp", "symbol"]].to_dict("records") == [
        {"timestamp": pd.Timestamp("2024-01-02"), "symbol": "AAPL"},
        {"timestamp": pd.Timestamp("2024-01-03"), "symbol": "MSFT"},
    ]
    assert normalized["side"].tolist() == ["buy", "sell"]


def test_trade_log_accepts_current_index_columns_and_restores_exit_time() -> None:
    records = pd.DataFrame(
        [
            {
                "Column": "AAPL",
                "Entry Index": pd.Timestamp("2024-01-02"),
                "Exit Index": pd.Timestamp("2024-01-03"),
                "Size": 1.0,
                "Avg Entry Price": 100.0,
                "Avg Exit Price": 101.0,
                "PnL": 1.0,
                "Direction": "Long",
            }
        ]
    )
    portfolio = SimpleNamespace(trades=SimpleNamespace(records_readable=records))

    normalized = extract_trade_log(portfolio)

    assert normalized.loc[0, "timestamp"] == pd.Timestamp("2024-01-02")
    assert normalized.loc[0, "exit_time"] == pd.Timestamp("2024-01-03")
