"""Tests for shared Backtrader and Zipline validation runners."""

from __future__ import annotations

import pandas as pd

from ml4t.backtest._validation.backtrader_runner import transactions_to_trade_log as bt_trade_log
from ml4t.backtest._validation.zipline_runner import (
    flatten_result_column,
    normalize_target_lookup,
)
from ml4t.backtest._validation.zipline_runner import (
    transactions_to_trade_log as zipline_trade_log,
)


def test_backtrader_transactions_to_trade_log_handles_flip():
    transactions = pd.DataFrame(
        {
            "amount": [10, -15, 5],
            "price": [100.0, 110.0, 108.0],
            "symbol": ["AAPL", "AAPL", "AAPL"],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

    trades_df = bt_trade_log(transactions)

    assert trades_df is not None
    assert len(trades_df) == 2
    assert trades_df["side"].tolist() == ["long", "short"]
    assert trades_df["quantity"].tolist() == [10.0, 5.0]
    assert trades_df["entry_price"].tolist() == [100.0, 110.0]
    assert trades_df["exit_price"].tolist() == [110.0, 108.0]


def test_zipline_normalize_target_lookup_strips_timezone():
    target_lookup_raw = {
        pd.Timestamp("2024-01-02", tz="UTC"): {"AAPL": 100.0},
        pd.Timestamp("2024-01-03"): {"MSFT": -50.0},
    }

    normalized = normalize_target_lookup(target_lookup_raw)

    assert list(normalized.keys()) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]


def test_zipline_flatten_result_column_reads_list_payloads():
    results = pd.DataFrame(
        {
            "transactions": [
                [{"symbol": "AAPL", "amount": 10, "price": 100.0}],
                [{"symbol": "AAPL", "amount": -10, "price": 101.0}],
            ]
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    flattened = flatten_result_column(results, "transactions")

    assert flattened["symbol"].tolist() == ["AAPL", "AAPL"]
    assert flattened["amount"].tolist() == [10, -10]
    assert pd.to_datetime(flattened["dt"]).tolist() == list(results.index)


def test_zipline_transactions_to_trade_log_handles_round_trip():
    transactions = pd.DataFrame(
        {
            "dt": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "amount": [10.0, -10.0],
            "price": [100.0, 102.0],
            "symbol": ["AAPL", "AAPL"],
        }
    )

    trades_df = zipline_trade_log(transactions)

    assert trades_df is not None
    assert len(trades_df) == 1
    record = trades_df.iloc[0]
    assert record["asset"] == "AAPL"
    assert record["side"] == "long"
    assert record["quantity"] == 10.0
    assert record["pnl"] == 20.0
