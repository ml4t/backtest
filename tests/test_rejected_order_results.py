from __future__ import annotations

from datetime import datetime

import polars as pl

from ml4t.backtest import BacktestConfig, Strategy, run_backtest
from ml4t.backtest.types import ExecutionMode


class _UnaffordableOrder(Strategy):
    def on_data(self, timestamp, data, context, broker) -> None:
        broker.submit_order("AAPL", 1_000_000.0)


class _NoOrders(Strategy):
    def on_data(self, timestamp, data, context, broker) -> None:
        pass


def _prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2)],
            "asset": ["AAPL"],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [1_000_000.0],
        }
    )


def _run(strategy: Strategy):
    return run_backtest(
        prices=_prices(),
        strategy=strategy,
        config=BacktestConfig(
            initial_cash=10_000.0,
            execution_mode=ExecutionMode.SAME_BAR,
        ),
    )


def test_unaffordable_order_is_preserved_in_public_result() -> None:
    result = _run(_UnaffordableOrder())

    assert len(result.rejected_orders) == 1
    rejected = result.rejected_orders[0]
    assert rejected.order_id
    assert rejected.asset == "AAPL"
    assert rejected.created_at == datetime(2024, 1, 2)
    assert rejected.requested_quantity == 1_000_000.0
    assert rejected.status.value == "rejected"
    assert rejected.rejection_code == "insufficient_cash"
    assert rejected.rejection_reason
    assert result.metrics["num_orders"] == 1
    assert result.metrics["num_rejected_orders"] == 1
    assert result.fills == []
    assert result.equity_curve[-1][1] == 10_000.0


def test_no_orders_and_all_orders_rejected_are_distinguishable() -> None:
    no_orders = _run(_NoOrders())
    all_rejected = _run(_UnaffordableOrder())

    assert no_orders.metrics["num_orders"] == 0
    assert no_orders.metrics["num_rejected_orders"] == 0
    assert all_rejected.metrics["num_orders"] == 1
    assert all_rejected.metrics["num_rejected_orders"] == 1


def test_rejected_orders_round_trip_through_result_artifact(tmp_path) -> None:
    result = _run(_UnaffordableOrder())

    frame = result.to_rejected_orders_dataframe()
    assert frame.to_dicts() == [
        {
            "order_id": result.rejected_orders[0].order_id,
            "symbol": "AAPL",
            "timestamp": datetime(2024, 1, 2),
            "requested_quantity": 1_000_000.0,
            "side": "buy",
            "order_type": "market",
            "limit_price": None,
            "stop_price": None,
            "trail_amount": None,
            "parent_id": None,
            "rebalance_id": None,
            "status": "rejected",
            "rejection_code": "insufficient_cash",
            "rejection_reason": result.rejected_orders[0].rejection_reason,
        }
    ]

    result.to_parquet(tmp_path)
    loaded = type(result).from_parquet(tmp_path)
    assert loaded.to_rejected_orders_dataframe().to_dicts() == frame.to_dicts()
