from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, Order, Strategy, run_backtest
from ml4t.backtest.config import ShareType
from ml4t.backtest.execution.limits import VolumeParticipationLimit
from ml4t.backtest.types import ExecutionMode, OrderSide, OrderStatus


class _UnaffordableOrder(Strategy):
    def __init__(self, quantity: float = 1_000_000.0) -> None:
        self.quantity = quantity
        self.submitted = False

    def on_data(self, timestamp, data, context, broker) -> None:
        if not self.submitted:
            broker.submit_order("AAPL", self.quantity)
            self.submitted = True


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


def _partial_then_unaffordable_prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "asset": ["AAPL", "AAPL"],
            "open": [50.0, 1_000.0],
            "high": [50.0, 1_000.0],
            "low": [50.0, 1_000.0],
            "close": [50.0, 1_000.0],
            "volume": [5.0, 1_000.0],
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
            "filled_quantity": 0.0,
            "remaining_quantity": 1_000_000.0,
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

    loaded.rejected_orders[0].rejection_reason = "Short selling not allowed"
    assert loaded.rejected_orders[0].rejection_code == "insufficient_cash"


def test_partially_filled_then_rejected_order_is_reconcilable() -> None:
    result = run_backtest(
        prices=_partial_then_unaffordable_prices(),
        strategy=_UnaffordableOrder(15.5),
        config=BacktestConfig(
            initial_cash=1_000.0,
            execution_mode=ExecutionMode.SAME_BAR,
            share_type=ShareType.INTEGER,
            partial_fills_allowed=True,
        ),
        execution_limits=VolumeParticipationLimit(max_participation=1.0),
    )

    assert len(result.fills) == 1
    assert len(result.rejected_orders) == 1
    rejected = result.rejected_orders[0]
    assert rejected.requested_quantity == 15.5
    assert rejected.filled_quantity == 5.0
    assert rejected.quantity == 10.0

    record = result.to_rejected_orders_dataframe().to_dicts()[0]
    assert record["requested_quantity"] == 15.5
    assert record["filled_quantity"] == 5.0
    assert record["remaining_quantity"] == 10.0


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("Insufficient cash to cover short", "insufficient_cash"),
        ("Insufficient buying power", "insufficient_buying_power"),
        ("Short selling not allowed", "account_restriction"),
        ("No price available", "price_unavailable"),
    ],
)
def test_rejection_code_classification(reason: str, expected: str) -> None:
    order = Order(
        asset="AAPL",
        side=OrderSide.BUY,
        quantity=1.0,
        status=OrderStatus.REJECTED,
        rejection_reason=reason,
    )

    assert order.rejection_code == expected
