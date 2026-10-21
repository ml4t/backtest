"""A declared asset/bar liquidity budget is shared by every actual execution."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, ExecutionMode, ExecutionPrice, FillOrdering
from ml4t.backtest.execution import SharedVolumeParticipationLimit, VolumeParticipationLimit


@pytest.mark.parametrize("shared,expected", [(False, [10.0, 10.0]), (True, [5.0, 5.0])])
def test_two_competing_orders_cannot_each_spend_the_same_shared_volume(shared, expected):
    times = [datetime(2025, 1, d, 15, tzinfo=UTC) for d in [2, 3]]
    frame = pl.DataFrame(
        {"timestamp": times, "asset": ["TEST"] * 2, "close": [100.0] * 2, "volume": [10.0] * 2}
    )

    class Orders(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 10)
                broker.submit_order("TEST", 10)

    limit = (SharedVolumeParticipationLimit if shared else VolumeParticipationLimit)(
        max_participation=0.5
    )
    result = Engine(
        DataFeed(prices_df=frame),
        Orders(),
        BacktestConfig(
            initial_cash=10000,
            calendar=None,
            execution_mode=ExecutionMode.SAME_BAR,
            execution_price=ExecutionPrice.CLOSE,
            commission_type=CommissionType.NONE,
        ),
        execution_limits=limit,
    ).run()
    assert (
        result.to_fills_dataframe()
        .group_by("timestamp")
        .agg(pl.col("quantity").sum())
        .sort("timestamp")["quantity"]
        .to_list()
        == expected
    )
    if shared:
        assert result.to_fills_dataframe()["order_id"].to_list() == ["ORD-1", "ORD-1"]
        assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 10000.0]


def test_shared_volume_is_per_asset_and_consumes_both_directions():
    instant = datetime(2025, 1, 2, 15, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "timestamp": [instant] * 2,
            "asset": ["TEST", "OTHER"],
            "close": [100.0, 100.0],
            "volume": [10.0, 10.0],
        }
    )

    class Orders(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("TEST", 4)
            broker.submit_order("TEST", -4)
            broker.submit_order("OTHER", 10)

    result = Engine(
        DataFeed(prices_df=frame),
        Orders(),
        BacktestConfig(
            initial_cash=10000,
            calendar=None,
            execution_mode=ExecutionMode.SAME_BAR,
            execution_price=ExecutionPrice.CLOSE,
            commission_type=CommissionType.NONE,
            fill_ordering=FillOrdering.FIFO,
        ),
        execution_limits=SharedVolumeParticipationLimit(max_participation=0.5),
    ).run()
    fills = result.to_fills_dataframe()
    assert fills["quantity"].to_list() == [4.0, 1.0, 5.0]
    assert fills["signed_quantity"].to_list() == [4.0, -1.0, 5.0]
    assert fills["asset"].to_list() == ["TEST", "TEST", "OTHER"]


def test_failed_immediate_callback_restores_shared_budget_with_cash_and_fills():
    class Failure(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("TEST", 10)
            assert len(broker.fills) == 1
            raise RuntimeError("decision failed")

    frame = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, 2, 15, tzinfo=UTC)],
            "asset": ["TEST"],
            "close": [100.0],
            "volume": [10.0],
        }
    )
    engine = Engine(
        DataFeed(prices_df=frame),
        Failure(),
        BacktestConfig(
            initial_cash=10000,
            calendar=None,
            execution_mode=ExecutionMode.SAME_BAR,
            execution_price=ExecutionPrice.CLOSE,
            commission_type=CommissionType.NONE,
            immediate_fill=True,
        ),
        execution_limits=SharedVolumeParticipationLimit(max_participation=0.5),
    )
    with pytest.raises(RuntimeError, match="decision failed"):
        engine.run()
    assert engine.broker.cash == 10000
    assert engine.broker.fills == []
    assert engine.broker.positions == {}
    assert engine.broker._order_state.volume_consumed_by_asset == {}


def test_shared_traded_volume_fraction_is_not_applied_again_to_quote_depth():
    instant = datetime(2025, 1, 2, 15, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "timestamp": [instant],
            "asset": ["TEST"],
            "close": [100.0],
            "volume": [100.0],
            "bid": [99.0],
            "ask": [101.0],
            "bid_size": [1.0],
            "ask_size": [1.0],
        }
    )

    class Orders(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("TEST", 10)

    result = Engine(
        DataFeed(
            prices_df=frame,
            bid_col="bid",
            ask_col="ask",
            bid_size_col="bid_size",
            ask_size_col="ask_size",
        ),
        Orders(),
        BacktestConfig(
            initial_cash=10000,
            calendar=None,
            execution_mode=ExecutionMode.SAME_BAR,
            execution_price=ExecutionPrice.QUOTE_SIDE,
            commission_type=CommissionType.NONE,
        ),
        execution_limits=SharedVolumeParticipationLimit(max_participation=0.5),
    ).run()
    assert result.to_fills_dataframe()["quantity"].to_list() == [1.0]
    assert result.to_fills_dataframe()["price"].to_list() == [101.0]
    assert result.to_equity_dataframe()["equity"].to_list() == [9999.0]
