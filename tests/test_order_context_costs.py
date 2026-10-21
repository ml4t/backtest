"""Partial fills must apply the selected fee basis without consuming quotes as fills."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, ExecutionMode, ExecutionPrice, SlippageType
from ml4t.backtest.execution import VolumeParticipationLimit


@pytest.mark.parametrize(
    "method,fees,nav",
    [
        ("per_share", [1.0, 1.0], 10003.0),
        ("per_order_per_share", [1.0, 0.0], 10004.0),
    ],
)
def test_partial_order_minimum_has_an_explicit_native_and_zipline_compatible_basis(
    method, fees, nav
):
    times = [datetime(2025, 1, day, 21, tzinfo=UTC) for day in [6, 7, 8]]
    frame = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * 3,
            "open": [102.0, 103.0, 104.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [10.0] * 3,
        }
    )

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 10)

    cfg = BacktestConfig(
        initial_cash=10000,
        calendar=None,
        execution_mode=ExecutionMode.NEXT_BAR,
        next_bar_execution_price=ExecutionPrice.CLOSE,
        commission_type=CommissionType(method),
        commission_per_share=0.01,
        commission_minimum=1.0,
        slippage_type=SlippageType.NONE,
    )
    engine = Engine(
        DataFeed(prices_df=frame),
        Buy(),
        cfg,
        execution_limits=VolumeParticipationLimit(max_participation=0.5),
    )
    result = engine.run()
    fills = result.to_fills_dataframe()
    assert fills["quantity"].to_list() == [5.0, 5.0]
    assert fills["timestamp"].to_list() == times[1:]
    assert fills["price"].to_list() == [105.0, 106.0]
    assert fills["commission"].to_list() == fees
    assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 9999.0, nav]
    if method == "per_order_per_share":
        assert engine.broker.orders[0].commission_paid == 1.0


def test_gatekeeper_estimates_do_not_advance_a_stateful_model_before_the_actual_fill():
    class StatefulFee:
        def __init__(self):
            self.calls = 0

        def calculate(self, asset, quantity, price):
            self.calls += 1
            return float(self.calls)

    model = StatefulFee()

    class Buy(Strategy):
        def on_start(self, broker):
            broker.commission_model = model
            broker.gatekeeper.commission_model = model

        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("TEST", 1)

    frame = pl.DataFrame(
        {"timestamp": [datetime(2025, 1, 2, 21, tzinfo=UTC)], "asset": ["TEST"], "close": [100.0]}
    )
    result = Engine(
        DataFeed(prices_df=frame),
        Buy(),
        BacktestConfig(
            initial_cash=10000,
            calendar=None,
            execution_mode=ExecutionMode.SAME_BAR,
            execution_price=ExecutionPrice.CLOSE,
        ),
    ).run()
    assert model.calls == 1
    assert result.to_fills_dataframe()["commission"].to_list() == [1.0]
    assert result.to_equity_dataframe()["equity"].to_list() == [9999.0]


def test_cumulative_notional_fee_uses_real_fill_prices_and_contract_multiplier_after_quotes():
    from ml4t.backtest.types import AssetClass, ContractSpec

    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]]
    frame = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * 3,
            "close": [100.0, 100.0, 110.0],
            "volume": [10.0] * 3,
        }
    )

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 10)
            for pending in broker.pending_orders:
                for _ in range(3):
                    assert broker.gatekeeper.validate_order(pending, data["TEST"]["close"])[0]

    cfg = BacktestConfig(
        initial_cash=100000,
        calendar=None,
        execution_mode=ExecutionMode.NEXT_BAR,
        next_bar_execution_price=ExecutionPrice.CLOSE,
        commission_type=CommissionType.PER_ORDER,
        commission_rate=0.001,
        commission_per_share=2.0,
        commission_per_trade=3.0,
        commission_minimum=0.0,
    )
    cfg = BacktestConfig.from_dict(cfg.to_dict(), strict=True)
    engine = Engine(
        DataFeed(prices_df=frame),
        Buy(),
        cfg,
        contract_specs={
            "TEST": ContractSpec("TEST", asset_class=AssetClass.FUTURE, multiplier=50.0)
        },
        execution_limits=VolumeParticipationLimit(max_participation=0.5),
    )
    result = engine.run()
    assert result.to_fills_dataframe()["quantity"].to_list() == [5.0, 5.0]
    assert result.to_fills_dataframe()["commission"].to_list() == [38.0, 37.5]
    assert result.to_equity_dataframe()["equity"].to_list() == [100000.0, 99962.0, 102424.5]
    assert engine.broker.orders[0].commission_paid == 75.5
