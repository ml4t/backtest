"""Conditional order triggers persist and actual executions obey their price bounds."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import ExecutionMode, SlippageType
from ml4t.backtest.models import NoSlippage
from ml4t.backtest.types import OrderStatus, OrderType


def test_buy_stop_limit_remains_triggered_when_a_later_bar_reaches_only_the_limit():
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]]
    frame = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * 3,
            "open": [100.0, 110.0, 103.0],
            "high": [101.0, 112.0, 104.0],
            "low": [99.0, 109.0, 101.0],
            "close": [100.0, 111.0, 102.0],
        }
    )
    orders = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                orders.append(
                    broker.submit_order(
                        "TEST", 10, order_type=OrderType.STOP_LIMIT, stop_price=105, limit_price=104
                    )
                )

    result = Engine(
        DataFeed(prices_df=frame),
        Buy(),
        BacktestConfig(calendar=None, initial_cash=10000, execution_mode=ExecutionMode.NEXT_BAR),
    ).run()
    fills = result.to_fills_dataframe()
    assert fills["price"].to_list() == [104.0]
    assert fills["timestamp"].to_list() == [times[2]]
    assert orders[0].status is OrderStatus.FILLED
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [10000.0, 10000.0, 8960.0]


def test_sell_stop_limit_survives_gap_below_limit_then_executes_on_a_later_rebound():
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8, 9]]
    frame = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * 4,
            "open": [100.0, 100.0, 90.0, 96.0],
            "high": [101.0, 101.0, 92.0, 97.0],
            "low": [99.0, 99.0, 89.0, 95.0],
            "close": [100.0, 100.0, 91.0, 96.0],
        }
    )

    class Sell(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 10)
            elif timestamp == times[1]:
                broker.submit_order(
                    "TEST", -10, order_type=OrderType.STOP_LIMIT, stop_price=95, limit_price=94
                )

    result = Engine(
        DataFeed(prices_df=frame),
        Sell(),
        BacktestConfig(calendar=None, initial_cash=10000, execution_mode=ExecutionMode.NEXT_BAR),
    ).run()
    fills = result.to_fills_dataframe()
    assert fills["price"].to_list() == [100.0, 94.0]
    assert fills["timestamp"].to_list() == [times[1], times[3]]
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 9940.0


def test_buy_limit_cannot_execute_above_its_limit_after_slippage():
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]]

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 1, order_type=OrderType.LIMIT, limit_price=100)
            elif timestamp == times[1]:
                broker.slippage_model = NoSlippage()

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["TEST"] * 3,
                    "close": [100.0] * 3,
                }
            )
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            execution_mode=ExecutionMode.NEXT_BAR,
            slippage_type=SlippageType.PERCENTAGE,
            slippage_rate=0.01,
        ),
    ).run()
    fills = result.to_fills_dataframe()
    assert fills["price"].to_list() == [100.0]
    assert fills["timestamp"].to_list() == [times[2]]


def test_triggered_stop_partial_remainder_is_market_eligible_without_retouching_stop(tmp_path):
    from ml4t.backtest.execution import VolumeParticipationLimit
    from ml4t.backtest.result import BacktestResult

    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]]
    orders = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                orders.append(
                    broker.submit_order("TEST", 10, order_type=OrderType.STOP, stop_price=105)
                )

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["TEST"] * 3,
                    "volume": [10.0] * 3,
                    "open": [100.0, 106.0, 100.0],
                    "high": [101.0, 108.0, 102.0],
                    "low": [99.0, 106.0, 99.0],
                    "close": [100.0, 107.0, 101.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(calendar=None, initial_cash=10000, execution_mode=ExecutionMode.NEXT_BAR),
        execution_limits=VolumeParticipationLimit(max_participation=0.5),
    ).run()
    fills = result.to_fills_dataframe()
    assert fills["quantity"].to_list() == [5.0, 5.0]
    assert fills["price"].to_list() == [106.0, 100.0]
    assert fills["stop_triggered_at"].to_list() == [times[1], times[1]]
    assert orders[0].stop_triggered and orders[0].stop_triggered_at == times[1]
    result.to_parquet(tmp_path / "actual")
    loaded = BacktestResult.from_parquet(tmp_path / "actual")
    assert loaded.to_fills_dataframe()["stop_triggered_at"].to_list() == [times[1], times[1]]
    assert loaded.to_fills_dataframe()["price"].to_list() == [106.0, 100.0]


def test_execution_price_mode_does_not_trigger_a_stop_from_an_unselected_intrabar_high():
    from ml4t.backtest.config import ConditionalOrderMode, ExecutionPrice

    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]]

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 1, order_type=OrderType.STOP, stop_price=105)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["TEST"] * 3,
                    "open": [100.0, 110.0, 100.0],
                    "high": [100.0, 112.0, 108.0],
                    "low": [100.0, 98.0, 99.0],
                    "close": [100.0, 100.0, 106.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            execution_mode=ExecutionMode.NEXT_BAR,
            next_bar_execution_price=ExecutionPrice.CLOSE,
            conditional_order_mode=ConditionalOrderMode.EXECUTION_PRICE,
        ),
    ).run()
    assert result.to_fills_dataframe()["price"].to_list() == [106.0]
    assert result.to_fills_dataframe()["timestamp"].to_list() == [times[2]]


def test_rejected_limit_impact_does_not_consume_the_shared_participation_budget():
    from ml4t.backtest.config import ConditionalOrderMode, ExecutionPrice
    from ml4t.backtest.execution import SharedVolumeParticipationLimit

    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7]]
    orders = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                for quantity in [10, 10, 10, 2]:
                    orders.append(
                        broker.submit_order(
                            "TEST", quantity, order_type=OrderType.LIMIT, limit_price=100.5
                        )
                    )

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["TEST"] * 2,
                    "close": [100.0] * 2,
                    "volume": [100.0] * 2,
                }
            )
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=10000,
            execution_mode=ExecutionMode.NEXT_BAR,
            next_bar_execution_price=ExecutionPrice.CLOSE,
            conditional_order_mode=ConditionalOrderMode.EXECUTION_PRICE,
            slippage_type=SlippageType.CUMULATIVE_VOLUME_SHARE,
            slippage_rate=0.1,
        ),
        execution_limits=SharedVolumeParticipationLimit(max_participation=1),
    ).run()
    assert result.to_fills_dataframe()["quantity"].to_list() == [10.0, 10.0, 2.0]
    assert result.to_fills_dataframe()["price"].to_list() == pytest.approx([100.1, 100.4, 100.484])
    assert orders[2].status is OrderStatus.PENDING and orders[2].filled_quantity == 0
    assert result.to_portfolio_state_dataframe()["cash"][-1] == pytest.approx(7794.032)


def test_conditional_price_method_survives_strict_config_serialization():
    from ml4t.backtest.config import ConditionalOrderMode

    config = BacktestConfig(conditional_order_mode=ConditionalOrderMode.EXECUTION_PRICE)
    assert (
        BacktestConfig.from_dict(config.to_dict(), strict=True).conditional_order_mode
        is ConditionalOrderMode.EXECUTION_PRICE
    )


def test_amending_stop_price_rearms_the_trigger_without_changing_order_identity():
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8, 9, 10]]
    orders = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                orders.append(
                    broker.submit_order(
                        "TEST", 1, order_type=OrderType.STOP_LIMIT, stop_price=105, limit_price=104
                    )
                )
            elif timestamp == times[1]:
                broker.update_order(orders[0].order_id, stop_price=120)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["TEST"] * 5,
                    "open": [100.0, 110.0, 103.0, 121.0, 103.0],
                    "high": [101.0, 112.0, 104.0, 122.0, 104.0],
                    "low": [99.0, 109.0, 101.0, 120.0, 101.0],
                    "close": [100.0, 111.0, 102.0, 121.0, 102.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.NEXT_BAR),
    ).run()
    assert result.to_fills_dataframe()["timestamp"].to_list() == [times[4]]
    assert orders[0].order_id == "ORD-1" and orders[0].stop_triggered_at == times[3]


@pytest.mark.parametrize(
    "style,values",
    [
        (OrderType.LIMIT, {}),
        (OrderType.STOP, {}),
        (OrderType.STOP_LIMIT, {"stop_price": 105, "limit_price": float("nan")}),
        (OrderType.LIMIT, {"limit_price": -1}),
        (OrderType.STOP, {"stop_price": True}),
    ],
)
def test_invalid_conditional_prices_cannot_create_phantom_orders_or_consume_identifiers(
    style, values
):
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7]]
    orders = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                with pytest.raises(ValueError, match="price"):
                    broker.submit_order("TEST", 1, order_type=style, **values)
                assert broker.orders == [] and broker.get_cash() == 100000
                orders.append(broker.submit_order("TEST", 1))

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["TEST"] * 2,
                    "close": [100.0, 100.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.NEXT_BAR),
    ).run()
    assert orders[0].order_id == "ORD-1"
    assert result.to_fills_dataframe()["quantity"].to_list() == [1.0]


def test_execution_price_stop_limit_matches_published_zipline_close_cash_and_nav():
    from ml4t.backtest.config import ConditionalOrderMode, ExecutionPrice

    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8, 10]]

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order(
                    "TEST", 10, order_type=OrderType.STOP_LIMIT, stop_price=105, limit_price=104
                )

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["TEST"] * 4,
                    "open": [100.0, 110.0, 103.0, 99.0],
                    "high": [101.0, 112.0, 104.0, 100.0],
                    "low": [99.0, 109.0, 101.0, 98.0],
                    "close": [100.0, 111.0, 102.0, 99.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=10000,
            execution_mode=ExecutionMode.NEXT_BAR,
            next_bar_execution_price=ExecutionPrice.CLOSE,
            conditional_order_mode=ConditionalOrderMode.EXECUTION_PRICE,
        ),
    ).run()
    assert result.to_fills_dataframe()["price"].to_list() == [102.0]
    assert result.to_fills_dataframe()["timestamp"].to_list() == [times[2]]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [
        10000.0,
        10000.0,
        8980.0,
        8980.0,
    ]
    assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 10000.0, 10000.0, 9970.0]


@pytest.mark.parametrize("quantity", [True, float("nan"), float("inf"), "1"])
def test_invalid_order_quantity_is_rejected_before_consuming_the_next_valid_identifier(quantity):
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7]]
    ids = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                with pytest.raises(ValueError, match="quantity"):
                    broker.submit_order("TEST", quantity)
                ids.append(broker.submit_order("TEST", 1).order_id)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["TEST"] * 2,
                    "close": [100.0, 100.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.NEXT_BAR),
    ).run()
    assert ids == ["ORD-1"]
    assert result.to_fills_dataframe()["quantity"].to_list() == [1.0]


@pytest.mark.parametrize(
    "configured,override,price,source",
    [
        ("close", None, 110.0, "open"),
        ("open", "close", 120.0, "close"),
    ],
)
def test_fill_price_source_follows_actual_next_bar_selection(configured, override, price, source):
    from ml4t.backtest.config import ExecutionPrice

    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7]]

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 1)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["TEST"] * 2,
                    "open": [90.0, 110.0],
                    "close": [100.0, 120.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            execution_mode=ExecutionMode.NEXT_BAR,
            execution_price=ExecutionPrice(configured),
            next_bar_execution_price=override,
        ),
    ).run()
    assert result.to_fills_dataframe()["price"].to_list() == [price]
    assert result.to_fills_dataframe()["price_source"].to_list() == [source]


@pytest.mark.parametrize("closing,expected_index", [(106.0, 2), (103.0, 1)])
def test_ohlc_stop_limit_cannot_assume_a_reversal_from_a_pretrigger_extreme(
    closing, expected_index
):
    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]]

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order(
                    "TEST", 1, order_type=OrderType.STOP_LIMIT, stop_price=105, limit_price=104
                )

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["TEST"] * 3,
                    "open": [100.0, 100.0, 103.0],
                    "high": [101.0, 106.0, 104.0],
                    "low": [99.0, 99.0, 102.0],
                    "close": [100.0, closing, 103.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.NEXT_BAR),
    ).run()
    assert result.to_fills_dataframe()["timestamp"].to_list() == [times[expected_index]]
    assert result.to_fills_dataframe()["price"].to_list() == [104.0]


def test_unpublished_high_cannot_activate_a_stop_before_the_field_becomes_known():
    from ml4t.backtest import RuntimeServices

    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7, 8]]

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 1, order_type=OrderType.STOP, stop_price=105)

    frame = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * 3,
            "close": [100.0] * 3,
            "high": [101.0, 110.0, 106.0],
            "high_published": [times[0], times[2], times[2]],
        }
    )
    result = Engine(
        DataFeed(prices_df=frame, feed_spec={"timestamp_semantics": "bar_close"}),
        Buy(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.NEXT_BAR),
        runtime=RuntimeServices(known_at_cols={"high": "high_published"}),
    ).run()
    assert result.to_fills_dataframe()["price"].to_list() == [105.0]
    assert result.to_fills_dataframe()["timestamp"].to_list() == [times[2]]
