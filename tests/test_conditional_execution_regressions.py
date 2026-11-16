"""Public strategy regressions for resting limits and bracket activation."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, OrderStatus, OrderType, Strategy
from ml4t.backtest.config import ExecutionMode, FillOrdering, SlippageType


def _feed(opens, highs, lows, closes):
    times = [datetime(2025, 1, 6, 15, tzinfo=UTC) + timedelta(minutes=i) for i in range(len(opens))]
    return times, DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": times,
                "asset": ["TEST"] * len(times),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
            }
        )
    )


@pytest.mark.parametrize("ordering", list(FillOrdering))
@pytest.mark.parametrize("direction", [1, -1])
def test_bracket_stop_cannot_trade_on_entry_observation(ordering, direction):
    trigger_price = 94.0 if direction == 1 else 106.0
    prices = [trigger_price] * 3
    times, feed = _feed(prices, prices, prices, prices)

    class Bracket(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                self.orders = broker.submit_bracket(
                    "TEST",
                    direction * 19,
                    100.0 + direction * 10,
                    100.0 - direction * 5,
                    validate_prices=False,
                )

    strategy = Bracket()
    result = Engine(
        feed,
        strategy,
        BacktestConfig(
            calendar=None,
            initial_cash=100_000,
            allow_short_selling=True,
            allow_leverage=True,
            fill_ordering=ordering,
            execution_mode=ExecutionMode.SAME_BAR,
        ),
    ).run()

    entry, profit, stop = strategy.orders
    assert [(fill.order_id, fill.timestamp) for fill in result.fills] == [
        (entry.order_id, times[0]),
        (stop.order_id, times[1]),
    ]
    assert profit.status is OrderStatus.CANCELLED


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("slippage", [SlippageType.FIXED, SlippageType.PERCENTAGE])
def test_resting_limit_uses_only_causal_favorable_open(direction, slippage):
    limit = 100.0
    adverse = limit + direction * 2
    favorable = limit - direction * 2
    times, feed = _feed(
        [favorable, adverse, favorable],
        [max(adverse, favorable)] * 3,
        [min(adverse, favorable)] * 3,
        [adverse] * 3,
    )

    class RestingLimit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order(
                    "TEST", direction, order_type=OrderType.LIMIT, limit_price=limit
                )

    result = Engine(
        feed,
        RestingLimit(),
        BacktestConfig(
            calendar=None,
            initial_cash=100_000,
            allow_short_selling=True,
            allow_leverage=True,
            execution_mode=ExecutionMode.SAME_BAR,
            slippage_type=slippage,
            slippage_fixed=0.5,
            slippage_rate=0.005,
        ),
    ).run()

    assert [fill.timestamp for fill in result.fills] == [times[2]]
    adjustment = 0.5 if slippage is SlippageType.FIXED else favorable * 0.005
    assert result.fills[0].price == pytest.approx(favorable + direction * adjustment)
    assert result.fills[0].price_source == "open"


@pytest.mark.parametrize("cancel_parent", [True, False])
def test_unfilled_bracket_entry_never_exposes_its_take_profit(cancel_parent):
    opens = [100.0, 110.0, 110.0]
    times, feed = _feed(opens, opens, opens, opens)

    class RestingBracket(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                self.orders = broker.submit_bracket(
                    "TEST",
                    5,
                    take_profit=105.0,
                    stop_loss=90.0,
                    entry_type=OrderType.LIMIT,
                    entry_limit=95.0,
                )
            if timestamp == times[1] and cancel_parent:
                assert broker.cancel_order(self.orders[0].order_id)

    strategy = RestingBracket()
    result = Engine(
        feed,
        strategy,
        BacktestConfig(
            calendar=None,
            initial_cash=100_000,
            allow_short_selling=True,
            allow_leverage=True,
            execution_mode=ExecutionMode.NEXT_BAR,
        ),
    ).run()
    assert not result.fills
    expected = OrderStatus.CANCELLED if cancel_parent else OrderStatus.PENDING
    assert [order.status for order in strategy.orders] == [expected] * 3
    if cancel_parent:
        assert [order.cancellation_reason for order in strategy.orders[1:]] == [
            "bracket_entry_cancelled",
            "bracket_entry_cancelled",
        ]
