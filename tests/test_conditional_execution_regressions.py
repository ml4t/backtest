"""Public strategy regressions for resting limits and bracket activation."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, OrderStatus, OrderType, Strategy
from ml4t.backtest.config import (
    CommissionType,
    ExecutionMode,
    ExecutionPrice,
    FillOrdering,
    SlippageType,
)
from ml4t.backtest.execution import VolumeParticipationLimit


def _feed(opens, highs, lows, closes, *, volumes=None):
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
                **({"volume": volumes} if volumes is not None else {}),
            }
        )
    )


@pytest.mark.parametrize("ordering", list(FillOrdering))
@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("immediate_fill", [True, False])
def test_bracket_stop_cannot_trade_on_entry_observation(ordering, direction, immediate_fill):
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
            immediate_fill=immediate_fill,
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
        [favorable, None, adverse, favorable],
        [max(adverse, favorable)] * 4,
        [min(adverse, favorable)] * 4,
        [adverse, favorable, adverse, adverse],
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

    assert [fill.timestamp for fill in result.fills] == [times[3]]
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


def test_bracket_exits_wait_for_final_partial_entry_fill():
    prices = [94.0] * 4
    times, feed = _feed(prices, prices, prices, prices, volumes=[10, 10, 20, 20])

    class Bracket(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                self.orders = broker.submit_bracket(
                    "TEST", 10, take_profit=110.0, stop_loss=95.0, validate_prices=False
                )

    strategy = Bracket()
    result = Engine(
        feed,
        strategy,
        BacktestConfig(calendar=None, initial_cash=10_000, execution_mode=ExecutionMode.SAME_BAR),
        execution_limits=VolumeParticipationLimit(max_participation=0.5),
    ).run()
    entry, profit, stop = strategy.orders
    assert [(fill.order_id, fill.timestamp, fill.quantity) for fill in result.fills] == [
        (entry.order_id, times[0], 5.0),
        (entry.order_id, times[1], 5.0),
        (stop.order_id, times[2], 10.0),
    ]
    assert profit.status is OrderStatus.CANCELLED


def test_rejected_bracket_entry_cancels_children_instead_of_leaving_orphans():
    prices = [100.0, 100.0, 115.0]
    times, feed = _feed(prices, prices, prices, prices)

    class Bracket(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                self.orders = broker.submit_bracket("TEST", 10, take_profit=110.0, stop_loss=95.0)

    strategy = Bracket()
    result = Engine(
        feed,
        strategy,
        BacktestConfig(calendar=None, initial_cash=100, execution_mode=ExecutionMode.NEXT_BAR),
    ).run()
    entry, *children = strategy.orders
    assert not result.fills
    assert entry.status is OrderStatus.REJECTED
    assert [child.status for child in children] == [OrderStatus.CANCELLED] * 2
    assert [child.cancellation_reason for child in children] == ["bracket_entry_rejected"] * 2


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("slippage", [SlippageType.FIXED, SlippageType.PERCENTAGE])
def test_opted_in_limit_clamp_fills_intrabar_touch_without_breaking_limit(direction, slippage):
    levels = [100.0, 100.0, 100.0]
    times, feed = _feed(levels, [102.0] * 3, [98.0] * 3, levels)

    class Limit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                self.order = broker.submit_order(
                    "TEST",
                    direction * 10,
                    order_type=OrderType.LIMIT,
                    limit_price=100.0 - direction,
                )

    strategy = Limit()
    engine = Engine(
        feed,
        strategy,
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_short_selling=True,
            allow_leverage=True,
            execution_mode=ExecutionMode.NEXT_BAR,
            slippage_type=slippage,
            slippage_fixed=0.5,
            slippage_rate=0.005,
            limit_slippage_policy="clamp",
            commission_type=CommissionType.PER_SHARE,
            commission_per_share=0.1,
        ),
    )
    result = engine.run()
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert fill.timestamp == times[1]
    assert fill.price == 100.0 - direction
    assert fill.slippage == 0.0
    assert fill.commission == 1.0
    assert engine.broker.cash == pytest.approx(100000 - direction * 10 * fill.price - 1)


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("kind", [OrderType.STOP, OrderType.LIMIT])
def test_trade_trigger_uses_trade_then_opposite_quote_and_records_trigger(direction, kind):
    from ml4t.backtest.config import ExecutionPrice

    trade = [100.0, 100.0 + direction * 4, 100.0 + direction * 6]
    times, _ = _feed(trade, trade, trade, trade)
    feed = DataFeed(
        bid_col="bid",
        ask_col="ask",
        prices_df=pl.DataFrame(
            {
                "timestamp": times,
                "asset": ["TEST"] * 3,
                "close": trade,
                "bid": [p - 2 for p in trade],
                "ask": [p + 2 for p in trade],
            }
        ),
    )
    # A buy stop or a sell take-profit is triggered by rising trades. Reverse for falling trades.
    side = direction if kind is OrderType.STOP else -direction

    class Conditional(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                kwargs = {
                    "stop_price" if kind is OrderType.STOP else "limit_price": 100 + direction * 5
                }
                self.order = broker.submit_order("TEST", side * 10, order_type=kind, **kwargs)

    strategy = Conditional()
    engine = Engine(
        feed,
        strategy,
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_short_selling=True,
            allow_leverage=True,
            execution_mode=ExecutionMode.NEXT_BAR,
            execution_price=ExecutionPrice.QUOTE_SIDE,
            conditional_order_mode="trade_quote",
        ),
    )
    result = engine.run()
    fills = result.fills
    # The quote crosses at event 1, but the trade does not. A limit quote outside its bound defers.
    if kind is OrderType.LIMIT:
        assert not fills
    else:
        assert len(fills) == 1
        assert fills[0].timestamp == times[2]
        assert fills[0].price == trade[2] + side * 2
        assert fills[0].trigger_price == trade[2]
        assert engine.broker.cash == pytest.approx(100000 - side * 10 * fills[0].price)


@pytest.mark.parametrize("direction", [1, -1])
def test_trade_quote_stop_partial_fills_keep_original_trade_trigger_and_can_cancel(
    direction, tmp_path
):
    from ml4t.backtest.result import BacktestResult

    trades = [100.0, 100.0 + direction * 6, 100.0, 100.0]
    times, _ = _feed(trades, trades, trades, trades)
    frame = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * 4,
            "close": trades,
            "bid": [p - 0.5 for p in trades],
            "ask": [p + 0.5 for p in trades],
            "volume": [10.0] * 4,
        }
    )

    class Stop(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                self.order = broker.submit_order(
                    "TEST",
                    direction * 10,
                    order_type=OrderType.STOP,
                    stop_price=100 + direction * 5,
                )
            if timestamp == times[2]:
                assert broker.cancel_order(self.order.order_id)

    strategy = Stop()
    engine = Engine(
        DataFeed(prices_df=frame, bid_col="bid", ask_col="ask"),
        strategy,
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_short_selling=True,
            allow_leverage=True,
            conditional_order_mode="trade_quote",
            execution_mode=ExecutionMode.NEXT_BAR,
            commission_type=CommissionType.PER_ORDER_PER_SHARE,
            commission_per_share=0.1,
            commission_minimum=1.0,
        ),
        execution_limits=VolumeParticipationLimit(max_participation=0.3),
    )
    result = engine.run()
    assert [f.quantity for f in result.fills] == [3.0, 3.0]
    assert [f.price for f in result.fills] == [
        trades[1] + direction * 0.5,
        trades[2] + direction * 0.5,
    ]
    assert [f.trigger_price for f in result.fills] == [trades[1]] * 2
    assert [f.stop_triggered_at for f in result.fills] == [times[1]] * 2
    assert sum(f.commission for f in result.fills) == 1.0
    assert strategy.order.status is OrderStatus.CANCELLED
    assert engine.broker.get_position("TEST").quantity == direction * 6
    assert engine.broker.cash == pytest.approx(
        100000 - direction * sum(f.quantity * f.price for f in result.fills) - 1
    )
    result.to_parquet(tmp_path / "run")
    restored = BacktestResult.from_parquet(tmp_path / "run")
    assert [f.trigger_price for f in restored.fills] == [trades[1]] * 2


def test_trade_quote_missing_quote_fails_without_an_invented_execution():
    trades = [100.0, 106.0]
    times, feed = _feed(trades, trades, trades, trades)

    class Stop(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 1, order_type=OrderType.STOP, stop_price=105)

    engine = Engine(
        feed, Stop(), BacktestConfig(calendar=None, conditional_order_mode="trade_quote")
    )
    with pytest.raises(ValueError, match="observed opposite quote"):
        engine.run()
    assert not engine.broker.fills


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("policy", ["defer", "clamp"])
def test_intrabar_bracket_profit_respects_limit_policy_and_cancels_sibling(direction, policy):
    trades = [100.0, 100.0, 100.0]
    times, feed = _feed(trades, [102.0] * 3, [98.0] * 3, trades)

    class Bracket(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                self.orders = broker.submit_bracket(
                    "TEST",
                    direction * 10,
                    take_profit=100 + direction,
                    stop_loss=100 - direction * 5,
                    validate_prices=False,
                )
            if timestamp == times[1] and policy == "defer":
                broker.cancel_order(self.orders[1].order_id)
                broker.cancel_order(self.orders[2].order_id)

    strategy = Bracket()
    engine = Engine(
        feed,
        strategy,
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_short_selling=True,
            allow_leverage=True,
            execution_mode=ExecutionMode.SAME_BAR,
            slippage_type=SlippageType.FIXED,
            slippage_fixed=0.1,
            limit_slippage_policy=policy,
            commission_type=CommissionType.PER_SHARE,
            commission_per_share=0.1,
        ),
    )
    result = engine.run()
    entry, profit, stop = strategy.orders
    assert stop.status is OrderStatus.CANCELLED
    if policy == "defer":
        assert len(result.fills) == 1
        assert profit.status is OrderStatus.CANCELLED
    else:
        assert profit.status is OrderStatus.FILLED
        assert [f.price for f in result.fills] == [100 + direction * 0.1, 100 + direction]
        assert [f.timestamp for f in result.fills] == times[:2]
        assert engine.broker.get_position("TEST") is None
        assert engine.broker.cash == pytest.approx(100000 + 10 * 0.9 - 2)


def test_conditional_execution_options_roundtrip_without_changing_defaults():
    config = BacktestConfig(
        calendar=None, conditional_order_mode="trade_quote", limit_slippage_policy="clamp"
    )
    restored = BacktestConfig.from_dict(config.to_dict())
    assert restored.conditional_order_mode.value == "trade_quote"
    assert restored.limit_slippage_policy.value == "clamp"
    assert BacktestConfig().limit_slippage_policy.value == "defer"


@pytest.mark.parametrize("direction", [1, -1])
def test_trade_quote_bracket_profit_fills_valid_quote_and_cancels_stop(direction):
    prices = [100.0, 100.0 + direction * 4, 100.0 + direction * 7]
    times, _ = _feed(prices, prices, prices, prices)
    frame = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["TEST"] * 3,
            "close": prices,
            "bid": [p - 0.5 for p in prices],
            "ask": [p + 0.5 for p in prices],
        }
    )

    class Bracket(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                self.orders = broker.submit_bracket(
                    "TEST",
                    direction * 10,
                    take_profit=100 + direction * 5,
                    stop_loss=100 - direction * 10,
                    validate_prices=False,
                )

    strategy = Bracket()
    engine = Engine(
        DataFeed(prices_df=frame, bid_col="bid", ask_col="ask"),
        strategy,
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_short_selling=True,
            allow_leverage=True,
            execution_mode=ExecutionMode.SAME_BAR,
            execution_price=ExecutionPrice.QUOTE_SIDE,
            conditional_order_mode="trade_quote",
            commission_type=CommissionType.PER_SHARE,
            commission_per_share=0.1,
        ),
    )
    result = engine.run()
    entry, profit, stop = strategy.orders
    assert profit.status is OrderStatus.FILLED
    assert stop.status is OrderStatus.CANCELLED
    assert result.fills[-1].price == prices[2] - direction * 0.5
    assert result.fills[-1].trigger_price == prices[2]
    assert engine.broker.get_position("TEST") is None
    assert engine.broker.cash == pytest.approx(100000 + 10 * 6 - 2)
