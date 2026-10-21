"""Custom execution models cannot spend unavailable cash or corrupt the native ledger."""

from datetime import UTC, datetime
from itertools import count

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import ExecutionMode, ShareType, ShortCashPolicy
from ml4t.backtest.execution import SharedVolumeParticipationLimit, VolumeParticipationLimit
from ml4t.backtest.models import VolatilityVolumeSlippage
from ml4t.backtest.types import OrderStatus


def _prices(days=(6,)):
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": [datetime(2025, 1, day, 21, tzinfo=UTC) for day in days],
                "asset": ["A"] * len(days),
                "open": [101.0] * len(days),
                "high": [102.0] * len(days),
                "low": [100.0] * len(days),
                "close": [101.0] * len(days),
                "volume": [20.0] * len(days),
            }
        ),
        feed_spec={
            "time_contract_version": 2,
            "calendar": None,
            "timestamp_semantics": "event_time",
        },
    )


class _Buy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        broker.submit_order("A", 1)


def _unchanged_account(engine):
    assert engine.broker.cash == 10000
    assert engine.broker.positions == {}
    assert engine.broker.fills == []


@pytest.mark.parametrize("capacity", [-1.0, float("nan")])
def test_invalid_custom_shared_capacity_cannot_silently_discard_or_overfill_an_order(capacity):
    class BrokenCapacity(SharedVolumeParticipationLimit):
        def bar_capacity(self, bar_volume, price):
            return capacity

    engine = Engine(
        _prices(),
        _Buy(),
        BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        execution_limits=BrokenCapacity(0.5),
    )
    with pytest.raises(ValueError, match="capacity must be finite and nonnegative"):
        engine.run()
    _unchanged_account(engine)


def test_noncloneable_stateful_price_model_is_rejected_before_a_quote_consumes_its_state():
    class Noncloneable:
        calls = 0

        def __deepcopy__(self, memo):
            return self

        def calculate(self, asset, quantity, price, volume):
            self.calls += 1
            return 0.1

    model = Noncloneable()
    engine = Engine(
        _prices(),
        _Buy(),
        BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        slippage_model=model,
    )
    with pytest.raises(TypeError, match="independent deepcopy quotes"):
        engine.run()
    assert model.calls == 0
    _unchanged_account(engine)


def test_shared_external_price_model_state_cannot_change_the_committed_price_after_its_quote():
    external_state = count(1)

    class SharedState:
        def calculate(self, asset, quantity, price, volume):
            return next(external_state) / 10

    engine = Engine(
        _prices(),
        _Buy(),
        BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        slippage_model=SharedState(),
    )
    with pytest.raises(ValueError, match="quoted deterministic cost"):
        engine.run()
    _unchanged_account(engine)


@pytest.mark.parametrize("services", [False, True])
def test_late_observed_ohlc_model_requires_explicit_runtime_instead_of_fabricated_observations(
    services,
):
    class Configure(_Buy):
        def on_start(self, broker):
            broker.slippage_model = VolatilityVolumeSlippage(0.05)

    engine = Engine(
        _prices(),
        Configure(),
        BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        runtime=RuntimeServices() if services else None,
    )
    if not services:
        with pytest.raises(ValueError, match="active raw observation service"):
            engine.run()
        _unchanged_account(engine)
    else:
        result = engine.run()
        assert [fill.price for fill in result.fills] == pytest.approx([101.07575])
        assert result.to_portfolio_state_dataframe()["cash"].to_list() == pytest.approx(
            [9898.92425]
        )


@pytest.mark.parametrize("reject", [False, True])
def test_unleveraged_locked_short_cover_cannot_spend_more_than_cash_at_its_final_price(reject):
    class Cost:
        calls = 0

        def calculate(self, asset, quantity, price, volume):
            self.calls += 1
            return 0.0 if self.calls == 1 else 150.0

    class Cover(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 6:
                broker.submit_order("A", -10)
            elif timestamp.day == 7:
                broker.submit_order("A", 10)

    model = Cost()
    engine = Engine(
        _prices((6, 7, 8)),
        Cover(),
        BacktestConfig(
            initial_cash=1010,
            calendar=None,
            execution_mode=ExecutionMode.NEXT_BAR,
            allow_short_selling=True,
            allow_leverage=False,
            short_cash_policy=ShortCashPolicy.LOCK_NOTIONAL,
            reject_on_insufficient_cash=reject,
        ),
        slippage_model=model,
    )
    result = engine.run()
    assert [fill.quantity for fill in result.fills] == [10]
    assert model.calls == 1
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [1010, 2020, 2020]
    assert result.to_portfolio_state_dataframe()["equity"].to_list() == [1010] * 3
    engine_orders = result.rejected_orders
    assert engine.broker.get_orders()[-1].status is (
        OrderStatus.REJECTED if reject else OrderStatus.CANCELLED
    )
    if reject:
        assert engine_orders[0].status is OrderStatus.REJECTED


def test_fractional_volume_limited_short_flip_opens_only_the_units_actually_filled():
    class Flip(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("A", -10 if timestamp.day == 6 else 20)

    engine = Engine(
        _prices((6, 7)),
        Flip(),
        BacktestConfig(
            initial_cash=1010,
            calendar=None,
            execution_mode=ExecutionMode.SAME_BAR,
            share_type=ShareType.FRACTIONAL,
            allow_short_selling=True,
            allow_leverage=False,
            short_cash_policy=ShortCashPolicy.LOCK_NOTIONAL,
            partial_fills_allowed=True,
        ),
        execution_limits=VolumeParticipationLimit(0.7),
    )
    result = engine.run()
    assert [fill.quantity for fill in result.fills] == [10, 14]
    assert engine.broker.get_position("A").quantity == 4
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [2020, 606]
    assert result.to_portfolio_state_dataframe()["equity"].to_list() == [1010, 1010]
