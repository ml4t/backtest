"""Native controls must reject actual Engine orders before they consume broker state."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import ExecutionMode


def _engine(
    strategy,
    *,
    calendar=None,
    times=None,
    execution_mode=ExecutionMode.NEXT_BAR,
    immediate_fill=False,
):
    times = times or [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [2, 3]]
    frame = pl.DataFrame(
        {"timestamp": times, "asset": ["TEST"] * len(times), "close": [100.0] * len(times)}
    )
    return Engine(
        DataFeed(prices_df=frame, feed_spec={"timestamp_semantics": "event_time"}),
        strategy,
        BacktestConfig(
            calendar=calendar,
            initial_cash=10000,
            execution_mode=execution_mode,
            immediate_fill=immediate_fill,
        ),
    )


def test_order_controls_reject_before_creating_an_order_or_consuming_an_identifier():
    from ml4t.backtest.controls import OrderConstraints, TradingControlViolation

    seen = []

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderConstraints(max_order_quantity=2))

        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 2:
                with pytest.raises(TradingControlViolation, match="quantity"):
                    broker.submit_order("TEST", 3)
                assert broker.orders == [] and broker.get_cash() == 10000
                seen.append(broker.submit_order("TEST", 2).order_id)

    result = _engine(Submit()).run()
    assert seen == ["ORD-1"]
    assert result.to_fills_dataframe()["quantity"].to_list() == [2.0]


@pytest.mark.parametrize("include_pending,second_allowed", [(True, False), (False, True)])
def test_position_control_distinguishes_pending_exposure_from_the_legacy_held_position_method(
    include_pending, second_allowed
):
    from ml4t.backtest.controls import OrderConstraints, TradingControlViolation

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(
                OrderConstraints(max_position_quantity=10, include_pending=include_pending)
            )

        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 2:
                broker.submit_order("TEST", 7)
                if second_allowed:
                    broker.submit_order("TEST", 4)
                else:
                    with pytest.raises(TradingControlViolation, match="position"):
                        broker.submit_order("TEST", 4)

    result = _engine(Submit()).run()
    assert result.to_fills_dataframe()["quantity"].to_list() == (
        [7.0, 4.0] if second_allowed else [7.0]
    )


def test_session_order_count_uses_one_cme_session_across_utc_midnight_and_resets_next_session():
    from ml4t.backtest.controls import OrderConstraints, TradingControlViolation

    times = [
        datetime(2025, 7, 6, 23, 30, tzinfo=UTC),
        datetime(2025, 7, 7, 0, 30, tzinfo=UTC),
        datetime(2025, 7, 7, 23, 30, tzinfo=UTC),
    ]
    seen = []

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderConstraints(max_orders_per_session=1))

        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[1]:
                with pytest.raises(TradingControlViolation, match="session"):
                    broker.submit_order("TEST", 1)
            else:
                seen.append(broker.submit_order("TEST", 1).order_id)

    _engine(Submit(), calendar="CME_Equity", times=times).run()
    assert seen == ["ORD-1", "ORD-2"]


def test_calendar_free_session_control_does_not_silently_count_civil_days():
    from ml4t.backtest.controls import OrderConstraints

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderConstraints(max_orders_per_session=1))

        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("TEST", 1)

    with pytest.raises(ValueError, match="calendar"):
        _engine(Submit()).run()


def test_failed_callback_restores_order_control_counts_together_with_pending_orders():
    from ml4t.backtest.controls import OrderConstraints

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderConstraints(max_orders_per_day=1))

        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("TEST", 1)
            raise RuntimeError("abort decision")

    engine = _engine(Submit())
    with pytest.raises(RuntimeError, match="abort decision"):
        engine.run()
    assert engine.broker.orders == [] and engine.broker.pending_orders == []
    assert engine.broker._order_state.control_counts == {}


def test_failed_callback_cannot_leave_registered_order_or_account_controls_active():
    from ml4t.backtest.controls import OrderConstraints

    class RejectAccount:
        def validate(self, account):
            raise ValueError("uncommitted account control")

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.register_trading_control(OrderConstraints(max_order_quantity=0.5))
            broker.register_account_control(RejectAccount())
            raise RuntimeError("abort control registration")

    engine = _engine(Submit())
    with pytest.raises(RuntimeError, match="abort control registration"):
        engine.run()
    engine.broker.validate_account_controls()
    order = engine.broker.submit_order("TEST", 1)
    assert order is not None and order.order_id == "ORD-1"


def test_long_only_and_dated_restrictions_see_current_positions_and_the_actual_order_time():
    from ml4t.backtest.controls import OrderConstraints, TradingControlViolation

    seen = []

    def restrict(request):
        return request.timestamp.day == 3

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderConstraints(long_only=True, restriction=restrict))

        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 2:
                with pytest.raises(TradingControlViolation, match="long-only"):
                    broker.submit_order("TEST", -1)
                broker.submit_order("TEST", 2)
            else:
                with pytest.raises(TradingControlViolation, match="restricted"):
                    broker.submit_order("TEST", 1)
                seen.append(broker.get_position("TEST").quantity)

    _engine(Submit()).run()
    assert seen == [2.0]


@pytest.mark.parametrize(
    "boundary,expected", [("after_callbacks", [0.0, 0.2]), ("after_fills", [0.2, 0.2])]
)
def test_account_control_boundary_distinguishes_pending_orders_from_actual_fills(
    boundary, expected
):
    from ml4t.backtest.controls import AccountConstraints

    seen = []

    class Capture(AccountConstraints):
        def validate(self, state):
            seen.append(state.leverage)
            super().validate(state)

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_account_control(Capture(max_leverage=1), boundary=boundary)

        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 2:
                broker.submit_order("TEST", 20)

    engine = _engine(Submit(), execution_mode=ExecutionMode.SAME_BAR)
    engine.run()
    assert seen == expected


def test_minimum_leverage_binds_only_after_the_declared_deadline():
    from ml4t.backtest.controls import AccountConstraints, AccountControlViolation

    class Observe(Strategy):
        def on_start(self, broker):
            broker.register_account_control(
                AccountConstraints(min_leverage=0.5, deadline=datetime(2025, 1, 2, 21, tzinfo=UTC))
            )

        def on_data(self, *args):
            pass

    engine = _engine(Observe())
    with pytest.raises(AccountControlViolation, match="minimum"):
        engine.run()
    assert [t for t, _ in engine.equity_curve] == [datetime(2025, 1, 2, 21, tzinfo=UTC)]


def test_updating_a_partially_filled_order_changes_its_remaining_executable_quantity():
    from ml4t.backtest.execution import VolumeParticipationLimit

    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [2, 3, 6]]
    frame = pl.DataFrame(
        {"timestamp": times, "asset": ["TEST"] * 3, "close": [100.0] * 3, "volume": [10.0] * 3}
    )
    ids = []

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                ids.append(broker.submit_order("TEST", 10).order_id)
            elif timestamp == times[1]:
                assert broker.get_position("TEST").quantity == 5
                assert broker.update_order(ids[0], quantity=2)

    result = Engine(
        DataFeed(prices_df=frame),
        Submit(),
        BacktestConfig(calendar=None, initial_cash=10000, execution_mode=ExecutionMode.NEXT_BAR),
        execution_limits=VolumeParticipationLimit(max_participation=0.5),
    ).run()
    assert result.to_fills_dataframe()["quantity"].to_list() == [5.0, 2.0]
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [10000.0, 9500.0, 9300.0]


def test_order_updates_cannot_bypass_size_controls_or_consume_another_order_count():
    from ml4t.backtest.controls import OrderConstraints, TradingControlViolation

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(
                OrderConstraints(
                    max_order_quantity=5, max_position_quantity=5, max_orders_per_day=1
                )
            )

        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 2:
                order = broker.submit_order("TEST", 3)
                with pytest.raises(TradingControlViolation, match="quantity"):
                    broker.update_order(order.order_id, quantity=6)
                assert order.quantity == 3
                assert broker.update_order(order.order_id, quantity=5)
                assert order.quantity == 5

    result = _engine(Submit()).run()
    assert result.to_fills_dataframe()["quantity"].to_list() == [5.0]


def test_pending_buys_do_not_authorize_a_short_sale_before_the_buys_fill():
    from ml4t.backtest.controls import OrderConstraints, TradingControlViolation

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderConstraints(long_only=True))

        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 2:
                broker.submit_order("TEST", 5)
                with pytest.raises(TradingControlViolation, match="long-only"):
                    broker.submit_order("TEST", -1)

    _engine(Submit()).run()


def test_opposing_pending_orders_do_not_hide_the_worst_case_position_size():
    from ml4t.backtest.controls import OrderConstraints, TradingControlViolation

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderConstraints(max_position_quantity=10))

        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 2:
                broker.submit_order("TEST", 6)
                broker.submit_order("TEST", -5)
                with pytest.raises(TradingControlViolation, match="position"):
                    broker.submit_order("TEST", 5)

    _engine(Submit()).run()


@pytest.mark.parametrize("basis,allowed", [("economic", False), ("quoted_units", True)])
def test_notional_control_uses_the_declared_multiplier_convention(basis, allowed):
    from ml4t.backtest.controls import OrderConstraints, TradingControlViolation
    from ml4t.backtest.types import ContractSpec

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(
                OrderConstraints(max_order_notional=1000, notional_basis=basis)
            )

        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 2:
                if allowed:
                    broker.submit_order("TEST", 1)
                else:
                    with pytest.raises(TradingControlViolation, match="notional"):
                        broker.submit_order("TEST", 1)

    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [2, 3]]
    frame = pl.DataFrame({"timestamp": times, "asset": ["TEST"] * 2, "close": [100.0] * 2})
    result = Engine(
        DataFeed(prices_df=frame),
        Submit(),
        BacktestConfig(calendar=None, initial_cash=10000, execution_mode=ExecutionMode.NEXT_BAR),
        contract_specs={"TEST": ContractSpec(symbol="TEST", multiplier=50)},
    ).run()
    assert result.to_fills_dataframe()["quantity"].to_list() == ([1.0] if allowed else [])


@pytest.mark.parametrize("quantity", [0, -1, float("nan"), float("inf"), True])
def test_invalid_remaining_order_updates_leave_the_original_pending_order_unchanged(quantity):
    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 2:
                order = broker.submit_order("TEST", 2)
                with pytest.raises(ValueError, match="remaining quantity"):
                    broker.update_order(order.order_id, quantity=quantity)
                assert order.quantity == 2

    result = _engine(Submit()).run()
    assert result.to_fills_dataframe()["quantity"].to_list() == [2.0]


def test_partial_fill_remainder_counts_toward_the_position_control():
    from ml4t.backtest.controls import OrderConstraints, TradingControlViolation
    from ml4t.backtest.execution import VolumeParticipationLimit

    times = [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [2, 3, 6]]
    frame = pl.DataFrame(
        {"timestamp": times, "asset": ["TEST"] * 3, "close": [100.0] * 3, "volume": [10.0] * 3}
    )

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderConstraints(max_position_quantity=10))

        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 10)
            elif timestamp == times[1]:
                with pytest.raises(TradingControlViolation, match="position"):
                    broker.submit_order("TEST", 1)

    result = Engine(
        DataFeed(prices_df=frame),
        Submit(),
        BacktestConfig(calendar=None, initial_cash=10000, execution_mode=ExecutionMode.NEXT_BAR),
        execution_limits=VolumeParticipationLimit(max_participation=0.5),
    ).run()
    assert result.to_fills_dataframe()["quantity"].to_list() == [5.0, 5.0]


def test_pre_open_order_count_uses_the_callbacks_upcoming_cme_session():
    from ml4t.specs import LifecyclePhase

    from ml4t.backtest import RuntimeServices
    from ml4t.backtest.controls import OrderConstraints

    times = [datetime(2025, 7, 7, 20, 30, tzinfo=UTC), datetime(2025, 7, 7, 23, tzinfo=UTC)]
    runtime = RuntimeServices(start=times[0])
    seen = []

    def before(timestamp, data, context, broker):
        seen.append(broker.submit_order("TEST", 1).created_at)

    runtime.scheduler.register(
        before, phase=LifecyclePhase.PRE_OPEN, at="session_open", minutes_before=45
    )

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderConstraints(max_orders_per_session=1))

        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 1)

    frame = pl.DataFrame({"timestamp": times, "asset": ["TEST"] * 2, "close": [100.0] * 2})
    Engine(
        DataFeed(prices_df=frame, feed_spec={"timestamp_semantics": "event_time"}),
        Submit(),
        BacktestConfig(calendar="CME_Equity", execution_mode=ExecutionMode.NEXT_BAR),
        runtime=runtime,
    ).run()
    assert seen == [datetime(2025, 7, 7, 21, 15, tzinfo=UTC)]


def test_maximum_leverage_validates_the_executed_account_without_rolling_back_completed_fills():
    from ml4t.backtest.controls import AccountConstraints, AccountControlViolation

    class Submit(Strategy):
        def on_start(self, broker):
            broker.register_account_control(AccountConstraints(max_leverage=1.5))

        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("TEST", 200)

    frame = pl.DataFrame(
        {"timestamp": [datetime(2025, 1, 2, 21, tzinfo=UTC)], "asset": ["TEST"], "close": [100.0]}
    )
    engine = Engine(
        DataFeed(prices_df=frame),
        Submit(),
        BacktestConfig(
            calendar=None,
            initial_cash=10000,
            allow_leverage=True,
            initial_margin=0.5,
            execution_mode=ExecutionMode.SAME_BAR,
        ),
    )
    with pytest.raises(AccountControlViolation, match="maximum"):
        engine.run()
    assert engine.broker.get_position("TEST").quantity == 200
    assert engine.broker.get_cash() == -10000
    assert engine.broker.get_account_value() == 10000
