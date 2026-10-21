"""Request quotas must not change the existing submitted-order count definition."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.controls import OrderConstraints, TradingControlViolation


def _feed(times=None):
    times = times or [datetime(2025, 1, d, 21, tzinfo=UTC) for d in [6, 7]]
    return DataFeed(
        prices_df=pl.DataFrame(
            {"timestamp": times, "asset": ["A"] * len(times), "close": [100.0] * len(times)}
        ),
        feed_spec={
            "time_contract_version": 2,
            "timestamp_semantics": "event_time",
            "calendar": None,
        },
    )


def test_request_count_resets_on_declared_utc_day_instead_of_delivered_timezone_date():
    from ml4t.backtest.controls import OrderRequestCount

    seen = []
    delivered_dates = []

    class Trade(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderRequestCount(1, timezone="UTC"))

        def on_data(self, timestamp, data, context, broker):
            delivered_dates.append(timestamp.date().isoformat())
            seen.append(broker.submit_order("A", 1).order_id)

    times = [
        datetime(2025, 1, 6, 18, 59, tzinfo=ZoneInfo("America/New_York")),
        datetime(2025, 1, 6, 19, 1, tzinfo=ZoneInfo("America/New_York")),
    ]
    prepared = _feed(times).prices.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York")
    )
    feed = DataFeed(
        prices_df=prepared,
        feed_spec={
            "time_contract_version": 2,
            "timestamp_semantics": "event_time",
            "calendar": None,
            "timezone": "America/New_York",
        },
    )
    Engine(feed, Trade(), BacktestConfig(calendar=None)).run()
    assert delivered_dates == ["2025-01-06", "2025-01-06"]
    assert seen == ["ORD-1", "ORD-2"]


def test_removing_another_rule_keeps_request_quota_and_unregistering_owner_removes_it():
    from ml4t.backtest.controls import OrderRequestCount

    seen = []

    class Trade(Strategy):
        def on_start(self, broker):
            self.first = OrderConstraints(max_order_quantity=100)
            self.count = OrderRequestCount(2)
            broker.register_trading_control(self.first)
            broker.register_trading_control(self.count)
            self.once = False

        def on_data(self, timestamp, data, context, broker):
            if self.once:
                return
            seen.append(broker.submit_order("A", 1).order_id)
            broker.unregister_trading_control(self.first)
            seen.append(broker.submit_order("A", 1).order_id)
            with pytest.raises(TradingControlViolation):
                broker.submit_order("A", 1)
            broker.unregister_trading_control(self.count)
            seen.append(broker.submit_order("A", 1).order_id)
            self.once = True

    Engine(_feed(), Trade(), BacktestConfig(calendar=None)).run()
    assert seen == ["ORD-1", "ORD-2", "ORD-3"]


def test_default_day_count_still_ignores_zero_and_pre_submission_validation_failures():
    seen = []

    class Trade(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(
                OrderConstraints(max_orders_per_day=2, max_order_quantity=5)
            )
            self.once = False

        def on_data(self, timestamp, data, context, broker):
            if self.once:
                return
            assert broker.submit_order("A", 0) is None
            with pytest.raises(TradingControlViolation):
                broker.submit_order("A", 6)
            seen.append(broker.submit_order("A", 1).order_id)
            seen.append(broker.submit_order("A", 1).order_id)
            self.once = True

    Engine(_feed(), Trade(), BacktestConfig(calendar=None)).run()
    assert seen == ["ORD-1", "ORD-2"]


def test_zero_request_consumes_explicit_request_quota_without_creating_an_order_or_cash_change():
    from ml4t.backtest.controls import OrderRequestCount

    class Trade(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderRequestCount(1))

        def on_data(self, timestamp, data, context, broker):
            assert broker.submit_order("A", 0) is None
            with pytest.raises(TradingControlViolation):
                broker.submit_order("A", 1)
            assert not broker.get_orders()

    result = Engine(_feed(), Trade(), BacktestConfig(calendar=None)).run()
    assert result.fills == []
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == [100000, 100000]
