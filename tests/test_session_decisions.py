"""Cross-sectional decisions over daily bars with different real close times."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.execution.rebalancer import TargetWeightExecutor
from ml4t.backtest.types import ExecutionMode, OrderType


def _prices(days: int = 3) -> pl.DataFrame:
    rows = []
    for day in range(days):
        session = date(2024, 1, 2) + timedelta(days=day)
        for asset, minute in (("ES", 0), ("ZC", 15)):
            rows.append(
                {
                    "timestamp": datetime(session.year, session.month, session.day, 21, minute),
                    "session_date": session,
                    "asset": asset,
                    "close": 100.0,
                    "open": 100.0,
                }
            )
    return pl.DataFrame(rows)


class _CrossSection(Strategy):
    def __init__(self) -> None:
        self.events: list[tuple[datetime, set[str]]] = []
        self.executor = TargetWeightExecutor()

    def on_data(self, timestamp, data, context, broker):
        self.events.append((timestamp, set(data)))
        self.executor.execute(dict.fromkeys(data, 0.5), data, broker)


def test_exact_timestamp_mode_reproduces_partial_cross_section():
    strategy = _CrossSection()
    engine = Engine(
        feed=DataFeed(prices_df=_prices(1)),
        strategy=strategy,
        config=BacktestConfig(execution_mode=ExecutionMode.SAME_BAR),
    )
    engine.run()

    assert [assets for _, assets in strategy.events] == [{"ES"}, {"ZC"}]
    assert any(order.asset == "ES" for order in engine.broker.get_pending_orders())


def test_explicit_session_date_combines_decision_and_defers_each_fill():
    strategy = _CrossSection()
    engine = Engine(
        feed=DataFeed(prices_df=_prices(), session_col="session_date"),
        strategy=strategy,
        config=BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
    )
    result = engine.run()

    assert strategy.events == [(datetime(2024, 1, day, 21, 15), {"ES", "ZC"}) for day in (2, 3, 4)]
    assert set(engine.broker.get_positions()) == {"ES", "ZC"}
    fills = result.to_fills_dataframe()
    assert fills.height == 2
    assert set(fills["timestamp"].to_list()) == {
        datetime(2024, 1, 3, 21),
        datetime(2024, 1, 3, 21, 15),
    }


def test_session_mode_rejects_same_bar_execution_before_run():
    feed = DataFeed(prices_df=_prices(), session_col="session_date")
    with pytest.raises(ValueError, match="NEXT_BAR"):
        Engine(
            feed=feed,
            strategy=_CrossSection(),
            config=BacktestConfig(execution_mode=ExecutionMode.SAME_BAR),
        )


def test_missing_or_duplicate_session_bars_fail_before_strategy_runs():
    prices = _prices()
    without_zc = prices.filter(
        ~((pl.col("session_date") == date(2024, 1, 3)) & (pl.col("asset") == "ZC"))
    )
    with pytest.raises(ValueError, match="missing bars.*ZC"):
        DataFeed(prices_df=without_zc, session_col="session_date")

    with pytest.raises(ValueError, match="duplicate bar"):
        DataFeed(prices_df=pl.concat([prices, prices.head(1)]), session_col="session_date")


def test_mixed_sessions_at_one_timestamp_fail_before_strategy_runs():
    prices = (
        _prices(1)
        .with_columns(
            pl.when(pl.col("asset") == "ZC")
            .then(pl.lit(date(2024, 1, 3)))
            .otherwise(pl.col("session_date"))
            .alias("session_date")
        )
        .with_columns(pl.lit(datetime(2024, 1, 2, 21)).alias("timestamp"))
    )
    with pytest.raises(ValueError, match="different sessions"):
        DataFeed(prices_df=prices, session_col="session_date")


def test_session_decision_moc_waits_for_next_matching_asset_bar():
    class SubmitMoc(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 2:
                broker.submit_order("ZC", 1, order_type=OrderType.MOC)

    result = Engine(
        feed=DataFeed(prices_df=_prices(), session_col="session_date"),
        strategy=SubmitMoc(),
        config=BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR),
    ).run()
    assert result.to_fills_dataframe()["timestamp"].to_list() == [datetime(2024, 1, 3, 21, 15)]


def test_explicit_sessions_handle_holiday_and_dst_with_real_utc_closes():
    rows = []
    for session, hour in (
        (date(2024, 3, 8), 21),
        (date(2024, 3, 11), 20),
        (date(2024, 3, 29), 20),
        (date(2024, 4, 1), 20),
    ):
        for asset, minute in (("ES", 0), ("ZC", 15)):
            rows.append(
                {
                    "timestamp": datetime(session.year, session.month, session.day, hour, minute),
                    "session_date": session,
                    "asset": asset,
                    "close": 100.0,
                    "open": 100.0,
                }
            )
    strategy = _CrossSection()
    engine = Engine(
        DataFeed(prices_df=pl.DataFrame(rows), session_col="session_date"),
        strategy,
        BacktestConfig(
            execution_mode=ExecutionMode.NEXT_BAR, calendar="NYSE", enforce_sessions=True
        ),
    )
    result = engine.run()
    assert [timestamp for timestamp, _ in strategy.events] == [
        datetime(2024, 3, 8, 21, 15),
        datetime(2024, 3, 11, 20, 15),
        datetime(2024, 4, 1, 20, 15),
    ]
    assert set(result.to_fills_dataframe()["timestamp"].to_list()) == {
        datetime(2024, 3, 11, 20),
        datetime(2024, 3, 11, 20, 15),
    }
