"""Distinct source nanoseconds must survive causal execution and saved account histories."""

from datetime import UTC, datetime

import polars as pl
import pytest
from ml4t.specs import FeedSpec, TimestampSemantics

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import CommissionType, ExecutionMode, ExecutionPrice, SlippageType
from ml4t.backtest.result import BacktestResult


def _instant_feed(**kwargs):
    spec = FeedSpec.from_any(kwargs.pop("feed_spec", None))
    if spec.timestamp_semantics is None:
        spec = spec.with_overrides(timestamp_semantics=TimestampSemantics.EVENT_TIME)
    return DataFeed(feed_spec=spec, **kwargs)


BASE = 1735830000000000000


def instants(offsets):
    return pl.Series("timestamp", [BASE + i for i in offsets], dtype=pl.Int64).cast(
        pl.Datetime("ns", "UTC")
    )


def config(**kwargs):
    return BacktestConfig(
        initial_cash=10000,
        calendar=kwargs.pop("calendar", None),
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.CLOSE,
        commission_type=CommissionType.NONE,
        slippage_type=SlippageType.NONE,
        **kwargs,
    )


class Trade(Strategy):
    def __init__(self):
        self.calls = 0

    def on_data(self, timestamp, data, context, broker):
        broker.submit_order("TEST", 1 if self.calls == 0 else -1)
        self.calls += 1


def epochs(frame, column="timestamp"):
    return frame[column].cast(pl.Datetime("ns", "UTC")).cast(pl.Int64).to_list()


def test_distinct_ns_events_keep_fill_account_holdings_and_artifact_identity(tmp_path):
    frame = pl.DataFrame(
        {"timestamp": instants([1, 2]), "asset": ["TEST"] * 2, "close": [100.0, 110.0]}
    )
    feed = _instant_feed(prices_df=frame)
    result = Engine(
        feed, Trade(), config(calendar="NYSE"), holdings_sampling="fill_and_session"
    ).run()
    assert len(feed) == 2
    assert epochs(result.to_equity_dataframe()) == [BASE + 1, BASE + 2]
    assert epochs(result.to_fills_dataframe()) == [BASE + 1, BASE + 2]
    assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 10010.0]
    assert epochs(result.to_trades_dataframe(), "entry_time") == [BASE + 1]
    assert epochs(result.to_trades_dataframe(), "exit_time") == [BASE + 2]
    assert epochs(result.to_holdings_dataframe(), "mark_timestamp") == [BASE + 1]
    path = tmp_path / "account"
    result.to_parquet(path)
    restored = BacktestResult.from_parquet(path)
    assert epochs(restored.to_equity_dataframe()) == [BASE + 1, BASE + 2]
    assert epochs(restored.to_fills_dataframe()) == [BASE + 1, BASE + 2]
    assert epochs(restored.to_trades_dataframe(), "exit_time") == [BASE + 2]
    assert epochs(restored.to_portfolio_state_dataframe()) == epochs(
        result.to_portfolio_state_dataframe()
    )
    assert epochs(restored.to_holdings_dataframe(), "mark_timestamp") == [BASE + 1]


def test_ns_known_at_and_history_cannot_merge_a_future_observation():
    frame = pl.DataFrame(
        {
            "timestamp": instants([1, 2]),
            "asset": ["TEST"] * 2,
            "close": [100.0, 999.0],
            "feature": [3.0, 9.0],
            "known": instants([2, 2]),
        }
    )
    runtime = RuntimeServices(known_at_cols={"feature": "known"})
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            h = runtime.market.history(["TEST"], ["close", "feature"], bar_count=2)
            seen.append((epochs(h), h["close"].to_list(), h["feature"].to_list()))

    Engine(_instant_feed(prices_df=frame), Read(), config(), runtime=runtime).run()
    assert seen == [
        ([BASE + 1], [100.0], [None]),
        ([BASE + 1, BASE + 2], [100.0, 999.0], [3.0, 9.0]),
    ]


def test_ns_cash_events_apply_to_their_actual_separate_quotes_and_survive_save(tmp_path):
    frame = pl.DataFrame(
        {"timestamp": instants([1, 2, 3]), "asset": ["TEST"] * 3, "close": [100.0, 110.0, 120.0]}
    )
    flows = pl.DataFrame(
        {
            "timestamp": instants([2, 3]),
            "event_id": ["first", "second"],
            "kind": ["delta"] * 2,
            "value": [500.0, 250.0],
        }
    )

    class Hold(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if not broker.positions:
                broker.submit_order("TEST", 1)

    result = Engine(_instant_feed(prices_df=frame, capital_flows=flows), Hold(), config()).run()
    assert epochs(result.to_capital_flows_dataframe()) == [BASE + 2, BASE + 3]
    assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 10510.0, 10770.0]
    assert result.to_equity_dataframe()["return"].to_list() == pytest.approx([0, 0.001, 10 / 10510])
    path = tmp_path / "flows"
    result.to_parquet(path)
    restored = BacktestResult.from_parquet(path)
    assert epochs(restored.to_capital_flows_dataframe()) == [BASE + 2, BASE + 3]
    assert (
        restored.to_equity_dataframe()["return"].to_list()
        == result.to_equity_dataframe()["return"].to_list()
    )


def test_ns_close_filter_keeps_only_inclusive_declared_closing_endpoint():
    close = int(datetime(2025, 1, 2, 21, tzinfo=UTC).timestamp()) * 10**9
    times = pl.Series("timestamp", [close - 1, close, close + 1], dtype=pl.Int64).cast(
        pl.Datetime("ns", "UTC")
    )
    frame = pl.DataFrame({"timestamp": times, "asset": ["TEST"] * 3, "close": [100.0] * 3})

    class Cash(Strategy):
        def on_data(self, *args):
            pass

    spec = FeedSpec(
        time_contract_version=2,
        calendar="NYSE",
        timestamp_semantics=TimestampSemantics.BAR_CLOSE,
        bar_label="right",
        data_frequency="1m",
    )
    result = Engine(
        _instant_feed(prices_df=frame, feed_spec=spec),
        Cash(),
        BacktestConfig(enforce_sessions=True),
    ).run()
    assert epochs(result.to_equity_dataframe()) == [close - 1, close]


def test_ns_funding_records_are_distinct_cash_movements_and_round_trip(tmp_path):
    frame = pl.DataFrame(
        {"timestamp": instants([1, 2, 3]), "asset": ["TEST"] * 3, "close": [100.0] * 3}
    )
    funding = pl.DataFrame(
        {"timestamp": instants([2, 3]), "asset": ["TEST"] * 2, "amount_per_unit": [1.0, 2.0]}
    )

    class Hold(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if not broker.positions:
                broker.submit_order("TEST", 1)

    result = Engine(_instant_feed(prices_df=frame), Hold(), config(), funding_df=funding).run()
    assert epochs(result.to_funding_dataframe()) == [BASE + 2, BASE + 3]
    assert result.to_funding_dataframe()["cash_delta"].to_list() == [-1.0, -2.0]
    assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 9999.0, 9997.0]
    path = tmp_path / "funding"
    result.to_parquet(path)
    restored = BacktestResult.from_parquet(path)
    assert epochs(restored.to_funding_dataframe()) == [BASE + 2, BASE + 3]
