"""Recorded strategy values must carry without overwriting account results."""

from datetime import UTC, datetime
from math import isnan

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.result import ArtifactError, BacktestResult


def prices():
    return pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, day, 16, tzinfo=UTC) for day in [2, 3, 6]],
            "asset": ["TEST"] * 3,
            "close": [100.0] * 3,
        }
    )


def test_recorded_values_carry_with_owned_keys_and_do_not_replace_account_cash(tmp_path):
    class Record(Strategy):
        def on_start(self, broker):
            broker.record(cash=42, score=float("nan"), detail={"side": [True, None]})

        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 3:
                broker.record(score=2.0)

    result = Engine(
        DataFeed(prices_df=prices()), Record(), BacktestConfig(initial_cash=10000, calendar=None)
    ).run()
    result.to_parquet(tmp_path)
    for actual in [result, BacktestResult.from_parquet(tmp_path)]:
        records = actual.to_recorded_records()
        assert [row["bar_index"] for row in records] == [0, 1, 2]
        assert [row["timestamp"] for row in records] == prices()["timestamp"].to_list()
        assert [row["values"]["cash"] for row in records] == [42, 42, 42]
        assert isnan(records[0]["values"]["score"])
        assert [row["values"]["score"] for row in records[1:]] == [2.0, 2.0]
        assert all(row["values"]["detail"] == {"side": [True, None]} for row in records)
        assert actual.to_portfolio_state_dataframe()["cash"].to_list() == [10000.0] * 3


def test_invalid_record_values_fail_before_mutation_and_callback_failure_rolls_back():
    class Record(Strategy):
        def on_start(self, broker):
            broker.record(score=1)

        def on_data(self, timestamp, data, context, broker):
            with pytest.raises(ValueError, match="record"):
                broker.record(score=2, bad=object())
            broker.record(score=3)
            raise RuntimeError("callback failed")

    engine = Engine(DataFeed(prices_df=prices()), Record(), BacktestConfig(calendar=None))
    with pytest.raises(RuntimeError, match="callback failed"):
        engine.run()
    assert engine.broker._execution_journal.recorded_values == {"score": "1"}
    assert engine.broker._execution_journal.recorded_rows == []


def test_unrecorded_strategy_reports_variables_unavailable(tmp_path):
    class Empty(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pass

    result = Engine(DataFeed(prices_df=prices()), Empty(), BacktestConfig(calendar=None)).run()
    result.to_parquet(tmp_path)
    with pytest.raises(ArtifactError, match="not recorded"):
        BacktestResult.from_parquet(tmp_path).to_recorded_dataframe()


def test_reserved_json_tags_inside_array_inputs_are_rejected_before_mutation():
    import numpy as np

    class Record(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.record(score=1)
            with pytest.raises(ValueError, match="reserved artifact"):
                broker.record(
                    score=2, literal=np.array([{"__ml4t_nonfinite_float__": "nan"}], dtype=object)
                )

    result = Engine(DataFeed(prices_df=prices()), Record(), BacktestConfig(calendar=None)).run()
    assert all(row["values"] == {"score": 1} for row in result.to_recorded_records())
