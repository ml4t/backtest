"""Capital flows must change available cash without becoming investment gains."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import ExecutionMode
from ml4t.backtest.result import BacktestResult


class InvestFirstBar(Strategy):
    def __init__(self):
        self.cash = []

    def on_data(self, timestamp, data, context, broker):
        self.cash.append(broker.cash)
        if len(self.cash) == 1:
            broker.submit_order("TEST", 100)


@pytest.mark.parametrize("kind,value", [("delta", 1000.0), ("target", 12000.0)])
def test_delta_and_target_changes_use_current_nav_and_preserve_investment_return(
    tmp_path, kind, value
):
    dates = [datetime(2025, 1, day, 16, tzinfo=UTC) for day in [2, 3, 6]]
    prices = pl.DataFrame(
        {"timestamp": dates, "asset": ["TEST"] * 3, "close": [100.0, 110.0, 98.0]}
    )
    flows = pl.DataFrame(
        {"timestamp": [dates[1]], "event_id": ["contribution"], "kind": [kind], "value": [value]}
    )
    strategy = InvestFirstBar()
    result = Engine(
        DataFeed(prices_df=prices, capital_flows=flows),
        strategy,
        BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        holdings_sampling="bar",
    ).run()
    assert strategy.cash == [10000, 1000, 1000]
    result.to_parquet(tmp_path)
    for actual in [result, BacktestResult.from_parquet(tmp_path)]:
        assert actual.to_portfolio_state_dataframe()["equity"].to_list() == [10000, 12000, 10800]
        assert actual.to_returns_series().to_list() == pytest.approx([0, 0.1, -0.1])
        assert actual.to_daily_returns(calendar="NYSE").to_list() == pytest.approx([0, 0.1, -0.1])
        assert actual.to_daily_returns(calendar="CME_Equity").to_list() == pytest.approx(
            [0, 0.1, -0.1]
        )
        assert actual.metrics["total_return"] == pytest.approx(-0.01)
        assert actual.metrics["final_value"] == 10800
        assert actual.metrics["total_external_flow"] == 1000
        change = actual.to_capital_flows_dataframe().row(0, named=True)
        assert change["cash_delta"] == 1000
        assert change["nav_before"] == 11000
        assert change["nav_after"] == 12000
        assert change["event_id"] == "contribution"


@pytest.mark.parametrize(
    "invalid", ["duplicate", "future", "missing_timestamp", "nan", "unknown_kind"]
)
def test_invalid_capital_schedule_is_rejected_before_strategy_or_account_mutation(invalid):
    timestamp = datetime(2025, 1, 2, 16, tzinfo=UTC)
    prices = pl.DataFrame({"timestamp": [timestamp], "asset": ["TEST"], "close": [100.0]})
    rows = [{"timestamp": timestamp, "event_id": "a", "kind": "delta", "value": 100.0}]
    if invalid == "duplicate":
        rows.append(dict(rows[0]))
    elif invalid == "future":
        rows[0]["known_at"] = datetime(2025, 1, 3, 16, tzinfo=UTC)
    elif invalid == "missing_timestamp":
        rows[0]["timestamp"] = datetime(2025, 1, 3, 16, tzinfo=UTC)
    elif invalid == "nan":
        rows[0]["value"] = float("nan")
    else:
        rows[0]["kind"] = "income"
    strategy = InvestFirstBar()
    with pytest.raises(ValueError, match="capital"):
        Engine(DataFeed(prices_df=prices, capital_flows=pl.DataFrame(rows)), strategy)
    assert strategy.cash == []


def test_failed_withdrawal_does_not_partially_apply_a_same_instant_deposit():
    timestamp = datetime(2025, 1, 2, 16, tzinfo=UTC)
    prices = pl.DataFrame({"timestamp": [timestamp], "asset": ["TEST"], "close": [100.0]})
    flows = pl.DataFrame(
        {
            "timestamp": [timestamp] * 2,
            "event_id": ["a", "b"],
            "kind": ["delta"] * 2,
            "value": [1000.0, -20000.0],
        }
    )
    engine = Engine(
        DataFeed(prices_df=prices, capital_flows=flows),
        InvestFirstBar(),
        BacktestConfig(initial_cash=10000, calendar=None),
    )
    with pytest.raises(ValueError, match="capital"):
        engine.run()
    assert engine.broker.get_cash() == 10000
    assert engine.broker.get_account_value() == 10000
    assert engine.capital_changes == []
    assert engine.broker.fills == []


def test_empty_account_and_later_recapitalization_do_not_create_a_return(tmp_path):
    dates = [datetime(2025, 1, day, 16, tzinfo=UTC) for day in [2, 3, 6]]
    prices = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 3, "close": [100.0] * 3})
    flows = pl.DataFrame(
        {
            "timestamp": dates[:2],
            "event_id": ["withdraw", "restore"],
            "kind": ["target", "delta"],
            "value": [0.0, 10000.0],
        }
    )

    class AllCash(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pass

    result = Engine(
        DataFeed(prices_df=prices, capital_flows=flows),
        AllCash(),
        BacktestConfig(initial_cash=10000, calendar=None),
    ).run()
    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.to_portfolio_state_dataframe()["equity"].to_list() == [0, 10000, 10000]
    assert restored.to_returns_series().to_list() == [0, 0, 0]
    assert restored.metrics["total_return"] == 0


def test_schedule_on_a_filtered_calendar_bar_is_not_silently_ignored():
    timestamp = datetime(2025, 1, 5, 16, tzinfo=UTC)  # Sunday.
    prices = pl.DataFrame({"timestamp": [timestamp], "asset": ["TEST"], "close": [100.0]})
    flows = pl.DataFrame(
        {"timestamp": [timestamp], "event_id": ["a"], "kind": ["delta"], "value": [100.0]}
    )
    engine = Engine(
        DataFeed(prices_df=prices, capital_flows=flows),
        InvestFirstBar(),
        BacktestConfig(initial_cash=10000, calendar="NYSE", enforce_sessions=True),
    )
    with pytest.raises(ValueError, match="outside accepted calendar"):
        engine.run()
    assert engine.broker.cash == 10000
    assert engine.capital_changes == []


@pytest.mark.parametrize("clock_source", ["context", "other_asset"])
def test_external_flow_rejects_an_unknown_held_asset_mark_before_changing_cash(clock_source):
    times = [datetime(2025, 1, 2, hour, tzinfo=UTC) for hour in [15, 16, 17]]
    prices = pl.DataFrame(
        {"timestamp": [times[0], times[2]], "asset": ["TEST"] * 2, "close": [100.0, 110.0]}
    )
    context = pl.DataFrame({"timestamp": [times[1]], "funds_available": [True]})
    if clock_source == "other_asset":
        prices = pl.concat(
            [prices, pl.DataFrame({"timestamp": [times[1]], "asset": ["OTHER"], "close": [2.0]})]
        )
        context = None
    flows = pl.DataFrame(
        {
            "timestamp": [times[1]],
            "event_id": ["contribution"],
            "kind": ["delta"],
            "value": [1000.0],
        }
    )

    class Hold(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 10)

    engine = Engine(
        DataFeed(prices_df=prices, context_df=context, capital_flows=flows),
        Hold(),
        BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
    )
    with pytest.raises(ValueError, match="current mark"):
        engine.run()
    assert engine.broker.cash == 9000
    assert engine.broker.positions["TEST"].quantity == 10
    assert engine.capital_changes == []


@pytest.mark.parametrize("mark", [0.0, 110.0])
def test_current_event_marks_determine_exact_investment_return_including_zero(mark, tmp_path):
    times = [datetime(2025, 1, 2, hour, tzinfo=UTC) for hour in [15, 16, 17]]
    prices = pl.DataFrame({"timestamp": times, "asset": ["TEST"] * 3, "close": [100.0, mark, mark]})
    flows = pl.DataFrame(
        {
            "timestamp": [times[1]],
            "event_id": ["contribution"],
            "kind": ["delta"],
            "value": [1000.0],
        }
    )

    class Hold(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("TEST", 10)

    result = Engine(
        DataFeed(prices_df=prices, capital_flows=flows),
        Hold(),
        BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
    ).run()
    before = 9000.0 + 10 * mark
    expected = -0.1 if mark == 0 else 0.01
    result.to_parquet(tmp_path)
    for actual in [result, BacktestResult.from_parquet(tmp_path)]:
        flow = actual.to_capital_flows_dataframe().row(0, named=True)
        assert flow["nav_before"] == before
        assert flow["nav_after"] == before + 1000
        assert actual.to_returns_series().to_list() == pytest.approx([0.0, expected, 0.0])
        assert actual.metrics["total_return"] == pytest.approx(expected)


def test_all_cash_capital_change_at_a_context_only_tick_has_an_independent_cash_valuation():
    times = [datetime(2025, 1, 2, hour, tzinfo=UTC) for hour in [15, 16, 17]]
    prices = pl.DataFrame(
        {"timestamp": [times[0], times[2]], "asset": ["TEST"] * 2, "close": [100.0, 110.0]}
    )
    context = pl.DataFrame({"timestamp": [times[1]], "funds_available": [True]})
    flows = pl.DataFrame(
        {
            "timestamp": [times[1]],
            "event_id": ["contribution"],
            "kind": ["delta"],
            "value": [1000.0],
        }
    )

    class Cash(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pass

    result = Engine(
        DataFeed(prices_df=prices, context_df=context, capital_flows=flows),
        Cash(),
        BacktestConfig(initial_cash=10000, calendar=None),
    ).run()
    assert result.to_portfolio_state_dataframe()["equity"].to_list() == [10000.0, 11000.0, 11000.0]
    assert result.to_returns_series().to_list() == [0.0, 0.0, 0.0]
