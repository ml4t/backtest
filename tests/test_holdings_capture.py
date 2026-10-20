"""Observed account snapshots must not be reconstructed from closed trades."""

from datetime import UTC, datetime

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, ExecutionMode
from ml4t.backtest.result import ArtifactError, BacktestResult
from ml4t.backtest.types import ContractSpec


class TradeAtBars(Strategy):
    def __init__(self, orders):
        self.orders = orders
        self.bar = 0

    def on_data(self, timestamp, data, context, broker):
        for asset, quantity in self.orders.get(self.bar, []):
            broker.submit_order(asset, quantity)
        self.bar += 1


def run_positions(orders, sampling="bar", **kwargs):
    dates = [datetime(2025, 1, 2, hour, tzinfo=UTC) for hour in [14, 15, 16]]
    prices = pl.DataFrame(
        {
            "timestamp": [t for t in dates for _ in range(2)],
            "asset": ["LONG", "SHORT"] * 3,
            "close": [100.0, 50.0, 110.0, 45.0, 120.0, 40.0],
        }
    )
    result = Engine(
        DataFeed(prices_df=prices),
        TradeAtBars(orders),
        BacktestConfig(
            initial_cash=10000,
            calendar=None,
            execution_mode=ExecutionMode.SAME_BAR,
            commission_type=CommissionType.NONE,
            allow_short_selling=True,
        ),
        holdings_sampling=sampling,
        **kwargs,
    ).run()
    return result, dates


def test_complete_holdings_retain_long_short_and_all_cash_observations(tmp_path):
    result, dates = run_positions(
        {0: [("LONG", 10), ("SHORT", -4)], 2: [("LONG", -10), ("SHORT", 4)]}
    )
    result.to_parquet(tmp_path)
    for actual in [result, BacktestResult.from_parquet(tmp_path)]:
        holdings = actual.to_holdings_dataframe()
        assert holdings["observation_id"].to_list() == [0, 0, 1, 1]
        assert holdings["timestamp"].to_list() == [dates[0], dates[0], dates[1], dates[1]]
        assert holdings["asset"].to_list() == ["LONG", "SHORT", "LONG", "SHORT"]
        assert holdings["quantity"].to_list() == [10, -4, 10, -4]
        assert holdings["mark"].to_list() == [100, 50, 110, 45]
        assert holdings["signed_notional"].to_list() == [1000, -200, 1100, -180]
        assert holdings["liquidation_value"].to_list() == [1000, -200, 1100, -180]
        account = actual.to_portfolio_state_dataframe()
        assert account["observation_id"].to_list() == [0, 1, 2]
        assert account["equity"].to_list() == [10000, 10120, 10240]
        assert account["cash"].to_list() == [9200, 9200, 10240]
        assert actual.holdings_sampling == "bar"
        assert actual.to_fills_dataframe()["fill_sequence"].to_list() == [0, 1, 2, 3]


def test_capture_distinguishes_complete_all_cash_from_unavailable(tmp_path):
    complete, dates = run_positions({})
    complete.to_parquet(tmp_path)
    for actual in [complete, BacktestResult.from_parquet(tmp_path)]:
        assert actual.to_holdings_dataframe().is_empty()
        assert actual.to_portfolio_state_dataframe()["timestamp"].to_list() == dates
        assert actual.holdings_sampling == "bar"
    unavailable, _ = run_positions({}, sampling="none")
    unavailable.to_parquet(tmp_path)
    with pytest.raises(ArtifactError, match="not captured"):
        BacktestResult.from_parquet(tmp_path).to_holdings_dataframe()


def test_contract_multiplier_and_currency_survive_observed_export(tmp_path):
    result, _ = run_positions(
        {0: [("LONG", 2)]},
        contract_specs={"LONG": ContractSpec(symbol="LONG", multiplier=10, currency="USD")},
    )
    result.to_parquet(tmp_path)
    holdings = BacktestResult.from_parquet(tmp_path).to_holdings_dataframe()
    assert holdings["multiplier"].to_list() == [10.0] * 3
    assert holdings["currency"].to_list() == ["USD"] * 3
    assert holdings["signed_notional"].to_list() == [2000, 2200, 2400]


def test_equal_time_fill_order_survives_reordered_storage(tmp_path):
    result, _ = run_positions({0: [("LONG", 10), ("SHORT", -4)]})
    result.to_parquet(tmp_path)
    path = tmp_path / "fills.parquet"
    pl.read_parquet(path).reverse().write_parquet(path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.to_fills_dataframe()["asset"].to_list() == ["LONG", "SHORT"]
    frame = pl.read_parquet(path).with_columns(pl.lit(0).alias("fill_sequence"))
    frame.write_parquet(path)
    with pytest.raises(ArtifactError, match="fill_sequence"):
        BacktestResult.from_parquet(tmp_path)


def test_missing_capture_never_loads_as_a_complete_empty_portfolio(tmp_path):
    result, _ = run_positions({0: [("LONG", 10)]})
    result.to_parquet(tmp_path)
    (tmp_path / "holdings.parquet").unlink()
    with pytest.raises(ArtifactError, match="holdings"):
        BacktestResult.from_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path, recovery=True)
    with pytest.raises(ArtifactError, match="not captured"):
        restored.to_holdings_dataframe()
    assert restored.holdings_sampling == "none"


def test_corrupted_holdings_cannot_claim_an_unobserved_account_key(tmp_path):
    result, _ = run_positions({0: [("LONG", 10)]})
    result.to_parquet(tmp_path)
    path = tmp_path / "holdings.parquet"
    pl.read_parquet(path).with_columns(
        (pl.col("observation_id") + 99).alias("observation_id")
    ).write_parquet(path)
    with pytest.raises(ArtifactError, match="actual account observations"):
        BacktestResult.from_parquet(tmp_path)


def test_fill_exports_keep_actual_multiplier_currency_and_signed_quantity(tmp_path):
    result, _ = run_positions(
        {0: [("LONG", 2), ("SHORT", -1)]},
        contract_specs={
            "LONG": ContractSpec(symbol="LONG", multiplier=10, currency="USD"),
            "SHORT": ContractSpec(symbol="SHORT", multiplier=20, currency="USD"),
        },
    )
    result.to_parquet(tmp_path)
    for actual in [result, BacktestResult.from_parquet(tmp_path)]:
        fills = actual.to_fills_dataframe()
        assert fills["multiplier"].to_list() == [10, 20]
        assert fills["currency"].to_list() == ["USD", "USD"]
        assert fills["signed_quantity"].to_list() == [2, -1]
        assert actual.returns_include_commissions is True


def test_fill_sampling_keeps_a_round_trip_that_is_flat_at_every_bar_end(tmp_path):
    dates = [datetime(2025, 1, 2, hour, tzinfo=UTC) for hour in [15, 16, 17]]
    prices = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 3, "close": [100.0] * 3})
    result = Engine(
        DataFeed(prices_df=prices),
        TradeAtBars({0: [("TEST", 10), ("TEST", -10)]}),
        BacktestConfig(
            initial_cash=10000,
            calendar="NYSE",
            execution_mode=ExecutionMode.SAME_BAR,
            allow_short_selling=True,
        ),
        holdings_sampling="fill_and_session",
    ).run()
    result.to_parquet(tmp_path)
    for actual in [result, BacktestResult.from_parquet(tmp_path)]:
        account = actual.to_portfolio_state_dataframe()
        assert account["observation_kind"].to_list() == [
            "after_fill",
            "after_fill",
            "bar",
            "bar",
            "session_last_observed",
        ]
        assert account["holdings_complete"].to_list() == [True, True, False, False, True]
        assert account["open_positions"].to_list() == [1, 0, 0, 0, 0]
        assert account["cash"].to_list() == [9000, 10000, 10000, 10000, 10000]
        holdings = actual.to_holdings_dataframe()
        assert holdings["observation_id"].to_list() == [0]
        assert holdings["quantity"].to_list() == [10]
        assert len(actual.equity_curve) == 3
        assert actual.to_returns_series().to_list() == [0, 0, 0]


def test_session_capture_marks_only_the_last_observed_bar_complete(tmp_path):
    dates = [datetime(2025, 1, day, hour, tzinfo=UTC) for day, hour in [(2, 15), (2, 16), (3, 15)]]
    prices = pl.DataFrame(
        {"timestamp": dates, "asset": ["TEST"] * 3, "close": [100.0, 110.0, 120.0]}
    )
    result = Engine(
        DataFeed(prices_df=prices),
        TradeAtBars({0: [("TEST", 10)]}),
        BacktestConfig(initial_cash=10000, calendar="NYSE", execution_mode=ExecutionMode.SAME_BAR),
        holdings_sampling="session",
    ).run()
    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.to_portfolio_state_dataframe()["holdings_complete"].to_list() == [
        False,
        True,
        True,
    ]
    assert restored.to_holdings_dataframe()["observation_id"].to_list() == [1, 2]
    assert restored.to_holdings_dataframe()["mark"].to_list() == [110, 120]


def test_failed_immediate_callback_cannot_leave_observed_fill_state():
    timestamp = datetime(2025, 1, 2, 16, tzinfo=UTC)
    prices = pl.DataFrame({"timestamp": [timestamp], "asset": ["TEST"], "close": [100.0]})

    class Failing(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("TEST", 10)
            raise RuntimeError("callback failed")

    engine = Engine(
        DataFeed(prices_df=prices),
        Failing(),
        BacktestConfig(calendar="NYSE", execution_mode=ExecutionMode.SAME_BAR, immediate_fill=True),
        holdings_sampling="fill_and_session",
    )
    with pytest.raises(RuntimeError, match="callback failed"):
        engine.run()
    assert engine.broker.fills == []
    assert engine.broker._execution_journal.fill_snapshots == []
    assert engine.broker.cash == engine.config.initial_cash


def test_missing_position_row_cannot_be_reported_as_a_complete_snapshot(tmp_path):
    result, _ = run_positions({0: [("LONG", 10), ("SHORT", -4)]})
    result.to_parquet(tmp_path)
    path = tmp_path / "holdings.parquet"
    pl.read_parquet(path).slice(1).write_parquet(path)
    with pytest.raises(ArtifactError, match="complete holdings"):
        BacktestResult.from_parquet(tmp_path)


@pytest.mark.parametrize("last_asset,last_price", [("TEST", 0.0), ("OTHER", 50.0)])
def test_observed_zero_mark_is_not_replaced_by_entry_price(last_asset, last_price):
    dates = [datetime(2025, 1, day, 16, tzinfo=UTC) for day in [2, 3, 6]]
    prices = pl.DataFrame(
        {
            "timestamp": dates,
            "asset": ["TEST", "TEST", last_asset],
            "close": [100.0, 0.0, last_price],
        }
    )
    result = Engine(
        DataFeed(prices_df=prices),
        TradeAtBars({0: [("TEST", 10)]}),
        BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
    ).run()
    assert result.to_portfolio_state_dataframe()["equity"].to_list() == [10000, 9000, 9000]
    assert result.metrics["final_value"] == 9000
    assert result.trades[0].pnl == -1000


def test_carried_mark_keeps_its_observation_time_and_source(tmp_path):
    dates = [datetime(2025, 1, day, 16, tzinfo=UTC) for day in [2, 3, 6]]
    prices = pl.DataFrame(
        {"timestamp": dates, "asset": ["TEST", "TEST", "OTHER"], "close": [100.0, 110.0, 50.0]}
    )
    result = Engine(
        DataFeed(prices_df=prices),
        TradeAtBars({0: [("TEST", 10)]}),
        BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        holdings_sampling="bar",
    ).run()
    result.to_parquet(tmp_path)
    holdings = BacktestResult.from_parquet(tmp_path).to_holdings_dataframe()
    assert holdings["mark_timestamp"].to_list() == [dates[0], dates[1], dates[1]]
    assert holdings["mark_source"][-1] == "last_observed"
    assert holdings["mark"].to_list() == [100, 110, 110]


def test_account_method_is_declared_and_old_artifacts_remain_unknown(tmp_path):
    from pathlib import Path

    result, _ = run_positions({0: [("LONG", 10)]})
    result.to_parquet(tmp_path)
    assert result.account_method == "signed_notional"
    assert BacktestResult.from_parquet(tmp_path).account_method == "signed_notional"
    legacy = BacktestResult.from_parquet(
        Path(__file__).parent / "fixtures/artifacts/v0.1.0b22", recovery=True
    )
    assert legacy.account_method is None


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_invalid_new_quote_does_not_replace_the_last_observed_finite_mark(invalid):
    dates = [datetime(2025, 1, day, 16, tzinfo=UTC) for day in [2, 3]]
    prices = pl.DataFrame({"timestamp": dates, "asset": ["TEST"] * 2, "close": [100.0, invalid]})

    class Observe(Strategy):
        def __init__(self):
            self.observed = []

        def on_data(self, timestamp, data, context, broker):
            self.observed.append(broker.get_last_price("TEST"))

    strategy = Observe()
    Engine(DataFeed(prices_df=prices), strategy, BacktestConfig(calendar=None)).run()
    assert strategy.observed == [100.0, 100.0]


def test_unusable_current_mark_uses_carried_valuation_without_fabricating_a_quote(tmp_path):
    dates = [datetime(2025, 1, day, 16, tzinfo=UTC) for day in [2, 3]]
    prices = pl.DataFrame(
        {"timestamp": dates, "asset": ["TEST"] * 2, "close": [100.0, float("nan")]}
    )
    result = Engine(
        DataFeed(prices_df=prices),
        TradeAtBars({0: [("TEST", 10)]}),
        BacktestConfig(initial_cash=10000, calendar=None, execution_mode=ExecutionMode.SAME_BAR),
        holdings_sampling="bar",
    ).run()
    result.to_parquet(tmp_path)
    holdings = BacktestResult.from_parquet(tmp_path).to_holdings_dataframe()
    assert holdings["mark"].to_list() == [100.0, 100.0]
    assert holdings["mark_timestamp"].to_list() == [dates[0], dates[0]]
    assert holdings["mark_source"][-1] == "last_observed"
    assert len(result.fills) == 1
    assert result.metrics["final_value"] == 10000
