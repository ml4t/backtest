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
