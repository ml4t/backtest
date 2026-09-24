"""Funding cash flows through a complete perpetual-futures backtest."""

import json
from datetime import datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.broker import Broker
from ml4t.backtest.config import DataFrequency
from ml4t.backtest.funding import FundingEvent
from ml4t.backtest.result import ArtifactReadError, BacktestResult
from ml4t.backtest.types import AssetClass, ContractSpec, ExecutionMode, OrderSide, Position


def _prices() -> pl.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
    values = [100.0, 110.0, 110.0]
    return pl.DataFrame(
        {
            "timestamp": dates,
            "asset": ["BTC-PERP"] * len(dates),
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": [1000] * len(dates),
        }
    )


def _run(quantity: int, funding: pl.DataFrame | None) -> BacktestResult:
    class Hold(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == datetime(2024, 1, 1):
                side = OrderSide.BUY if quantity > 0 else OrderSide.SELL
                broker.submit_order("BTC-PERP", abs(quantity), side)

    return Engine(
        feed=DataFeed(prices_df=_prices()),
        strategy=Hold(),
        config=BacktestConfig(
            initial_cash=1000,
            execution_mode=ExecutionMode.SAME_BAR,
            allow_short_selling=True,
            allow_leverage=True,
        ),
        contract_specs={
            "BTC-PERP": ContractSpec(
                symbol="BTC-PERP", asset_class=AssetClass.FUTURE, multiplier=2.0
            )
        },
        funding_df=funding,
    ).run()


@pytest.mark.parametrize("quantity,expected", [(1, -2.2), (-1, 2.2)])
def test_funding_uses_held_position_and_current_causal_mark(quantity, expected, tmp_path):
    funding = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2)],
            "asset": ["BTC-PERP"],
            "rate": [0.01],
        }
    )
    baseline = _run(quantity, None)
    result = _run(quantity, funding)

    payments = result.to_funding_dataframe()
    assert payments.height == 1
    payment = payments.row(0, named=True)
    assert payment["quantity"] == quantity
    assert payment["mark_price"] == 110.0
    assert payment["multiplier"] == 2.0
    assert payment["cash_delta"] == pytest.approx(expected)
    assert result.metrics["total_funding"] == pytest.approx(expected)
    assert result.metrics["num_funding_events"] == 1
    assert result.metrics["final_value"] - baseline.metrics["final_value"] == pytest.approx(
        expected
    )
    assert result.to_portfolio_state_dataframe()["cash"][
        -1
    ] - baseline.to_portfolio_state_dataframe()["cash"][-1] == pytest.approx(expected)
    assert [trade.pnl for trade in result.trades] == [trade.pnl for trade in baseline.trades]
    assert result.metrics["total_costs"] == baseline.metrics["total_costs"]
    assert len(result.fills) == len(baseline.fills) == 1

    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.to_funding_dataframe().equals(payments)


def test_position_opened_at_funding_timestamp_does_not_pay():
    class OpenAtFunding(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == datetime(2024, 1, 2):
                broker.submit_order("BTC-PERP", 1)

    funding = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2)],
            "asset": ["BTC-PERP"],
            "rate": [0.01],
        }
    )
    result = Engine(
        feed=DataFeed(prices_df=_prices()),
        strategy=OpenAtFunding(),
        config=BacktestConfig(initial_cash=1000, execution_mode=ExecutionMode.SAME_BAR),
        funding_df=funding,
    ).run()
    assert result.to_funding_dataframe()["cash_delta"].to_list() == [0.0]
    assert len(result.fills) == 1


@pytest.mark.parametrize(
    "rows",
    [
        [(datetime(2024, 1, 2), "BTC-PERP", 0.01)] * 2,
        [(datetime(2024, 1, 4), "BTC-PERP", 0.01)],
        [(datetime(2024, 1, 2), "BTC-PERP", float("nan"))],
    ],
)
def test_invalid_funding_inputs_fail_before_the_run(rows):
    funding = pl.DataFrame(rows, schema=["timestamp", "asset", "rate"], orient="row")
    with pytest.raises(ValueError, match="funding"):
        _run(1, funding)


def test_amount_per_unit_is_account_currency_per_underlying_unit():
    funding = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2)],
            "asset": ["BTC-PERP"],
            "amount_per_unit": [0.5],
        }
    )
    baseline = _run(1, None)
    result = _run(1, funding)
    payment = result.to_funding_dataframe().row(0, named=True)
    assert payment["cash_delta"] == -1.0  # 1 contract x 2 underlying units x $0.50
    assert result.metrics["final_value"] - baseline.metrics["final_value"] == -1.0


def test_multiple_funding_payments_are_atomic_if_a_mark_is_missing():
    broker = Broker(initial_cash=1000)
    timestamp = datetime(2024, 1, 2)
    broker.positions["A"] = Position("A", 1.0, 100.0, datetime(2024, 1, 1))
    broker.positions["B"] = Position("B", 1.0, 100.0, datetime(2024, 1, 1))
    broker._last_prices["A"] = 100.0
    before_cash = broker.cash
    events = [FundingEvent(timestamp, "A", rate=0.01), FundingEvent(timestamp, "B", rate=0.01)]

    with pytest.raises(ValueError, match="funding mark"):
        broker._apply_funding(events)

    assert broker.cash == before_cash
    assert broker.account._lock_notional_free_cash == before_cash


def test_funding_artifact_rejects_missing_cash_flow_component(tmp_path):
    funding = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2)],
            "asset": ["BTC-PERP"],
            "rate": [0.01],
        }
    )
    _run(1, funding).to_parquet(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["components"].pop("funding")
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ArtifactReadError, match="funding.parquet is required"):
        BacktestResult.from_parquet(tmp_path)


def test_funding_event_on_a_filtered_session_is_rejected_before_cash_changes():
    dates = [datetime(2024, 1, 5), datetime(2024, 1, 6), datetime(2024, 1, 8)]
    prices = pl.DataFrame(
        {
            "timestamp": dates,
            "asset": ["BTC-PERP"] * len(dates),
            "open": [100.0] * len(dates),
            "high": [100.0] * len(dates),
            "low": [100.0] * len(dates),
            "close": [100.0] * len(dates),
            "volume": [1000] * len(dates),
        }
    )
    funding = pl.DataFrame({"timestamp": [dates[1]], "asset": ["BTC-PERP"], "rate": [0.01]})

    class Noop(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pass

    engine = Engine(
        feed=DataFeed(prices_df=prices),
        strategy=Noop(),
        config=BacktestConfig(
            initial_cash=1000,
            calendar="NYSE",
            enforce_sessions=True,
            data_frequency=DataFrequency.DAILY,
        ),
        funding_df=funding,
    )

    with pytest.raises(ValueError, match="excluded by session filtering"):
        engine.run()
    assert engine.broker.cash == 1000.0


@pytest.mark.parametrize("rate,amount", [(None, None), (0.01, 0.5)])
def test_funding_event_requires_one_payment_convention(rate, amount):
    funding = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2)],
            "asset": ["BTC-PERP"],
            "rate": [rate],
            "amount_per_unit": [amount],
        }
    )
    with pytest.raises(ValueError, match="exactly one"):
        _run(1, funding)


def test_funding_without_same_time_asset_bar_uses_last_prior_mark():
    first = datetime(2024, 1, 1)
    event_time = datetime(2024, 1, 2)
    last = datetime(2024, 1, 3)
    prices = pl.DataFrame(
        {
            "timestamp": [first, first, event_time, last, last],
            "asset": ["BTC-PERP", "OTHER", "OTHER", "BTC-PERP", "OTHER"],
            "open": [100.0, 1.0, 1.0, 120.0, 1.0],
            "high": [100.0, 1.0, 1.0, 120.0, 1.0],
            "low": [100.0, 1.0, 1.0, 120.0, 1.0],
            "close": [100.0, 1.0, 1.0, 120.0, 1.0],
            "volume": [1000] * 5,
        }
    )

    class Hold(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == first:
                broker.submit_order("BTC-PERP", 1)

    result = Engine(
        feed=DataFeed(prices_df=prices),
        strategy=Hold(),
        config=BacktestConfig(initial_cash=1000, execution_mode=ExecutionMode.SAME_BAR),
        funding_df=pl.DataFrame({"timestamp": [event_time], "asset": ["BTC-PERP"], "rate": [0.01]}),
    ).run()
    payment = result.to_funding_dataframe().row(0, named=True)
    assert payment["mark_price"] == 100.0
    assert payment["cash_delta"] == -1.0
