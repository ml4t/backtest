"""Explicit per-asset fees must use committed partial fills and contract units."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from ml4t.backtest import (
    AssetClass,
    BacktestConfig,
    ContractSpec,
    DataFeed,
    Engine,
    Strategy,
    run_backtest,
)
from ml4t.backtest.broker import Broker
from ml4t.backtest.execution import VolumeParticipationLimit
from ml4t.backtest.models import PerOrderCommission, PerOrderPerShareCommission


def _model(method="per_contract"):
    from ml4t.backtest.models import AssetCommission

    return AssetCommission(
        {
            "ES": PerOrderCommission(per_share=2.0, per_trade=3.0, minimum=8.0)
            if method == "per_contract"
            else PerOrderCommission(per_trade=3.0)
        },
        default=PerOrderPerShareCommission(per_share=0.2, minimum=1.0),
    )


def _prices():
    return pl.DataFrame(
        [
            {
                "timestamp": datetime(2025, 3, day, 21, tzinfo=UTC),
                "asset": asset,
                "close": 100.0,
                "volume": 2.0,
            }
            for day in [3, 4, 5, 6]
            for asset in ["A", "ES"]
        ]
    )


@pytest.mark.parametrize("method", ["per_contract", "per_future_trade"])
@pytest.mark.parametrize("entry", ["engine", "from_config", "run_backtest"])
def test_mixed_asset_fees_charge_exchange_once_and_do_not_multiply_per_contract_cost(entry, method):
    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 3:
                broker.submit_order("A", 3)
                broker.submit_order("ES", 3)

    config = BacktestConfig(calendar=None, initial_cash=100000)
    kwargs = {
        "commission_model": _model(method),
        "execution_limits": VolumeParticipationLimit(0.5),
        "contract_specs": {"ES": ContractSpec("ES", AssetClass.FUTURE, multiplier=50.0)},
    }
    if entry == "run_backtest":
        result = run_backtest(_prices(), Buy(), config=config, **kwargs)
    else:
        constructor = Engine if entry == "engine" else Engine.from_config
        result = constructor(DataFeed(prices_df=_prices()), Buy(), config, **kwargs).run()
    fills = result.to_fills_dataframe()
    assert fills.filter(pl.col("asset") == "ES")["commission"].to_list() == (
        [8.0, 0.0, 1.0] if method == "per_contract" else [3.0, 0.0, 0.0]
    )
    assert fills.filter(pl.col("asset") == "A")["commission"].to_list() == [1.0, 0.0, 0.0]
    assert fills["quantity"].to_list() == [1.0] * 6
    nav = result.to_equity_dataframe()["equity"].to_list()
    assert nav == (
        [100000.0, 99991.0, 99991.0, 99990.0]
        if method == "per_contract"
        else [100000.0, 99996.0, 99996.0, 99996.0]
    )
    reference = json.loads(
        (Path(__file__).parent / "fixtures/zipline_original/future-fee-reference.json").read_text()
    )["methods"][method]
    assert nav == reference["nav"]
    captured = json.loads(
        (Path(__file__).parent / "fixtures/zipline_original/future-fee-reference.json").read_text()
    )
    assert [point.isoformat() for point in _prices()["timestamp"].unique().sort()] == captured[
        "closes"
    ]
    # Signed-notional native account is deliberately separate from variation-cash accounting.
    assert result.to_portfolio_state_dataframe()["cash"][-1] == (
        84690.0 if method == "per_contract" else 84696.0
    )


def test_broker_config_accepts_the_same_explicit_native_fee_model():
    model = _model()
    broker = Broker.from_config(BacktestConfig(calendar=None), commission_model=model)
    assert broker.commission_model is model


def test_fee_estimates_do_not_advance_the_selected_asset_models_actual_fill_state():
    from ml4t.backtest.models import AssetCommission, NoCommission

    class CountingFee:
        def __init__(self):
            self.fills = 0

        def calculate(self, asset, quantity, price):
            self.fills += 1
            return float(self.fills)

    selected = CountingFee()
    model = AssetCommission({"ES": selected}, default=NoCommission())

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 3:
                broker.submit_order("ES", 3)

    result = Engine(
        DataFeed(prices_df=_prices()),
        Buy(),
        BacktestConfig(calendar=None),
        commission_model=model,
        execution_limits=VolumeParticipationLimit(0.5),
    ).run()
    assert result.to_fills_dataframe()["commission"].to_list() == [1.0, 2.0, 3.0]
    assert selected.fills == 3


def test_invalid_explicit_fee_model_is_rejected_before_any_callback():
    calls = []

    class Read(Strategy):
        def on_start(self, broker):
            calls.append("started")

        def on_data(self, *args):
            calls.append("bar")

    with pytest.raises(TypeError, match="CommissionModel"):
        Engine(DataFeed(prices_df=_prices()), Read(), commission_model=object()).run()
    assert calls == []
