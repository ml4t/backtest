"""Raw actions must change the real account once, with causal entitlement and payment."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.types import ContractSpec

START = datetime(2025, 3, 3, 21, tzinfo=UTC)
TIMES = [START + timedelta(days=i) for i in range(5)]
RAW = {"price_basis": "raw", "share_basis": "source"}


def action(kind="split", **terms):
    return {
        "source": "declared_fixture",
        "event_id": "action-1",
        "revision_id": "r1",
        "asset": "A",
        "action_kind": kind,
        "available_at": TIMES[0] - timedelta(days=1),
        "effective_at": TIMES[2],
        "currency": "USD",
        "share_basis": "pre_event",
        "split_ratio_convention": "new_per_old",
        **terms,
    }


def run_actions(events, closes, *, quantity=10, policy="declared_terms", sell=False):
    observed = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("A", quantity)
            if sell and timestamp == TIMES[2]:
                broker.submit_order("A", -quantity)
            observed.append(
                (
                    timestamp,
                    broker.cash,
                    broker.get_account_value(),
                    broker.get_position("A").quantity if broker.get_position("A") else 0,
                )
            )

    prices = pl.DataFrame(
        {"timestamp": TIMES, "asset": ["A"] * 5, "close": [float(p) for p in closes]}
    )
    feed = DataFeed(prices_df=prices, corporate_actions=pl.DataFrame(events), input_manifest=RAW)
    config = BacktestConfig(
        initial_cash=10000,
        calendar=None,
        allow_short_selling=True,
        commission_rate=0,
        slippage_rate=0,
        corporate_action_policy=policy,
    )
    engine = Engine(
        feed, Buy(), config, holdings_sampling="bar", contract_specs={"A": ContractSpec("A")}
    )
    return engine, engine.run(), observed


@pytest.mark.parametrize("quantity", [10, -10])
@pytest.mark.parametrize("ratio", [2.0, 0.5])
def test_raw_split_conserves_actual_quantity_cash_basis_and_nav(quantity, ratio):
    engine, result, _ = run_actions(
        [action(split_ratio=ratio)], [100, 100, *([100 / ratio] * 3)], quantity=quantity
    )
    position = engine.broker.get_position("A")
    assert position.quantity == quantity * ratio
    assert position.entry_price == 100 / ratio
    assert engine.broker.cash == 10000 - quantity * 100
    assert result.to_equity_dataframe()["equity"].to_list() == [10000] * 5
    assert len(result.fills) == 1
    assert result.fills[0].quantity == abs(quantity)
    applications = result.to_actions_dataframe()
    assert applications.filter(pl.col("operation") == "split").height == 1


@pytest.mark.parametrize("quantity", [10, -10])
@pytest.mark.parametrize("policy", ["declared_terms", "zipline_3_2"])
def test_dividend_entitlement_survives_sale_and_only_changes_cash_on_payment(quantity, policy):
    event = action("cash_dividend", cash_per_share=1.0, payment_date=TIMES[4])
    engine, result, observed = run_actions(
        [event], [100, 100, 99, 99, 99], quantity=quantity, policy=policy, sell=True
    )
    assert engine.broker.get_position("A") is None
    assert engine.broker.cash == 10000
    expected_ex_nav = 10000 if policy == "declared_terms" else 10000 - quantity
    assert observed[2][2] == expected_ex_nav
    assert observed[2][1] == 10000 - quantity * 100
    applications = result.to_actions_dataframe()
    assert applications.filter(pl.col("operation") == "payment")["cash_delta"].to_list() == [
        quantity
    ]
    entitlements = result.to_entitlements_dataframe()
    assert entitlements["eligible_quantity"].to_list() == [quantity]
    assert entitlements["paid"].to_list() == [True]


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_invalid_split_is_rejected_before_strategy_or_account_mutation(ratio):
    with pytest.raises(ValueError, match="split_ratio"):
        run_actions([action(split_ratio=ratio)], [100] * 5)


def test_raw_action_against_adjusted_prices_is_rejected_before_execution():
    prices = pl.DataFrame({"timestamp": TIMES, "asset": ["A"] * 5, "close": [100.0] * 5})
    feed = DataFeed(
        prices_df=prices,
        corporate_actions=pl.DataFrame([action(split_ratio=2)]),
        input_manifest={"price_basis": "split_adjusted"},
    )
    with pytest.raises(ValueError, match="raw"):

        class Cash(Strategy):
            def on_data(self, *args):
                pass

        Engine(feed, Cash(), BacktestConfig(calendar=None))
