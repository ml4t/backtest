"""Raw actions must change the real account once, with causal entitlement and payment."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import ShortCashPolicy
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
    event = action("cash_dividend", cash_per_share=1.0, payment_at=TIMES[4])
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


@pytest.mark.parametrize("policy", ["declared_terms", "zipline_3_2"])
def test_action_artifact_keeps_paid_lineage_and_actual_receivables_without_fabricating_legacy_history(
    tmp_path, policy
):
    from ml4t.backtest.result import BacktestResult

    _, result, _ = run_actions(
        [action("cash_dividend", cash_per_share=1.0, payment_at=TIMES[4])],
        [100, 100, 99, 99, 99],
        policy=policy,
        sell=True,
    )
    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.to_actions_dataframe().equals(result.to_actions_dataframe())
    assert restored.to_entitlements_dataframe().equals(result.to_entitlements_dataframe())
    account = restored.to_portfolio_state_dataframe()
    assert account["receivables_value"].to_list() == (
        [0, 0, 10, 10, 0] if policy == "declared_terms" else [0] * 5
    )
    assert (
        account["equity"].to_list()
        == (account["cash"] + account["net_exposure"] + account["receivables_value"]).to_list()
    )
    terms = __import__("json").loads(restored.to_actions_dataframe()["terms_json"][0])
    assert terms["cash_per_share"] == 1
    assert terms["available_at"] == (TIMES[0] - timedelta(days=1)).isoformat()


@pytest.mark.no_invariant_check
def test_artifact_rejects_a_payment_journal_omission_from_the_actual_engine_producer(
    monkeypatch, tmp_path
):
    from ml4t.backtest.corporate_actions import CorporateActionLedger
    from ml4t.backtest.result import ArtifactWriteError

    original = CorporateActionLedger._record

    def lose_payment(self, event, timestamp, operation, **values):
        if operation != "payment":
            original(self, event, timestamp, operation, **values)

    monkeypatch.setattr(CorporateActionLedger, "_record", lose_payment)
    _, result, _ = run_actions(
        [action("cash_dividend", cash_per_share=1.0, payment_at=TIMES[4])], [100, 100, 99, 99, 99]
    )
    with pytest.raises(ArtifactWriteError, match="incomplete"):
        result.to_parquet(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("quantity", [10, -10])
def test_declared_fractional_cash_in_lieu_recognizes_the_actual_gain_instead_of_disappearing_basis(
    quantity,
):
    engine, result, _ = run_actions(
        [action(split_ratio=1 / 3, fractional_policy="cash_in_lieu", cash_in_lieu_price=310.0)],
        [100, 100, 300, 300, 300],
        quantity=quantity,
    )
    assert engine.broker.get_position("A").quantity == (3 if quantity > 0 else -3)
    assert result.equity_curve[-1][1] == pytest.approx(
        10000 + (10 / 3 - 3) * 10 * (1 if quantity > 0 else -1)
    )


def test_cash_only_conversion_records_realized_gain_without_inventing_a_trade_fill():
    engine, result, _ = run_actions(
        [action("merger", quantity_ratio=0.0, cash_component=120.0)], [100] * 5
    )
    assert engine.broker.get_position("A") is None
    assert engine.broker.cash == 10200
    assert len(result.fills) == 1
    assert result.to_actions_dataframe()["realized_pnl"].sum() == 200


def test_revision_published_after_the_original_effective_instant_cannot_erase_the_original_action():
    old = action(split_ratio=2.0)
    later = action(split_ratio=2.0, revision_id="r2", available_at=TIMES[3], effective_at=TIMES[4])
    with pytest.raises(ValueError, match="Post-effective"):
        run_actions([old, later], [100, 100, 50, 50, 50])


def conversion_run(
    event, *, initial_b=0, b_price=160.0, buy_c=0, rebuy_source=False, policy="declared_terms"
):
    submitted = []

    class Hold(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("A", 10)
                if initial_b:
                    broker.submit_order("B", initial_b)
            if timestamp == TIMES[3]:
                if buy_c:
                    submitted.append(broker.submit_order("C", buy_c))
                if rebuy_source:
                    submitted.append(broker.submit_order("A", 1))

    prices = pl.DataFrame(
        {
            "timestamp": TIMES * 3,
            "asset": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "close": [100.0, 100.0, 80.0, 80.0, 80.0] + [float(b_price)] * 5 + [100.0] * 5,
        }
    )
    config = BacktestConfig(
        initial_cash=10000,
        calendar=None,
        allow_short_selling=True,
        short_cash_policy=ShortCashPolicy.LOCK_NOTIONAL,
        commission_rate=0,
        slippage_rate=0,
        corporate_action_policy=policy,
    )
    engine = Engine(
        DataFeed(prices_df=prices, corporate_actions=pl.DataFrame([event]), input_manifest=RAW),
        Hold(),
        config,
        holdings_sampling="bar",
        contract_specs={a: ContractSpec(a) for a in ["A", "B", "C"]},
    )
    return engine, engine.run(), submitted


def test_security_distribution_that_closes_a_short_releases_the_existing_collateral_for_real_orders():
    event = action("stock_dividend", destination_asset="B", quantity_ratio=0.2, payment_at=TIMES[3])
    engine, result, submitted = conversion_run(event, initial_b=-2, b_price=100, buy_c=90)
    assert submitted[0].status.value == "filled"
    assert engine.broker.get_position("C").quantity == 90
    assert engine.broker.cash == 200
    assert engine.broker.account._lock_notional_short_basis == {}
    assert result.equity_curve[-1][1] == 10000


def test_cash_conversion_cancels_source_orders_and_prevents_reentering_the_converted_identity():
    event = action("merger", quantity_ratio=0.0, cash_component=120.0)
    engine, result, submitted = conversion_run(event, rebuy_source=True)
    assert submitted[0].status.value == "rejected"
    assert submitted[0].rejection_code == "instrument_converted"
    assert engine.broker.get_position("A") is None
    assert len(result.fills) == 1
    assert result.equity_curve[-1][1] == 10200


@pytest.mark.parametrize("policy", ["declared_terms", "zipline_3_2"])
def test_delayed_cash_and_security_conversion_conserves_cash_and_carries_book_basis(policy):
    event = action(
        "merger",
        destination_asset="B",
        quantity_ratio=0.5,
        cash_component=20.0,
        payment_at=TIMES[4],
    )
    engine, result, _ = conversion_run(event, b_price=160, policy=policy)
    assert engine.broker.get_position("A") is None
    assert engine.broker.get_position("B").quantity == 5
    assert engine.broker.get_position("B").entry_price == 160
    assert engine.broker.cash == 9200
    assert result.equity_curve[-1][1] == 10000
    assert len(result.fills) == 1


def test_split_on_a_sparse_panel_retains_the_original_price_observation_time():
    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("A", 10)

    frame = pl.DataFrame(
        {"timestamp": TIMES[:3], "asset": ["A", "A", "B"], "close": [100.0, 100.0, 1.0]}
    )
    result = Engine(
        DataFeed(
            prices_df=frame,
            corporate_actions=pl.DataFrame([action(split_ratio=2.0)]),
            input_manifest=RAW,
        ),
        Buy(),
        BacktestConfig(initial_cash=10000, calendar=None, commission_rate=0, slippage_rate=0),
        contract_specs={"A": ContractSpec("A")},
        holdings_sampling="bar",
    ).run()
    held = result.to_holdings_dataframe().filter(
        (pl.col("timestamp") == TIMES[2]) & (pl.col("asset") == "A")
    )
    assert held["mark_source"].to_list() == ["action_adjusted_previous_mark"]
    assert held["mark_timestamp"].to_list() == [TIMES[1]]
    assert result.equity_curve[-1][1] == 10000
    assert len(result.fills) == 1


def test_own_share_dividend_rebases_only_the_account_mark_without_creating_a_future_quote():
    from ml4t.specs import FeedSpec, LifecyclePhase

    from ml4t.backtest import RuntimeServices
    from ml4t.backtest.execution.schedule import RebalanceSchedule

    runtime = RuntimeServices()
    seen = []

    def observe(timestamp, data, context, broker):
        seen.append(
            (
                broker.get_account_value(),
                runtime.market.current(["A"], ["close"], forward_fill=True)["close"][0],
                broker.get_last_price("A"),
                broker.account.total_equity,
            )
        )

    runtime.scheduler.register(
        observe,
        phase=LifecyclePhase.INTRABAR,
        at="clock",
        schedule=RebalanceSchedule.explicit_timestamps([TIMES[2] - timedelta(hours=1)]),
    )

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("A", 10)

    frame = pl.DataFrame(
        {
            "timestamp": TIMES,
            "asset": ["A"] * 5,
            "close": [100.0, 100.0, 100 / 1.2, 100 / 1.2, 100 / 1.2],
        }
    )
    event = action(
        "stock_dividend",
        destination_asset="A",
        quantity_ratio=0.2,
        effective_at=TIMES[2] - timedelta(hours=2),
        payment_at=TIMES[4],
    )
    result = Engine(
        DataFeed(
            prices_df=frame,
            corporate_actions=pl.DataFrame([event]),
            input_manifest=RAW,
            feed_spec=FeedSpec(
                calendar=None, time_contract_version=2, timestamp_semantics="event_time"
            ),
        ),
        Buy(),
        BacktestConfig(initial_cash=10000, calendar=None, commission_rate=0, slippage_rate=0),
        contract_specs={"A": ContractSpec("A")},
        runtime=runtime,
    ).run()
    assert seen[0][0] == pytest.approx(10000, rel=0, abs=1e-9)
    assert seen[0][1] == 100
    assert seen[0][2] == 100
    assert seen[0][3] == pytest.approx(10000, rel=0, abs=1e-9)
    assert result.equity_curve[-1][1] == pytest.approx(10000, rel=0, abs=1e-9)
