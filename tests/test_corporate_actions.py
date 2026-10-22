"""Raw actions must change the real account once, with causal entitlement and payment."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy, run_backtest
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


def test_convenience_runner_consumes_raw_actions_instead_of_silently_losing_the_share_change():
    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("A", 10)

    result = run_backtest(
        pl.DataFrame(
            {"timestamp": TIMES, "asset": ["A"] * 5, "close": [100.0, 100.0, 50.0, 50.0, 50.0]}
        ),
        Buy(),
        config=BacktestConfig(
            calendar=None, initial_cash=10000, commission_rate=0, slippage_rate=0
        ),
        corporate_actions=pl.DataFrame([action(split_ratio=2.0)]),
        input_manifest=RAW,
        contract_specs={"A": ContractSpec("A")},
        holdings_sampling="bar",
    )
    assert result.to_equity_dataframe()["equity"].to_list() == [10000] * 5
    assert result.to_actions_dataframe().filter(pl.col("operation") == "split")[
        "quantity_after"
    ].to_list() == [20]


def test_physical_entitlement_overrides_civil_effective_label_when_calendars_are_disabled():
    event = action(
        "cash_dividend",
        cash_per_share=1.0,
        effective_at=TIMES[2].date(),
        entitlement_at=TIMES[2],
        payment_at=TIMES[4],
    )
    engine, result, _ = run_actions([event], [100, 100, 99, 99, 99])
    assert engine.broker.cash == 9010
    assert result.to_entitlements_dataframe()["entitlement_at"].to_list() == [TIMES[2]]


def test_duplicate_revision_pays_once_and_a_pre_effective_revision_replaces_earlier_terms():
    first = action("cash_dividend", cash_per_share=1.0, payment_at=TIMES[4])
    revised = {**first, "revision_id": "r2", "available_at": TIMES[1], "cash_per_share": 2.0}
    engine, result, _ = run_actions([first, first, revised, revised], [100, 100, 98, 98, 98])
    assert engine.broker.cash == 9020
    assert result.to_entitlements_dataframe()["revision_id"].to_list() == ["r2"]
    assert result.to_actions_dataframe().filter(pl.col("operation") == "payment").height == 1


def test_conflicting_same_revision_does_not_choose_arbitrary_distribution_terms():
    first = action("cash_dividend", cash_per_share=1.0, payment_at=TIMES[4])
    with pytest.raises(ValueError, match="Conflicting terms"):
        run_actions([first, {**first, "cash_per_share": 2.0}], [100] * 5)


@pytest.mark.parametrize("share_basis,expected", [("pre_event", 10.0), ("post_event", 20.0)])
def test_same_clock_split_eligibility_uses_the_explicit_share_basis(share_basis, expected):
    dividend = action(
        "cash_dividend",
        event_id="dividend",
        share_basis=share_basis,
        cash_per_share=1.0,
        payment_at=TIMES[4],
    )
    _, result, _ = run_actions([action(split_ratio=2.0), dividend], [100, 100, 49, 49, 49])
    assert result.to_entitlements_dataframe()["eligible_quantity"].to_list() == [expected]
    assert result.to_actions_dataframe().filter(pl.col("operation") == "payment")[
        "cash_delta"
    ].to_list() == [expected]


def test_same_clock_failure_rolls_back_prior_split_position_cash_and_action_journal():
    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("A", 10)

    # Both declarations are individually finite, but their composition overflows actual holdings.
    events = [action(split_ratio=2.0), action(event_id="later", split_ratio=1e308)]
    engine = Engine(
        DataFeed(
            prices_df=pl.DataFrame({"timestamp": TIMES, "asset": ["A"] * 5, "close": [100.0] * 5}),
            corporate_actions=pl.DataFrame(events),
            input_manifest=RAW,
        ),
        Buy(),
        BacktestConfig(calendar=None, initial_cash=10000, commission_rate=0, slippage_rate=0),
        contract_specs={"A": ContractSpec("A")},
    )
    with pytest.raises(ValueError, match="nonfinite"):
        engine.run()
    assert engine.broker.cash == 9000
    assert engine.broker.get_position("A").quantity == 10
    assert engine.broker.get_position("A").entry_price == 100
    assert engine._corporate_actions.records == []


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


def test_after_fill_and_session_observations_capture_the_actual_unpaid_receivable(tmp_path):
    from ml4t.backtest import BacktestResult

    class Trade(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("A", 10)
            if timestamp == TIMES[2]:
                broker.submit_order("A", -10)

    feed = DataFeed(
        prices_df=pl.DataFrame(
            {"timestamp": TIMES, "asset": ["A"] * 5, "close": [100.0, 100.0, 99.0, 99.0, 99.0]}
        ),
        corporate_actions=pl.DataFrame(
            [action("cash_dividend", cash_per_share=1.0, payment_at=TIMES[4])]
        ),
        input_manifest=RAW,
        feed_spec={
            "time_contract_version": 2,
            "calendar": "NYSE",
            "timezone": "UTC",
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "data_frequency": "1d",
        },
    )
    result = Engine(
        feed,
        Trade(),
        BacktestConfig(initial_cash=10000, commission_rate=0, slippage_rate=0),
        contract_specs={"A": ContractSpec("A")},
        holdings_sampling="fill_and_session",
    ).run()
    account = result.to_portfolio_state_dataframe()
    fills = account.filter(pl.col("observation_kind") == "after_fill")
    assert fills["timestamp"].to_list() == [TIMES[1], TIMES[3]]
    assert fills["cash"].to_list() == [9000, 9990]
    assert fills["receivables_value"].to_list() == [0, 10]
    assert account["equity"].to_list() == [10000] * 7
    result.to_parquet(tmp_path)
    assert BacktestResult.from_parquet(tmp_path).to_portfolio_state_dataframe().equals(account)


@pytest.mark.no_invariant_check
@pytest.mark.parametrize("lost_observations", [None, []])
def test_missing_receivable_observations_from_the_producer_cannot_be_exported_as_complete(
    monkeypatch, tmp_path, lost_observations
):
    from ml4t.backtest import Broker
    from ml4t.backtest.result import ArtifactError

    original = Broker._corporate_receivable_observations
    monkeypatch.setattr(
        Broker, "_corporate_receivable_observations", lambda *_args: lost_observations
    )
    event = action("cash_dividend", cash_per_share=1.0, payment_at=TIMES[4])
    _, result, _ = run_actions([event], [100, 100, 99, 99, 99])
    destination = tmp_path / "result"
    with pytest.raises(ArtifactError, match="receivables"):
        result.to_parquet(destination)
    assert not destination.exists()
    monkeypatch.setattr(Broker, "_corporate_receivable_observations", original)
    _, valid, _ = run_actions([event], [100, 100, 99, 99, 99])
    valid.to_parquet(destination)


def test_actual_legacy_artifact_does_not_invent_zero_action_history():
    from pathlib import Path

    from ml4t.backtest import BacktestResult

    result = BacktestResult.from_parquet(
        Path(__file__).parent / "fixtures/artifacts/v0.1.0b22", recovery=True
    )
    with pytest.raises(ValueError, match="captured"):
        result.to_actions_dataframe()
    with pytest.raises(ValueError, match="captured"):
        result.to_entitlements_dataframe()


@pytest.mark.parametrize("fault", ["cash_delta", "terms", "nonobject_terms", "nav", "schema"])
def test_reader_rejects_actual_parquet_writer_faults_before_exposing_action_results(
    tmp_path, monkeypatch, fault
):
    import json
    from pathlib import Path

    from ml4t.backtest.result import ArtifactManifestError, ArtifactReadError, BacktestResult

    _, result, _ = run_actions(
        [action("cash_dividend", cash_per_share=1.0, payment_at=TIMES[4])],
        [100, 100, 99, 99, 99],
    )
    writer = pl.DataFrame.write_parquet

    def lose_financial_fact(frame, file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            name = Path(file).name
            if name == "corporate_actions.parquet" and fault == "cash_delta":
                frame = frame.with_columns(pl.lit(0.0).alias("cash_delta"))
            elif name == "corporate_actions.parquet" and fault == "terms":
                frame = frame.with_columns(
                    pl.col("terms_json").str.replace(
                        '"cash_per_share": 1.0', '"cash_per_share": 2.0'
                    )
                )
            elif name == "corporate_actions.parquet" and fault == "nonobject_terms":
                frame = frame.with_columns(pl.lit("[]").alias("terms_json"))
            elif name == "portfolio_state.parquet" and fault == "nav":
                frame = frame.with_columns((pl.col("equity") + 1.0).alias("equity"))
        return writer(frame, file, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", lose_financial_fact)
    result.to_parquet(tmp_path)
    if fault == "schema":
        path = tmp_path / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["schema_version"] = 3
        path.write_text(json.dumps(manifest))
    with pytest.raises(
        (ArtifactReadError, ArtifactManifestError), match="Corporate|corporate|NAV|Payment"
    ):
        BacktestResult.from_parquet(tmp_path)
    monkeypatch.setattr(pl.DataFrame, "write_parquet", writer)
    result.to_parquet(tmp_path)
    assert BacktestResult.from_parquet(tmp_path).to_entitlements_dataframe()[
        "cash_amount"
    ].to_list() == [10]


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
