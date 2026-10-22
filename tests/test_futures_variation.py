"""Public Engine acceptance for opt-in variation cash; independent dollar oracles."""

from datetime import UTC, datetime, timedelta

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
from ml4t.backtest.config import CommissionType, ExecutionMode

TIMES = [datetime(2025, 3, 3, 21, tzinfo=UTC) + timedelta(days=i) for i in range(4)]


def run_future(
    quantity,
    *,
    mode="on_mark",
    settlements=None,
    legacy=False,
    runner="direct",
    sampling="bar",
    config_options=None,
):
    reads = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("ES", quantity)
            before = broker.cash
            for _ in range(3):
                broker.get_account_value()
            assert broker.cash == before
            reads.append(broker.cash)

    options = (
        {} if legacy else {"accounting_method": "futures_variation", "settlement_policy": mode}
    )
    spec = ContractSpec(
        "ES", asset_class=AssetClass.FUTURE, multiplier=50, currency="USD", **options
    )
    frame = pl.DataFrame(
        {"timestamp": TIMES, "asset": ["ES"] * 4, "close": [4000.0, 4000.0, 4010.0, 4010.0]}
    )
    config = BacktestConfig(
        calendar="NYSE" if sampling in {"session", "fill_and_session"} else None,
        initial_cash=100000,
        allow_leverage=True,
        allow_short_selling=True,
        commission_type=CommissionType.PER_TRADE,
        commission_per_trade=2,
        slippage_rate=0,
        fixed_margin_schedule={"ES": (15000, 12000)},
        **(config_options or {}),
    )
    kwargs = {} if settlements is None else {"settlement_events": settlements}
    if runner == "convenience":
        return (
            None,
            run_backtest(
                frame,
                Buy(),
                config=config,
                contract_specs={"ES": spec},
                holdings_sampling=sampling,
                **kwargs,
            ),
            reads,
        )
    constructor = Engine.from_config if runner == "from_config" else Engine
    engine = constructor(
        DataFeed(prices_df=frame),
        Buy(),
        config,
        contract_specs={"ES": spec},
        holdings_sampling=sampling,
        **kwargs,
    )
    return engine, engine.run(), reads


@pytest.mark.parametrize(
    "flags",
    [
        {"next_bar_submission_precheck": True},
        {"next_bar_submission_precheck": True, "next_bar_simple_cash_check": True},
        {"buying_power_reservation": True},
        {"next_bar_queue_shadow_validation": True},
        {
            "next_bar_submission_precheck": True,
            "next_bar_queue_shadow_validation": True,
            "buying_power_reservation": True,
        },
    ],
)
@pytest.mark.parametrize("quantity", [1, -1])
@pytest.mark.parametrize("mode", ["scheduled", "on_mark"])
def test_public_submission_and_reservation_options_do_not_spend_futures_principal(
    flags, quantity, mode
):
    _, result, reads = run_future(quantity, mode=mode, config_options=flags)
    marked_cash = 99998 + quantity * 500 if mode == "on_mark" else 99998
    assert reads == [100000, 99998, marked_cash, marked_cash]
    assert len(result.fills) == 1
    assert result.to_portfolio_state_dataframe()["initial_margin_required"][-1] == 15000


@pytest.mark.parametrize("cash,filled", [(30004, 2), (30002, 1)])
@pytest.mark.parametrize(
    "flags",
    [
        {"next_bar_submission_precheck": True},
        {"buying_power_reservation": True},
        {"next_bar_queue_shadow_validation": True},
        {"next_bar_submission_precheck": True, "buying_power_reservation": True},
    ],
)
def test_projected_future_reservations_charge_fees_once_and_share_actual_margin(
    cash, filled, flags
):
    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("ES", 1)
                broker.submit_order("NQ", 1)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": [t for t in TIMES[:2] for _ in range(2)],
                    "asset": ["ES", "NQ"] * 2,
                    "close": [4000.0, 15000.0] * 2,
                }
            )
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=cash,
            allow_leverage=True,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
            fixed_margin_schedule={"ES": (15000, 12000), "NQ": (15000, 12000)},
            **flags,
        ),
        contract_specs={
            asset: ContractSpec(
                asset,
                asset_class=AssetClass.FUTURE,
                multiplier=multiplier,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
            for asset, multiplier in [("ES", 50), ("NQ", 20)]
        },
    ).run()
    assert len(result.fills) == filled
    assert result.to_portfolio_state_dataframe()["cash"][-1] == cash - filled * 2
    assert result.to_portfolio_state_dataframe()["initial_margin_required"][-1] == filled * 15000


@pytest.mark.parametrize("gap", [False, True])
def test_intraday_reservation_refreshes_from_actual_gap_rejections_not_phantom_positions(gap):
    times = [TIMES[0] + timedelta(minutes=15 * i) for i in range(3)]

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("ES", 1)
            if timestamp == times[1]:
                broker.submit_order("NQ", 1)

    es_price = 10000.0 if gap else 4000.0
    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": [t for t in times for _ in range(2)],
                    "asset": ["ES", "NQ"] * 3,
                    "close": [4000.0, 100.0, es_price, 100.0, es_price, 100.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=20002,
            allow_leverage=True,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
            next_bar_submission_precheck=True,
        ),
        contract_specs={
            "ES": ContractSpec(
                "ES",
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                margin_pct=(0.05, 0.04),
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            ),
            "NQ": ContractSpec(
                "NQ",
                asset_class=AssetClass.FUTURE,
                multiplier=20,
                margin=15000,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            ),
        },
    ).run()
    assert [fill.asset for fill in result.fills] == ["NQ" if gap else "ES"]
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 20000


@pytest.mark.parametrize("sampling", ["none", "bar", "session", "fill_and_session"])
def test_variation_artifact_retains_liquidation_margin_and_actual_payouts(tmp_path, sampling):
    from ml4t.backtest import BacktestResult

    _, result, _ = run_future(1, sampling=sampling)
    assert result.account_method == "per_instrument"
    accounts = result.to_portfolio_state_dataframe()
    assert accounts["positions_value"].to_list() == [0] * accounts.height
    assert accounts["initial_margin_required"][-1] == 15000
    assert accounts["maintenance_margin_required"][-1] == 12000
    assert accounts["available_funds"][-1] == 85498
    assert accounts["net_exposure"][-1] == 200500
    settlements = result.to_futures_settlements_dataframe()
    assert settlements["cash_delta"].sum() == 500
    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.account_method == "per_instrument"
    assert restored.instrument_accounting == result.instrument_accounting
    assert restored.to_portfolio_state_dataframe().equals(accounts)
    assert restored.to_futures_settlements_dataframe().equals(settlements)


@pytest.mark.parametrize("quantity", [1, -1])
def test_mark_variation_has_no_principal_cash_movement_and_reads_cannot_pay_again(quantity):
    engine, result, reads = run_future(quantity)
    assert reads == [100000, 99998, 99998 + quantity * 500, 99998 + quantity * 500]
    assert result.to_equity_dataframe()["equity"].to_list() == reads
    assert engine.broker.get_position("ES").quantity == quantity
    holdings = result.to_holdings_dataframe()
    assert holdings["signed_notional"][-1] == quantity * 200500
    assert holdings["liquidation_value"][-1] == 0


@pytest.mark.parametrize("quantity", [1, -1])
def test_scheduled_variation_keeps_unpaid_pnl_out_of_cash_until_declared_payment(quantity):
    engine, result, reads = run_future(quantity, mode="scheduled")
    assert reads == [100000, 99998, 99998, 99998]
    assert result.to_equity_dataframe()["equity"].to_list() == [
        100000,
        99998,
        99998 + quantity * 500,
        99998 + quantity * 500,
    ]
    assert result.to_holdings_dataframe()["liquidation_value"][-1] == quantity * 500


@pytest.mark.parametrize("quantity", [1, -1])
def test_explicit_settlement_transfers_pnl_once_without_changing_nav(quantity):
    event = pl.DataFrame(
        [
            {
                "timestamp": TIMES[3],
                "asset": "ES",
                "available_at": TIMES[2],
                "settlement_price": 4010.0,
                "currency": "USD",
                "source": "declared-settlement",
                "event_id": "day3",
                "revision_id": "r1",
            }
        ]
    )
    _, result, reads = run_future(quantity, mode="scheduled", settlements=event)
    assert reads[-2:] == [99998, 99998 + quantity * 500]
    assert result.to_equity_dataframe()["equity"].to_list()[-2:] == [99998 + quantity * 500] * 2


def test_default_signed_notional_account_does_not_change_when_variation_is_added():
    _, result, reads = run_future(1, legacy=True)
    assert reads == [100000, -100002, -100002, -100002]
    assert result.to_equity_dataframe()["equity"].to_list() == [100000, 99998, 100498, 100498]


@pytest.mark.parametrize("runner", ["from_config", "convenience"])
def test_public_constructors_forward_actual_scheduled_settlement_events(runner):
    events = pl.DataFrame(
        [
            {
                "timestamp": TIMES[3],
                "asset": "ES",
                "available_at": TIMES[2],
                "settlement_price": 4010.0,
                "currency": "USD",
                "source": "declared-settlement",
                "event_id": "day3",
                "revision_id": "r1",
            }
        ]
    )
    _, result, reads = run_future(1, mode="scheduled", settlements=events, runner=runner)
    assert reads[-2:] == [99998, 100498]
    assert result.to_equity_dataframe()["equity"].to_list()[-2:] == [100498, 100498]


@pytest.mark.parametrize("mode,liquidation", [("scheduled", 500.0), ("on_mark", 0.0)])
def test_futures_account_snapshot_keeps_exposure_distinct_from_liquidation_and_margin(
    mode, liquidation
):
    engine, _, _ = run_future(1, mode=mode)
    account = engine.broker.get_account_snapshot()
    assert account.positions_value == liquidation
    assert account.net_exposure == 200500
    assert account.equity == 100498
    assert account.initial_margin_required == 15000
    assert account.maintenance_margin_required == 12000
    assert account.available_funds == 85498
    assert account.futures_unsettled_pnl == liquidation
    assert account.margin_breach is False


@pytest.mark.parametrize("mode", ["scheduled", "on_mark"])
def test_a_later_nonfinite_payout_cannot_leave_an_earlier_future_partially_settled(mode):
    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("A", 1)
                broker.submit_order("C", 1)

    # Each declared price/multiplier is finite. Their later P&L product overflows.
    frame = pl.DataFrame(
        {
            "timestamp": [t for t in TIMES for _ in range(2)],
            "asset": ["A", "C"] * 4,
            "close": [100.0, 1.0, 100.0, 1.0, 110.0, 3.0, 110.0, 3.0],
        }
    )
    events = (
        None
        if mode == "on_mark"
        else pl.DataFrame(
            [
                {
                    "timestamp": TIMES[2],
                    "asset": asset,
                    "available_at": TIMES[1],
                    "settlement_price": price,
                    "currency": "USD",
                    "source": "declared-settlement",
                    "event_id": asset,
                    "revision_id": "r1",
                }
                for asset, price in [("A", 110.0), ("C", 3.0)]
            ]
        )
    )
    specs = {
        asset: ContractSpec(
            asset,
            asset_class=AssetClass.FUTURE,
            multiplier=multiplier,
            accounting_method="futures_variation",
            settlement_policy=mode,
        )
        for asset, multiplier in [("A", 1.0), ("C", 1e308)]
    }
    engine = Engine(
        DataFeed(prices_df=frame),
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            commission_rate=0,
            slippage_rate=0,
            fixed_margin_schedule={"A": (10.0, 5.0), "C": (1.0, 0.5)},
        ),
        contract_specs=specs,
        settlement_events=events,
    )
    with pytest.raises(ValueError, match="nonfinite"):
        engine.run()
    assert engine.broker.cash == 100000
    assert engine.broker.get_position("A").variation_basis == 100
    assert engine.broker.get_position("C").variation_basis == 1
    assert engine._futures_ledger.records == []


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"next_bar_submission_precheck": True},
        {"buying_power_reservation": True},
        {"next_bar_queue_shadow_validation": True},
    ],
)
def test_two_scheduled_settlements_scale_in_partial_close_and_final_close_do_not_pay_twice(flags):
    times = [TIMES[0] + timedelta(days=i) for i in range(7)]
    cash = []

    class Trade(Strategy):
        def on_data(self, timestamp, data, context, broker):
            plan = {times[0]: 2, times[1]: 2, times[2]: -1, times[4]: -3}
            if timestamp in plan:
                broker.submit_order("ES", plan[timestamp])
            cash.append(broker.cash)

    events = pl.DataFrame(
        [
            {
                "timestamp": timestamp,
                "available_at": timestamp,
                "asset": "ES",
                "settlement_price": price,
                "currency": "USD",
                "source": "declared-settlement",
                "event_id": str(i),
                "revision_id": "r1",
            }
            for i, (timestamp, price) in enumerate([(times[4], 4030.0), (times[6], 4045.0)])
        ]
    )
    engine = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["ES"] * 7,
                    "close": [4000.0, 4000.0, 4020.0, 4020.0, 4030.0, 4040.0, 4045.0],
                }
            )
        ),
        Trade(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
            fixed_margin_schedule={"ES": (15000, 12000)},
            **flags,
        ),
        contract_specs={
            "ES": ContractSpec(
                "ES",
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                accounting_method="futures_variation",
            )
        },
        settlement_events=events,
        holdings_sampling="bar",
    )
    result = engine.run()
    assert cash == [100000, 99998, 99996, 100494, 103494, 104992, 104992]
    assert result.to_equity_dataframe()["equity"].to_list() == [
        100000,
        99998,
        101996,
        101994,
        103494,
        104992,
        104992,
    ]
    assert engine.broker.get_position("ES") is None
    assert [row["cash_delta"] for row in engine._futures_ledger.records] == [500, 3000, 1500, 0]
    assert [fill.quantity for fill in result.fills] == [2, 2, 1, 3]
    assert sum(fill.commission for fill in result.fills) == 8


def test_mixed_equity_future_nav_uses_liquidation_value_while_margin_uses_notional():
    snapshots = []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("A", 10)
                broker.submit_order("ES", 1)
            snapshots.append(broker.get_account_snapshot())

    prices = pl.DataFrame(
        {
            "timestamp": [t for t in TIMES for _ in range(2)],
            "asset": ["A", "ES"] * 4,
            "close": [100.0, 4000.0, 100.0, 4000.0, 110.0, 4010.0, 110.0, 4010.0],
        }
    )
    result = Engine(
        DataFeed(prices_df=prices),
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
            fixed_margin_schedule={"ES": (15000, 12000)},
        ),
        contract_specs={
            "A": ContractSpec("A"),
            "ES": ContractSpec(
                "ES",
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            ),
        },
        holdings_sampling="bar",
    ).run()
    final = snapshots[-1]
    assert final.cash == 99496
    assert final.equity == 100596
    assert final.net_exposure == 201600
    assert final.positions_value == 1100
    assert final.futures_unsettled_pnl == 0
    assert final.initial_margin_required == 15550
    assert final.maintenance_margin_required == 12275
    assert final.available_funds == 85046
    assert result.to_equity_dataframe()["equity"].to_list() == [100000, 99996, 100596, 100596]


def test_settlement_price_is_a_cash_transfer_basis_and_does_not_invent_a_tradable_quote():
    events = pl.DataFrame(
        [
            {
                "timestamp": TIMES[3],
                "asset": "ES",
                "available_at": TIMES[3],
                "settlement_price": 4015.0,
                "currency": "USD",
                "source": "declared-settlement",
                "event_id": "day3",
                "revision_id": "r1",
            }
        ]
    )
    engine, result, cash = run_future(1, mode="scheduled", settlements=events)
    assert cash[-1] == 100748
    assert engine.broker.get_last_price("ES") == 4010
    assert result.to_equity_dataframe()["equity"].to_list()[-1] == 100498
    assert result.to_holdings_dataframe()["liquidation_value"][-1] == -250


def test_unpaid_dividend_cannot_finance_future_margin_before_its_actual_payment(tmp_path):
    from ml4t.specs import FeedSpec

    from ml4t.backtest import BacktestResult, BenchmarkData

    snapshots, attempts = [], []

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("A", 10)
                broker.submit_order("ES", 1)
            if timestamp in TIMES[2:]:
                snapshots.append(broker.get_account_snapshot())
                attempts.append(broker.submit_order("F2", 1))

    prices = pl.DataFrame(
        {
            "timestamp": [t for t in TIMES for _ in range(3)],
            "asset": ["A", "ES", "F2"] * 4,
            "close": [
                100.0,
                4000.0,
                100.0,
                100.0,
                4000.0,
                100.0,
                99.0,
                4010.0,
                100.0,
                99.0,
                4010.0,
                100.0,
            ],
        }
    )
    events = pl.DataFrame(
        [
            {
                "source": "declared-example",
                "event_id": "dividend",
                "revision_id": "r1",
                "asset": "A",
                "action_kind": "cash_dividend",
                "available_at": TIMES[0],
                "effective_at": TIMES[2],
                "payment_at": TIMES[3],
                "currency": "USD",
                "share_basis": "pre_event",
                "cash_per_share": 1.0,
            }
        ]
    )
    specs = {
        "A": ContractSpec("A"),
        **{
            asset: ContractSpec(
                asset,
                asset_class=AssetClass.FUTURE,
                multiplier=multiplier,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
            for asset, multiplier in [("ES", 50.0), ("F2", 1.0)]
        },
    }
    engine = Engine(
        DataFeed(
            prices_df=prices,
            corporate_actions=events,
            input_manifest={"price_basis": "raw"},
            feed_spec=FeedSpec(timestamp_semantics="event_time"),
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            execution_mode=ExecutionMode.SAME_BAR,
            immediate_fill=True,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
            fixed_margin_schedule={"ES": (15000, 12000), "F2": (84993, 80000)},
        ),
        contract_specs=specs,
        holdings_sampling="bar",
        benchmark=BenchmarkData(
            pl.DataFrame({"timestamp": TIMES, "return": [0.0, 0.01, 0.0, -0.01]}),
            source_id="declared-example-index",
            return_method="close_to_close",
            reinvestment="none",
            feed_spec=FeedSpec(timestamp_semantics="event_time"),
        ),
    )
    result = engine.run()
    assert [row.available_funds for row in snapshots] == [84991, 85001]
    assert [o.status.value for o in attempts] == ["rejected", "filled"]
    assert result.to_fills_dataframe().filter(pl.col("asset") == "F2")["timestamp"].to_list() == [
        TIMES[3]
    ]
    assert result.to_equity_dataframe()["equity"].to_list()[-1] == 100494
    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.to_portfolio_state_dataframe().equals(result.to_portfolio_state_dataframe())
    assert restored.to_actions_dataframe().equals(result.to_actions_dataframe())
    assert restored.to_futures_settlements_dataframe().equals(
        result.to_futures_settlements_dataframe()
    )
    assert restored.to_benchmark_dataframe().equals(result.to_benchmark_dataframe())


def test_failing_callback_removes_its_future_payout_and_fill_observation_only():
    class Fail(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("ES", 1 if timestamp == TIMES[0] else -1)
            if timestamp == TIMES[1]:
                raise RuntimeError("callback failed")

    engine = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": TIMES[:2],
                    "asset": ["ES"] * 2,
                    "close": [4000.0, 4010.0],
                }
            )
        ),
        Fail(),
        BacktestConfig(
            calendar="NYSE",
            allow_leverage=True,
            initial_cash=100000,
            execution_mode=ExecutionMode.SAME_BAR,
            immediate_fill=True,
            commission_rate=0,
            slippage_rate=0,
        ),
        contract_specs={
            "ES": ContractSpec(
                "ES",
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                margin=15000,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
        },
        holdings_sampling="fill_and_session",
    )
    with pytest.raises(RuntimeError, match="callback failed"):
        engine.run()
    assert engine.broker.cash == 100500
    assert engine.broker.get_position("ES").quantity == 1
    assert len(engine.broker.fills) == 1
    assert [row["phase"] for row in engine._futures_ledger.records] == ["before_callbacks"]
    journal = engine.broker._execution_journal
    assert set(journal.accounting_snapshots) == {(TIMES[0], 0), (TIMES[0], None)}


def test_portable_future_payout_rejects_changed_cash_amount(tmp_path):
    from ml4t.backtest import BacktestResult
    from ml4t.backtest.result import ArtifactReadError

    _, result, _ = run_future(1)
    result.to_parquet(tmp_path)
    path = tmp_path / "futures_settlements.parquet"
    pl.read_parquet(path).with_columns(
        (pl.col("cash_delta") + 1).alias("cash_delta")
    ).write_parquet(path)
    with pytest.raises(ArtifactReadError, match="cash does not match"):
        BacktestResult.from_parquet(tmp_path)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("currency", "EUR", "currency must match"),
        ("available_at", TIMES[3] + timedelta(seconds=1), "before publication"),
        ("settlement_price", float("nan"), "price must be finite"),
        ("timestamp", TIMES[3].replace(tzinfo=None), "feed-compatible physical"),
    ],
)
def test_invalid_settlement_terms_fail_before_any_strategy_order(field, value, message):
    event = {
        "timestamp": TIMES[3],
        "available_at": TIMES[2],
        "asset": "ES",
        "settlement_price": 4010.0,
        "currency": "USD",
        "source": "declared-settlement",
        "event_id": "day3",
        "revision_id": "r1",
    }
    with pytest.raises(ValueError, match=message):
        run_future(1, mode="scheduled", settlements=pl.DataFrame([{**event, field: value}]))
    _, result, _ = run_future(1, mode="scheduled", settlements=pl.DataFrame([event]))
    assert result.to_futures_settlements_dataframe()["cash_delta"].to_list() == [500]
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 100498


def test_retried_identical_settlement_pays_once_but_conflicting_revision_cannot_replace_it():
    event = {
        "timestamp": TIMES[3],
        "available_at": TIMES[2],
        "asset": "ES",
        "settlement_price": 4010.0,
        "currency": "USD",
        "source": "declared-settlement",
        "event_id": "day3",
        "revision_id": "r1",
    }
    _, result, _ = run_future(1, mode="scheduled", settlements=pl.DataFrame([event, event]))
    assert result.to_futures_settlements_dataframe()["cash_delta"].to_list() == [500]
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 100498
    with pytest.raises(ValueError, match="Conflicting or revised"):
        run_future(
            1,
            mode="scheduled",
            settlements=pl.DataFrame(
                [
                    event,
                    {**event, "revision_id": "r2", "settlement_price": 4020.0},
                ]
            ),
        )


def test_actual_market_loss_records_margin_breach_without_fabricated_liquidation():
    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("ES", 1)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": TIMES,
                    "asset": ["ES"] * 4,
                    "close": [4000.0, 4000.0, 2100.0, 2100.0],
                }
            )
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            allow_leverage=True,
            initial_cash=100000,
            commission_rate=0,
            slippage_rate=0,
        ),
        contract_specs={
            "ES": ContractSpec(
                "ES",
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                margin=15000,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
        },
        holdings_sampling="bar",
    ).run()
    accounts = result.to_portfolio_state_dataframe()
    assert accounts["cash"].to_list() == [100000, 100000, 5000, 5000]
    assert accounts["margin_breach"].to_list() == [False, False, True, True]
    assert accounts["available_funds"][-1] == -10000
    assert len(result.fills) == 1
    assert result.to_holdings_dataframe()["quantity"][-1] == 1


def test_insufficient_initial_margin_rejects_without_fee_position_or_payout():
    engine, result, reads = run_future(7)
    assert reads == [100000] * 4
    assert not result.fills
    assert not engine.broker.positions
    assert result.to_futures_settlements_dataframe().is_empty()
    assert any("margin" in order.rejection_reason.lower() for order in result.rejected_orders)


def test_partial_entry_sizes_whole_futures_by_declared_margin_instead_of_principal():
    _, result, reads = run_future(7, config_options={"partial_fills_allowed": True})
    assert [fill.quantity for fill in result.fills] == [6]
    assert reads == [100000, 99998, 102998, 102998]
    assert result.to_portfolio_state_dataframe()["initial_margin_required"][-1] == 90000


def test_single_cash_variation_account_cannot_sum_different_declared_currencies():
    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            broker.submit_order("ES", 1)
            broker.submit_order("FDAX", 1)

    specs = {
        asset: ContractSpec(
            asset,
            asset_class=AssetClass.FUTURE,
            multiplier=multiplier,
            currency=currency,
            margin=1000,
            accounting_method="futures_variation",
            settlement_policy="on_mark",
        )
        for asset, multiplier, currency in [("ES", 50, "USD"), ("FDAX", 25, "EUR")]
    }
    with pytest.raises(ValueError, match="single.*currency|same.*currency"):
        Engine(
            DataFeed(
                prices_df=pl.DataFrame(
                    {
                        "timestamp": [t for t in TIMES[:2] for _ in range(2)],
                        "asset": ["ES", "FDAX"] * 2,
                        "close": [4000.0, 19000.0, 4010.0, 19010.0],
                    }
                )
            ),
            Buy(),
            BacktestConfig(calendar=None, allow_leverage=True),
            contract_specs=specs,
        ).run()
