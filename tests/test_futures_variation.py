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


def run_future(quantity, *, mode="on_mark", settlements=None, legacy=False, runner="direct"):
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
        calendar=None,
        initial_cash=100000,
        allow_leverage=True,
        allow_short_selling=True,
        commission_type=CommissionType.PER_TRADE,
        commission_per_trade=2,
        slippage_rate=0,
        fixed_margin_schedule={"ES": (15000, 12000)},
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
                holdings_sampling="bar",
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
        holdings_sampling="bar",
        **kwargs,
    )
    return engine, engine.run(), reads


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


def test_two_scheduled_settlements_scale_in_partial_close_and_final_close_do_not_pay_twice():
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


def test_unpaid_dividend_cannot_finance_future_margin_before_its_actual_payment():
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
        DataFeed(prices_df=prices, corporate_actions=events, input_manifest={"price_basis": "raw"}),
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
    )
    result = engine.run()
    assert [row.available_funds for row in snapshots] == [84991, 85001]
    assert [o.status.value for o in attempts] == ["rejected", "filled"]
    assert result.to_fills_dataframe().filter(pl.col("asset") == "F2")["timestamp"].to_list() == [
        TIMES[3]
    ]
    assert result.to_equity_dataframe()["equity"].to_list()[-1] == 100494
