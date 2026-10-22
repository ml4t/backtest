"""Public Engine roll execution acceptance; prices are raw individual contracts."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import (
    AssetClass,
    BacktestConfig,
    BacktestResult,
    ContractSpec,
    DataFeed,
    Engine,
    Strategy,
)
from ml4t.backtest.config import CommissionType, ExecutionMode
from ml4t.backtest.controls import OrderConstraints, TradingControlViolation
from ml4t.backtest.result import ArtifactReadError

TIMES = [datetime(2025, 3, 3, 21, tzinfo=UTC) + timedelta(days=i) for i in range(6)]


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("scenario", ["complete", "missing_new_quote", "rejected_open"])
@pytest.mark.parametrize("legacy", [False, True], ids=["variation", "signed-notional"])
def test_dated_roll_links_actual_legs_and_fees_without_using_adjustment_marks(
    direction, scenario, legacy, tmp_path
):
    orders = []

    class Roll(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("capture:ESH25", direction)
            if timestamp in TIMES[2:4]:
                legs = broker.roll_position(
                    "capture:ESH25",
                    "capture:ESM25",
                    roll_id="selection-1",
                    decision_at=TIMES[2],
                    available_at=TIMES[1],
                )
                if timestamp == TIMES[2]:
                    orders.extend(legs)
                else:
                    assert [order.order_id for order in legs] == [
                        order.order_id for order in orders
                    ]

    prices = pl.DataFrame(
        {
            "timestamp": [t for t in TIMES for _ in range(2)],
            "asset": ["capture:ESH25", "capture:ESM25"] * 6,
            "close": [
                4000.0,
                4010.0,
                4000.0,
                4010.0,
                4010.0,
                4020.0,
                4020.0,
                4030.0,
                4030.0,
                4040.0,
                4040.0,
                4050.0,
            ],
        }
    )
    if scenario == "missing_new_quote":
        prices = prices.filter(
            (pl.col("asset") != "capture:ESM25") | (pl.col("timestamp") < TIMES[3])
        )
    result = Engine(
        DataFeed(prices_df=prices, input_manifest={"price_basis": "raw"}),
        Roll(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            allow_short_selling=True,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
        ),
        contract_specs={
            asset: ContractSpec(
                asset,
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                margin=200000
                if scenario == "rejected_open" and asset == "capture:ESM25"
                else 15000,
                **(
                    {}
                    if legacy
                    else {
                        "accounting_method": "futures_variation",
                        "settlement_policy": "on_mark",
                    }
                ),
            )
            for asset in ["capture:ESH25", "capture:ESM25"]
        },
        holdings_sampling="bar",
    ).run()
    expected = [
        (TIMES[1], "capture:ESH25", "buy" if direction > 0 else "sell", 1, 4000),
        (TIMES[3], "capture:ESH25", "sell" if direction > 0 else "buy", 1, 4020),
        (TIMES[3], "capture:ESM25", "buy" if direction > 0 else "sell", 1, 4030),
    ]
    if scenario != "complete":
        expected = expected[:2]
    assert [
        (f.timestamp, f.asset, f.side.value, f.quantity, f.price) for f in result.fills
    ] == expected
    assert len(orders) == 2
    closing = (
        100000
        + direction * (2000 if scenario == "complete" else 1000)
        - (6 if scenario == "complete" else 4)
    )
    expected_cash = (
        closing - direction * 4050 * 50 if legacy and scenario == "complete" else closing
    )
    assert result.to_portfolio_state_dataframe()["cash"][-1] == expected_cash
    assert result.to_equity_dataframe()["equity"][-1] == closing
    rolls = result.to_rolls_dataframe()
    assert rolls["roll_id"].to_list() == ["selection-1", "selection-1"]
    assert rolls["leg"].to_list() == ["close", "open"]
    assert rolls["status"].to_list() == [
        "filled",
        {
            "complete": "filled",
            "missing_new_quote": "pending",
            "rejected_open": "rejected",
        }[scenario],
    ]
    assert (
        rolls["roll_state"].to_list() == ["completed" if scenario == "complete" else "partial"] * 2
    )
    assert rolls["commission"].to_list() == [2, 2 if scenario == "complete" else 0]
    assert rolls["filled_at"].to_list() == [TIMES[3], TIMES[3] if scenario == "complete" else None]
    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.to_rolls_dataframe().equals(rolls)
    table = tmp_path / "futures_rolls.parquet"
    rolls.with_columns((pl.col("commission") + 1).alias("commission")).write_parquet(table)
    with pytest.raises(ArtifactReadError, match="fees disagree"):
        BacktestResult.from_parquet(tmp_path)
    if scenario != "complete":
        rolls.with_columns(pl.lit("completed").alias("roll_state")).write_parquet(table)
        with pytest.raises(ArtifactReadError, match="completion status"):
            BacktestResult.from_parquet(tmp_path)


@pytest.mark.parametrize("immediate", [False, True])
@pytest.mark.parametrize("failure", ["blocked_open", "future_availability", "invalid_quantity"])
def test_failed_roll_can_be_caught_without_leaving_orders_fees_or_phantom_margin(
    failure, immediate
):
    class Roll(Strategy):
        def on_start(self, broker):
            broker.register_trading_control(OrderConstraints(blocked_assets=frozenset({"NEW"})))

        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("OLD", 1)
            if timestamp == TIMES[2]:
                kwargs = (
                    {"available_at": TIMES[3]}
                    if failure == "future_availability"
                    else {"new_quantity": float("nan")}
                    if failure == "invalid_quantity"
                    else {}
                )
                error = TradingControlViolation if failure == "blocked_open" else ValueError
                with pytest.raises(error):
                    broker.roll_position("OLD", "NEW", roll_id="failed", **kwargs)
                assert broker.get_position("OLD").quantity == 1
                assert not broker.get_orders(pending_only=True)
                broker.submit_order("OLD", -1)
                broker.submit_order("THIRD", 1)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": [t for t in TIMES for _ in range(3)],
                    "asset": ["OLD", "NEW", "THIRD"] * 6,
                    "close": [
                        value
                        for i in range(6)
                        for value in [
                            [4000.0, 4000.0, 4010.0, 4020.0, 4030.0, 4040.0][i],
                            100.0,
                            100.0,
                        ]
                    ],
                }
            )
        ),
        Roll(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
            execution_mode=ExecutionMode.SAME_BAR if immediate else ExecutionMode.NEXT_BAR,
            immediate_fill=immediate,
            buying_power_reservation=True,
        ),
        contract_specs={
            asset: ContractSpec(
                asset,
                asset_class=AssetClass.FUTURE,
                multiplier=multiplier,
                margin=margin,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
            for asset, multiplier, margin in [
                ("OLD", 50, 15000),
                ("NEW", 50, 15000),
                ("THIRD", 20, 100000),
            ]
        },
    ).run()
    assert [fill.asset for fill in result.fills] == ["OLD", "OLD", "THIRD"]
    assert (
        result.to_portfolio_state_dataframe()["cash"][-1]
        == 100000 + (500 if immediate else 1000) - 6
    )
    assert sum(fill.commission for fill in result.fills) == 6
    with pytest.raises(ValueError, match="no captured"):
        result.to_rolls_dataframe()


@pytest.mark.parametrize("finish", [False, True])
def test_volume_limited_roll_retains_each_actual_partial_fill_and_final_completion(
    finish, tmp_path
):
    from ml4t.backtest.execution import VolumeParticipationLimit

    class Roll(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("OLD", 3)
            if timestamp == TIMES[2]:
                broker.roll_position("OLD", "NEW", roll_id="partial")

    times = TIMES if finish else TIMES[:4]
    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": [t for t in times for _ in range(2)],
                    "asset": ["OLD", "NEW"] * len(times),
                    "close": [
                        4000.0,
                        4010.0,
                        4000.0,
                        4010.0,
                        4010.0,
                        4020.0,
                        4020.0,
                        4030.0,
                        4030.0,
                        4040.0,
                        4040.0,
                        4050.0,
                    ][: len(times) * 2],
                    "volume": [4.0] * (len(times) * 2),
                }
            )
        ),
        Roll(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
        ),
        contract_specs={
            asset: ContractSpec(
                asset,
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                margin=15000,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
            for asset in ["OLD", "NEW"]
        },
        execution_limits=VolumeParticipationLimit(max_participation=0.5),
        holdings_sampling="bar",
    ).run()
    rolls = result.to_rolls_dataframe()
    assert rolls["filled_quantity"].to_list() == [3 if finish else 2] * 2
    assert rolls["roll_state"].to_list() == ["completed" if finish else "partial"] * 2
    assert rolls["commission"].to_list() == [4 if finish else 2] * 2
    assert result.to_portfolio_state_dataframe()["cash"][-1] == (105488 if finish else 102492)
    assert sum(f.commission for f in result.fills) == (12 if finish else 8)
    result.to_parquet(tmp_path)
    assert BacktestResult.from_parquet(tmp_path).to_rolls_dataframe().equals(rolls)


def test_fractional_contract_roll_roundtrips_real_partial_fill_arithmetic(tmp_path):
    from ml4t.backtest.config import ShareType
    from ml4t.backtest.execution import VolumeParticipationLimit

    times = [TIMES[0] + timedelta(minutes=i) for i in range(10)]

    class Roll(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("OLD", 0.3)
            if timestamp == times[3]:
                broker.roll_position("OLD", "NEW", roll_id="fractional")

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": [t for t in times for _ in range(2)],
                    "asset": ["OLD", "NEW"] * len(times),
                    "close": [4000.0] * (len(times) * 2),
                    "volume": [0.2] * (len(times) * 2),
                }
            )
        ),
        Roll(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            share_type=ShareType.FRACTIONAL,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
        ),
        contract_specs={
            asset: ContractSpec(
                asset,
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                margin=15000,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
            for asset in ["OLD", "NEW"]
        },
        execution_limits=VolumeParticipationLimit(max_participation=0.5),
        holdings_sampling="bar",
    ).run()
    assert result.to_rolls_dataframe()["roll_state"].to_list() == ["completed"] * 2
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 99982
    result.to_parquet(tmp_path)
    assert (
        BacktestResult.from_parquet(tmp_path)
        .to_rolls_dataframe()
        .equals(result.to_rolls_dataframe())
    )


def test_roll_artifact_preserves_mixed_equity_dividend_benchmark_and_variation_ledgers(tmp_path):
    from ml4t.specs import FeedSpec

    from ml4t.backtest.benchmark import BenchmarkData

    class Mixed(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("OLD", 1)
                broker.submit_order("EQUITY", 10)
            if timestamp == TIMES[2]:
                broker.roll_position("OLD", "NEW", roll_id="mixed")

    spec = FeedSpec(timestamp_semantics="event_time")
    actions = pl.DataFrame(
        [
            {
                "source": "capture",
                "event_id": "dividend",
                "revision_id": "r1",
                "asset": "EQUITY",
                "action_kind": "cash_dividend",
                "available_at": TIMES[0],
                "effective_at": TIMES[2],
                "payment_at": TIMES[4],
                "currency": "USD",
                "share_basis": "pre_event",
                "cash_per_share": 1.0,
            }
        ]
    )
    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": [t for t in TIMES for _ in range(3)],
                    "asset": ["OLD", "NEW", "EQUITY"] * 6,
                    "close": [
                        4000.0,
                        4010.0,
                        100.0,
                        4000.0,
                        4010.0,
                        100.0,
                        4010.0,
                        4020.0,
                        100.0,
                        4020.0,
                        4030.0,
                        100.0,
                        4030.0,
                        4040.0,
                        100.0,
                        4040.0,
                        4050.0,
                        100.0,
                    ],
                }
            ),
            corporate_actions=actions,
            input_manifest={"price_basis": "raw"},
            feed_spec=spec,
        ),
        Mixed(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
        ),
        contract_specs={
            "EQUITY": ContractSpec("EQUITY"),
            **{
                asset: ContractSpec(
                    asset,
                    asset_class=AssetClass.FUTURE,
                    multiplier=50,
                    margin=15000,
                    accounting_method="futures_variation",
                    settlement_policy="on_mark",
                )
                for asset in ["OLD", "NEW"]
            },
        },
        holdings_sampling="bar",
        benchmark=BenchmarkData(
            pl.DataFrame({"timestamp": TIMES, "return": [0.0] * 6}),
            source_id="declared-index",
            return_method="close_to_close",
            reinvestment="none",
            feed_spec=spec,
        ),
    ).run()
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 101002
    assert result.to_equity_dataframe()["equity"][-1] == 102002
    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    for name in [
        "rolls",
        "actions",
        "entitlements",
        "benchmark",
        "futures_settlements",
        "portfolio_state",
        "holdings",
    ]:
        method = f"to_{name}_dataframe"
        assert getattr(restored, method)().equals(getattr(result, method)())


@pytest.mark.parametrize("clock", ["decision_at", "available_at"])
def test_missing_pandas_roll_clock_is_rejected_before_order_or_account_mutation(clock):
    import pandas as pd

    times = [t.replace(tzinfo=None) for t in TIMES]

    class Roll(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("OLD", 1)
            if timestamp == times[2]:
                missing = pd.to_datetime(pd.Series([None]))[0]
                with pytest.raises(ValueError, match="physical instants"):
                    broker.roll_position("OLD", "NEW", roll_id="missing", **{clock: missing})
                assert not broker.get_pending_orders()
                assert broker.get_position("OLD").quantity == 1
                assert broker.get_position("NEW") is None
                assert broker.cash == 100000

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": [t for t in times for _ in range(2)],
                    "asset": ["OLD", "NEW"] * 6,
                    "close": [100.0] * 12,
                }
            )
        ),
        Roll(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            commission_rate=0,
            slippage_rate=0,
        ),
        contract_specs={
            asset: ContractSpec(
                asset,
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                margin=15000,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
            for asset in ["OLD", "NEW"]
        },
    ).run()
    assert len(result.fills) == 1
