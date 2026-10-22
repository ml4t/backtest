"""Declared physical boundaries route ordinary orders, never synthetic expiry prices."""

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from ml4t.backtest import (
    AssetClass,
    BacktestConfig,
    ContractSpec,
    DataFeed,
    Engine,
    RuntimeServices,
    Strategy,
)
from ml4t.backtest.config import CommissionType, ExecutionMode
from ml4t.backtest.controls import TradingControlViolation
from ml4t.backtest.types import OrderType

TIMES = [datetime(2025, 7, 7, 21, tzinfo=UTC) + timedelta(days=i) for i in range(5)]


@pytest.mark.parametrize("action", ["close", "roll"])
@pytest.mark.parametrize("immediate", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("offset_hours", [0, 1])
def test_notice_before_expiry_uses_next_real_quote_and_prevents_reopening(
    action, immediate, missing, offset_hours
):
    from ml4t.backtest import FuturesLifecyclePolicy

    boundary = TIMES[2] - timedelta(hours=offset_hours)
    policy = FuturesLifecyclePolicy(
        event_id="notice-17-r1",
        asset="capture:17",
        effective_at=boundary,
        available_at=TIMES[0],
        reason="first_notice",
        source="capture",
        revision_id="r1",
        replacement_asset="capture:23" if action == "roll" else None,
    )
    runtime = RuntimeServices(futures_lifecycle=[policy])
    pending = []

    class Buy(Strategy):
        def on_end(self, broker):
            pending.extend(broker.get_pending_orders())

        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("capture:17", 1)
            if timestamp == TIMES[1]:
                broker.submit_order("capture:17", 1, order_type=OrderType.LIMIT, limit_price=1)
            if timestamp == TIMES[2]:
                with pytest.raises(TradingControlViolation, match="lifecycle"):
                    broker.submit_order("capture:17", 1)

    prices = pl.DataFrame(
        {
            "timestamp": [t for t in TIMES for _ in range(2)],
            "asset": ["capture:17", "capture:23"] * 5,
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
            ],
        }
    )
    if missing:
        prices = prices.filter((pl.col("asset") != "capture:17") | (pl.col("timestamp") < TIMES[2]))
    result = Engine(
        DataFeed(
            prices_df=prices,
            feed_spec={
                "time_contract_version": 2,
                "calendar": None,
                "timestamp_semantics": "event_time",
            },
        ),
        Buy(),
        BacktestConfig(
            calendar=None,
            initial_cash=100000,
            allow_leverage=True,
            execution_mode=ExecutionMode.SAME_BAR if immediate else ExecutionMode.NEXT_BAR,
            immediate_fill=immediate,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
        ),
        runtime=runtime,
        contract_specs={
            asset: ContractSpec(
                asset,
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                margin=15000,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
            for asset in ["capture:17", "capture:23"]
        },
        holdings_sampling="bar",
    ).run()
    assert result.fills[0].price == 4000
    closes = [f for f in result.fills if f.asset == "capture:17" and f.side.value == "sell"]
    assert [(f.timestamp, f.price) for f in closes] == ([] if missing else [(TIMES[2], 4010)])
    if offset_hours:
        assert all(f.timestamp != boundary for f in result.fills)
    assert all(f.price != 1 for f in result.fills)
    assert not any(order.limit_price == 1 for order in pending)
    if action == "roll":
        rolls = result.to_rolls_dataframe()
        assert rolls["decision_at"].to_list() == [boundary] * 2
        assert rolls["available_at"].to_list() == [TIMES[0]] * 2
        assert rolls["roll_state"].to_list() == ["partial" if missing else "completed"] * 2
    elif not missing:
        assert result.to_portfolio_state_dataframe()["cash"][-1] == 100496
    if missing:
        assert any(order.asset == "capture:17" for order in pending)


@pytest.mark.parametrize(
    "field,value",
    [
        ("effective_at", None),
        ("available_at", None),
        ("available_at", TIMES[4]),
        ("source", ""),
        ("revision_id", ""),
        ("reason", "unknown"),
        ("replacement_asset", "capture:17"),
    ],
)
def test_unknown_or_noncausal_lifecycle_facts_fail_before_runtime_registration(field, value):
    from ml4t.backtest import FuturesLifecyclePolicy

    args = {
        "event_id": "e",
        "asset": "capture:17",
        "effective_at": TIMES[2],
        "available_at": TIMES[0],
        "reason": "expiration",
        "source": "capture",
        "revision_id": "r1",
    }
    args[field] = value
    with pytest.raises((TypeError, ValueError)):
        FuturesLifecyclePolicy(**args)


def test_notice_after_captured_data_preserves_open_position_without_invented_terminal_fill():
    from ml4t.backtest import FuturesLifecyclePolicy

    runtime = RuntimeServices(
        futures_lifecycle=[
            FuturesLifecyclePolicy(
                event_id="future-notice",
                asset="OLD",
                effective_at=TIMES[-1] + timedelta(days=1),
                available_at=TIMES[0],
                reason="first_notice",
                source="capture",
                revision_id="r1",
            )
        ]
    )

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == TIMES[0]:
                broker.submit_order("OLD", 1)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {"timestamp": TIMES, "asset": ["OLD"] * 5, "close": [4000.0] * 5}
            ),
            feed_spec={
                "time_contract_version": 2,
                "calendar": None,
                "timestamp_semantics": "event_time",
            },
        ),
        Buy(),
        BacktestConfig(calendar=None, initial_cash=100000, allow_leverage=True, slippage_rate=0),
        runtime=runtime,
        contract_specs={
            "OLD": ContractSpec(
                "OLD",
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                accounting_method="futures_variation",
                margin=15000,
            )
        },
        holdings_sampling="bar",
    ).run()
    assert len(result.fills) == 1
    assert result.to_holdings_dataframe()["quantity"][-1] == 1


@pytest.mark.parametrize("clock", ["effective_at", "available_at"])
def test_pandas_missing_lifecycle_timestamp_cannot_be_treated_as_a_known_boundary(clock):
    import pandas as pd

    from ml4t.backtest import FuturesLifecyclePolicy

    args = {
        "event_id": "e",
        "asset": "capture:17",
        "effective_at": TIMES[2].replace(tzinfo=None),
        "available_at": TIMES[0].replace(tzinfo=None),
        "reason": "expiration",
        "source": "capture",
        "revision_id": "r1",
    }
    args[clock] = pd.to_datetime(pd.Series([None]))[0]
    with pytest.raises((TypeError, ValueError), match="known physical"):
        FuturesLifecyclePolicy(**args)
