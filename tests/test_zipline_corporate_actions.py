"""Original 3.2 account and callback traces consumed through the native migration adapter."""

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.api import (
    get_datetime,
    get_order,
    order,
    set_benchmark,
    set_commission,
    set_slippage,
    sid,
)
from ml4t.backtest.compat.zipline.assets import Asset
from ml4t.backtest.compat.zipline.finance.commission import PerShare
from ml4t.backtest.compat.zipline.finance.slippage import NoSlippage, VolumeShareSlippage
from ml4t.backtest.types import ContractSpec

ROOT = Path(__file__).parent / "fixtures/corporate_actions"
REFERENCE = json.loads((ROOT / "reference.json").read_text())
BOUNDARIES = json.loads((ROOT / "boundary-reference.json").read_text())


def prepared_feed(reference, *, declarations=True, ex_close=None, prepared_actions=None):
    snapshot = ROOT / reference["input"]["name"] / "snapshot"
    frames = {
        name: pl.read_parquet(snapshot / f"tables/{name}/data.parquet")
        for name in ["daily_bars", "actions", "zipline_equities"]
    }
    clocks = dict(
        zip(
            [pd.Timestamp(d).date() for d in reference["sessions"]],
            [pd.Timestamp(d) for d in reference["closes"]],
        )
    )
    prices = frames["daily_bars"].with_columns(
        pl.col("timestamp").replace_strict(clocks, return_dtype=pl.Datetime("ns", "UTC"))
    )
    if ex_close is not None:
        prices = prices.with_columns(
            [
                pl.when(
                    (pl.col("source_sid") == 7)
                    & (pl.col("timestamp") == pd.Timestamp(reference["closes"][2]))
                )
                .then(float(ex_close))
                .otherwise(pl.col(name))
                .alias(name)
                for name in ["open", "high", "low", "close"]
            ]
        )
    actions = frames["actions"]
    if declarations:
        actions = pl.read_parquet(ROOT / reference["input"]["name"] / "native_actions.parquet")
    if prepared_actions is not None:
        actions = prepared_actions
    assets = [
        Asset(row["asset"], row["symbol"], row["sid"])
        for row in frames["zipline_equities"].to_dicts()
    ]
    feed = DataFeed(
        prices_df=prices,
        corporate_actions=actions,
        input_manifest=json.loads((snapshot / "manifest.json").read_text()),
        feed_spec=FeedSpec(
            time_contract_version=2,
            calendar="NYSE",
            timezone="UTC",
            data_frequency="1d",
            timestamp_semantics="bar_close",
            bar_label="right",
        ),
    )
    return feed, assets


def run_reference_case(reference, **options):
    case = reference["input"]
    feed, assets = prepared_feed(reference, **options)
    trace, premarket = [], []

    def capture(context):
        return {
            "timestamp": get_datetime().isoformat(),
            "cash": context.portfolio.cash,
            "nav": context.portfolio.portfolio_value,
            "positions": [
                {
                    "sid": int(asset),
                    "quantity": p.amount,
                    "cost_basis": p.cost_basis,
                    "last_sale_price": p.last_sale_price,
                }
                for asset, p in context.portfolio.positions.items()
            ],
            "orders": [
                {
                    "amount": get_order(key).amount,
                    "filled": get_order(key).filled,
                    "commission": get_order(key).commission,
                    "status": int(get_order(key).status),
                    "limit": get_order(key).limit,
                    "stop": get_order(key).stop,
                }
                for key in context.orders
            ],
        }

    def initialize(context):
        set_benchmark(None)
        set_commission(us_equities=PerShare(0, 0))
        set_slippage(
            us_equities=VolumeShareSlippage(volume_limit=4 / 100000, price_impact=0)
            if case.get("partial")
            else NoSlippage()
        )
        context.orders, context.closed = [], False

    def before(context, data):
        premarket.append(capture(context))

    def handle(context, data):
        if not context.orders:
            context.orders.append(
                order(
                    sid(7),
                    case["quantity"],
                    limit_price=100 if case.get("partial") else None,
                    stop_price=90
                    if case.get("partial") and case["quantity"] > 0
                    else 110
                    if case.get("partial")
                    else None,
                )
            )
            if case.get("destination_quantity"):
                context.orders.append(order(sid(8), case["destination_quantity"]))
        if (
            case.get("close_on_ex")
            and get_datetime().date() == pd.Timestamp(reference["sessions"][2]).date()
            and not context.closed
        ):
            context.orders.append(order(sid(7), -case["quantity"]))
            context.closed = True
        trace.append(capture(context))

    performance = run_algorithm(
        feed=feed,
        assets=assets,
        initialize=initialize,
        handle_data=handle,
        before_trading_start=before,
        capital_base=10000,
        contract_specs={a.key: ContractSpec(a.key, currency="USD") for a in assets},
    )
    return performance, trace, premarket


@pytest.mark.parametrize(
    "reference", REFERENCE["cases"] + BOUNDARIES["cases"], ids=lambda c: c["input"]["name"]
)
def test_installed_data_preparation_flows_directly_into_the_action_enabled_engine(reference):
    adjustments = pytest.importorskip("ml4t.data.adjustments")
    storage = pytest.importorskip("ml4t.data.storage")
    snapshot = storage.read_snapshot(ROOT / reference["input"]["name"] / "snapshot")
    declared = snapshot.tables["actions"].with_columns(
        pl.lit(pd.Timestamp(reference["closes"][0])).alias("available_at"),
        pl.lit("USD").alias("currency"),
        pl.lit("pre_event").alias("share_basis"),
    )
    actual = adjustments.prepare_action_inputs(
        declared,
        feed_spec=FeedSpec(
            time_contract_version=2,
            timestamp_col="effective_at",
            entity_col="asset",
            calendar="NYSE",
            timestamp_semantics="session_label",
        ),
        required_terms=["available_at", "currency", "share_basis"],
    )
    retained = pl.read_parquet(ROOT / reference["input"]["name"] / "native_actions.parquet")
    assert actual.equals(retained)
    assert actual["missing_terms"].to_list() == [[]]
    performance, _, _ = run_reference_case(reference, prepared_actions=actual)
    assert performance["cash"].to_list() == reference["cash"]
    assert performance["portfolio_value"].to_list() == reference["nav"]


@pytest.mark.parametrize("reference", BOUNDARIES["cases"], ids=lambda c: c["input"]["name"])
def test_real_exported_stock_record_date_and_price_moves_match_closing_ledger_without_premarket_lookahead(
    reference,
):
    performance, trace, premarket = run_reference_case(reference)
    assert performance["portfolio_value"].to_list() == reference["nav"]
    assert performance["cash"].to_list() == reference["cash"]
    for got, wanted in zip(trace, reference["trace"], strict=True):
        wanted = dict(wanted)
        wanted["orders"] = [{k: v for k, v in row.items() if k != "id"} for row in wanted["orders"]]
        assert got == wanted
    fills = [fill for transactions in performance["transactions"] for fill in transactions]
    assert all(fill["commission"] == 0 for fill in fills)
    assert all(fill["commission"] is None for fill in reference["fills"])
    assert [
        (fill["dt"].isoformat(), int(fill["sid"]), fill["amount"], fill["price"]) for fill in fills
    ] == [
        (fill["timestamp"], fill["sid"], fill["quantity"], fill["price"])
        for fill in reference["fills"]
    ]
    case = reference["input"]
    if case.get("extra_move"):
        causal_mark = 99.0 if case.get("dividend") else 80.0
        assert premarket[2]["positions"][0]["last_sale_price"] == causal_mark
        assert reference["premarket"][2]["positions"][0]["last_sale_price"] == causal_mark + 7
        assert premarket[2]["nav"] == 9000 + 10 * causal_mark
    if case.get("record_offset"):
        # The real source retains record_date separately; selling after ex does not forfeit payment.
        assert performance["cash"].to_list()[-1] == 10000


@pytest.mark.parametrize("reference", REFERENCE["cases"], ids=lambda c: c["input"]["name"])
def test_actual_exported_actions_match_original_cash_nav_orders_positions_and_full_fill_stream(
    reference,
):
    performance, trace, premarket = run_reference_case(reference)
    assert performance["portfolio_value"].to_list() == reference["nav"]
    assert performance["cash"].to_list() == reference["cash"]
    for actual, expected in [(trace, reference["trace"]), (premarket, reference["premarket"])]:
        assert len(actual) == len(expected)
        for got, wanted in zip(actual, expected, strict=True):
            wanted = dict(wanted)
            wanted["orders"] = [
                {k: v for k, v in entry.items() if k != "id"} for entry in wanted["orders"]
            ]
            assert got == wanted
    fills = [
        {
            "timestamp": fill["dt"].isoformat(),
            "sid": int(fill["sid"]),
            "quantity": fill["amount"],
            "price": fill["price"],
            "commission": fill["commission"],
        }
        for transactions in performance["transactions"]
        for fill in transactions
    ]
    # Original transactions leave commission unset under NoCommission; their order ledger
    # above supplies the actual zero charge. Native fills report that zero explicitly.
    assert all(fill["commission"] == 0 for fill in fills)
    assert all(fill["commission"] is None for fill in reference["fills"])
    assert [{k: v for k, v in fill.items() if k != "commission"} for fill in fills] == [
        {k: v for k, v in fill.items() if k != "commission"} for fill in reference["fills"]
    ]
    result = performance.attrs["backtest_result"]
    assert result.corporate_actions is not None
    assert result.config.corporate_action_policy == "zipline_3_2"


@pytest.mark.parametrize("known_terms,error", [(False, "available_at"), (True, "currency")])
def test_unknown_terms_in_the_actual_data_export_are_rejected_before_strategy_start(
    known_terms, error
):
    reference = next(c for c in REFERENCE["cases"] if c["input"]["name"] == "long_cash_after_sale")
    feed, assets = prepared_feed(reference, declarations=False)
    if known_terms:
        feed.corporate_actions = feed.corporate_actions.with_columns(
            pl.lit(pd.Timestamp(reference["closes"][0])).alias("available_at")
        )
    started = []

    class Never(Strategy):
        def on_start(self, broker):
            started.append(True)

        def on_data(self, *args):
            pass

    with pytest.raises(ValueError, match=error):
        Engine(
            feed,
            Never(),
            BacktestConfig(),
            contract_specs={a.key: ContractSpec(a.key) for a in assets},
        )
    assert started == []


def test_pre_market_distribution_mark_cannot_read_the_future_ex_session_close():
    reference = next(c for c in REFERENCE["cases"] if c["input"]["name"] == "long_cash_after_sale")
    _, _, before = run_reference_case(reference, ex_close=321)
    assert before[2]["nav"] == 9990
    assert before[2]["positions"][0]["last_sale_price"] == 99
