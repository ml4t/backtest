"""Compare raw-date lifecycle accounting with actual published Zipline 3.2.0 output."""

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
    FuturesLifecyclePolicy,
    RuntimeServices,
    Strategy,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/dated_futures/zipline_3_2.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda c: c["name"])
def test_explicit_raw_quote_close_reconciles_original_expiry_cash_and_nav(case):
    times = [datetime.fromisoformat(row["timestamp"]) for row in case["trace"]]
    boundary = datetime(2025, 7, 10, 21, tzinfo=UTC)

    class Buy(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == times[0]:
                broker.submit_order("17", 1)

    result = Engine(
        DataFeed(
            prices_df=pl.DataFrame(
                {
                    "timestamp": times,
                    "asset": ["17"] * len(times),
                    "close": [4000.0 + i * 10 for i in range(len(times))],
                }
            ),
            feed_spec={
                "time_contract_version": 2,
                "calendar": "us_futures",
                "timestamp_semantics": "bar_close",
            },
        ),
        Buy(),
        BacktestConfig(
            initial_cash=100000, allow_leverage=True, commission_rate=0, slippage_rate=0
        ),
        runtime=RuntimeServices(
            futures_lifecycle=[
                FuturesLifecyclePolicy(
                    event_id="declared-close-r1",
                    asset="17",
                    effective_at=boundary,
                    available_at=times[0],
                    reason="auto_close",
                    source="synthetic-reference",
                    revision_id="r1",
                )
            ]
        ),
        contract_specs={
            "17": ContractSpec(
                "17",
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                margin=15000,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
        },
        holdings_sampling="bar",
    ).run()
    assert result.to_portfolio_state_dataframe()["cash"].to_list() == case["cash"]
    assert result.to_equity_dataframe()["equity"].to_list() == case["nav"]
    assert [
        (
            f.timestamp.isoformat(),
            f.asset,
            f.quantity * (1 if f.side.value == "buy" else -1),
            f.price,
        )
        for f in result.fills
    ] == [(f["timestamp"], str(f["sid"]), f["quantity"], f["price"]) for f in case["fills"]]
    assert case["cash"] == [100000, 100000, 100500, 101000, 101000, 101000]
