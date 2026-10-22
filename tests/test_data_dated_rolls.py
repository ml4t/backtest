"""Actual Data selectors must stay causal when routed to raw dated-contract fills."""

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest
from ml4t.specs import FeedSpec, LifecyclePhase

from ml4t.backtest import (
    AssetClass,
    BacktestConfig,
    ContractSpec,
    DataFeed,
    Engine,
    RuntimeServices,
    Strategy,
)
from ml4t.backtest.config import CommissionType
from ml4t.backtest.execution import RebalanceSchedule

TIMES = [datetime(2025, 7, 7, 20, tzinfo=UTC) + timedelta(days=i) for i in range(5)]
SPEC = FeedSpec(time_contract_version=2, calendar="CME_Equity", timestamp_semantics="event_time")


def contract_inputs():
    bars = pl.DataFrame(
        [
            {
                "timestamp": t,
                "asset": asset,
                "symbol": f"TEST{asset}",
                "close": 4000.0 + i * 10 + (10 if asset == 23 else 0),
                "volume": [100.0, 40.0, 30.0, 20.0, 10.0][i]
                if asset == 17
                else [50.0, 80.0, 90.0, 100.0, 110.0][i],
                "open_interest": [100.0, 40.0, 30.0, 20.0, 10.0][i]
                if asset == 17
                else [50.0, 80.0, 90.0, 100.0, 110.0][i],
                "currency": "USD",
                "price_quote_unit": "index_points",
                "mark_kind": "trade_close",
                "available_at": t,
                "open_interest_available_at": TIMES[3] if asset == 23 and i > 0 else t,
                "source": "capture",
                "revision_id": "r1",
            }
            for i, t in enumerate(TIMES)
            for asset in [17, 23]
        ]
    )
    definitions = pl.DataFrame(
        [
            {
                "asset": asset,
                "symbol": f"TEST{asset}",
                "root": "TEST",
                "exchange": "CME",
                "instrument_type": "future",
                "valid_from": date(2025, 7, 1),
                "available_at": datetime(2025, 6, 1, tzinfo=UTC),
                "source": "capture",
                "revision_id": "r1",
                "expiration": date(2025, 7, 15) if asset == 17 else date(2025, 9, 15),
                "first_notice_date": date(2025, 7, 10) if asset == 17 else date(2025, 9, 10),
                "last_trade_date": date(2025, 7, 14) if asset == 17 else date(2025, 9, 14),
                "currency": "USD",
                "multiplier": 50.0,
                "tick_size": 0.25,
                "price_quote_unit": "index_points",
                "settlement_type": "physical",
                "volume_unit": "contracts",
                "open_interest_unit": "contracts",
            }
            for asset in [17, 23]
        ]
    )
    return bars, definitions


@pytest.mark.parametrize("perturb_future", [False, True])
def test_data_previous_session_volume_delivers_only_causal_selection_and_raw_execution(
    perturb_future,
):
    futures = pytest.importorskip("ml4t.data.futures")
    bars, definitions = contract_inputs()
    if perturb_future:
        bars = bars.with_columns(
            pl.when(pl.col("timestamp") >= TIMES[3])
            .then(1e12)
            .otherwise(pl.col("volume"))
            .alias("volume"),
            pl.when(pl.col("timestamp") >= TIMES[3])
            .then(1e12)
            .otherwise(pl.col("open_interest"))
            .alias("open_interest"),
        )
    prepared = futures.prepare_futures_inputs(bars, definitions, as_of=TIMES[-1], feed_spec=SPEC)
    raw = prepared.bars.with_columns(
        pl.concat_str(pl.lit("capture:"), pl.col("asset").cast(pl.String)).alias("asset")
    )
    runtime = RuntimeServices()
    decisions = [t + timedelta(minutes=30) for t in TIMES[1:3]]
    seen = []

    def decide(timestamp, data, context, broker):
        current = futures.prepare_futures_inputs(
            bars,
            definitions,
            as_of=timestamp,
            feed_spec=SPEC,
            roll_strategy=futures.HighestVolumeRoll(),
        )
        selection = current.selections.row(0, named=True)
        seen.append(selection)
        asset = f"capture:{selection['selected_instrument_id']}"
        if len(seen) == 1:
            broker.submit_order(asset, 1)
        else:
            broker.roll_position(
                "capture:17",
                asset,
                roll_id="volume-r1",
                decision_at=selection["decision_at"],
                available_at=selection["metric_available_at"],
                selection=selection,
            )

    runtime.scheduler.register(
        decide,
        phase=LifecyclePhase.INTRABAR,
        at="clock",
        schedule=RebalanceSchedule.explicit_timestamps(decisions),
    )

    class Noop(Strategy):
        def on_data(self, *args):
            pass

    result = Engine(
        DataFeed(prices_df=raw, feed_spec=prepared.metadata["normalized_feed_spec"]),
        Noop(),
        BacktestConfig(
            initial_cash=100000,
            allow_leverage=True,
            commission_type=CommissionType.PER_TRADE,
            commission_per_trade=2,
            slippage_rate=0,
        ),
        runtime=runtime,
        contract_specs={
            f"capture:{asset}": ContractSpec(
                f"capture:{asset}",
                asset_class=AssetClass.FUTURE,
                multiplier=50,
                margin=15000,
                accounting_method="futures_variation",
                settlement_policy="on_mark",
            )
            for asset in [17, 23]
        },
        holdings_sampling="bar",
    ).run()
    assert [s["selected_instrument_id"] for s in seen] == [17, 23]
    assert [s["metric_observed_at"] for s in seen] == TIMES[:2]
    assert [(f.timestamp, f.asset, f.price) for f in result.fills] == [
        (TIMES[2], "capture:17", 4020),
        (TIMES[3], "capture:17", 4030),
        (TIMES[3], "capture:23", 4040),
    ]
    assert result.to_portfolio_state_dataframe()["cash"][-1] == 100994
    rolls = result.to_rolls_dataframe()
    assert rolls["decision_at"].to_list() == [decisions[1]] * 2
    assert rolls["available_at"].to_list() == [TIMES[1]] * 2
    assert rolls["filled_at"].to_list() == [TIMES[3]] * 2
    oi = futures.prepare_futures_inputs(
        bars,
        definitions,
        as_of=decisions[1],
        feed_spec=SPEC,
        roll_strategy=futures.HighestOpenInterestRoll(),
    )
    assert oi.selections["selected_instrument_id"].to_list() == [17]
