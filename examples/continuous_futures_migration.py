"""Prepare continuous research history using Data; keep raw dated execution separate."""

import argparse
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl
from ml4t.specs import FeedSpec

from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import ContinuousFuture, PreparedFutures, run_algorithm
from ml4t.backtest.compat.zipline.assets import Future
from ml4t.backtest.compat.zipline.finance import commission, slippage


def continuous_history_provider(
    bars: pl.DataFrame,
    source: PreparedFutures,
    feed_spec: FeedSpec,
    *,
    roll_marks: str = "previous_session",
) -> Any:
    """Build a causal daily history provider for this single-root synthetic workflow.

    Previous-session paired closes reproduce the captured Zipline convention.
    Same-session marks expose Data's build_roll_events convention explicitly.
    """
    from ml4t.data import adjustments as history
    from ml4t.data.futures import adjustment as adjustments
    from ml4t.data.futures import roll

    if roll_marks not in {"previous_session", "same_session"}:
        raise ValueError("Declare previous_session or same_session roll marks")
    adjustment = source.handle.adjustment

    def provide(*, as_of, observation_before=None):
        selected = []
        known = bars.filter(pl.col("timestamp") <= as_of)
        if observation_before is not None:
            known = known.filter(pl.col("timestamp") < observation_before)
        for point in known["timestamp"].unique().sort():
            prepared = source.provider(as_of=point)
            if prepared.selections.is_empty():
                continue
            identity = prepared.selections["selected_instrument_id"][0]
            selected.append({"date": point.date(), "symbol": str(identity)})
        raw = known.with_columns(
            pl.col("timestamp").dt.date().alias("date"),
            pl.col("asset").cast(pl.String).alias("symbol"),
            *[pl.col("close").alias(name) for name in ["open", "high", "low"]],
        )
        selection = pl.DataFrame(selected)
        events = roll.build_roll_events(raw, selection)
        if roll_marks == "previous_session":
            previous_events = []
            for event in events:
                paired = raw.filter(
                    (pl.col("date") < event.date)
                    & pl.col("symbol").is_in([event.old_symbol, event.new_symbol])
                )
                common = paired.group_by("date").len().filter(pl.col("len") == 2)["date"]
                if common.is_empty():
                    raise ValueError(
                        "Previous-session roll marks require an observed common session"
                    )
                paired = paired.filter(pl.col("date") == common.max())
                previous_events.append(
                    replace(
                        event,
                        old_close=paired.filter(pl.col("symbol") == event.old_symbol)["close"][0],
                        new_close=paired.filter(pl.col("symbol") == event.new_symbol)["close"][0],
                    )
                )
            events = previous_events
        method = {
            None: adjustments.NoAdjustment,
            "add": adjustments.BackAdjustment,
            "mul": adjustments.RatioAdjustment,
        }[adjustment]()
        continuous = method.adjust(raw.join(selection, on=["date", "symbol"]), events)
        prices = continuous.select(
            "timestamp",
            pl.lit(as_of).alias("available_at"),
            "source",
            "revision_id",
            pl.lit(source.handle.key).alias("asset"),
            pl.col("adjusted_close").alias("close"),
            "volume",
        )
        # Data owns adjustments and validates physical source availability again.
        return history.prepare_adjusted_history(
            prices,
            None,
            as_of=as_of,
            feed_spec=feed_spec.with_overrides(entity_col="asset"),
            basis="raw",
        )

    return provide


def synthetic_inputs():
    times = [datetime(2025, 7, 7, 20, tzinfo=UTC) + timedelta(days=i) for i in range(5)]
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
                "open_interest_available_at": times[3] if asset == 23 and i > 0 else t,
                "source": "capture",
                "revision_id": "r1",
            }
            for i, t in enumerate(times)
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


def run(output: Path, *, adjustment: str | None = "mul") -> dict[str, Any]:
    from ml4t.data.futures import HighestVolumeRoll, prepare_futures_inputs
    from ml4t.data.storage import SnapshotTable, export_snapshot, read_snapshot

    bars, definitions = synthetic_inputs()
    spec = FeedSpec(
        entity_col="asset",
        calendar="CME_Equity",
        time_contract_version=2,
        timestamp_semantics="event_time",
        data_frequency="daily",
    )
    declarations = {
        "source_revision": "continuous-example-r1",
        "identity_namespace": "capture",
        "units": {"close": "index_points", "volume": "contracts", "multiplier": "50 USD/point"},
        "as_of": bars["timestamp"].max().isoformat(),
        "price_basis": "raw",
        "share_basis": "source",
        "revisions": {"capture": "r1"},
    }
    captured = export_snapshot(
        output / "inputs",
        {
            "prices": SnapshotTable(
                bars,
                role="prices",
                feed_spec=spec,
                metadata=declarations,
            ),
            "definitions": SnapshotTable(
                definitions,
                role="instruments",
                metadata=declarations,
            ),
        },
        effective_config={"roll_marks": "previous_session", "adjustment": adjustment},
        artifacts={},
        calendar="CME_Equity",
    )
    inputs = read_snapshot(output / "inputs")
    bars, definitions = inputs.tables["prices"], inputs.tables["definitions"]

    def prepare(*, as_of):
        return prepare_futures_inputs(
            bars, definitions, as_of=as_of, feed_spec=spec, roll_strategy=HighestVolumeRoll()
        )

    assets = {
        identity: Future(
            f"capture:{identity}",
            f"TEST{identity}",
            identity,
            root_symbol="TEST",
            multiplier=50,
            tick_size=0.25,
        )
        for identity in [17, 23]
    }
    research = ContinuousFuture("research:TEST", root_symbol="TEST", adjustment=adjustment)
    source = PreparedFutures(research, prepare, assets, "CME", captured.snapshot_id)
    points = bars["timestamp"].unique().sort()
    prepared = prepare(as_of=points[-1])
    raw = prepared.bars.with_columns(
        pl.concat_str(pl.lit("capture:"), pl.col("asset").cast(pl.String)).alias("asset")
    )
    trace = []

    def initialize(context):
        context.set_commission(us_futures=commission.PerFutureTrade(0))
        context.set_slippage(us_futures=slippage.NoSlippage())
        context.cf = context.continuous_future("TEST", adjustment=adjustment)

    def handle(context, data):
        contract = data.current(context.cf, "contract")
        trace.append(
            {
                "timestamp": context.get_datetime().isoformat(),
                "contract": contract.key,
                "chain": [asset.key for asset in data.current_chain(context.cf)],
                "raw_price": data.current(context.cf, "price"),
                "history": data.history(context.cf, "price", 2, "1d").tolist(),
            }
        )
        if len(trace) == 1:
            context.order(contract, 1)

    result = run_algorithm(
        feed=DataFeed(prices_df=raw, feed_spec=prepared.metadata["normalized_feed_spec"]),
        assets=list(assets.values()),
        continuous_futures=[source],
        initialize=initialize,
        handle_data=handle,
        runtime=RuntimeServices(
            start=points[2],
            history_sources={"research": continuous_history_provider(bars, source, spec)},
            history_assets={"research": [research.key]},
        ),
        history_sources={"1d": "research"},
        history_unit="observation",
    ).attrs["backtest_result"]
    result.to_parquet(output / "result")
    report = {
        "input_snapshot_id": captured.snapshot_id,
        "random_seed": None,
        "packages": {name: version(name) for name in ["ml4t-data", "ml4t-backtest", "ml4t-specs"]},
        "roll_policy": prepared.metadata["selection_policy"],
        "roll_marks": "previous_session",
        "adjustment": adjustment,
        "trace": trace,
        "config": result.config.to_dict(),
        "fills": result.to_fills_dataframe().to_dicts(),
    }
    (output / "workflow.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--adjustment", choices=["none", "add", "mul"], default="mul")
    args = parser.parse_args()
    run(args.output, adjustment=None if args.adjustment == "none" else args.adjustment)
