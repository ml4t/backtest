"""Independent prepared read grids must not become executable observations."""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset
from ml4t.backtest.config import ExecutionMode

ROOT = Path(__file__).parent / "fixtures/history_sources"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
ASSET = Asset("A", "A", 7)


def _feed():
    times = [datetime.fromisoformat(s["metadata"]["as_of"]) for s in MANIFEST["snapshots"]]
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": times,
                "asset": ["A"] * 3,
                "close": [201.0, 202.0, 203.0],
                "volume": [1000.0] * 3,
            }
        ),
        feed_spec={**MANIFEST["feed_spec"], "data_frequency": "1m"},
    )


def _provider(*, as_of):
    snapshot = next(
        s for s in MANIFEST["snapshots"] if pd.Timestamp(s["metadata"]["as_of"]) == as_of
    )
    path = ROOT / snapshot["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST["files"][path.name]
    return SimpleNamespace(prices=pl.read_parquet(path), metadata=dict(snapshot["metadata"]))


@pytest.mark.parametrize("actual", [False, True], ids=["captured", "actual_data"])
def test_daily_read_grid_uses_actual_data_output_while_orders_fill_only_on_minute_quotes(actual):
    provider = _provider
    if actual:
        module = pytest.importorskip("ml4t.data.adjustments")

        def provider(*, as_of):
            result = module.prepare_adjusted_history(
                pl.read_parquet(ROOT / "raw.parquet"),
                None,
                as_of=as_of,
                feed_spec=FeedSpec.from_any(MANIFEST["feed_spec"]),
                basis="raw",
            )
            assert result.prices.equals(_provider(as_of=as_of).prices)
            return result

    calls, views, trace = [], [], []

    def prepared(*, as_of):
        calls.append(as_of)
        return provider(as_of=as_of)

    runtime = RuntimeServices(history_sources={"daily": prepared})

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            view = runtime.history_source("daily")
            views.append(view)
            assert runtime.history_source("daily") is view
            assert view.history(["A"], ["close"], bar_count=2, unit="expected_bar")[
                "close"
            ].to_list() == [90.0, 100.0]
            trace.append(runtime.market.current(["A"], ["close"])["close"][0])
            metadata = view.prepared_metadata
            assert metadata["price_basis"] == "raw"
            metadata["feed_spec"]["calendar"] = "corrupt"
            assert view.prepared_metadata["feed_spec"]["calendar"] == "NYSE"
            if len(trace) == 1:
                broker.submit_order("A", 1)

    result = Engine(
        _feed(), Read(), BacktestConfig(execution_mode=ExecutionMode.NEXT_BAR), runtime=runtime
    ).run()
    assert trace == [201.0, 202.0, 203.0]
    assert calls == _feed().prices["timestamp"].to_list()
    assert result.to_fills_dataframe()["price"].to_list() == [202.0]
    with pytest.raises(RuntimeError, match="active run"):
        views[0].history(["A"], ["close"], bar_count=2)
    with pytest.raises(RuntimeError, match="active run"):
        runtime.history_source("daily")


def test_original_callback_history_selects_the_declared_daily_source_without_resampling():
    observed = []

    def handle(context, data):
        daily = data.history(ASSET, "price", 2, "1d")
        assert daily.index.tolist() == [datetime(2024, 7, 1), datetime(2024, 7, 2)]
        observed.append((daily.tolist(), data.current(ASSET, "price")))
        if len(observed) == 1:
            context.order(ASSET, 1)

    result = run_algorithm(
        feed=_feed(),
        assets=[ASSET],
        handle_data=handle,
        runtime=RuntimeServices(history_sources={"daily": _provider}),
        history_sources={"1d": "daily"},
    )
    assert observed == [([90.0, 100.0], 201.0), ([90.0, 100.0], 202.0), ([90.0, 100.0], 203.0)]
    fills = [fill for period in result["transactions"] for fill in period]
    assert [fill["price"] for fill in fills] == pytest.approx([202.101])


@pytest.mark.parametrize(
    "fault,error",
    [
        ("future", "future"),
        ("publication", "unavailable"),
        ("retrospective", "causal"),
        ("cutoff", "as_of"),
        ("imputed", "imputed"),
        ("labels", "physical"),
        ("identity", "asset"),
        ("missing_clock", "version-2"),
    ],
)
def test_invalid_producer_snapshot_fails_before_any_order(fault, error):
    orders = []

    def provide(*, as_of):
        value = _provider(as_of=as_of)
        if fault == "future":
            value.prices = value.prices.with_columns(
                pl.lit(as_of + timedelta(minutes=1)).alias("timestamp")
            )
        elif fault == "publication":
            value.prices = value.prices.with_columns(
                pl.lit(as_of + timedelta(minutes=1)).alias("available_at")
            )
        elif fault == "imputed":
            value.prices = value.prices.with_columns(pl.lit(True).alias("is_imputed"))
        elif fault == "identity":
            value.prices = value.prices.with_columns(pl.lit("UNKNOWN").alias("asset"))
        elif fault == "cutoff":
            value.metadata["as_of"] = (as_of + timedelta(minutes=1)).isoformat()
        elif fault == "retrospective":
            value.metadata["perspective"] = "retrospective"
        else:
            value.metadata["feed_spec"] = {
                **value.metadata["feed_spec"],
                "bar_label": None,
                "bar_grid_anchor": None,
                "partial_bar_policy": None,
                **(
                    {"timestamp_semantics": "session_label"}
                    if fault == "labels"
                    else {"time_contract_version": 1}
                ),
            }
        return value

    runtime = RuntimeServices(history_sources={"daily": provide})

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            runtime.history_source("daily").history(["A"], ["close"], bar_count=2)
            orders.append(broker.submit_order("A", 1))

    with pytest.raises(ValueError, match=error):
        Engine(_feed(), Read(), runtime=runtime).run()
    assert orders == []


def test_naive_execution_clock_and_aware_data_snapshots_refer_to_the_same_instant():
    feed = _feed()
    frame = feed.prices.with_columns(
        pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
    )
    seen, retained = [], []
    runtime = RuntimeServices(history_sources={"daily": _provider})

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if retained:
                with pytest.raises(RuntimeError, match="no longer active"):
                    retained[-1].history(["A"], ["close"], bar_count=2)
            source = runtime.history_source("daily")
            retained.append(source)
            values = source.history(["A"], ["close"], bar_count=2, unit="expected_bar")
            assert values["timestamp"].dt.hour().to_list() == [16, 16]
            assert pd.Timestamp(source.prepared_metadata["as_of"]) == pd.Timestamp(
                timestamp
            ).tz_localize("America/New_York")
            seen.append(values["close"].to_list())

    Engine(
        DataFeed(
            prices_df=frame, feed_spec=feed.feed_spec.with_overrides(timezone="America/New_York")
        ),
        Read(),
        runtime=runtime,
    ).run()
    assert seen == [[90.0, 100.0]] * 3


def test_source_registration_does_not_relax_same_grid_prepared_history_guards():
    runtime = RuntimeServices(history_provider=_provider, history_sources={"daily": _provider})

    class Read(Strategy):
        def on_data(self, *args):
            runtime.market.history(["A"], ["close"], bar_count=2, view="prepared")

    with pytest.raises(ValueError, match="observed execution keys"):
        Engine(_feed(), Read(), runtime=runtime).run()
