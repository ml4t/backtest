"""Consume actual Data-prepared history without changing executable observations."""

import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from ml4t.specs import FeedSpec
from polars.testing import assert_frame_equal

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import ExecutionMode

ROOT = Path(__file__).parent / "fixtures" / "prepared_history"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def _inputs():
    return pl.read_parquet(ROOT / "raw.parquet"), FeedSpec.from_any(MANIFEST["input_feed_spec"])


def _snapshot(name):
    snapshot = next(s for s in MANIFEST["snapshots"] if s["name"] == name)
    path = ROOT / (name + ".parquet")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == snapshot["parquet_sha256"]
    return SimpleNamespace(prices=pl.read_parquet(path), metadata=dict(snapshot["metadata"]))


def _provider(*, as_of):
    return _snapshot("before" if as_of.day == 2 else "after")


def test_captured_data_history_adjusts_only_the_read_view_and_does_not_reprice_fills():
    raw, spec = _inputs()
    calls, seen = [], []

    def provide(*, as_of):
        calls.append(as_of)
        return _provider(as_of=as_of)

    runtime = RuntimeServices(history_provider=provide)

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            prepared = runtime.market.history(
                ["TEST"], ["close", "volume"], bar_count=2, view="prepared"
            )
            current = runtime.market.current(["TEST"], ["close"], view="prepared")
            observed = runtime.market.history(["TEST"], ["close"], bar_count=2)
            metadata = runtime.market.prepared_metadata
            assert metadata["price_basis"] == "split_adjusted"
            assert metadata["as_of"] == timestamp.isoformat()
            metadata["feed_spec"]["price_col"] = "corrupted"
            assert runtime.market.prepared_metadata["feed_spec"]["price_col"] == "adj_close"
            seen.append(
                (
                    prepared["close"].to_list(),
                    prepared["volume"].to_list(),
                    observed["close"].to_list(),
                    current["close"].to_list(),
                )
            )
            if timestamp.day == 2:
                broker.submit_order("TEST", 1)

    result = Engine(
        DataFeed(prices_df=raw, feed_spec=spec),
        Read(),
        BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR, initial_cash=10000),
        runtime=runtime,
    ).run()
    assert seen == [
        ([200.0], [10.0], [200.0], [200.0]),
        ([100.0, 100.0], [20.0, 20.0], [200.0, 100.0], [100.0]),
    ]
    assert calls == raw["timestamp"].to_list()
    assert result.to_fills_dataframe()["price"].to_list() == [200.0]
    assert result.to_equity_dataframe()["equity"].to_list() == [10000.0, 9900.0]
    # The raw corporate-action ledger is a separate account method, not a history side effect.
    assert_frame_equal(raw, pl.read_parquet(ROOT / "raw.parquet"))


@pytest.mark.parametrize(
    "fault,error",
    [
        ("retrospective", "causal"),
        ("later_cutoff", "as_of"),
        ("calendar", "calendar"),
        ("session_labels", "timestamp role"),
        ("future_revision", "unavailable"),
        ("unknown_key", "observed execution keys"),
        ("imputed", "imputed"),
    ],
)
def test_prepared_history_rejects_wrong_perspectives_or_unavailable_input_before_an_order(
    fault, error
):
    raw, spec = _inputs()
    runtime = RuntimeServices()
    orders = []

    def provide(*, as_of):
        prepared = _snapshot(
            "retrospective"
            if fault == "retrospective"
            else "after"
            if fault == "later_cutoff"
            else "before"
        )
        if fault != "later_cutoff":
            prepared.metadata["as_of"] = as_of.isoformat()
        if fault in {"calendar", "session_labels"}:
            feed = FeedSpec.from_any(prepared.metadata["feed_spec"])
            feed = replace(
                feed,
                **(
                    {"calendar": "NYSE"}
                    if fault == "calendar"
                    else {"timestamp_semantics": "session_label"}
                ),
            )
            prepared.metadata["feed_spec"] = feed.to_dict()
        elif fault == "future_revision":
            prepared.prices = prepared.prices.with_columns(
                pl.lit(as_of + timedelta(days=1)).alias("available_at")
            )
        elif fault == "unknown_key":
            prepared.prices = prepared.prices.with_columns(pl.lit("UNKNOWN").alias("asset"))
        elif fault == "imputed":
            prepared.prices = prepared.prices.with_columns(pl.lit(True).alias("is_imputed"))
        return prepared

    runtime.history_provider = provide

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            runtime.market.history(["TEST"], ["close"], bar_count=2, view="prepared")
            orders.append(broker.submit_order("TEST", 1))

    with pytest.raises(ValueError, match=error):
        Engine(
            DataFeed(prices_df=raw, feed_spec=spec),
            Read(),
            BacktestConfig(calendar=None),
            runtime=runtime,
        ).run()
    assert orders == []
    with pytest.raises(RuntimeError, match="active run"):
        runtime.market.current(["TEST"], ["close"], view="prepared")


def test_prepared_view_requires_an_explicit_provider_instead_of_guessing_adjustment_terms():
    raw, spec = _inputs()
    runtime = RuntimeServices()

    class Read(Strategy):
        def on_data(self, *args):
            runtime.market.history(["TEST"], ["close"], bar_count=1, view="prepared")

    with pytest.raises(ValueError, match="history_provider"):
        Engine(
            DataFeed(prices_df=raw, feed_spec=spec),
            Read(),
            BacktestConfig(calendar=None),
            runtime=runtime,
        ).run()


@pytest.mark.parametrize("append", [False, True])
def test_actual_data_producer_matches_captured_views_and_preserves_future_append_invariance(append):
    adjustments = pytest.importorskip("ml4t.data.adjustments")
    raw, spec = _inputs()
    actions = pl.read_parquet(ROOT / "actions.parquet")
    future = raw.tail(1).with_columns(
        (pl.col("timestamp") + timedelta(days=3)).alias("timestamp"),
        (pl.col("available_at") + timedelta(days=3)).alias("available_at"),
        pl.lit(99999.0).alias("close"),
    )
    source = pl.concat([raw, future]) if append else raw
    observed = []

    def prepare(*, as_of):
        result = adjustments.prepare_adjusted_history(
            source, actions, as_of=as_of, feed_spec=spec, **MANIFEST["method"]
        )
        assert_frame_equal(result.prices, _provider(as_of=as_of).prices)
        return result

    runtime = RuntimeServices(history_provider=prepare, end=raw["timestamp"][-1])

    class Read(Strategy):
        def on_data(self, *args):
            observed.append(
                runtime.market.history(["TEST"], ["close"], bar_count=2, view="prepared")[
                    "close"
                ].to_list()
            )

    Engine(
        DataFeed(prices_df=source, feed_spec=spec),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert observed == [[200.0], [100.0, 100.0]]


def test_actual_data_provider_and_runtime_do_not_round_adjacent_nanosecond_observations():
    adjustments = pytest.importorskip("ml4t.data.adjustments")
    epoch = 1735851600000000000
    clocks = pl.Series("timestamp", [epoch + 1, epoch + 2]).cast(pl.Datetime("ns", "UTC"))
    raw = pl.DataFrame(
        {
            "timestamp": clocks,
            "asset": ["TEST"] * 2,
            "close": [100.0, 101.0],
            "source": ["captured"] * 2,
            "revision_id": ["r1"] * 2,
            "available_at": clocks.alias("available_at"),
        }
    )
    spec = FeedSpec(
        time_contract_version=2, calendar=None, entity_col="asset", timestamp_semantics="event_time"
    )

    def prepare(*, as_of):
        return adjustments.prepare_adjusted_history(
            raw, None, as_of=as_of, feed_spec=spec, basis="raw"
        )

    runtime = RuntimeServices(history_provider=prepare)
    seen = []

    class Read(Strategy):
        def on_data(self, *args):
            history = runtime.market.history(["TEST"], ["close"], bar_count=2, view="prepared")
            seen.append((history["timestamp"].cast(pl.Int64).to_list(), history["close"].to_list()))

    Engine(
        DataFeed(prices_df=raw, feed_spec=spec),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen == [([epoch + 1], [100.0]), ([epoch + 1, epoch + 2], [100.0, 101.0])]


def test_actual_data_provider_can_return_empty_history_before_the_first_available_observation():
    adjustments = pytest.importorskip("ml4t.data.adjustments")
    raw, spec = _inputs()
    actions = pl.read_parquet(ROOT / "actions.parquet")

    def prepare(*, as_of, observation_before=None):
        visible = (
            raw
            if observation_before is None
            else raw.filter(pl.col(spec.timestamp_col) < observation_before)
        )
        return adjustments.prepare_adjusted_history(
            visible, actions, as_of=as_of, feed_spec=spec, **MANIFEST["method"]
        )

    runtime = RuntimeServices(
        history_provider=prepare, start=raw["timestamp"][0] - timedelta(hours=1)
    )
    seen = []

    class Read(Strategy):
        def on_start(self, broker):
            seen.append(runtime.market.history(["TEST"], ["close"], bar_count=1, view="prepared"))

        def on_data(self, *args):
            pass

    Engine(
        DataFeed(prices_df=raw, feed_spec=spec),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen[0].is_empty()
    assert seen[0].columns == ["timestamp", "asset", "close"]
