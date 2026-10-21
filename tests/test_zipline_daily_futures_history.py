"""The legacy daily execution close cannot borrow a later full-calendar quote."""

import io
import json
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import polars as pl
import pytest

from ml4t.backtest import DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.compat.zipline import PreparedAssetLookup, run_algorithm

ROOT = Path(__file__).parent / "fixtures/zipline_daily_futures"
REFERENCE = json.loads((ROOT / "reference.json").read_text())


def _snapshot(actual=False, *, root=ROOT):
    with tarfile.open(root / "snapshot.tar.gz") as archive:
        if actual:
            storage = pytest.importorskip("ml4t.data.storage")
            with TemporaryDirectory(prefix="ml4t-daily-futures-") as directory:
                archive.extractall(directory, filter="data")
                return storage.read_snapshot(directory)
        manifest = json.loads(archive.extractfile("./manifest.json").read())
        return SimpleNamespace(
            snapshot_id=manifest["snapshot_id"],
            tables={
                name: pl.read_parquet(
                    io.BytesIO(archive.extractfile("./" + entry["files"][0]["path"]).read())
                )
                for name, entry in manifest["content"]["tables"].items()
            },
        )


def _feed(snapshot, reference=REFERENCE):
    raw = snapshot.tables["daily_bars"]
    instants = [
        datetime.fromisoformat(row["timestamp"])
        for row in reference["trace"]
        if row["kind"] == "handle"
    ]
    clocks = pl.DataFrame({"session": raw["timestamp"].unique().sort(), "timestamp": instants})
    frame = (
        raw.rename({"timestamp": "session"})
        .join(clocks, on="session")
        .with_columns(
            # These synthetic quotes explicitly become known at the reference execution close.
            pl.col("timestamp").alias("available_at")
        )
    )
    return DataFeed(
        prices_df=frame,
        feed_spec={
            "calendar": "us_futures",
            "time_contract_version": 2,
            "data_frequency": "daily",
            "timestamp_semantics": "event_time",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )


@pytest.mark.parametrize("actual", [False, True], ids=["captured", "installed_data"])
@pytest.mark.parametrize("naive", [False, True], ids=["utc", "naive_new_york"])
@pytest.mark.parametrize("case", ["", "dst"], ids=["standard", "dst_weekend"])
def test_original_daily_futures_current_history_and_pre_event_clocks_match_without_future_quotes(
    actual,
    naive,
    case,
):
    reference = json.loads((ROOT / case / "reference.json").read_text())
    snapshot = _snapshot(actual, root=ROOT / case)
    feed = _feed(snapshot, reference)
    if naive:
        feed = DataFeed(
            prices_df=feed.prices.with_columns(
                pl.col("timestamp")
                .dt.convert_time_zone("America/New_York")
                .dt.replace_time_zone(None)
            ),
            feed_spec=feed.feed_spec.with_overrides(timezone="America/New_York"),
        )
    lookup = PreparedAssetLookup(snapshot)
    asset = lookup.assets[0]
    calls = []
    runtime = RuntimeServices()
    if actual:
        module = pytest.importorskip("ml4t.data.adjustments")

        def provide(*, as_of, observation_before=None):
            prices = feed.prices
            if observation_before is not None:
                bound = observation_before
                if prices.schema["timestamp"].time_zone is None:
                    bound = bound.tz_convert(feed.feed_spec.timezone).tz_localize(None)
                prices = prices.filter(pl.col("timestamp") < bound)
            return module.prepare_adjusted_history(
                prices, None, as_of=as_of, feed_spec=feed.feed_spec, basis="raw"
            )

        runtime = RuntimeServices(history_provider=provide)

    def before(context, data):
        day = len([row for row in calls if row["kind"] == "handle"])
        if day:
            previous = data.current(asset, ["close", "volume"])
            assert previous.tolist() == [99.0 + day, 1000.0]
        calls.append({"kind": "before", "timestamp": context.get_datetime().isoformat()})

    def handle(context, data):
        history = data.history(asset, ["close", "volume"], 1, "1d")
        calls.append(
            {
                "kind": "handle",
                "timestamp": context.get_datetime().isoformat(),
                "price": data.current(asset, "price"),
                "history": history.to_numpy().tolist(),
                "history_labels": [day.isoformat() for day, _asset in history.index],
            }
        )

    result = run_algorithm(
        feed=feed,
        asset_lookup=lookup,
        handle_data=handle,
        before_trading_start=before,
        runtime=runtime,
        history_view="prepared" if actual else "raw",
    )
    assert calls == reference["trace"]
    assert not result.attrs["backtest_result"].fills


def test_explicit_native_daily_read_offset_preserves_default_calendar_grid_and_later_quote_boundary():
    feed = _feed(_snapshot())
    point = feed.prices["timestamp"].max()
    key = "daily-futures-clock-reference:8"
    future = feed.prices.filter(pl.col("timestamp") == point).with_columns(
        (pl.col("timestamp") + timedelta(hours=1)).alias("timestamp"),
        (pl.col("available_at") + timedelta(hours=1)).alias("available_at"),
        pl.lit(999999.0).alias("close"),
    )
    runtime = RuntimeServices(start=point, end=point)
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            with pytest.raises(ValueError, match="off-grid"):
                runtime.market.history([key], ["close"], bar_count=1, unit="expected_bar")
            selected = runtime.market.history(
                [key],
                ["close"],
                bar_count=1,
                unit="expected_bar",
                session_close_offset=-timedelta(hours=1),
            )
            assert selected["timestamp"][0] == point
            assert selected["close"][0] == 102.0
            assert runtime.market.current([key], ["price"])["price"][0] == 102.0
            seen.append(timestamp)

    result = Engine(
        DataFeed(prices_df=pl.concat([feed.prices, future]), feed_spec=feed.feed_spec),
        Read(),
        runtime=runtime,
    ).run()
    assert seen == [point] and not result.fills


@pytest.mark.parametrize(
    "unit,offset,frequency,label,error",
    [
        ("observation", -timedelta(hours=1), "daily", "right", "requires expected_bar"),
        ("expected_bar", timedelta(minutes=1), "daily", "right", "nonpositive timedelta"),
        ("expected_bar", "-1h", "daily", "right", "nonpositive timedelta"),
        ("expected_bar", -timedelta(days=1), "daily", "right", "follow.*session open"),
        ("expected_bar", -timedelta(hours=1), "1m", "right", "daily right labels"),
        ("expected_bar", -timedelta(hours=1), "daily", "left", "daily right labels"),
    ],
)
def test_daily_read_endpoint_rejects_ambiguous_or_invalid_clocks_before_any_order(
    unit, offset, frequency, label, error
):
    feed = _feed(_snapshot())
    runtime = RuntimeServices()
    submitted = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            runtime.market.history(
                ["daily-futures-clock-reference:8"],
                ["close"],
                bar_count=1,
                unit=unit,
                session_close_offset=offset,
            )
            submitted.append(broker.submit_order("daily-futures-clock-reference:8", 1))

    with pytest.raises(ValueError, match=error):
        Engine(
            DataFeed(
                prices_df=feed.prices,
                feed_spec=feed.feed_spec.with_overrides(data_frequency=frequency, bar_label=label),
            ),
            Read(),
            runtime=runtime,
        ).run()
    assert submitted == []


def test_a_known_after_hours_quote_cannot_be_reclassified_as_a_later_regular_session_close():
    point = datetime.fromisoformat("2025-03-03T22:00:00+00:00")
    feed = DataFeed(
        prices_df=pl.DataFrame({"timestamp": [point], "asset": ["A"], "close": [100.0]}),
        feed_spec={
            "calendar": "NYSE",
            "time_contract_version": 2,
            "data_frequency": "daily",
            "timestamp_semantics": "event_time",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )
    runtime = RuntimeServices()
    observed, submitted = [], []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            assert runtime.market.current(["A"], ["price"])["price"][0] == 100.0
            observed.append(timestamp)
            runtime.market.history(
                ["A"],
                ["close"],
                bar_count=1,
                unit="expected_bar",
                session_close_offset=timedelta(hours=1),
            )
            submitted.append(broker.submit_order("A", 1))

    with pytest.raises(ValueError, match="nonpositive timedelta"):
        Engine(feed, Read(), runtime=runtime).run()
    assert observed == [point] and submitted == []


def test_earlier_daily_expected_grid_keeps_a_missing_session_instead_of_counting_older_observations():
    feed = _feed(_snapshot())
    point = feed.prices["timestamp"].max()
    absent = point - timedelta(days=1)
    frame = feed.prices.filter(pl.col("timestamp") != absent)
    runtime = RuntimeServices(start=point, end=point)
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            expected = runtime.market.history(
                ["daily-futures-clock-reference:8"],
                ["close", "volume"],
                bar_count=2,
                unit="expected_bar",
                session_close_offset=-timedelta(hours=1),
            )
            observed = runtime.market.history(
                ["daily-futures-clock-reference:8"],
                ["close"],
                bar_count=2,
            )
            assert expected["timestamp"].to_list() == [absent, point]
            assert expected["close"].to_list() == [None, 102.0]
            assert expected["volume"].to_list() == [None, 1000]
            assert observed["close"].to_list() == [100.0, 102.0]
            seen.append(timestamp)

    result = Engine(
        DataFeed(prices_df=frame, feed_spec=feed.feed_spec), Read(), runtime=runtime
    ).run()
    assert seen == [point] and not result.fills
