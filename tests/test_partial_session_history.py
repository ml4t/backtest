"""A minute callback must never read its session's future daily OHLCV."""

import io
import json
import tarfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from ml4t.backtest import DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.compat.zipline import PreparedAssetLookup, run_algorithm

ROOT = Path(__file__).parent / "fixtures/zipline_partial_session"
REFERENCE = json.loads((ROOT / "reference.json").read_text())
FIELDS = ["price", "open", "high", "low", "close", "volume"]


def _snapshot(actual=False):
    with tarfile.open(ROOT / "snapshot.tar.gz") as archive:
        if actual:
            storage = pytest.importorskip("ml4t.data.storage")
            with TemporaryDirectory(prefix="ml4t-partial-session-") as directory:
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


def _feed(snapshot):
    return DataFeed(
        prices_df=snapshot.tables["minute_bars"].with_columns(
            # The controlled source publishes each synthetic input at its bar close.
            pl.col("timestamp").alias("available_at")
        ),
        feed_spec={
            "calendar": "NYSE",
            "data_frequency": "1m",
            "time_contract_version": 2,
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )


def _services(feed, actual):
    start = datetime.fromisoformat(REFERENCE["trace"][0]["timestamp"])
    if not actual:
        return RuntimeServices(start=start)
    adjustments = pytest.importorskip("ml4t.data.adjustments")

    def provide(*, as_of, observation_before=None):
        prices = feed.prices
        if observation_before is not None:
            prices = prices.filter(pl.col("timestamp") < observation_before)
        return adjustments.prepare_adjusted_history(
            prices, None, as_of=as_of, feed_spec=feed.feed_spec, basis="raw"
        )

    return RuntimeServices(start=start, history_provider=provide)


@pytest.mark.parametrize("actual", [False, True], ids=["captured", "installed_data"])
@pytest.mark.parametrize("entry", ["native", "original_callback"])
def test_actual_original_partial_daily_fields_use_only_minutes_observed_through_each_callback(
    actual, entry
):
    snapshot = _snapshot(actual)
    feed = _feed(snapshot)
    lookup = PreparedAssetLookup(snapshot)
    chosen = sorted(lookup.assets, key=lambda asset: asset.legacy_sid)
    targets = {datetime.fromisoformat(row["timestamp"]): row for row in REFERENCE["trace"]}
    services = _services(feed, actual)
    seen = []

    def inspect(timestamp, data=None):
        if timestamp not in targets:
            return
        expected = targets[timestamp]["cases"]["default"]
        if entry == "original_callback":
            frame = data.history(chosen, FIELDS, 2, "1d")
            assert [[day.isoformat(), int(asset)] for day, asset in frame.index] == expected[
                "index"
            ]
            assert frame.columns.tolist() == expected["columns"]
        else:
            native = services.market.session_bars(
                [asset.key for asset in chosen],
                FIELDS,
                bar_count=2,
                view="prepared" if actual else "raw",
                forward_fill={"price": True},
                finite_prices_only=True,
            )
            frame = pd.DataFrame(native.select(FIELDS).rows(), columns=FIELDS, dtype="float64")
            assert (
                native["timestamp"].to_list()
                == [datetime.fromisoformat("2024-07-02T20:00:00+00:00")] * 2 + [timestamp] * 2
            )
            assert native["session_closed"].to_list() == [True, True] + [timestamp.hour == 17] * 2
        pd.testing.assert_frame_equal(
            pd.DataFrame(frame.to_numpy(), columns=FIELDS, dtype="float64"),
            pd.DataFrame(expected["values"], columns=FIELDS, dtype="float64"),
            check_exact=True,
        )
        seen.append(timestamp)

    if entry == "native":

        class Read(Strategy):
            def on_data(self, timestamp, data, context, broker):
                inspect(timestamp)

        result = Engine(feed, Read(), runtime=services).run()
        assert not result.fills
    else:

        def handle(context, data):
            inspect(context.get_datetime().to_pydatetime(), data)

        output = run_algorithm(
            feed=feed,
            asset_lookup=lookup,
            runtime=services,
            handle_data=handle,
            history_view="prepared" if actual else "raw",
            start="2024-07-03",
            end="2024-07-03",
        )
        assert not output.attrs["backtest_result"].fills
    assert seen == list(targets)
    assert snapshot.snapshot_id == REFERENCE["snapshot_id"]


@pytest.mark.parametrize("gap", ["missing_row", "unknown_volume", "nonfinite_volume"])
def test_unknown_minute_volume_cannot_become_a_known_partial_session_total(gap):
    from datetime import timedelta

    feed = _feed(_snapshot())
    first = datetime.fromisoformat(REFERENCE["trace"][0]["timestamp"])
    second = first + timedelta(minutes=1)
    key = "partial-daily-reference:7"
    target = (pl.col("asset") == key) & (pl.col("timestamp") == second)
    prices = feed.prices
    if gap == "missing_row":
        prices = prices.filter(~target)
    else:
        prices = prices.with_columns(
            pl.when(target)
            .then(None if gap == "unknown_volume" else float("nan"))
            .otherwise(pl.col("volume"))
            .alias("volume")
        )
    runtime = RuntimeServices(start=first, end=second)
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == second:
                bars = runtime.market.session_bars(
                    [key], ["price", "volume"], bar_count=1, forward_fill={"price": True}
                )
                assert bars["volume"][0] is None
                assert bars["expected_bars"][0] == 2
                assert bars["observed_bars"][0] == (1 if gap == "missing_row" else 2)
                assert bars["price"][0] == (201.0 if gap == "missing_row" else 202.0)
                seen.append(timestamp)

    result = Engine(
        DataFeed(prices_df=prices, feed_spec=feed.feed_spec), Read(), runtime=runtime
    ).run()
    assert seen == [second] and not result.fills


def test_partial_aggregation_respects_each_fields_publication_instant_and_future_prefix():
    from datetime import timedelta

    feed = _feed(_snapshot())
    first = datetime.fromisoformat(REFERENCE["trace"][0]["timestamp"])
    second, third = first + timedelta(minutes=1), first + timedelta(minutes=2)
    key = "partial-daily-reference:7"
    target = (pl.col("asset") == key) & (pl.col("timestamp") == second)
    frame = feed.prices.with_columns(
        pl.when(target).then(pl.lit(third)).otherwise(pl.col("timestamp")).alias("published_at"),
        pl.when(target).then(500.0).otherwise(pl.col("high")).alias("high"),
    )

    def execute(prices):
        runtime = RuntimeServices(
            start=first, end=third, known_at_cols={"high": "published_at", "volume": "published_at"}
        )
        seen = []

        class Read(Strategy):
            def on_data(self, timestamp, data, context, broker):
                bars = runtime.market.session_bars([key], ["high", "volume"], bar_count=1)
                seen.append((timestamp, bars["high"][0], bars["volume"][0]))

        Engine(DataFeed(prices_df=prices, feed_spec=feed.feed_spec), Read(), runtime=runtime).run()
        return seen

    expected = [(first, 203.0, 1000.0), (second, 203.0, None), (third, 500.0, 3000.0)]
    assert execute(frame) == expected
    changed = frame.with_columns(
        pl.when(pl.col("timestamp") > third).then(1e12).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.col("timestamp") > third).then(1e12).otherwise(pl.col("volume")).alias("volume"),
    )
    assert execute(changed) == expected


@pytest.mark.parametrize("fault", ["no_calendar", "off_grid"])
def test_session_aggregation_needs_a_declared_calendar_and_accurate_intraday_grid(fault):
    from datetime import timedelta

    feed = _feed(_snapshot())
    first = datetime.fromisoformat(REFERENCE["trace"][0]["timestamp"])
    spec, prices = feed.feed_spec, feed.prices
    if fault == "no_calendar":
        spec = spec.with_overrides(calendar=None)
    else:
        prices = prices.with_columns(
            (pl.col("timestamp") + timedelta(seconds=30)).alias("timestamp")
        )
    runtime = RuntimeServices(start=first)

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            runtime.market.session_bars(["partial-daily-reference:7"], ["close"], bar_count=1)

    with pytest.raises(ValueError, match="declared calendar|off-grid"):
        Engine(DataFeed(prices_df=prices, feed_spec=spec), Read(), runtime=runtime).run()


@pytest.mark.parametrize("naive", [False, True], ids=["utc", "naive_new_york"])
def test_before_trading_daily_aggregation_reads_only_completed_warmup_sessions(naive):
    feed = _feed(_snapshot())
    prices, spec = feed.prices, feed.feed_spec
    if naive:
        prices = prices.with_columns(
            pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
        )
        spec = spec.with_overrides(timezone="America/New_York")
    lookup = PreparedAssetLookup(_snapshot())
    chosen = sorted(lookup.assets, key=lambda asset: asset.legacy_sid)
    calls = []

    def before(context, data):
        frame = data.history(chosen, ["price", "volume"], 2, "1d")
        assert [[day.isoformat(), int(asset)] for day, asset in frame.index] == [
            ["2024-07-01T00:00:00", 7],
            ["2024-07-01T00:00:00", 8],
            ["2024-07-02T00:00:00", 7],
            ["2024-07-02T00:00:00", 8],
        ]
        assert frame.to_numpy().tolist() == [
            [90.0, 390000.0],
            [38.0, 390000.0],
            [100.0, 390000.0],
            [40.0, 390000.0],
        ]
        calls.append(context.get_datetime())

    result = run_algorithm(
        feed=DataFeed(prices_df=prices, feed_spec=spec),
        asset_lookup=lookup,
        before_trading_start=before,
        start="2024-07-03",
        end="2024-07-03",
    )
    assert calls == [pd.Timestamp("2024-07-03T12:45:00Z")]
    assert not result.attrs["backtest_result"].fills


def test_actual_data_split_adjustment_is_applied_before_session_aggregation_without_repricing_fills():
    from datetime import timedelta

    from ml4t.backtest import BacktestConfig, ExecutionMode
    from ml4t.backtest.config import ExecutionPrice

    adjustments = pytest.importorskip("ml4t.data.adjustments")
    feed = _feed(_snapshot(True))
    first = datetime.fromisoformat(REFERENCE["trace"][0]["timestamp"])
    key = "partial-daily-reference:7"
    actions = pl.DataFrame(
        {
            "asset": [key],
            "event_id": ["split"],
            "revision_id": ["r1"],
            "action_kind": ["split"],
            "source": ["controlled-session-test"],
            "effective_at": [first - timedelta(minutes=1)],
            "available_at": [first - timedelta(hours=1)],
            "split_ratio": [2.0],
        }
    )
    calls, seen = [], []

    def provide(*, as_of, observation_before=None):
        prices = feed.prices
        if observation_before is not None:
            prices = prices.filter(pl.col("timestamp") < observation_before)
        result = adjustments.prepare_adjusted_history(
            prices,
            actions,
            as_of=as_of,
            feed_spec=feed.feed_spec,
            basis="split_adjusted",
            action_coverage="complete",
        )
        calls.append(result.metadata["price_basis"])
        return result

    runtime = RuntimeServices(start=first, end=first, history_provider=provide)

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            raw = runtime.market.session_bars([key], ["close", "volume"], bar_count=2)
            prepared = runtime.market.session_bars(
                [key], ["close", "volume"], bar_count=2, view="prepared"
            )
            assert raw["close"].to_list() == [100.0, 201.0]
            assert raw["volume"].to_list() == [390000.0, 1000.0]
            assert prepared["close"].to_list() == [50.0, 201.0]
            assert prepared["volume"].to_list() == [780000.0, 1000.0]
            assert runtime.market.current([key], ["price"])["price"][0] == 201.0
            seen.append(timestamp)
            broker.submit_order(key, 1)

    result = Engine(
        feed,
        Read(),
        BacktestConfig(execution_mode=ExecutionMode.SAME_BAR, execution_price=ExecutionPrice.CLOSE),
        runtime=runtime,
    ).run()
    assert seen == [first]
    assert [fill.price for fill in result.fills] == [201.0]
    assert result.to_portfolio_state_dataframe()["equity"].to_list() == [100000.0]
    assert calls == ["split_adjusted"]


def test_integer_session_volume_above_float_precision_keeps_the_exact_input_units():
    from datetime import timedelta

    feed = _feed(_snapshot())
    first = datetime.fromisoformat(REFERENCE["trace"][0]["timestamp"])
    second = first + timedelta(minutes=1)
    key = "partial-daily-reference:7"
    large = 2**53 + 1
    prices = feed.prices.with_columns(pl.lit(large, dtype=pl.Int64).alias("volume"))
    runtime = RuntimeServices(start=first, end=second)
    observed = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            frame = runtime.market.session_bars([key], ["volume"], bar_count=1)
            assert frame["volume"].dtype == pl.Int64
            observed.append(frame["volume"][0])

    Engine(DataFeed(prices_df=prices, feed_spec=feed.feed_spec), Read(), runtime=runtime).run()
    assert observed == [large, 2 * large]


@pytest.mark.parametrize("ffill", [False, True])
def test_price_carry_is_explicit_and_never_carries_missing_session_ohlc_or_volume(ffill):
    feed = _feed(_snapshot())
    first = datetime.fromisoformat(REFERENCE["trace"][0]["timestamp"])
    key = "partial-daily-reference:8"
    runtime = RuntimeServices(start=first, end=first)
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            bars = runtime.market.session_bars(
                [key], FIELDS, bar_count=1, forward_fill={"price": ffill}
            )
            assert bars["price"][0] == (40.0 if ffill else None)
            assert bars.select("open", "high", "low", "close").rows() == [(None,) * 4]
            assert bars["volume"][0] == 0
            seen.append(timestamp)

    result = Engine(feed, Read(), runtime=runtime).run()
    assert seen == [first] and not result.fills
