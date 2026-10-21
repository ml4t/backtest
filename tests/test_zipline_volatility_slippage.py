"""Actual daily-history and futures-impact options must produce the original execution ledger."""

import io
import json
import tarfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import polars as pl
import pytest

from ml4t.backtest import DataFeed
from ml4t.backtest.compat.zipline import PreparedAssetLookup, run_algorithm
from ml4t.backtest.compat.zipline.finance import commission, slippage

ROOT = Path(__file__).parent / "fixtures/zipline_volatility"
REFERENCE = json.loads((ROOT / "reference.json").read_text())


def _snapshot(actual=False, *, directory=ROOT):
    with tarfile.open(directory / "snapshot.tar.gz") as archive:
        if actual:
            storage = pytest.importorskip("ml4t.data.storage")
            with TemporaryDirectory(prefix="ml4t-volatility-facts-") as directory:
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
    frame = snapshot.tables["daily_bars"]
    clocks = pl.DataFrame(
        {
            "label": frame["timestamp"].unique().sort(),
            "timestamp": [datetime.fromisoformat(value) for value in REFERENCE["closes"]],
        }
    )
    return DataFeed(
        prices_df=frame.rename({"timestamp": "label"}).join(clocks, on="label"),
        feed_spec={
            "calendar": "NYSE",
            "data_frequency": "daily",
            "time_contract_version": 2,
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )


@pytest.mark.parametrize("actual", [False, True], ids=["captured_data", "installed_data"])
@pytest.mark.parametrize("case", sorted(REFERENCE["cases"]))
def test_actual_original_mixed_fills_match_sparse_and_complete_history_scalar_root_default_eta(
    actual, case
):
    snapshot = _snapshot(actual)
    reference = REFERENCE["cases"][case]
    direction = int(case.split("/")[1])

    def initialize(context):
        context.n = 0
        context.set_commission(
            us_equities=commission.PerShare(0), us_futures=commission.PerFutureTrade(0)
        )
        context.set_slippage(
            us_equities=slippage.FixedBasisPointsSlippage(5, 0.1),
            us_futures=slippage.VolatilityVolumeShare(**reference["settings"]),
        )

    def handle(context, data):
        if context.n in [0, 20]:
            context.order(context.sid(7), direction * 8)
            context.order(context.sid(reference["asset_sid"]), direction * 8)
        context.n += 1

    result = run_algorithm(
        feed=_feed(snapshot),
        asset_lookup=PreparedAssetLookup(snapshot),
        initialize=initialize,
        handle_data=handle,
        capital_base=1000000.0,
    )
    native = result.attrs["backtest_result"]
    actual_fills = [
        (
            fill.timestamp.isoformat(),
            int(fill.asset.rsplit(":", 1)[1]),
            fill.quantity * (1 if fill.side.value == "buy" else -1),
        )
        for fill in native.fills
    ]
    assert actual_fills == [
        (fill["timestamp"], fill["sid"], fill["amount"]) for fill in reference["fills"]
    ]
    assert [fill.price for fill in native.fills] == pytest.approx(
        [fill["price"] for fill in reference["fills"]], rel=0, abs=1e-10
    )
    assert result["portfolio_value"].tolist() == pytest.approx(reference["nav"], rel=0, abs=1e-7)
    assert [fill.price_source for fill in native.fills] == [
        "close",
        "mid",
        "mid",
        "close",
        "mid",
        "mid",
    ]
    assert native.fills[1].price == (102.0765 if direction > 0 else 101.9235)
    assert snapshot.snapshot_id == REFERENCE["snapshot_id"]


def test_actual_reference_known_zero_volume_missing_close_uses_only_prior_window_close_for_returns():
    directory = ROOT.parent / "zipline_volatility_missing"
    reference = json.loads((directory / "reference.json").read_text())["cases"]["scalar/1"]
    snapshot = _snapshot(directory=directory)

    def initialize(context):
        context.n = 0
        context.set_commission(
            us_equities=commission.PerShare(0), us_futures=commission.PerFutureTrade(0)
        )
        context.set_slippage(
            us_equities=slippage.FixedBasisPointsSlippage(5, 0.1),
            us_futures=slippage.VolatilityVolumeShare(0.05, eta=0.5),
        )

    def handle(context, data):
        if context.n in [0, 20]:
            context.order(context.sid(7), 8)
            context.order(context.sid(8), 8)
        context.n += 1

    output = run_algorithm(
        feed=_feed(snapshot),
        asset_lookup=PreparedAssetLookup(snapshot),
        initialize=initialize,
        handle_data=handle,
        capital_base=1000000.0,
    )
    native = output.attrs["backtest_result"]
    assert [fill.price for fill in native.fills] == pytest.approx(
        [fill["price"] for fill in reference["fills"]], rel=0, abs=1e-10
    )
    assert output["portfolio_value"].tolist() == pytest.approx(reference["nav"], rel=0, abs=1e-7)
    missing = datetime.fromisoformat(REFERENCE["closes"][10])
    raw = _feed(snapshot).prices.filter(
        (pl.col("asset") == "volatility-volume-reference:8") & (pl.col("timestamp") == missing)
    )
    assert raw["close"][0] is None and raw["volume"][0] == 0
    assert not any(fill.timestamp == missing for fill in native.fills)


@pytest.mark.parametrize("mode", ["per_order", "overfill"])
def test_future_participation_is_per_order_but_total_execution_never_exceeds_reported_volume(mode):
    snapshot = _snapshot()
    reference = json.loads((ROOT / "competing-reference.json").read_text())["cases"][mode]

    def initialize(context):
        context.sent = False
        context.set_commission(us_futures=commission.PerFutureTrade(0))
        context.set_slippage(
            us_futures=slippage.VolatilityVolumeShare(reference["volume_limit"], eta=0.5)
        )

    def handle(context, data):
        if not context.sent:
            context.order(context.sid(8), reference["order_quantity"])
            context.order(context.sid(8), reference["order_quantity"])
            context.sent = True

    native = run_algorithm(
        feed=_feed(snapshot),
        asset_lookup=PreparedAssetLookup(snapshot),
        initialize=initialize,
        handle_data=handle,
        capital_base=1000000.0,
    ).attrs["backtest_result"]
    fills = native.to_fills_dataframe()
    first = fills.filter(pl.col("timestamp") == datetime.fromisoformat(REFERENCE["closes"][1]))
    if mode == "per_order":
        assert first["quantity"].to_list() == [4.0, 4.0]
        assert first["price"].to_list() == [fill["price"] for fill in reference["fills"]]
    else:
        assert reference["first_bar_filled"] == 132 > reference["first_bar_volume"]
        assert first["quantity"].to_list() == [66.0, 44.0]
        second = fills.filter(pl.col("timestamp") == datetime.fromisoformat(REFERENCE["closes"][2]))
        assert second["quantity"].to_list() == [34.0, 56.0]
    assert first["quantity"].sum() <= reference["first_bar_volume"]


def test_pending_current_and_future_daily_closes_cannot_enter_prior_session_volatility():
    snapshot = _snapshot()
    base = _feed(snapshot)

    def execute(prices):
        def initialize(context):
            context.n = 0
            context.set_commission(us_futures=commission.PerFutureTrade(0))
            context.set_slippage(us_futures=slippage.VolatilityVolumeShare(0.05, eta=0.5))

        def handle(context, data):
            if context.n == 20:
                context.order(context.sid(8), 8)
            context.n += 1

        return run_algorithm(
            feed=DataFeed(prices_df=prices, feed_spec=base.feed_spec),
            asset_lookup=PreparedAssetLookup(snapshot),
            initialize=initialize,
            handle_data=handle,
            capital_base=1000000.0,
        ).attrs["backtest_result"]

    baseline = execute(base.prices)
    last_but_one = datetime.fromisoformat(REFERENCE["closes"][-2])
    changed = base.prices.with_columns(
        pl.when(
            (pl.col("asset") == "volatility-volume-reference:8")
            & (pl.col("timestamp") == last_but_one)
        )
        .then(108.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    updated = execute(changed)
    assert baseline.fills[0].price == updated.fills[0].price
    assert baseline.fills[1].price != updated.fills[1].price


def test_explicit_no_calendar_counts_prior_observations_across_a_missing_session():
    import math

    import pandas as pd

    snapshot = _snapshot()
    base = _feed(snapshot)
    omitted = datetime.fromisoformat(REFERENCE["closes"][10])
    submitted = datetime.fromisoformat(REFERENCE["closes"][20])
    executed = datetime.fromisoformat(REFERENCE["closes"][21])
    prices = base.prices.filter(pl.col("timestamp") != omitted)

    def run(calendar):
        def initialize(context):
            context.set_commission(us_futures=commission.PerFutureTrade(0))
            context.set_slippage(us_futures=slippage.VolatilityVolumeShare(0.05, eta=0.5))

        def handle(context, data):
            if context.get_datetime().to_pydatetime() == submitted:
                context.order(context.sid(8), 8)

        return run_algorithm(
            feed=DataFeed(
                prices_df=prices, feed_spec=base.feed_spec.with_overrides(calendar=calendar)
            ),
            asset_lookup=PreparedAssetLookup(snapshot),
            initialize=initialize,
            handle_data=handle,
            capital_base=1000000.0,
        ).attrs["backtest_result"]

    observations, sessions = run(None), run("NYSE")
    past = prices.filter(
        (pl.col("asset") == "volatility-volume-reference:8") & (pl.col("timestamp") < executed)
    ).sort("timestamp")
    assert len(past) == 20
    returns = pd.Series(past["close"].to_list()).pct_change(fill_method=None).iloc[1:]
    expected = (
        100.0
        + 100.0
        * 0.5
        * returns.std(ddof=1)
        * math.sqrt(252.0)
        * math.sqrt(5.0 / past["volume"].mean())
        / 10000.0
    )
    assert observations.config.resolved_calendar is None
    assert observations.fills[0].price == pytest.approx(expected, rel=0, abs=1e-10)
    assert sessions.fills[0].price == 100.075


@pytest.mark.parametrize("field", ["high", "low"])
@pytest.mark.parametrize("availability", ["missing", "unpublished", "published"])
def test_native_volatility_impact_requires_actual_published_high_low_before_filling(
    field, availability
):
    from datetime import timedelta

    from ml4t.backtest import BacktestConfig, Engine, RuntimeServices, Strategy
    from ml4t.backtest.config import ExecutionMode, ExecutionPrice
    from ml4t.backtest.models import AssetSlippage, NoSlippage, VolatilityVolumeSlippage

    base = _feed(_snapshot())
    key = "volatility-volume-reference:8"

    class Submit(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp == datetime.fromisoformat(REFERENCE["closes"][0]):
                broker.submit_order(key, 1)

    frame = base.prices
    runtime = None
    if availability == "missing":
        frame = frame.drop(field)
    else:
        frame = frame.with_columns(
            (pl.col("timestamp") + timedelta(seconds=availability == "unpublished")).alias(
                "published_at"
            )
        )
        runtime = RuntimeServices(known_at_cols={field: "published_at"})
    engine = Engine(
        DataFeed(prices_df=frame, feed_spec=base.feed_spec),
        Submit(),
        BacktestConfig(initial_cash=1000000.0, execution_mode=ExecutionMode.NEXT_BAR),
        slippage_model=AssetSlippage(
            {key: VolatilityVolumeSlippage(0.5)},
            default=NoSlippage(),
            price_sources={key: ExecutionPrice.MID},
        ),
        runtime=runtime,
    )
    if availability == "published":
        result = engine.run()
        assert len(result.fills) == 1
        assert result.fills[0].price == 102.0765
        return
    with pytest.raises(ValueError, match="observed positive high and low"):
        engine.run()
    assert engine.broker.get_fills() == []
    assert engine.broker.cash == 1000000.0


@pytest.mark.parametrize("measure", ["weight", "quantity"])
def test_opening_targets_cannot_read_completed_high_low_before_the_auction(measure):
    from datetime import UTC, date

    from ml4t.specs import (
        AssetTarget,
        CanonicalTargetIntent,
        IntentReason,
        LifecyclePhase,
        ResidualPolicy,
        RoundingPolicy,
        TargetMeasure,
    )

    from ml4t.backtest import BacktestConfig, Engine, Strategy, UnsupportedPreOpenPolicyError
    from ml4t.backtest.models import AssetSlippage, NoSlippage, VolatilityVolumeSlippage

    key = "volatility-volume-reference:8"
    decision = datetime(2025, 2, 3, 23, tzinfo=UTC)

    class Submit(Strategy):
        def on_prepare(self, broker, config=None):
            broker.register_target_intent(
                CanonicalTargetIntent(
                    intent_id="opening",
                    decision_time=decision,
                    information_cutoff=decision,
                    effective_session=date(2025, 2, 4),
                    effective_phase=LifecyclePhase.PRE_OPEN,
                    targets=(AssetTarget(key, TargetMeasure(measure), 1.0),),
                    idempotency_key="opening-key",
                    measure=TargetMeasure(measure),
                    cash_buffer=0.0,
                    rounding=RoundingPolicy.TOWARD_ZERO,
                    residual=ResidualPolicy.KEEP_CASH,
                    reason=IntentReason.REBALANCE,
                )
            )

        def on_data(self, *args):
            pass

    feed = _feed(_snapshot())
    engine = Engine(
        feed,
        Submit(),
        BacktestConfig(initial_cash=1000000.0),
        slippage_model=AssetSlippage({key: VolatilityVolumeSlippage(0.5)}, default=NoSlippage()),
    )
    with pytest.raises(UnsupportedPreOpenPolicyError, match="completed-bar high/low"):
        engine.run()
    assert not engine.broker.get_fills()
    assert engine.broker.cash == 1000000.0


@pytest.mark.parametrize(
    "revision", [False, True], ids=["frozen_history", "newly_available_revision"]
)
def test_minute_futures_consume_actual_data_completed_daily_source_without_resampling_execution(
    revision,
):
    import math

    import pandas as pd

    from ml4t.backtest import RuntimeServices

    adjustments = pytest.importorskip("ml4t.data.adjustments")
    snapshot = _snapshot(True)
    daily = _feed(snapshot)
    points = [
        datetime.fromisoformat("2025-03-06T13:31:00+00:00"),
        datetime.fromisoformat("2025-03-06T13:32:00+00:00"),
        datetime.fromisoformat("2025-03-06T13:33:00+00:00"),
    ]
    minute_frame = pl.DataFrame(
        [
            {
                "timestamp": point,
                "asset": asset,
                "close": 105.0,
                "high": 110.0,
                "low": 90.0,
                "volume": 100.0,
            }
            for point in points
            for asset in [
                "volatility-volume-reference:7",
                "volatility-volume-reference:8",
                "volatility-volume-reference:9",
            ]
        ]
    )
    calls = []
    daily_prices = daily.prices.with_columns(
        pl.col("timestamp").alias("available_at"), pl.lit("base").alias("revision_id")
    )
    if revision:
        correction = daily_prices.filter(
            (pl.col("asset") == "volatility-volume-reference:8")
            & (pl.col("timestamp") == datetime.fromisoformat(REFERENCE["closes"][20]))
        ).with_columns(
            (pl.col("close") + 1.0).alias("close"),
            pl.lit("correction").alias("revision_id"),
            pl.lit(points[2]).alias("available_at"),
        )
        daily_prices = pl.concat([daily_prices, correction], how="vertical_relaxed")

    def provide(*, as_of, observation_before=None):
        # This synthetic fixture explicitly declares observation availability at its close.
        prices = daily_prices
        if observation_before is not None:
            prices = prices.filter(pl.col("timestamp") < observation_before)
        prepared = adjustments.prepare_adjusted_history(
            prices, None, as_of=as_of, feed_spec=daily.feed_spec, basis="raw"
        )
        last = prepared.prices.filter(pl.col("asset") == "volatility-volume-reference:8").sort(
            "timestamp"
        )[-1]
        calls.append((as_of, prepared.prices["timestamp"].max(), last["close"][0]))
        return prepared

    def initialize(context):
        context.sent = False
        context.set_commission(us_futures=commission.PerFutureTrade(0))
        context.set_slippage(us_futures=slippage.VolatilityVolumeShare(0.05, eta=0.5))

    def handle(context, data):
        if not context.sent:
            context.order(context.sid(8), 8)
            context.sent = True

    native = run_algorithm(
        feed=DataFeed(
            prices_df=minute_frame, feed_spec=daily.feed_spec.with_overrides(data_frequency="1m")
        ),
        asset_lookup=PreparedAssetLookup(snapshot),
        initialize=initialize,
        handle_data=handle,
        runtime=RuntimeServices(history_sources={"daily": provide}),
        history_sources={"1d": "daily"},
        capital_base=1000000.0,
    ).attrs["backtest_result"]
    assert [fill.timestamp for fill in native.fills] == points[1:]
    assert [fill.quantity for fill in native.fills] == [5.0, 3.0]
    reference_price = REFERENCE["cases"]["scalar/1"]["fills"][-2]["price"]
    assert native.fills[0].price == pytest.approx(reference_price, rel=0, abs=1e-10)
    prior = (
        daily.prices.filter(
            (pl.col("asset") == "volatility-volume-reference:8") & (pl.col("timestamp") < points[0])
        )
        .sort("timestamp")
        .tail(20)
    )
    values = prior["close"].to_list()
    values[-1] = 103.0 if revision else 102.0
    expected_volatility = pd.Series(values).pct_change(fill_method=None).iloc[1:].std(
        ddof=1
    ) * math.sqrt(252.0)
    expected_price = (
        100.0
        + 100.0 * 0.5 * expected_volatility * math.sqrt(3.0 / prior["volume"].mean()) / 10000.0
    )
    assert native.fills[1].price == pytest.approx(expected_price, rel=0, abs=1e-10)
    assert all(
        as_of >= latest and latest == datetime.fromisoformat(REFERENCE["closes"][20])
        for as_of, latest, _close in calls
    )
    assert [(as_of, close) for as_of, _latest, close in calls] == [
        (points[1], 102.0),
        (points[2], 103.0 if revision else 102.0),
    ]
