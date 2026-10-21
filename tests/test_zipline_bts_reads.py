"""A pre-event read view must expose completed bars without publishing a current quote."""

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.api import calendars, date_rules, schedule_function, time_rules
from ml4t.backtest.compat.zipline.assets import Asset

ROOT = Path(__file__).parent / "fixtures/zipline_original/bts-fields"
REFERENCE = json.loads((ROOT / "us-futures-clock-reference.json").read_text())
FIELDS = ["price", "open", "high", "low", "close", "volume", "last_traded"]


def _clean(value):
    if pd.isna(value):
        return None
    return value.isoformat() if isinstance(value, pd.Timestamp) else float(value)


def test_before_trading_ohlcv_last_trade_and_staleness_match_real_original_completed_minute_reads():
    source = pl.read_csv(ROOT / "us-futures-grid.csv", try_parse_dates=True)
    runtime = RuntimeServices()
    asset = Asset("ES", "ESH25", 8)
    trace = []
    count = 0

    def observe(kind, context, data):
        fields = data.current(asset, FIELDS)
        values = {key: _clean(value) for key, value in fields.items()}
        trace.append(
            {
                "kind": kind,
                "timestamp": context.get_datetime().isoformat(),
                "price": values["price"],
                "bars_seen": count,
                "fields": values,
                "is_stale": bool(data.is_stale(asset)),
                "can_trade": bool(data.can_trade(asset)),
            }
        )
        if kind == "before_trading_start":
            assert runtime.market.current(["ES"], ["close", "volume"]).is_empty()
            assert data.is_stale([asset]).tolist() == [False]
            assert data.current([asset], FIELDS).iloc[0].map(_clean).to_dict() == values

    def initialize(context):
        schedule_function(
            lambda c, d: observe("open", c, d),
            date_rules.every_day(),
            time_rules.market_open(),
            calendar=calendars.US_FUTURES,
        )
        schedule_function(
            lambda c, d: observe("close", c, d),
            date_rules.every_day(),
            time_rules.market_close(),
            calendar=calendars.US_FUTURES,
        )

    def handle(context, data):
        nonlocal count
        count += 1

    result = run_algorithm(
        feed=DataFeed(
            prices_df=source,
            feed_spec={
                "time_contract_version": 2,
                "calendar": "us_futures",
                "data_frequency": "1m",
                "timestamp_semantics": "bar_close",
                "bar_label": "right",
                "bar_grid_anchor": "session_open",
                "partial_bar_policy": "include",
            },
        ),
        assets=[asset],
        runtime=runtime,
        initialize=initialize,
        handle_data=handle,
        before_trading_start=lambda c, d: observe("before_trading_start", c, d),
    )
    assert trace == REFERENCE["trace"]
    assert count == 1260
    assert result.attrs["backtest_result"].to_fills_dataframe().is_empty()


def test_pre_event_projection_can_consume_actual_causal_data_adjustments_without_repricing_raw_quotes():
    import pytest
    from ml4t.specs import FeedSpec, LifecyclePhase

    from ml4t.backtest import BacktestConfig, Engine, Strategy
    from ml4t.backtest.compat.zipline.data import BarData
    from ml4t.backtest.execution import RebalanceSchedule

    module = pytest.importorskip("ml4t.data.adjustments")
    root = Path(__file__).parent / "fixtures/prepared_history"
    manifest = json.loads((root / "manifest.json").read_text())
    raw = pl.read_parquet(root / "raw.parquet")
    actions = pl.read_parquet(root / "actions.parquet")
    spec = FeedSpec.from_any(manifest["input_feed_spec"])
    cutoff = raw["timestamp"][-1]

    def provide(*, as_of, observation_before=None):
        visible = (
            raw
            if observation_before is None
            else raw.filter(pl.col(spec.timestamp_col) < observation_before)
        )
        return module.prepare_adjusted_history(
            visible, actions, as_of=as_of, feed_spec=spec, **manifest["method"]
        )

    runtime = RuntimeServices(start=cutoff, history_provider=provide)
    adjusted = BarData(runtime, read_policy="zipline", history_view="prepared")
    observed = BarData(runtime, read_policy="zipline")
    asset = Asset("TEST", "TEST", 7)
    seen = []

    def inspect(timestamp, data, context, broker):
        assert runtime.market.current(["TEST"], ["close"]).is_empty()
        assert adjusted.current(asset, "price") == 100.0
        assert adjusted.current(asset, ["price", "close", "volume"]).tolist() == [
            100.0,
            100.0,
            20.0,
        ]
        assert observed.current(asset, ["price", "close", "volume"]).tolist() == [
            200.0,
            200.0,
            10.0,
        ]
        seen.append(timestamp)

    runtime.scheduler.register(
        inspect,
        phase=LifecyclePhase.INTRABAR,
        at="clock",
        schedule=RebalanceSchedule.explicit_timestamps([cutoff]),
    )

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            assert runtime.market.current(["TEST"], ["close"])["close"][0] == 100.0

    result = Engine(
        DataFeed(prices_df=raw, feed_spec=spec),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert seen == [cutoff]
    assert result.to_fills_dataframe().is_empty()


@pytest.mark.parametrize(
    "future_close,expected_prior", [(100.0, 200.0 * 100.0 / 110.0), (40.0, 160.0)]
)
def test_pre_event_cash_factor_cannot_use_the_pending_quote_but_after_event_can_use_it(
    future_close, expected_prior
):
    from datetime import date

    import pytest
    from ml4t.specs import FeedSpec, LifecyclePhase

    from ml4t.backtest import BacktestConfig, Engine, Strategy
    from ml4t.backtest.compat.zipline.data import BarData
    from ml4t.backtest.execution import RebalanceSchedule

    module = pytest.importorskip("ml4t.data.adjustments")
    root = Path(__file__).parent / "fixtures/prepared_history"
    raw = pl.read_parquet(root / "raw.parquet").with_columns(pl.lit("USD").alias("currency"))
    actions = (
        pl.read_parquet(root / "actions.parquet")
        .drop("split_ratio")
        .with_columns(
            pl.lit("cash_dividend").alias("action_kind"),
            pl.lit(10.0).alias("cash_per_share"),
            pl.lit("USD").alias("currency"),
            pl.lit(date(2025, 1, 3)).alias("ex_date"),
            pl.lit(date(2025, 1, 6)).alias("payment_date"),
            pl.lit("post_event").alias("share_basis"),
            pl.lit("cash1").alias("source_record_id"),
        )
    )
    spec = FeedSpec.from_any(
        json.loads((root / "manifest.json").read_text())["input_feed_spec"]
    ).with_overrides(calendar="NYSE", data_frequency="daily")
    cutoff = raw["timestamp"][-1]
    prices = raw.with_columns(
        pl.when(pl.col("timestamp") == cutoff)
        .then(pl.lit(future_close))
        .otherwise(pl.col("close"))
        .alias("close")
    )
    calls, seen = [], []

    def provide(*, as_of, observation_before=None):
        calls.append((as_of, observation_before))
        visible = (
            prices
            if observation_before is None
            else prices.filter(pl.col(spec.timestamp_col) < observation_before)
        )
        return module.prepare_adjusted_history(
            visible,
            actions,
            as_of=as_of,
            feed_spec=spec,
            basis="total_return",
            action_coverage="complete",
        )

    runtime = RuntimeServices(start=cutoff, history_provider=provide)
    reader = BarData(
        runtime, read_policy="zipline", history_view="prepared", history_unit="observation"
    )

    def before(timestamp, data, context, broker):
        assert runtime.market.current(["TEST"], ["close"]).is_empty()
        with pytest.raises(ValueError, match="actually available ex-date close"):
            reader.current(Asset("TEST", "TEST", 7), "price")
        seen.append("no_current_close")

    runtime.scheduler.register(
        before,
        phase=LifecyclePhase.INTRABAR,
        at="clock",
        schedule=RebalanceSchedule.explicit_timestamps([cutoff]),
    )

    class Read(Strategy):
        def on_data(self, *args):
            frame = runtime.market.history(["TEST"], ["close"], bar_count=2, view="prepared")
            assert frame["close"].to_list() == pytest.approx([expected_prior, future_close])

    Engine(
        DataFeed(prices_df=prices, feed_spec=spec),
        Read(),
        BacktestConfig(calendar="NYSE"),
        runtime=runtime,
    ).run()
    assert seen == ["no_current_close"]
    assert calls[0] == (cutoff, cutoff)
    assert calls[-1] == (cutoff, None)


@pytest.mark.parametrize("named", [False, True], ids=["same_grid", "named_source"])
@pytest.mark.parametrize("naive", [False, True], ids=["utc", "naive_new_york"])
def test_observation_boundary_preserves_availability_equality_and_normalizes_actual_data_clock(
    named, naive
):
    from ml4t.specs import FeedSpec, LifecyclePhase

    from ml4t.backtest import BacktestConfig, Engine, Strategy
    from ml4t.backtest.execution import RebalanceSchedule

    module = pytest.importorskip("ml4t.data.adjustments")
    root = Path(__file__).parent / "fixtures/prepared_history"
    manifest = json.loads((root / "manifest.json").read_text())
    raw = pl.read_parquet(root / "raw.parquet")
    spec = FeedSpec.from_any(manifest["input_feed_spec"])
    cutoff = raw["timestamp"][-1]
    source = raw.with_columns(pl.lit(cutoff).alias("available_at"))
    feed_frame = source
    if naive:
        feed_frame = source.with_columns(
            pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None),
            pl.col("available_at")
            .dt.convert_time_zone("America/New_York")
            .dt.replace_time_zone(None),
        )
        spec = spec.with_overrides(timezone="America/New_York")
    calls = []

    def provide(*, as_of, observation_before=None):
        calls.append((as_of, observation_before))
        visible = (
            source
            if observation_before is None
            else source.filter(pl.col("timestamp") < observation_before.astimezone(cutoff.tzinfo))
        )
        return module.prepare_adjusted_history(
            visible, None, as_of=as_of, feed_spec=spec, basis="raw"
        )

    runtime = RuntimeServices(
        start=feed_frame["timestamp"][-1],
        history_sources={"past": provide} if named else None,
        history_provider=None if named else provide,
        known_at_cols={"close": "available_at", "volume": "available_at"},
    )
    seen = []

    def inspect(timestamp, data, context, broker):
        view = runtime.history_source("past") if named else runtime.market
        values = view.history(
            ["TEST"], ["close"], bar_count=1, **({} if named else {"view": "prepared"})
        )
        assert values["close"].to_list() == [200.0]
        assert pd.Timestamp(view.prepared_metadata["as_of"]) == cutoff
        seen.append(timestamp)

    runtime.scheduler.register(
        inspect,
        phase=LifecyclePhase.INTRABAR,
        at="clock",
        schedule=RebalanceSchedule.explicit_timestamps([cutoff]),
    )

    class Read(Strategy):
        def on_data(self, *args):
            pass

    Engine(
        DataFeed(prices_df=feed_frame, feed_spec=spec),
        Read(),
        BacktestConfig(calendar=None),
        runtime=runtime,
    ).run()
    assert len(seen) == 1
    assert calls[0][0] == calls[0][1] == cutoff


@pytest.mark.parametrize("named", [False, True], ids=["same_grid", "named_source"])
def test_provider_ignoring_pre_event_boundary_cannot_publish_pending_actual_data_observation(named):
    from ml4t.specs import FeedSpec, LifecyclePhase

    from ml4t.backtest import BacktestConfig, Engine, Strategy
    from ml4t.backtest.execution import RebalanceSchedule

    module = pytest.importorskip("ml4t.data.adjustments")
    root = Path(__file__).parent / "fixtures/prepared_history"
    spec = FeedSpec.from_any(json.loads((root / "manifest.json").read_text())["input_feed_spec"])
    raw = pl.read_parquet(root / "raw.parquet")
    cutoff = raw["timestamp"][-1]

    def provide(*, as_of, observation_before=None):
        return module.prepare_adjusted_history(raw, None, as_of=as_of, feed_spec=spec, basis="raw")

    runtime = RuntimeServices(
        start=cutoff,
        history_sources={"past": provide} if named else None,
        history_provider=None if named else provide,
    )

    def inspect(timestamp, data, context, broker):
        view = runtime.history_source("past") if named else runtime.market
        view.history(["TEST"], ["close"], bar_count=1, **({} if named else {"view": "prepared"}))

    runtime.scheduler.register(
        inspect,
        phase=LifecyclePhase.INTRABAR,
        at="clock",
        schedule=RebalanceSchedule.explicit_timestamps([cutoff]),
    )

    class Read(Strategy):
        def on_data(self, *args):
            pass

    with pytest.raises(ValueError, match="observation excluded by observation_before"):
        Engine(
            DataFeed(prices_df=raw, feed_spec=spec),
            Read(),
            BacktestConfig(calendar=None),
            runtime=runtime,
        ).run()
