"""Legacy synthetic futures clocks must preserve observed inputs and native cleanup."""

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.api import calendars, date_rules, schedule_function, time_rules
from ml4t.backtest.compat.zipline.assets import Asset

ROOT = Path(__file__).parent / "fixtures/zipline_original"


@pytest.mark.parametrize("naive", [False, True], ids=["utc", "naive_new_york"])
@pytest.mark.parametrize("fixture", ["", "us-futures-dst"], ids=["standard", "dst_weekend"])
@pytest.mark.parametrize("restrict", [False, True], ids=["whole_capture", "explicit_first_day"])
def test_actual_original_future_clock_counts_prices_and_callbacks_match_with_dst_and_user_windows(
    fixture, restrict, naive
):
    root = ROOT / fixture
    reference = json.loads((root / "us-futures-clock-reference.json").read_text())
    frame = pl.read_csv(root / "us-futures-grid.csv", try_parse_dates=True)
    if naive:
        frame = frame.with_columns(
            pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.replace_time_zone(None)
        )
    asset = Asset("ES", "ESH25", 8)
    trace = []
    count = 0
    manual = None
    if restrict:
        opened = pd.Timestamp(reference["trace"][0]["timestamp"])
        closed = pd.Timestamp(reference["trace"][2]["timestamp"]) + pd.Timedelta(minutes=1)
        if naive:
            opened = opened.tz_convert("America/New_York").tz_localize(None)
            closed = closed.tz_convert("America/New_York").tz_localize(None)
        manual = ((opened, closed),)
    runtime = RuntimeServices(execution_windows=manual)

    def observe(kind, context, data):
        price = data.current(asset, "price")
        trace.append(
            {
                "kind": kind,
                "timestamp": context.get_datetime().isoformat(),
                "price": None if pd.isna(price) else float(price),
                "bars_seen": count,
            }
        )

    def initialize(context):
        context.set_max_leverage(1.0)
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

    def before(context, data):
        observe("before_trading_start", context, data)
        with pytest.raises(RuntimeError, match="before_trading_start"):
            context.order(asset, 1)

    def handle(context, data):
        nonlocal count
        count += 1

    performance = run_algorithm(
        feed=DataFeed(
            prices_df=frame,
            feed_spec={
                "time_contract_version": 2,
                "calendar": "us_futures",
                "timezone": "America/New_York" if naive else "UTC",
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
        before_trading_start=before,
    )
    assert trace == (reference["trace"][:3] if restrict else reference["trace"])
    assert count == (630 if restrict else 1260)
    assert len(performance) == count
    result = performance.attrs["backtest_result"]
    assert result.to_fills_dataframe().is_empty()
    assert result.to_equity_dataframe().height == count
    assert runtime.execution_windows == manual
    assert runtime.start is None and runtime.end is None
    assert runtime.scheduler._registrations == []


def test_futures_account_minimum_is_checked_at_execution_close_and_failed_run_restores_runtime():
    from datetime import timedelta

    from ml4t.backtest.controls import AccountControlViolation

    reference = json.loads((ROOT / "us-futures-clock-reference.json").read_text())
    frame = pl.read_csv(ROOT / "us-futures-grid.csv", try_parse_dates=True)
    runtime = RuntimeServices()
    seen = []

    def initialize(context):
        context.set_min_leverage(0.1, timedelta(0))

    with pytest.raises(AccountControlViolation, match="minimum gross leverage"):
        run_algorithm(
            feed=DataFeed(
                prices_df=frame,
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
            assets=[Asset("ES", "ESH25", 8)],
            runtime=runtime,
            initialize=initialize,
            handle_data=lambda context, _data: seen.append(context.get_datetime()),
        )
    assert seen[-1] == pd.Timestamp(reference["trace"][2]["timestamp"]) + timedelta(minutes=1)
    assert len(seen) == 630
    assert runtime.execution_windows is None and runtime.start is None and runtime.end is None
    assert runtime._engine is None
    assert runtime.scheduler._registrations == []
