"""Legacy stale presentation must not bypass native lifetime or execution boundaries."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

_ROOT = Path(__file__).parent / "fixtures/stale_instruments"
_MANIFEST = json.loads((_ROOT / "manifest.json").read_text())
_REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/zipline_original/stale-read-reference.json").read_text()
)
_ASSETS = [Asset("A", "A", 7), Asset("B", "B", 8)]


class _CapturedRegistry:
    def lifetimes(self, on, *, as_of, universe=None):
        assert universe is None
        match = next(
            row
            for row in _MANIFEST["snapshots"]
            if (row["on"], row["as_of"]) == (on.isoformat(), as_of.isoformat())
        )
        return pl.read_parquet(_ROOT / match["file"])

    def resolve(self, *args, **kwargs):
        raise AssertionError("Read fixture does not perform symbol lookup")


def _runtime(actual=False):
    spec = FeedSpec.from_any(_MANIFEST["feed_spec"])
    source = _CapturedRegistry()
    if actual:
        module = pytest.importorskip("ml4t.data.assets")
        source = module.InstrumentRegistry.from_frame(
            pl.read_parquet(_ROOT / "identities.parquet"), feed_spec=spec
        )
    return RuntimeServices(instruments=source, instrument_feed_spec=spec)


def _feed(nan=False):
    rows = [
        {
            "timestamp": datetime.fromisoformat(point),
            "asset": asset.key,
            "close": _REFERENCE["price"][str(int(asset))][index],
            "volume": _REFERENCE["volume"][str(int(asset))][index],
        }
        for index, point in enumerate(_REFERENCE["closes"])
        for asset in _ASSETS
    ]
    frame = pl.DataFrame(rows)
    if nan:
        frame = frame.with_columns(pl.col("close").fill_null(float("nan")))
    return DataFeed(
        prices_df=frame,
        feed_spec={
            "time_contract_version": 2,
            "calendar": "NYSE",
            "timestamp_semantics": "bar_close",
            "bar_label": "right",
            "data_frequency": "1d",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )


def _clean(value):
    if pd.isna(value):
        return None
    return value.isoformat() if isinstance(value, pd.Timestamp) else float(value)


@pytest.mark.parametrize("actual", [False, True], ids=["captured", "actual_data"])
@pytest.mark.parametrize("nan", [False, True], ids=["null", "nan"])
def test_price_close_volume_last_trade_and_lifetime_masks_match_actual_published_workflow(
    actual, nan
):
    trace = []

    def handle(context, data):
        frame = data.current(_ASSETS, ["price", "close", "volume", "last_traded"])
        item = {
            "timestamp": context.get_datetime().isoformat(),
            "can_trade": data.can_trade(_ASSETS).tolist(),
            "is_stale": data.is_stale(_ASSETS).tolist(),
            "current": {
                "columns": list(frame),
                "dtypes": [str(dtype) for dtype in frame.dtypes],
                "values": [[_clean(value) for value in row] for row in frame.to_numpy()],
            },
        }
        if trace:
            history = data.history(
                _ASSETS, ["price", "close", "volume"], bar_count=2, frequency="1d"
            )
            item["history"] = {
                "index": [[point.isoformat(), int(asset)] for point, asset in history.index],
                "index_names": list(history.index.names),
                "columns": list(history),
                "values": [[_clean(value) for value in row] for row in history.to_numpy()],
            }
        trace.append(item)

    performance = run_algorithm(
        feed=_feed(nan), assets=_ASSETS, runtime=_runtime(actual), handle_data=handle
    )
    assert trace == _REFERENCE["trace"]
    assert performance.attrs["backtest_result"].fills == []


def test_stale_price_view_cannot_enable_fills_and_native_quote_reads_stay_missing():
    runtime = _runtime()
    seen = []

    def handle(context, data):
        point = context.get_datetime()
        price = data.current(_ASSETS[0], "price")
        if point.day == 2:
            assert price == 100.0
            assert data.can_trade(_ASSETS[0]) is True
            assert runtime.market.can_trade("A") is False
            assert runtime.market.current(["A"], ["price"])["price"][0] is None
            context.order(_ASSETS[0], 1)
        seen.append(data.is_stale(_ASSETS[0]))

    result = run_algorithm(feed=_feed(), assets=_ASSETS, runtime=runtime, handle_data=handle).attrs[
        "backtest_result"
    ]
    assert seen == [False, True, True]
    assert result.fills == []
    assert result.to_equity_dataframe()["equity"].to_list() == [100000.0] * 3


def test_history_ffill_false_preserves_missing_prices_and_last_trade_scalar_series_dtypes():
    runtime = _runtime()
    seen = []

    def handle(context, data):
        single = data.current(_ASSETS[0], "last_traded")
        multiple = data.current(_ASSETS, "last_traded")
        assert str(multiple.dtype) == "datetime64[ns]"
        assert single == pd.Timestamp("2025-07-01")
        if context.get_datetime().day == 2:
            stale = data.history(_ASSETS[0], "price", 2, "1d")
            raw = data.history(_ASSETS[0], "price", 2, "1d", ffill=False)
            assert stale.tolist() == [100.0, 100.0]
            assert raw.iloc[0] == 100.0 and pd.isna(raw.iloc[1])
            seen.append(True)

    run_algorithm(feed=_feed(), assets=_ASSETS, runtime=runtime, handle_data=handle)
    assert seen == [True]


def test_native_bar_data_mode_preserves_raw_prices_and_observation_recency_definition():
    from ml4t.backtest import BacktestConfig, Engine, Strategy
    from ml4t.backtest.compat.zipline.data import BarData

    runtime = _runtime()
    view = BarData(runtime)
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(
                (
                    view.current(_ASSETS[0], "price"),
                    view.can_trade(_ASSETS[0]),
                    view.is_stale(_ASSETS[0]),
                )
            )

    Engine(_feed(), Read(), BacktestConfig(), runtime=runtime).run()
    assert seen[0] == (100.0, True, False)
    assert pd.isna(seen[1][0]) and seen[1][1:] == (False, True)
    assert seen[2] == (110.0, True, False)


def test_explicit_finite_price_view_retains_source_provenance_without_mutating_raw_quotes():
    from ml4t.backtest import BacktestConfig, Engine, Strategy

    runtime = _runtime()
    source = _feed(nan=True)
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day == 2:
                raw = runtime.market.current(["A"], ["price"])
                stale = runtime.market.current(
                    ["A"], ["price"], forward_fill=True, finite_prices_only=True
                )
                assert raw["price"].is_nan()[0]
                assert stale["price"][0] == 100.0
                assert stale["price_observed_at"][0] == datetime.fromisoformat(
                    _REFERENCE["closes"][0]
                )
                assert source.prices.filter(pl.col("asset") == "A")["close"].is_nan().to_list() == [
                    False,
                    True,
                    False,
                ]
                seen.append(True)

    Engine(source, Read(), BacktestConfig(), runtime=runtime).run()
    assert seen == [True]


def test_known_price_presentation_cannot_publish_future_quotes_into_a_liquid_execution_bar():
    times = [datetime.fromisoformat(point) for point in _REFERENCE["closes"]]
    prices = pl.DataFrame(
        {
            "timestamp": times,
            "asset": ["A"] * 3,
            "close": [100.0, 101.0, None],
            "volume": [1000] * 3,
            "published": [times[0], times[2], times[2]],
        }
    )
    feed = DataFeed(prices_df=prices, feed_spec=_feed().feed_spec)
    runtime = RuntimeServices(known_at_cols={"close": "published"})
    seen = []

    def handle(context, data):
        if len(seen) == 0:
            context.order(_ASSETS[0], 1)
        if len(seen) == 1:
            assert data.current(_ASSETS[0], "price") == 100.0
            assert runtime.market.current(["A"], ["close"])["close"][0] is None
            assert runtime.market.current(["A"], ["volume"])["volume"][0] == 1000
        seen.append(context.get_datetime())

    result = run_algorithm(
        feed=feed, assets=[_ASSETS[0]], runtime=runtime, handle_data=handle
    ).attrs["backtest_result"]
    assert len(seen) == 3 and result.fills == []
    assert result.to_equity_dataframe()["equity"].to_list() == [100000.0] * 3
