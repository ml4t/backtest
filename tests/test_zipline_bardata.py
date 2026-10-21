"""Native read containers match actual published Zipline3.2, including axis order."""

import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from ml4t.specs import FeedSpec

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/zipline_bardata/reference.json").read_text()
)


def _assets():
    from ml4t.backtest.compat.zipline.assets import Asset

    return [Asset("A", symbol="ALPHA", legacy_sid=0), Asset("B", symbol="BETA", legacy_sid=1)]


def _feed(append_future=False, frequency="1d"):
    rows = []
    for asset, closes in REFERENCE["inputs"]["closes"].items():
        for day, close, volume in zip([6, 7, 8], closes, [1000, 2000, 3000], strict=True):
            rows.append(
                {
                    "timestamp": (
                        datetime(2025, 1, day, 21, tzinfo=UTC)
                        if frequency == "1d"
                        else datetime(2025, 1, 6, 14, 31 + day - 6, tzinfo=UTC)
                    ),
                    "asset": asset,
                    "open": close - 2,
                    "high": close + 1,
                    "low": close - 3,
                    "close": close,
                    "volume": volume,
                }
            )
    if append_future:
        rows.append(
            {
                "timestamp": datetime(2025, 1, 10, 21, tzinfo=UTC),
                "asset": "A",
                "open": 9999.0,
                "high": 10001.0,
                "low": 9998.0,
                "close": 10000.0,
                "volume": 999999,
            }
        )
    return DataFeed(
        prices_df=pl.DataFrame(rows),
        feed_spec=FeedSpec(
            time_contract_version=2,
            calendar="NYSE",
            data_frequency=frequency,
            timestamp_semantics="bar_close",
            bar_grid_anchor="session_open",
            bar_label="right",
            partial_bar_policy="include",
        ),
    )


def _freeze(value):
    from ml4t.backtest.compat.zipline.assets import Asset

    def label(item):
        if isinstance(item, Asset):
            return item.key
        if isinstance(item, tuple):
            return [label(v) for v in item]
        if isinstance(item, pd.Timestamp):
            return item.isoformat()
        return item

    if isinstance(value, pd.DataFrame):
        return {
            "kind": "DataFrame",
            "index": [label(v) for v in value.index],
            "index_names": list(value.index.names),
            "columns": [label(v) for v in value.columns],
            "column_names": list(value.columns.names),
            "dtypes": [str(t) for t in value.dtypes],
            "index_dtypes": [
                str(value.index.get_level_values(i).dtype) for i in range(value.index.nlevels)
            ],
            "values": value.to_numpy().tolist(),
        }
    if isinstance(value, pd.Series):
        return {
            "kind": "Series",
            "name": label(value.name),
            "index": [label(v) for v in value.index],
            "index_names": list(value.index.names),
            "dtype": str(value.dtype),
            "index_dtypes": [
                str(value.index.get_level_values(i).dtype) for i in range(value.index.nlevels)
            ],
            "values": value.to_numpy().tolist(),
        }
    return {"kind": "scalar", "value": float(value)}


@pytest.mark.parametrize("append_future", [False, True])
@pytest.mark.parametrize("frequency", ["1d", "1m"])
def test_actual_engine_current_and_history_match_all_four_published_shapes_and_permutations(
    append_future,
    frequency,
):
    from ml4t.backtest.compat.zipline.data import BarData

    target = (
        datetime(2025, 1, 8, 21, tzinfo=UTC)
        if frequency == "1d"
        else datetime(2025, 1, 6, 14, 33, tzinfo=UTC)
    )
    reference = (
        REFERENCE
        if frequency == "1d"
        else json.loads(
            (Path(__file__).parent / "fixtures/zipline_bardata/minute-reference.json").read_text()
        )
    )
    runtime = RuntimeServices(end=target)
    observed, saved = {}, []
    a, b = _assets()

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp != target:
                return
            view = BarData(runtime)
            saved.append(view)
            for name, assets, fields in [
                ("scalar_scalar", a, "price"),
                ("scalar_fields", a, ["price", "volume"]),
                ("assets_scalar", [a, b], "price"),
                ("assets_fields", [a, b], ["price", "volume"]),
                ("reverse_scalar_fields", a, ["volume", "price"]),
                ("reverse_assets_scalar", [b, a], "price"),
                ("reverse_assets_fields", [b, a], ["volume", "price"]),
            ]:
                observed["current_" + name] = _freeze(view.current(assets, fields))
                observed["history_" + name] = _freeze(view.history(assets, fields, 3, frequency))

    result = Engine(
        _feed(append_future, frequency), Read(), BacktestConfig(calendar="NYSE"), runtime=runtime
    ).run()
    assert observed == reference["values"]
    assert result.to_fills_dataframe().is_empty()
    with pytest.raises(RuntimeError, match="active run"):
        saved[0].current(a, "price")


def test_dated_display_metadata_and_legacy_sid_cannot_change_native_identity():
    from ml4t.backtest.compat.zipline.assets import Asset

    before, after = Asset("A", symbol="OLD", legacy_sid=0), Asset("A", symbol="NEW", legacy_sid=0)
    reused = Asset("B", symbol="OLD", legacy_sid=1)
    assert before == after and hash(before) == hash(after)
    assert before != reused and len({before, after, reused}) == 2
    assert int(before) == 0
    with pytest.raises(FrozenInstanceError):
        before.symbol = "CHANGED"
    unknown = Asset("X")
    assert unknown.symbol is None and unknown.sid == "X"
    with pytest.raises(TypeError, match="No legacy integer"):
        int(unknown)


def test_cross_frequency_history_cannot_silently_count_daily_rows_as_minutes():
    from ml4t.backtest.compat.zipline.data import BarData

    runtime = RuntimeServices()
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            with pytest.raises(ValueError, match="explicitly prepared source grid"):
                BarData(runtime).history(_assets(), "price", 3, "1m")
            seen.append(timestamp)

    Engine(_feed(), Read(), BacktestConfig(calendar="NYSE"), runtime=runtime).run()
    assert len(seen) == 3


def test_preopen_current_is_missing_and_history_never_reads_the_future_daily_close():
    from ml4t.specs import LifecyclePhase

    from ml4t.backtest.compat.zipline.data import BarData

    pre = datetime(2025, 1, 7, 13, 45, tzinfo=UTC)
    runtime = RuntimeServices(start=pre)
    a = _assets()[0]
    seen = []

    def before(timestamp, data, context, broker):
        if timestamp != pre:
            return
        view = BarData(runtime)
        assert pd.isna(view.current(a, "price"))
        assert view.current([a], ["price", "volume"]).isna().all().all()
        history = view.history(a, "price", 1, "1d")
        assert history.index.tolist() == [pd.Timestamp("2025-01-06")]
        assert history.iloc[0] == 100.0
        with pytest.raises(ValueError, match="before captured source coverage"):
            view.history(a, "price", 2, "1d")
        seen.append(timestamp)

    runtime.scheduler.register(
        before, phase=LifecyclePhase.PRE_OPEN, at="session_open", minutes_before=45
    )

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pass

    Engine(_feed(), Read(), BacktestConfig(calendar="NYSE"), runtime=runtime).run()
    assert seen == [pre]


def test_merging_bundle_sid_columns_cannot_alias_distinct_native_assets():
    from ml4t.backtest.compat.zipline.assets import Asset
    from ml4t.backtest.compat.zipline.data import BarData

    mappings = pl.DataFrame({"key": ["A", "B"], "legacy_sid": [0, 0]})
    assets = [Asset(**row) for row in mappings.iter_rows(named=True)]
    runtime = RuntimeServices()

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            with pytest.raises(ValueError, match="Legacy SID mapping is ambiguous"):
                BarData(runtime).current(assets, "price")
            assert broker.get_orders() == ()

    Engine(_feed(), Read(), BacktestConfig(calendar="NYSE"), runtime=runtime).run()


@pytest.mark.parametrize("unit", [None, "observation", "expected_bar"])
def test_explicit_calendar_none_uses_observations_and_preserves_physical_indexes(unit):
    from ml4t.backtest.compat.zipline.data import BarData

    runtime = RuntimeServices()
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            if timestamp.day != 7:
                return
            kwargs = {} if unit is None else {"history_unit": unit}
            reader = BarData(runtime, **kwargs)
            if unit == "expected_bar":
                with pytest.raises(ValueError, match="require a declared calendar"):
                    reader.history(_assets()[0], "price", 2, "1d")
            else:
                frame = reader.history(_assets()[0], "price", 2, "1d")
                assert frame.index.tolist() == list(
                    pd.to_datetime(["2025-01-06T21:00:00Z", "2025-01-07T21:00:00Z"])
                )
                assert frame.tolist() == [100.0, 101.0]
            seen.append(timestamp)

    result = Engine(_feed(), Read(), BacktestConfig(calendar=None), runtime=runtime).run()
    assert result.config.resolved_calendar is None and len(seen) == 1
