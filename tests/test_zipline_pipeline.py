"""Prepared Engineer factors must preserve the reference Pipeline's premarket snapshot."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from ml4t.backtest import DataFeed, RuntimeServices
from ml4t.backtest.compat.zipline import PreparedPipeline, run_algorithm
from ml4t.backtest.compat.zipline.assets import Asset

_ROOT = Path(__file__).parent / "fixtures/prepared_pipeline"
_MANIFEST = json.loads((_ROOT / "manifest.json").read_text())
_REFERENCE = json.loads((_ROOT / "reference.json").read_text())


def _feed():
    rows = [
        {
            "timestamp": datetime.fromisoformat(timestamp),
            "asset": asset,
            "open": price,
            "high": price + 1,
            "low": price - 1,
            "close": price,
            "volume": _REFERENCE["input"][sid]["volume"][index],
        }
        for sid, asset in [("0", "A"), ("1", "B")]
        for index, (timestamp, price) in enumerate(
            zip(_REFERENCE["input_close_instants"], _REFERENCE["input"][sid]["close"], strict=True)
        )
    ]
    return DataFeed(
        prices_df=pl.DataFrame(rows),
        feed_spec={
            "calendar": "NYSE",
            "time_contract_version": 2,
            "timestamp_semantics": "bar_close",
            "data_frequency": "1d",
            "bar_label": "right",
            "bar_grid_anchor": "session_open",
            "partial_bar_policy": "include",
        },
    )


def _prepared():
    return PreparedPipeline(
        pl.read_parquet(_ROOT / "panel.parquet"),
        feature_spec=_MANIFEST["feature_spec"],
        decisions=pl.read_parquet(_ROOT / "decisions.parquet"),
    )


def _serialize(frame):
    return {
        "index": [int(asset) for asset in frame.index],
        "index_name": frame.index.name,
        "columns": list(frame.columns),
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "values": frame.to_numpy().tolist(),
    }


def test_prepared_pipeline_matches_published_eager_lazy_screens_values_dtypes_and_market_cache():
    pipeline = _prepared()
    pre, market = [], []

    def initialize(context):
        assert context.attach_pipeline(pipeline, "alpha") is pipeline
        context.attach_pipeline(pipeline, "lazy", eager=False)

    def before(context, data):
        pre.append(
            {
                "timestamp": context.get_datetime().isoformat(),
                "alpha": _serialize(context.pipeline_output("alpha")),
                "lazy": _serialize(context.pipeline_output("lazy")),
            }
        )

    def handle(context, data):
        market.append(
            {
                "timestamp": context.get_datetime().isoformat(),
                "alpha": _serialize(context.pipeline_output("alpha")),
            }
        )

    runtime = RuntimeServices()
    frame = run_algorithm(
        feed=_feed(),
        assets=[Asset("A", "A", 0), Asset("B", "B", 1)],
        initialize=initialize,
        before_trading_start=before,
        handle_data=handle,
        start="2025-07-02",
        end="2025-07-03",
        capital_base=1000,
        runtime=runtime,
    )
    assert pre == _REFERENCE["before"]
    assert market == _REFERENCE["market"]
    assert len(frame) == 2
    assert runtime.start is None and runtime.end is None


def test_screened_prepared_factors_drive_native_next_close_orders_and_account_records():
    selected = []

    def handle(context, _data):
        frame = context.pipeline_output("eager")
        selected.append([asset.key for asset in frame.index])
        if len(selected) == 1:
            for asset in frame.index:
                context.order(asset, 1)

    performance = _execute(_prepared(), handle=handle)
    result = performance.attrs["backtest_result"]
    fills = result.to_fills_dataframe()
    assert selected == [["A"], ["A", "B"]]
    assert fills["asset"].to_list() == ["A"]
    assert fills["quantity"].to_list() == [1.0]
    assert fills["timestamp"][0].isoformat() == "2025-07-03T17:00:00+00:00"
    # Published default equity impact is 5bps; per-share commission is .001.
    price = 40.0 * (1.0 + 5.0 / 10000.0)
    assert fills["price"][0] == pytest.approx(price)
    assert fills["commission"].to_list() == [0.001]
    assert performance.cash.tolist() == pytest.approx([1000.0, 999.999 - price])
    assert performance.portfolio_value.tolist() == pytest.approx([1000.0, 1039.999 - price])


def _execute(pipeline, before=None, handle=None, runtime=None, initialize=None):
    def setup(context):
        context.attach_pipeline(pipeline, "eager")
        context.attach_pipeline(pipeline, "lazy", eager=False)
        if initialize is not None:
            initialize(context)

    return run_algorithm(
        feed=_feed(),
        assets=[Asset("A", "A", 0), Asset("B", "B", 1)],
        initialize=setup,
        before_trading_start=before,
        handle_data=handle,
        start="2025-07-02",
        end="2025-07-03",
        runtime=runtime,
        capital_base=1000,
    )


@pytest.mark.parametrize("future_value", [25.0, 999.0])
def test_late_publication_future_values_and_returned_frame_mutation_cannot_change_pre_snapshot(
    future_value,
):
    panel = pl.read_parquet(_ROOT / "panel.parquet")
    first = panel["decision_time"].min()
    panel = panel.with_columns(
        pl.when((pl.col("asset") == "A") & (pl.col("decision_time") == first))
        .then(pl.col("decision_time") + timedelta(minutes=1))
        .otherwise(pl.col("available_at"))
        .alias("available_at"),
        pl.when(pl.col("decision_time") > first)
        .then(pl.lit(future_value))
        .otherwise(pl.col("mean"))
        .alias("mean"),
    )
    pipeline = PreparedPipeline(panel, feature_spec=_MANIFEST["feature_spec"])
    before, market = [], []

    def pre(context, data):
        for name in ["eager", "lazy"]:
            output = context.pipeline_output(name)
            before.append((name, _serialize(output)))
            if not output.empty:
                output.iloc[0, 0] = -1000.0

    def handle(context, data):
        market.append([_serialize(context.pipeline_output(name)) for name in ["eager", "lazy"]])

    _execute(pipeline, pre, handle)
    assert all(row["index"] == [] for _, row in before[:2])
    assert all(row["index"] == [] for row in market[0])
    assert all(row["values"] == [[future_value, 30.0], [future_value, 60.0]] for row in market[1])
    assert market[1][0] == before[2][1] == market[1][1] == before[3][1]


def test_declared_empty_decision_preserves_column_dtypes_and_cannot_reuse_yesterdays_values():
    panel = pl.read_parquet(_ROOT / "panel.parquet")
    decisions = pl.read_parquet(_ROOT / "decisions.parquet")
    panel = panel.filter(pl.col("decision_time") == panel["decision_time"].min())
    pipeline = PreparedPipeline(panel, feature_spec=_MANIFEST["feature_spec"], decisions=decisions)
    seen = []
    _execute(
        pipeline, lambda context, _data: seen.append(_serialize(context.pipeline_output("eager")))
    )
    assert seen[0] == _REFERENCE["before"][0]["alpha"]
    assert seen[1] == {
        "index": [],
        "index_name": None,
        "columns": ["mean", "latest"],
        "dtypes": ["float64", "float64"],
        "values": [],
    }
    empty = PreparedPipeline(
        panel.head(0), feature_spec=_MANIFEST["feature_spec"], decisions=decisions
    )
    seen.clear()
    _execute(empty, lambda context, _data: seen.append(context.pipeline_output("eager")))
    assert len(seen) == 2 and all(frame.empty for frame in seen)


def test_nan_factor_and_declared_column_order_are_preserved_with_sorted_legacy_asset_index():
    panel = (
        pl.read_parquet(_ROOT / "panel.parquet")
        .reverse()
        .with_columns(pl.lit(float("nan")).alias("mean"))
    )
    pipeline = PreparedPipeline(
        panel, feature_spec=_MANIFEST["feature_spec"], columns={"close": "latest", "signal": "mean"}
    )
    seen = []
    _execute(pipeline, lambda context, _data: seen.append(context.pipeline_output("eager")))
    assert [int(asset) for asset in seen[1].index] == [0, 1]
    assert list(seen[1]) == ["close", "signal"]
    assert seen[1]["close"].tolist() == [30.0, 60.0]
    assert seen[1]["signal"].isna().all()


@pytest.mark.parametrize("eager", [True, False])
def test_missing_decision_is_an_explicit_error_instead_of_a_previous_session_snapshot(eager):
    panel = pl.read_parquet(_ROOT / "panel.parquet")
    pipeline = PreparedPipeline(
        panel.filter(pl.col("decision_time") > panel["decision_time"].min()),
        feature_spec=_MANIFEST["feature_spec"],
    )

    def initialize(context):
        context.attach_pipeline(pipeline, "missing", eager=eager)

    with pytest.raises(ValueError, match="no declared decision"):
        run_algorithm(
            feed=_feed(),
            assets=[Asset("A", "A", 0), Asset("B", "B", 1)],
            initialize=initialize,
            start="2025-07-02",
            end="2025-07-03",
            before_trading_start=lambda context, _data: context.pipeline_output("missing"),
        )


def test_pipeline_api_is_setup_scoped_and_caller_runtime_survives_callback_failure():
    from ml4t.backtest.compat.zipline.api import attach_pipeline, pipeline_output

    pipeline, runtime = _prepared(), RuntimeServices()
    seen = []

    def initialize(context):
        with pytest.raises(RuntimeError, match="during initialize"):
            pipeline_output("eager")
        with pytest.raises(ValueError, match="unique"):
            attach_pipeline(pipeline, "eager")
        with pytest.raises(ValueError, match="chunks controls"):
            attach_pipeline(pipeline, "chunks", chunks=2)
        with pytest.raises(TypeError, match="Boolean"):
            attach_pipeline(pipeline, "eager_bad", eager=1)
        with pytest.raises(TypeError, match="Prepare feature"):
            attach_pipeline(object(), "unprepared")

    def before(context, data):
        with pytest.raises(RuntimeError, match="during initialize"):
            attach_pipeline(pipeline, "late")
        with pytest.raises(ValueError, match="No pipeline"):
            pipeline_output("unknown")
        seen.append(pipeline_output("eager").shape)
        raise RuntimeError("user callback fails")

    with pytest.raises(RuntimeError, match="user callback fails"):
        _execute(pipeline, before, runtime=runtime, initialize=initialize)
    assert seen == [(1, 2)]
    assert runtime.start is None and runtime.end is None
    assert runtime._auxiliary_sources == {}
    assert len(_execute(pipeline, runtime=runtime)) == 2
    with pytest.raises(RuntimeError, match="active migration callback"):
        pipeline_output("eager")


def test_unmapped_eligible_stable_identity_fails_without_dropping_the_asset():
    pipeline = _prepared()
    with pytest.raises(ValueError, match="unmapped stable asset keys.*B"):
        run_algorithm(
            feed=_feed(),
            assets=[Asset("A", "A", 0)],
            initialize=lambda context: context.attach_pipeline(pipeline, "alpha"),
            start="2025-07-02",
            end="2025-07-03",
        )


def test_undeclared_or_naive_decisions_fail_at_preparation_before_any_callback():
    panel = pl.read_parquet(_ROOT / "panel.parquet")
    with pytest.raises(ValueError, match="undeclared decision"):
        PreparedPipeline(
            panel,
            feature_spec=_MANIFEST["feature_spec"],
            decisions=pl.read_parquet(_ROOT / "decisions.parquet").head(1),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        PreparedPipeline(
            panel,
            feature_spec=_MANIFEST["feature_spec"],
            decisions=pd.DataFrame({"decision_time": [datetime(2025, 7, 2)]}),
        )


def test_actual_engineer_preparation_reproduces_captured_factors_screen_and_delivery():
    preparation = pytest.importorskip("ml4t.engineer.features.preparation")
    from ml4t.engineer.artifacts import FeatureSpec
    from ml4t.engineer.features.rolling import finite_mean
    from ml4t.specs import FeedSpec
    from polars.testing import assert_frame_equal

    panel = preparation.prepare_feature_panel(
        pl.read_parquet(_ROOT / "history.parquet"),
        features={"mean": finite_mean("close", period=2), "latest": pl.col("close")},
        lookbacks={"mean": 2, "latest": 1},
        input_columns=["close"],
        decisions=pl.read_parquet(_ROOT / "decisions.parquet"),
        universe=pl.read_parquet(_ROOT / "universe.parquet"),
        price_basis="raw",
        feed_spec=FeedSpec.from_any(_MANIFEST["input_feed_spec"]),
    )
    assert_frame_equal(panel, pl.read_parquet(_ROOT / "panel.parquet"))
    seen = []
    pipeline = PreparedPipeline(
        panel, feature_spec=FeatureSpec.from_mapping(_MANIFEST["feature_spec"])
    )
    _execute(
        pipeline, lambda context, _data: seen.append(_serialize(context.pipeline_output("eager")))
    )
    assert seen == [row["alpha"] for row in _REFERENCE["before"]]
