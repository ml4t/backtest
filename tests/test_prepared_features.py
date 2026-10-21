"""Actual Engineer panels retain their decision, publication and universe boundaries."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.backtest import BacktestConfig, DataFeed, Engine, RuntimeServices, Strategy
from ml4t.backtest.config import ExecutionMode

ROOT = Path(__file__).parent / "fixtures" / "prepared_features"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def _panel():
    return pl.read_parquet(ROOT / "panel.parquet")


def _feed(panel=None, specification=None):
    frame = _panel() if panel is None else panel
    source = _panel()
    prices = source.select(
        pl.col("decision_time").alias("timestamp"), "asset", pl.lit(100.0).alias("close")
    )
    return DataFeed(
        prices_df=prices,
        signals_df=frame,
        feature_spec=MANIFEST["feature_spec"] if specification is None else specification,
        feed_spec={
            "time_contract_version": 2,
            "calendar": None,
            "timestamp_semantics": "event_time",
        },
    )


@pytest.mark.parametrize("container", ["polars", "pandas", "indexed_pandas", "lazy", "chunked"])
def test_actual_feature_panel_uses_decision_clock_and_never_trades_ineligible_assets(container):
    panel = _panel()
    if container in {"pandas", "indexed_pandas"}:
        panel = pd.DataFrame(panel.to_dict(as_series=False))
        if container == "indexed_pandas":
            panel = panel.set_index(["decision_time", "asset"])
    elif container == "lazy":
        panel = panel.lazy()
    elif container == "chunked":
        panel = pl.concat([panel.head(2), panel.tail(2)], rechunk=False)
    feed = _feed(panel)
    seen = []

    class Choose(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append([data[asset]["signals"]["average"] for asset in ["A", "B"]])
            for asset in ["A", "B"]:
                if data[asset]["signals"]["average"] is not None:
                    broker.submit_order(asset, 1)

    result = Engine(
        feed, Choose(), BacktestConfig(calendar=None, execution_mode=ExecutionMode.SAME_BAR)
    ).run()
    assert seen == [[2.0, None], [3.0, None]]
    fills = result.to_fills_dataframe()
    assert fills["asset"].to_list() == ["A", "A"]
    assert fills["price"].to_list() == [100.0, 100.0]
    assert fills["timestamp"].to_list() == _panel()["decision_time"].unique().sort().to_list()
    assert feed.signals["feature_observation_time"].to_list() == _panel()["timestamp"].to_list()


def test_undeclared_columns_are_excluded_from_strategy_inputs_and_identified_in_projection():
    panel = _panel().with_columns(pl.lit(999.0).alias("future_label"))
    feed = _feed(panel)
    assert "future_label" not in feed.signals
    assert feed.feature_metadata["projection"]["excluded_columns"] == ["future_label"]


def test_nullable_unsigned_identity_and_nanosecond_decisions_survive_pandas_projection():
    panel = pd.DataFrame(_panel().head(2).to_dict(as_series=False))
    panel["asset"] = pd.Series([2**63 + 1, 2**63 + 2], dtype="UInt64")
    clocks = pd.to_datetime(["2025-07-09T17:00:00.000000001Z", "2025-07-09T17:00:00.000000002Z"])
    panel["decision_time"] = clocks
    raw_prices = pl.DataFrame(
        {
            "timestamp": pl.Series(clocks.asi8, dtype=pl.Int64).cast(pl.Datetime("ns", "UTC")),
            "asset": pl.Series([2**63 + 1, 2**63 + 2], dtype=pl.UInt64),
            "close": [100.0, 100.0],
        }
    )
    feed = DataFeed(
        prices_df=raw_prices,
        signals_df=panel,
        feature_spec=MANIFEST["feature_spec"],
        feed_spec={"timestamp_semantics": "event_time", "calendar": None},
    )
    assert feed.signals.schema["asset"] == pl.UInt64
    assert feed.signals["asset"].to_list() == [2**63 + 1, 2**63 + 2]
    assert feed.signals["timestamp"].cast(pl.Int64).to_list() == clocks.asi8.tolist()


def test_shared_observation_decision_and_publication_column_is_projected_without_overwrite():
    panel = _panel().with_columns(pl.col("decision_time").alias("timestamp"))
    specification = deepcopy(MANIFEST["feature_spec"])
    specification["schema"].update(decision_time_col="timestamp", available_at_col="timestamp")
    specification["definition"]["feed_spec"]["timestamp_semantics"] = "event_time"
    feed = _feed(panel, specification)
    assert feed.signals["timestamp"].to_list() == panel["timestamp"].to_list()
    assert feed.signals["feature_observation_time"].to_list() == panel["timestamp"].to_list()


def test_explicit_runtime_queries_inherit_manifest_masks_without_mutating_reusable_configuration():
    runtime = RuntimeServices()
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(runtime.signals.current(["A", "B"], ["average"])["average"].to_list())

    Engine(_feed(), Read(), BacktestConfig(calendar=None), runtime=runtime).run()
    assert seen == [[2.0, None], [3.0, None]]
    assert runtime.signal_known_at_cols == {} and runtime.signal_eligible_cols == {}


def test_late_publication_hides_the_actual_feature_payload_without_changing_its_saved_value():
    panel = _panel().with_columns(
        pl.when(pl.col("asset") == "A")
        .then(pl.col("decision_time") + pl.duration(seconds=1))
        .otherwise(pl.col("available_at"))
        .alias("available_at")
    )
    seen = []

    class Read(Strategy):
        def on_data(self, timestamp, data, context, broker):
            seen.append(data["A"]["signals"]["average"])

    Engine(_feed(panel), Read(), BacktestConfig(calendar=None)).run()
    assert seen == [None, None]
    assert panel["average"].to_list() == [2.0, 20.0, 3.0, 30.0]


def test_feature_metadata_is_a_defensive_copy_of_the_original_producer_contract():
    feed = _feed()
    metadata = feed.feature_metadata
    assert metadata["spec"] == MANIFEST["feature_spec"]
    assert metadata["projection"]["observation_col"] == "feature_observation_time"
    metadata["spec"]["definition"]["price_basis"] = "invented"
    assert feed.feature_metadata["spec"]["definition"]["price_basis"] == "raw"


@pytest.mark.parametrize(
    "fault,error",
    [
        ("duplicate", "Duplicate"),
        ("missing", "missing"),
        ("basis", "price_basis"),
        ("naive", "aware"),
        ("eligibility", "Boolean"),
    ],
)
def test_invalid_producer_panel_is_rejected_before_strategy_initialization(fault, error):
    panel = _panel()
    if fault == "duplicate":
        panel = pl.concat([panel, panel.head(1)])
    elif fault == "missing":
        panel = panel.drop("available_at")
    elif fault == "basis":
        panel = panel.with_columns(pl.lit("other_basis").alias("price_basis"))
    elif fault == "naive":
        panel = panel.with_columns(pl.col("decision_time").dt.replace_time_zone(None))
    else:
        panel = panel.with_columns(pl.col("eligible").cast(pl.Int64))
    with pytest.raises(ValueError, match=error):
        _feed(panel)


def test_future_outcome_artifact_cannot_be_used_as_a_prepared_feature_contract():
    specification = deepcopy(MANIFEST["feature_spec"])
    specification["kind"] = "labels"
    with pytest.raises(ValueError, match="FeatureSpec"):
        _feed(specification=specification)


def test_runtime_cannot_replace_declared_feature_eligibility_with_another_column():
    class Noop(Strategy):
        def on_data(self, timestamp, data, context, broker):
            pass

    runtime = RuntimeServices(signal_eligible_cols={"average": "member"})
    engine = Engine(_feed(), Noop(), BacktestConfig(calendar=None), runtime=runtime)
    with pytest.raises(ValueError, match="conflicts with feature_spec"):
        engine.run()
    assert engine.broker.orders == [] and engine.broker.fills == []


def test_captured_panel_can_be_rebuilt_by_the_actual_installed_engineer_producer():
    preparation = pytest.importorskip("ml4t.engineer.features.preparation")
    rolling = pytest.importorskip("ml4t.engineer.features.rolling")
    artifacts = pytest.importorskip("ml4t.engineer.artifacts")
    from ml4t.specs import FeedSpec

    for name, digest in MANIFEST["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    spec = artifacts.FeatureSpec.from_mapping(MANIFEST["feature_spec"])
    for lazy in [False, True]:
        history = pl.read_parquet(ROOT / "history.parquet")
        actual = preparation.prepare_feature_panel(
            history.lazy() if lazy else history,
            features={"average": rolling.finite_mean("close", period=3)},
            lookbacks={"average": 3},
            input_columns=["close"],
            decisions=pl.read_parquet(ROOT / "decisions.parquet"),
            universe=pl.read_parquet(ROOT / "universe.parquet"),
            price_basis="raw",
            feed_spec=FeedSpec.from_any(MANIFEST["input_feed_spec"]),
        )
        actual = actual.collect() if isinstance(actual, pl.LazyFrame) else actual
        assert_frame_equal(actual, _panel())
        assert _feed(actual, spec).feature_metadata["spec"] == spec.to_dict()
