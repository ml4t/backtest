"""Prepared benchmark definitions must survive Engine, artifacts and migration presentation."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ml4t.backtest import (
    BacktestConfig,
    BacktestResult,
    DataFeed,
    Engine,
    RuntimeServices,
    Strategy,
)
from ml4t.backtest.compat.zipline import run_algorithm, to_performance_frame
from ml4t.backtest.compat.zipline.assets import Asset

_REFERENCE = json.loads(
    (
        Path(__file__).parent / "fixtures/zipline_original/benchmark-dividend-reference.json"
    ).read_text()
)
_ASSET = Asset("A", "A", 7)
_SPEC = {
    "time_contract_version": 2,
    "timestamp_semantics": "bar_close",
    "bar_label": "right",
    "calendar": "NYSE",
    "data_frequency": "1d",
}


def _input():
    return pl.DataFrame(
        {
            "timestamp": [datetime.fromisoformat(t) for t in _REFERENCE["closes"][1:]],
            "return": [95.0 / 90.0 - 1.0, 104.5 / 95.0 - 1.0, 99.0 / 104.5 - 1.0],
        }
    )


def _benchmark(frame=None, **kwargs):
    from ml4t.backtest import BenchmarkData

    return BenchmarkData(
        _input() if frame is None else frame,
        source_id="fixture://dividend-benchmark",
        return_method="total_return",
        reinvestment="zipline_previous_close_cash_factor",
        asset=kwargs.pop("asset", "A"),
        feed_spec=_SPEC,
        **kwargs,
    )


def _feed():
    return DataFeed(
        prices_df=pl.DataFrame(
            {
                "timestamp": [datetime.fromisoformat(t) for t in _REFERENCE["closes"]],
                "asset": ["A"] * 4,
                "close": _REFERENCE["raw_close"],
                "volume": [1000] * 4,
            }
        ),
        feed_spec=_SPEC,
    )


class _Cash(Strategy):
    def on_data(self, *_):
        pass


def _native(benchmark):
    return Engine(
        _feed(),
        _Cash(),
        BacktestConfig(initial_cash=1000),
        benchmark=benchmark,
        runtime=RuntimeServices(start=datetime.fromisoformat(_REFERENCE["closes"][1])),
    ).run()


def test_real_dividend_benchmark_matches_published_periods_and_does_not_change_cash_or_nav(
    tmp_path,
):
    benchmark = _benchmark()

    def initialize(context):
        context.set_benchmark(context.sid(7))

    performance = run_algorithm(
        feed=_feed(),
        assets=[_ASSET],
        benchmark=benchmark,
        initialize=initialize,
        start="2025-07-01",
        end="2025-07-03",
        capital_base=1000,
    )
    assert performance.benchmark_return.tolist() == pytest.approx(
        _REFERENCE["expected_period_returns"]
    )
    assert performance.benchmark_period_return.tolist() == pytest.approx(
        _REFERENCE["benchmark_period_return"]
    )
    assert performance.portfolio_value.tolist() == _REFERENCE["portfolio_value"]
    assert performance.cash.tolist() == [1000.0] * 3
    assert performance.returns.tolist() == [0.0] * 3
    result = performance.attrs["backtest_result"]
    assert result.fills == []
    assert_frame_equal(
        result.to_benchmark_dataframe(),
        _input()
        .rename({"return": "benchmark_return"})
        .with_columns(pl.col("timestamp").cast(pl.Datetime("ns", "UTC"))),
    )
    result.to_parquet(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == 4
    assert manifest["benchmark_metadata"]["reinvestment"] == "zipline_previous_close_cash_factor"
    restored = BacktestResult.from_parquet(tmp_path)
    pd.testing.assert_series_equal(
        to_performance_frame(restored).benchmark_return, performance.benchmark_return
    )
    assert restored.benchmark_metadata == result.benchmark_metadata
    assert_frame_equal(restored.to_benchmark_dataframe(), result.to_benchmark_dataframe())


def test_native_engine_retains_exact_prepared_returns_without_a_callback_adapter():
    benchmark = _benchmark()
    result = _native(benchmark)
    assert result.benchmark_metadata["source_id"] == "fixture://dividend-benchmark"
    assert result.benchmark_metadata["return_scale"] == "decimal"
    assert (
        result.to_benchmark_dataframe()["benchmark_return"].to_list()
        == _input()["return"].to_list()
    )
    assert result.to_equity_dataframe()["equity"].to_list() == [1000.0] * 3


def test_action_enabled_artifact_keeps_the_real_benchmark_instead_of_rejecting_the_combined_schema(
    tmp_path,
):
    from ml4t.backtest import ContractSpec

    feed = _feed()
    times = [datetime.fromisoformat(t) for t in _REFERENCE["closes"]]
    feed.corporate_actions = pl.DataFrame(
        [
            {
                "source": "declared-benchmark-fixture",
                "event_id": "distribution",
                "revision_id": "r1",
                "asset": "A",
                "action_kind": "cash_dividend",
                "currency": "USD",
                "available_at": times[0],
                "effective_at": times[2],
                "payment_at": times[3],
                "share_basis": "pre_event",
                "cash_per_share": 1.0,
            }
        ]
    )
    feed.input_manifest = {"price_basis": "raw"}
    result = Engine(
        feed,
        _Cash(),
        BacktestConfig(initial_cash=1000),
        benchmark=_benchmark(),
        runtime=RuntimeServices(start=times[1]),
        contract_specs={"A": ContractSpec("A")},
    ).run()
    result.to_parquet(tmp_path)
    restored = BacktestResult.from_parquet(tmp_path)
    assert_frame_equal(restored.to_benchmark_dataframe(), result.to_benchmark_dataframe())
    assert_frame_equal(restored.to_entitlements_dataframe(), result.to_entitlements_dataframe())


def test_public_convenience_and_config_constructors_deliver_the_same_benchmark():
    from ml4t.backtest import run_backtest

    feed = _feed()
    prices = feed.prices.filter(pl.col("timestamp") >= _input()["timestamp"][0])
    for result in [
        run_backtest(
            prices, _Cash(), config=BacktestConfig(), feed_spec=_SPEC, benchmark=_benchmark()
        ),
        Engine.from_config(
            DataFeed(prices_df=prices, feed_spec=_SPEC),
            _Cash(),
            BacktestConfig(),
            benchmark=_benchmark(),
        ).run(),
    ]:
        assert (
            result.to_benchmark_dataframe()["benchmark_return"].to_list()
            == _input()["return"].to_list()
        )


def test_benchmark_rejects_implicit_session_labels_before_callbacks():
    called = []

    class Record(_Cash):
        def on_start(self, broker):
            called.append("start")

    labels = _feed().prices.with_columns(
        pl.col("timestamp").dt.date().cast(pl.Datetime("ns")).alias("timestamp")
    )
    with pytest.raises(ValueError, match="physical event_time/bar_close"):
        Engine(
            DataFeed(
                prices_df=labels,
                feed_spec={**_SPEC, "timestamp_semantics": "session_label", "bar_label": None},
            ),
            Record(),
            BacktestConfig(),
            benchmark=_benchmark(),
        ).run()
    assert called == []


def test_benchmark_gap_is_rejected_before_strategy_callbacks_instead_of_filling_zero_or_stale_returns():
    calls = []

    class Record(Strategy):
        def on_start(self, broker):
            calls.append("started")

        def on_data(self, *_):
            calls.append("market")

    with pytest.raises(ValueError, match="no return for observed period"):
        Engine(
            _feed(),
            Record(),
            BacktestConfig(),
            benchmark=_benchmark(_input().head(2)),
            runtime=RuntimeServices(start=datetime.fromisoformat(_REFERENCE["closes"][1])),
        ).run()
    assert calls == []


def test_asset_benchmark_requires_matching_prepared_stable_key_and_explicit_none_disables_it():
    for benchmark in [None, _benchmark(asset="OTHER")]:
        with pytest.raises(ValueError, match="prepared benchmark returns"):
            run_algorithm(
                feed=_feed(),
                assets=[_ASSET],
                benchmark=benchmark,
                initialize=lambda context: context.set_benchmark(_ASSET),
                start="2025-07-01",
                end="2025-07-03",
            )
    frame = run_algorithm(
        feed=_feed(),
        assets=[_ASSET],
        benchmark=_benchmark(),
        initialize=lambda context: context.set_benchmark(None),
    )
    assert "benchmark_return" not in frame
    assert frame.attrs["backtest_result"].benchmark is None
    assert (
        frame.attrs["backtest_result"].config.metadata["benchmark_status"] == "explicitly_disabled"
    )


def test_supplied_returns_are_used_on_omission_and_benchmark_setting_remains_initialize_only():
    seen = []

    def handle(context, data):
        with pytest.raises(RuntimeError, match="during initialize"):
            context.set_benchmark(None)
        seen.append(context.get_datetime())

    frame = run_algorithm(
        feed=_feed(),
        assets=[_ASSET],
        benchmark=_benchmark(),
        handle_data=handle,
        start="2025-07-01",
        end="2025-07-03",
    )
    assert len(seen) == 3
    assert frame.benchmark_period_return.tolist() == pytest.approx(
        _REFERENCE["benchmark_period_return"]
    )


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_period_returns_are_rejected_instead_of_disabling_optional_analyses(
    bad,
):
    raw = _input().with_columns(
        pl.when(pl.int_range(pl.len()) == 1)
        .then(pl.lit(bad))
        .otherwise(pl.col("return"))
        .alias("return")
    )
    with pytest.raises(ValueError, match="finite decimal"):
        _benchmark(raw)


def test_duplicate_or_naive_benchmark_clocks_cannot_be_silently_aggregated():
    with pytest.raises(ValueError, match="known and unique"):
        _benchmark(pl.concat([_input(), _input().head(1)]))
    with pytest.raises(ValueError, match="timezone-aware physical"):
        _benchmark(_input().with_columns(pl.col("timestamp").dt.replace_time_zone(None)))


def test_nanosecond_benchmark_periods_remain_distinct_and_foreign_timezone_instants_align():
    from ml4t.backtest import BenchmarkData

    points = pd.to_datetime(["2025-07-01T14:00:00.000000001Z", "2025-07-01T14:00:00.000000002Z"])
    frame = pd.DataFrame(
        {"timestamp": points.tz_convert("America/New_York"), "return": [0.1, -0.1]}
    )
    benchmark = BenchmarkData(
        frame,
        source_id="fixture://ns",
        asset=None,
        return_method="total_return",
        reinvestment="declared_fixture",
        feed_spec={**_SPEC, "calendar": None},
    )
    prices = pl.DataFrame(
        {
            "timestamp": pl.Series(points.asi8).cast(pl.Datetime("ns", "UTC")),
            "asset": ["A"] * 2,
            "close": [100.0, 100.0],
        }
    )
    result = Engine(
        DataFeed(prices_df=prices, feed_spec={**_SPEC, "calendar": None}),
        _Cash(),
        BacktestConfig(calendar=None),
        benchmark=benchmark,
    ).run()
    assert (
        result.to_benchmark_dataframe()["timestamp"].cast(pl.Int64).to_list()
        == points.asi8.tolist()
    )
    assert to_performance_frame(result).benchmark_period_return.tolist() == pytest.approx(
        [0.1, -0.01]
    )


def test_benchmark_schema_four_content_and_provenance_corruption_fail_strict_reads(tmp_path):
    result = _native(_benchmark())
    result.to_parquet(tmp_path)
    path = tmp_path / "benchmark.parquet"
    frame = pl.read_parquet(path)
    frame.with_columns((pl.col("benchmark_return") + 0.5).alias("benchmark_return")).write_parquet(
        path
    )
    with pytest.raises(Exception, match="retained digest"):
        BacktestResult.from_parquet(tmp_path)
    recovered = BacktestResult.from_parquet(tmp_path, recovery=True)
    assert recovered.benchmark is None
    assert any(
        item.component == "benchmark" and item.code == "component_invalid"
        for item in recovered.artifact_diagnostics
    )
    result.to_parquet(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["benchmark_metadata"].pop("reinvestment")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(Exception, match="provenance"):
        BacktestResult.from_parquet(tmp_path)


def test_benchmarkless_artifacts_keep_schema_three_and_do_not_fabricate_a_benchmark(tmp_path):
    result = Engine(_feed(), _Cash(), BacktestConfig()).run()
    result.to_parquet(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["schema_version"] == 3
    restored = BacktestResult.from_parquet(tmp_path)
    assert restored.benchmark is None and restored.benchmark_metadata is None
    assert "benchmark_period_return" not in to_performance_frame(restored)
    with pytest.raises(ValueError, match="no prepared benchmark"):
        restored.to_benchmark_dataframe()


def test_schema_four_cannot_silently_lose_benchmark_component_and_reports_recovery_omission(
    tmp_path,
):
    from ml4t.backtest import ArtifactIncompleteError

    _native(_benchmark()).to_parquet(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["components"].pop("benchmark")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactIncompleteError, match="benchmark"):
        BacktestResult.from_parquet(tmp_path)
    recovered = BacktestResult.from_parquet(tmp_path, recovery=True)
    assert recovered.benchmark is None
    assert any(
        item.component == "benchmark" and item.code == "component_missing"
        for item in recovered.artifact_diagnostics
    )


def test_captured_data_benchmark_returns_keep_the_declared_dividend_method_in_native_accounts():
    root = Path(__file__).parent / "fixtures/prepared_benchmark"
    manifest = json.loads((root / "manifest.json").read_text())
    benchmark = _benchmark(
        pl.read_parquet(root / "returns.parquet"), provenance=manifest["preparation_metadata"]
    )
    result = _native(benchmark)
    produced = pl.read_parquet(root / "returns.parquet")["return"].to_list()
    assert result.to_benchmark_dataframe()["benchmark_return"].to_list() == produced
    assert produced == pytest.approx(_REFERENCE["expected_period_returns"], rel=0, abs=1e-15)
    assert result.benchmark_metadata["provenance"]["factor_source"] == "supplied"
    assert result.benchmark_metadata["provenance"]["perspective"] == "retrospective"
    assert to_performance_frame(result).benchmark_period_return.iloc[-1] == pytest.approx(0.1)


def test_actual_data_preparation_reproduces_the_captured_reference_factor_benchmark():
    adjustments = pytest.importorskip("ml4t.data.adjustments")
    from ml4t.specs import FeedSpec

    root = Path(__file__).parent / "fixtures/prepared_benchmark"
    manifest = json.loads((root / "manifest.json").read_text())
    prepared = adjustments.prepare_adjusted_history(
        pl.read_parquet(root / "raw.parquet"),
        None,
        as_of=None,
        feed_spec=FeedSpec.from_any(manifest["feed_spec"]),
        basis="total_return",
        action_coverage="complete",
        factors=pl.read_parquet(root / "factors.parquet"),
        factor_method=manifest["method"],
    )
    assert_frame_equal(prepared.prices, pl.read_parquet(root / "adjusted.parquet"))
    consumer = FeedSpec.from_any(prepared.metadata["feed_spec"])
    returns = prepared.prices.select(
        "timestamp", pl.col(consumer.price_col).pct_change().alias("return")
    ).slice(1)
    assert_frame_equal(returns, pl.read_parquet(root / "returns.parquet"))
    assert (
        _native(_benchmark(returns)).to_benchmark_dataframe()["benchmark_return"].to_list()
        == returns["return"].to_list()
    )


def test_later_benchmark_rows_and_caller_changes_cannot_rewrite_earlier_observed_returns():
    raw = _input()
    benchmark = _benchmark(raw)
    raw.replace_column(1, pl.Series("return", [999.0] * 3))
    returned = benchmark.frame
    returned.replace_column(1, pl.Series("benchmark_return", [-999.0] * 3))
    benchmark.metadata["provenance"]["fake"] = True
    result = _native(benchmark)
    assert (
        result.to_benchmark_dataframe()["benchmark_return"].to_list()
        == _input()["return"].to_list()
    )
    future = (
        _input()
        .tail(1)
        .with_columns(
            (pl.col("timestamp") + pl.duration(days=5)).alias("timestamp"),
            pl.lit(999.0).alias("return"),
        )
    )
    extended = _native(_benchmark(pl.concat([_input(), future])))
    assert_frame_equal(result.to_benchmark_dataframe(), extended.to_benchmark_dataframe())
    frame = to_performance_frame(result)
    frame.attrs["benchmark_metadata"]["reinvestment"] = "mutated"
    assert result.benchmark_metadata["reinvestment"] == "zipline_previous_close_cash_factor"
