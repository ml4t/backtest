"""Behavior contracts for the frozen real-strategy ML4T runner."""

from __future__ import annotations

import copy
import importlib.util
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest


def _load_runner():
    path = Path(__file__).parents[2] / "validation" / "real_strategy_runner.py"
    spec = importlib.util.spec_from_file_location("ml4t_real_strategy_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _load_input():
    path = Path(__file__).parents[2] / "validation" / "real_strategy_input.py"
    spec = importlib.util.spec_from_file_location("ml4t_real_strategy_input", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_evidence():
    path = Path(__file__).parents[2] / "validation" / "real_strategy_evidence.py"
    spec = importlib.util.spec_from_file_location("ml4t_real_strategy_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _load_benchmark():
    path = Path(__file__).parents[2] / "validation" / "real_strategy_benchmark.py"
    spec = importlib.util.spec_from_file_location("ml4t_real_strategy_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.skipif(sys.platform != "linux", reason="Publication timings identify a Linux host")
def test_benchmark_report_accepts_linux_with_blank_processor_query(monkeypatch) -> None:
    benchmark = _load_benchmark()
    validation = Path(__file__).parents[2] / "validation"
    correctness = json.loads((validation / "REAL_STRATEGY_RESULTS.json").read_text())
    measurements = json.loads((validation / "REAL_STRATEGY_PERFORMANCE.json").read_text())
    retained = {
        (record["case_study"], record["framework"]): record for record in measurements["records"]
    }

    def measured_side(*, case_study, framework, side, **_settings):
        key = "framework_engine" if side == "framework" else "ml4t_engine"
        return copy.deepcopy(retained[(case_study, framework)][key])

    # Reuse real measured samples to isolate report production from native engine runtimes.
    monkeypatch.setattr(benchmark, "_measure_side", measured_side)
    monkeypatch.setattr(benchmark.platform, "processor", lambda: "")
    report = benchmark.build_report(
        correctness=correctness,
        bundle_root=validation,
        warmups=1,
        samples=10,
    )

    assert benchmark.report_failures(report, correctness) == []


class _Broker:
    def __init__(self) -> None:
        self.cash = 1_000.0
        self.positions = {"PERP": SimpleNamespace(quantity=2.0, multiplier=3.0)}

    @staticmethod
    def get_mark_price(_symbol: str, *, quantity: float) -> float:
        assert quantity == 2.0
        return 10.0


def test_lean_funding_schedule_skips_entry_boundary_then_settles() -> None:
    runner = _load_runner()
    funding = pl.DataFrame(
        {
            "symbol": ["PERP", "PERP"],
            "timestamp": [
                datetime(2024, 1, 1, 8, tzinfo=UTC),
                datetime(2024, 1, 1, 16, tzinfo=UTC),
            ],
            "funding_rate": [0.01, 0.02],
        }
    )
    ledger = runner.FundingSettlementLedger(funding, skip_first_after_entry=True)
    broker = _Broker()

    ledger.settle(datetime(2024, 1, 1, 8, tzinfo=UTC), broker)
    assert broker.cash == 1_000.0

    ledger.settle(datetime(2024, 1, 1, 16, tzinfo=UTC), broker)
    assert broker.cash == 998.8
    assert ledger.metrics() == {
        "funding_pnl": -1.2,
        "funding_events": 1,
        "funding_settlements": 2,
    }


def test_lean_funding_schedule_observes_flat_positions_between_settlements() -> None:
    runner = _load_runner()
    funding = pl.DataFrame(
        {
            "symbol": ["PERP", "PERP"],
            "timestamp": [
                datetime(2024, 1, 1, 8, tzinfo=UTC),
                datetime(2024, 1, 1, 16, tzinfo=UTC),
            ],
            "funding_rate": [0.01, 0.02],
        }
    )
    ledger = runner.FundingSettlementLedger(funding, skip_first_after_entry=True)
    broker = _Broker()

    ledger.settle(datetime(2024, 1, 1, 8, tzinfo=UTC), broker)
    broker.positions["PERP"].quantity = 0.0
    ledger.settle(datetime(2024, 1, 1, 12, tzinfo=UTC), broker)
    broker.positions["PERP"].quantity = 2.0
    ledger.settle(datetime(2024, 1, 1, 16, tzinfo=UTC), broker)

    assert broker.cash == 1_000.0
    assert ledger.metrics()["funding_events"] == 0


def test_lean_comparison_preserves_margin_account_and_disables_equity_costs() -> None:
    runner = _load_runner()
    source = {
        "account": {"allow_short_selling": True, "allow_leverage": False},
        "cash": {"initial": 125_000.0},
        "calendar": {"calendar": "NYSE", "data_frequency": "daily"},
        "feed": {"calendar": "NYSE", "data_frequency": "daily"},
        "commission": {"model": "percentage", "rate": 0.001},
        "slippage": {"model": "percentage", "rate": 0.001},
        "orders": {"rebalance_headroom_pct": 0.99},
        "metadata": {},
    }

    config = runner._comparison_config({"backtest_config": source}, "lean")

    assert config.allow_leverage is True
    assert config.commission_rate == 0.0
    assert config.slippage_rate == 0.0
    assert config.rebalance_headroom_pct == 1.0
    assert config.enforce_sessions is True


def test_real_strategy_comparator_detects_one_quantum_mutation() -> None:
    evidence = _load_evidence()
    framework = [{"timestamp": "2024-01-01", "price": "10.00000000"}]
    mutated = [{"timestamp": "2024-01-01", "price": "10.00000001"}]

    result = evidence.compare_records(framework, mutated, numeric_fields=("price",))

    assert result["passed"] is False
    assert result["first_divergence"] == {
        "kind": "record",
        "index": 0,
        "field": "price",
        "framework": framework[0],
        "ml4t": mutated[0],
        "canonical_gap": "0.00000001",
    }


def test_real_strategy_comparator_rounds_account_money_to_cents() -> None:
    evidence = _load_evidence()
    framework = [{"timestamp": "2024-01-01", "equity": "1000.00000000"}]
    residual = [{"timestamp": "2024-01-01", "equity": "1000.00000015"}]

    result = evidence.compare_records(framework, residual, numeric_fields=("equity",))

    assert result["passed"] is True


def test_real_strategy_comparator_rounds_fractional_share_residuals() -> None:
    evidence = _load_evidence()
    framework = [{"timestamp": "2024-01-01", "quantity": "10.00000000"}]
    residual = [{"timestamp": "2024-01-01", "quantity": "10.00000036"}]

    result = evidence.compare_records(framework, residual, numeric_fields=("quantity",))

    assert result["passed"] is True


def test_real_strategy_comparator_rejects_one_quantity_quantum() -> None:
    evidence = _load_evidence()
    framework = [{"timestamp": "2024-01-01", "quantity": "10.00000"}]
    changed = [{"timestamp": "2024-01-01", "quantity": "10.00001"}]

    result = evidence.compare_records(framework, changed, numeric_fields=("quantity",))

    assert result["passed"] is False
    assert result["first_divergence"]["canonical_gap"] == "0.00001"


def test_real_strategy_comparator_rejects_one_cent_account_money_difference() -> None:
    evidence = _load_evidence()
    framework = [{"final_value": "1000.00"}]
    changed = [{"final_value": "1000.01"}]

    result = evidence.compare_records(framework, changed, numeric_fields=("final_value",))

    assert result["passed"] is False
    assert result["first_divergence"]["canonical_gap"] == "0.01"


def test_real_strategy_comparator_rejects_money_gap_that_rounds_to_one_cent() -> None:
    evidence = _load_evidence()
    framework = [{"final_value": "1000.000"}]
    changed = [{"final_value": "1000.006"}]

    result = evidence.compare_records(framework, changed, numeric_fields=("final_value",))

    assert result["passed"] is False
    assert result["first_divergence"]["canonical_gap"] == "0.01"


def test_real_strategy_comparator_rejects_missing_valuation_timestamp() -> None:
    evidence = _load_evidence()
    framework = [
        {"timestamp": "2024-01-01", "equity": "1000.0"},
        {"timestamp": "2024-01-02", "equity": "1000.0"},
    ]
    ml4t = [{"timestamp": "2024-01-01", "equity": "1000.0"}]

    surface, _, _ = evidence._value_surface(framework, ml4t, field="equity")

    assert surface["passed"] is False
    assert surface["coverage_passed"] is False
    assert surface["first_divergence"] == {
        "kind": "timestamp_coverage",
        "framework_only": "2024-01-02",
        "ml4t_only": None,
    }


def test_comparison_market_applies_production_daily_calendar() -> None:
    comparison_input = _load_input()
    market = pl.DataFrame(
        {
            "timestamp": [datetime(2018, 12, 4), datetime(2018, 12, 5)],
            "symbol": ["ES", "ES"],
            "close": [100.0, 101.0],
        }
    )
    spec = {
        "backtest_config": {
            "calendar": {
                "calendar": "CME",
                "data_frequency": "daily",
                "enforce_sessions": True,
                "timezone": "UTC",
            }
        }
    }

    filtered = comparison_input.filter_comparison_market(market, spec)

    assert filtered["timestamp"].to_list() == [datetime(2018, 12, 4)]


def test_comparison_market_normalizes_missing_ohlcv_consistently() -> None:
    comparison_input = _load_input()
    market = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "symbol": ["A", "A"],
            "open": [None, 11.0],
            "high": [None, 12.0],
            "low": [None, 10.0],
            "close": [10.0, None],
            "volume": [None, 100.0],
        }
    )
    spec = {
        "backtest_config": {
            "calendar": {
                "calendar": "NYSE",
                "data_frequency": "daily",
                "enforce_sessions": False,
                "timezone": "UTC",
            }
        }
    }

    filtered = comparison_input.filter_comparison_market(market, spec)

    assert filtered.to_dicts() == [
        {
            "timestamp": datetime(2024, 1, 2),
            "symbol": "A",
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 0.0,
        }
    ]


def test_zipline_comparison_enforces_exchange_sessions() -> None:
    runner = _load_runner()
    source = {
        "account": {"allow_short_selling": True, "allow_leverage": False},
        "cash": {"initial": 100_000.0},
        "calendar": {
            "calendar": "NYSE",
            "data_frequency": "daily",
            "enforce_sessions": False,
            "timezone": "UTC",
        },
        "feed": {"calendar": "NYSE", "data_frequency": "daily"},
    }

    config = runner._comparison_config({"backtest_config": source}, "zipline_strict")

    assert config.calendar == "NYSE"
    assert config.enforce_sessions is True
    assert source["calendar"]["enforce_sessions"] is False


def test_fx_comparison_uses_native_usd_quoted_pairs() -> None:
    comparison_input = _load_input()
    market = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1)] * 3,
            "symbol": ["EUR_USD", "USD_JPY", "EUR_GBP"],
            "close": [1.1, 145.0, 0.85],
        }
    )
    targets = market.select("timestamp", "symbol").with_columns(pl.lit(0.1).alias("weight"))
    spec = {
        "backtest_config": {
            "calendar": {
                "calendar": "FX",
                "data_frequency": "daily",
                "enforce_sessions": False,
                "timezone": "UTC",
            },
            "metadata": {"case_study": "fx_pairs"},
        }
    }

    filtered_market = comparison_input.filter_comparison_market(market, spec)
    filtered_targets = comparison_input.filter_comparison_targets(targets, spec)

    assert filtered_market["symbol"].to_list() == ["EUR_USD"]
    assert filtered_targets["symbol"].to_list() == ["EUR_USD"]


def test_real_strategy_benchmark_interval_is_deterministic() -> None:
    benchmark = _load_benchmark()
    values = [float(value) for value in range(1, 11)]

    first = benchmark.bootstrap_median_interval(values, draws=1_000, seed=7)
    second = benchmark.bootstrap_median_interval(values, draws=1_000, seed=7)

    assert first == second
    assert first[0] <= 5.5 <= first[1]


def test_real_strategy_performance_evidence_fails_stale_source() -> None:
    """An engine source that is neither measured-under nor certified is refused."""
    benchmark = _load_benchmark()
    validation = Path(__file__).parents[2] / "validation"
    correctness = json.loads((validation / "REAL_STRATEGY_RESULTS.json").read_text())
    report = json.loads((validation / "REAL_STRATEGY_PERFORMANCE.json").read_text())
    changed = copy.deepcopy(report)
    changed.setdefault("provenance", {})["ml4t_engine_source_sha256"] = "0" * 64
    changed["provenance"]["certified_equivalent_sources"] = []

    assert benchmark.report_failures(report, correctness) == []
    failures = benchmark.report_failures(changed, correctness)

    assert any("neither the source these" in failure for failure in failures)


def test_performance_impact_review_allows_reuse_for_a_later_engine_source() -> None:
    """A reviewed change can retain measurements without changing their measured source."""
    benchmark = _load_benchmark()
    validation = Path(__file__).parents[2] / "validation"
    correctness = json.loads((validation / "REAL_STRATEGY_RESULTS.json").read_text())
    report = json.loads((validation / "REAL_STRATEGY_PERFORMANCE.json").read_text())
    report["provenance"]["ml4t_engine_source_sha256"] = "0" * 64
    report["provenance"]["certified_equivalent_sources"] = []
    benchmark.certify_source(
        report,
        correctness,
        validation / "REAL_STRATEGY_RESULTS.json",
        "Test a later source with unchanged measured behavior",
        performance_impact="none",
    )
    provenance = report["provenance"]
    certified = provenance["certified_equivalent_sources"]

    current = benchmark._tree_digest(benchmark.PROJECT_ROOT / "src/ml4t/backtest")
    assert current != provenance["ml4t_engine_source_sha256"]
    assert current in {entry["engine_source_sha256"] for entry in certified}
    assert benchmark.report_failures(report, correctness) == []


def test_malformed_certification_is_refused() -> None:
    """A certification missing its evidence cannot publish anything."""
    benchmark = _load_benchmark()
    validation = Path(__file__).parents[2] / "validation"
    correctness = json.loads((validation / "REAL_STRATEGY_RESULTS.json").read_text())
    report = json.loads((validation / "REAL_STRATEGY_PERFORMANCE.json").read_text())
    report["provenance"]["ml4t_engine_source_sha256"] = "0" * 64
    report["provenance"]["certified_equivalent_sources"] = []
    benchmark.certify_source(
        report,
        correctness,
        validation / "REAL_STRATEGY_RESULTS.json",
        "Test a later source with unchanged measured behavior",
        performance_impact="none",
    )

    for field in ("reason", "correctness_evidence_sha256", "correctness_pairs_passed"):
        changed = copy.deepcopy(report)
        del changed["provenance"]["certified_equivalent_sources"][0][field]
        failures = benchmark.report_failures(changed, correctness)
        assert any("Certification 0 lacks" in failure for failure in failures), field


def test_correctness_alone_cannot_certify_performance() -> None:
    benchmark = _load_benchmark()
    validation = Path(__file__).parents[2] / "validation"
    correctness = json.loads((validation / "REAL_STRATEGY_RESULTS.json").read_text())
    report = json.loads((validation / "REAL_STRATEGY_PERFORMANCE.json").read_text())
    before = copy.deepcopy(report)

    with pytest.raises(ValueError, match="performance-impact review"):
        benchmark.certify_source(
            report, correctness, validation / "REAL_STRATEGY_RESULTS.json", "Outputs still match"
        )

    assert report == before


@pytest.mark.parametrize("impact", ["material", "unknown"])
def test_material_or_unknown_performance_impact_cannot_reuse_timings(impact) -> None:
    benchmark = _load_benchmark()
    validation = Path(__file__).parents[2] / "validation"
    correctness = json.loads((validation / "REAL_STRATEGY_RESULTS.json").read_text())
    report = json.loads((validation / "REAL_STRATEGY_PERFORMANCE.json").read_text())
    before = copy.deepcopy(report)

    with pytest.raises(ValueError, match="requires fresh measurements"):
        benchmark.certify_source(
            report,
            correctness,
            validation / "REAL_STRATEGY_RESULTS.json",
            "Changed timed execution",
            performance_impact=impact,
        )

    assert report == before


def test_review_cli_preserves_measurements_and_their_original_correctness_run(tmp_path) -> None:
    benchmark = _load_benchmark()
    validation = Path(__file__).parents[2] / "validation"
    original = json.loads((validation / "REAL_STRATEGY_PERFORMANCE.json").read_text())
    report_path = tmp_path / "performance.json"
    report_path.write_text(json.dumps(original))

    result = subprocess.run(
        [
            sys.executable,
            str(validation / "real_strategy_benchmark.py"),
            "--certify-source",
            "--performance",
            str(report_path),
            "--performance-impact",
            "none",
            "--reason",
            "Exercise the reviewed-reuse workflow without invoking any measured process.",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    reviewed = json.loads(report_path.read_text())
    for field in (
        "generated_at",
        "records",
        "environment",
        "timing_policy",
        "correctness_evidence_generated_at",
    ):
        assert reviewed[field] == original[field], field
    assert reviewed["provenance"]["sources"] == original["provenance"]["sources"]
    correctness = json.loads((validation / "REAL_STRATEGY_RESULTS.json").read_text())
    assert benchmark.report_failures(reviewed, correctness) == []


def test_performance_review_cannot_authorize_different_measurements_or_runner(
    tmp_path, monkeypatch
):
    benchmark = _load_benchmark()
    validation = Path(__file__).parents[2] / "validation"
    correctness = json.loads((validation / "REAL_STRATEGY_RESULTS.json").read_text())
    report = json.loads((validation / "REAL_STRATEGY_PERFORMANCE.json").read_text())
    benchmark.certify_source(
        report,
        correctness,
        validation / "REAL_STRATEGY_RESULTS.json",
        "Review only this report and the current runner.",
        performance_impact="none",
    )
    assert benchmark.report_failures(report, correctness) == []

    different = copy.deepcopy(report)
    result = different["records"][0]["ml4t_engine"]
    result["samples_seconds"] = [sample * 2 for sample in result["samples_seconds"]]
    result["median_seconds"] = statistics.median(result["samples_seconds"])
    result["minimum_seconds"] = min(result["samples_seconds"])
    result["maximum_seconds"] = max(result["samples_seconds"])
    result["ci_95_seconds"] = list(benchmark.bootstrap_median_interval(result["samples_seconds"]))
    different["records"][0]["framework_to_ml4t_median_ratio"] /= 2
    assert any(
        "performance-impact review" in failure
        for failure in benchmark.report_failures(different, correctness)
    )

    runner = tmp_path / "real_strategy_runner.py"
    runner.write_text(
        benchmark.TIMING_SOURCE_PATHS["ml4t"]
        .read_text()
        .replace("    result = engine.run()", "    time.sleep(1)\n    result = engine.run()")
    )
    monkeypatch.setitem(benchmark.TIMING_SOURCE_PATHS, "ml4t", runner)
    assert any(
        "performance-impact review" in failure
        for failure in benchmark.report_failures(report, correctness)
    )


def test_failed_performance_review_does_not_partially_mutate_report() -> None:
    benchmark = _load_benchmark()
    validation = Path(__file__).parents[2] / "validation"
    correctness = json.loads((validation / "REAL_STRATEGY_RESULTS.json").read_text())
    report = json.loads((validation / "REAL_STRATEGY_PERFORMANCE.json").read_text())
    report["records"][0]["input_bundle_sha256"] = "0" * 64
    before = copy.deepcopy(report)

    with pytest.raises(ValueError, match="timing input differs"):
        benchmark.certify_source(
            report,
            correctness,
            validation / "REAL_STRATEGY_RESULTS.json",
            "The timing sample must belong to the compared frozen input.",
            performance_impact="none",
        )

    assert report == before


@pytest.mark.parametrize(
    "measured_directly",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                sys.platform != "linux", reason="Publication timings identify a Linux host"
            ),
        ),
    ],
)
def test_unreviewed_dependency_change_invalidates_timings(
    tmp_path, monkeypatch, measured_directly
) -> None:
    benchmark = _load_benchmark()
    validation = Path(__file__).parents[2] / "validation"
    correctness = json.loads((validation / "REAL_STRATEGY_RESULTS.json").read_text())
    report = json.loads((validation / "REAL_STRATEGY_PERFORMANCE.json").read_text())
    if measured_directly:
        retained = {
            (record["case_study"], record["framework"]): record for record in report["records"]
        }

        def measured_side(*, case_study, framework, side, **_settings):
            key = "framework_engine" if side == "framework" else "ml4t_engine"
            return copy.deepcopy(retained[(case_study, framework)][key])

        monkeypatch.setattr(benchmark, "_measure_side", measured_side)
        report = benchmark.build_report(
            correctness=correctness, bundle_root=validation, warmups=1, samples=10
        )
    else:
        benchmark.certify_source(
            report,
            correctness,
            validation / "REAL_STRATEGY_RESULTS.json",
            "Retain only the reviewed locked dependencies.",
            performance_impact="none",
        )
    assert benchmark.report_failures(report, correctness) == []

    lock = tmp_path / "uv.lock"
    dependency = "validation/environments/vectorbt_pro/uv.lock"
    original = benchmark.DEPENDENCY_PATHS[dependency].read_bytes()
    lock.write_bytes((validation / "environments/vectorbt_pro_2025_12_31/uv.lock").read_bytes())
    assert lock.read_bytes() != original
    monkeypatch.setitem(benchmark.DEPENDENCY_PATHS, dependency, lock)

    assert any(
        "performance-impact review" in failure
        for failure in benchmark.report_failures(report, correctness)
    )


def test_qualification_resolver_permission_preserves_unchanged_measured_dependencies(
    tmp_path, monkeypatch
) -> None:
    """uv's actual prerelease resolve must not invalidate an unchanged package graph."""
    import os
    import shutil
    import tomllib

    benchmark = _load_benchmark()
    validation = Path(__file__).parents[2] / "validation"
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "qualification-lock-fixture"\nversion = "0.0.0"\n'
        'requires-python = ">=3.12"\ndependencies = []\n'
    )
    command = ["uv", "lock", "--directory", str(tmp_path), "--python", sys.executable, "--offline"]
    environment = {**os.environ, "UV_CACHE_DIR": str(tmp_path / "empty-uv-cache")}
    baseline = subprocess.run(
        [*command, "--prerelease", "disallow"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert baseline.returncode == 0, baseline.stderr
    dependency = "uv.lock"
    monkeypatch.setitem(benchmark.DEPENDENCY_PATHS, dependency, tmp_path / "uv.lock")
    correctness = json.loads((validation / "REAL_STRATEGY_RESULTS.json").read_text())
    report = json.loads((validation / "REAL_STRATEGY_PERFORMANCE.json").read_text())
    benchmark.certify_source(
        report,
        correctness,
        validation / "REAL_STRATEGY_RESULTS.json",
        "Review unchanged selected packages through shared qualification's resolver permission.",
        performance_impact="none",
    )
    assert benchmark.report_failures(report, correctness) == []
    before = tomllib.loads((tmp_path / "uv.lock").read_text())
    before_bytes = (tmp_path / "uv.lock").read_bytes()
    completed = subprocess.run(
        [*command, "--prerelease", "allow"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    after = tomllib.loads((tmp_path / "uv.lock").read_text())
    assert (tmp_path / "uv.lock").read_bytes() != before_bytes
    assert before.pop("options", {}) in ({}, {"prerelease-mode": "disallow"})
    assert after.pop("options") == {"prerelease-mode": "allow"}
    assert after == before  # No selected version, artifact, hash, marker or source changed.
    assert benchmark.report_failures(report, correctness) == []

    # Another real uv lock selects materially different dependencies and must fail.
    shutil.copy2(validation / "environments/vectorbt_pro_2025_12_31/uv.lock", tmp_path / "uv.lock")
    assert any(
        "performance-impact review" in failure
        for failure in benchmark.report_failures(report, correctness)
    )
