"""Selected cross-framework benchmark tests.

These are opt-in and intended for comparison environments where optional
framework dependencies are installed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.requires_comparison]

_RUN_COMPARISON = os.getenv("ML4T_RUN_COMPARISON_BENCHMARKS") == "1"


def _load_benchmark_suite():
    suite_path = Path(__file__).resolve().parents[2] / "validation" / "benchmark_suite.py"
    module_name = "ml4t_validation_benchmark_suite"
    spec = importlib.util.spec_from_file_location(module_name, suite_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_framework_pair(framework: str, runner_name: str, artifact_dir: Path):
    if framework == "vectorbt":
        pytest.importorskip("vectorbt")
    elif framework == "backtrader":
        pytest.importorskip("backtrader")
    elif framework == "nautilus":
        pytest.importorskip("nautilus_trader")
    elif framework == "zipline":
        pytest.importorskip("zipline")
        pytest.importorskip("exchange_calendars")
        os.environ["ZIPLINE_ROOT"] = str(artifact_dir / "zipline-root")
        Path(os.environ["ZIPLINE_ROOT"]).mkdir(parents=True, exist_ok=True)

    suite = _load_benchmark_suite()
    config = suite.BenchmarkConfig(
        name=f"Selected-{framework}",
        n_bars=80,
        n_assets=15,
        frequency="D",
        top_n=3,
        bottom_n=3,
        rebalance_freq=1,
    )
    price_data, signals, dates = suite.generate_benchmark_data(config, seed=123)

    execution_mode = "next_bar" if framework in {"backtrader", "zipline"} else "same_bar"
    ml4t_result = suite.benchmark_ml4t(
        config, price_data, signals, dates, execution_mode=execution_mode
    )
    framework_result = getattr(suite, runner_name)(config, price_data, signals, dates)

    assert ml4t_result.error is None, ml4t_result.error
    assert framework_result.error is None, framework_result.error
    artifact = suite.compare_benchmark_results_exact(
        framework_result,
        ml4t_result,
        initial_cash=config.initial_cash,
    )
    artifact_path = artifact_dir / f"{framework}-exact-comparison.json"
    suite.write_exact_comparison_artifact(artifact, artifact_path)

    assert artifact["passed"], json.dumps(artifact, indent=2, sort_keys=True)


@pytest.mark.skipif(
    not _RUN_COMPARISON,
    reason="Set ML4T_RUN_COMPARISON_BENCHMARKS=1 to enable cross-framework benchmarks.",
)
def test_selected_scenario_vs_vectorbt_oss(tmp_path: Path):
    _run_framework_pair("vectorbt", "benchmark_vectorbt_oss", tmp_path)


@pytest.mark.skipif(
    not _RUN_COMPARISON,
    reason="Set ML4T_RUN_COMPARISON_BENCHMARKS=1 to enable cross-framework benchmarks.",
)
def test_selected_scenario_vs_backtrader(tmp_path: Path):
    _run_framework_pair("backtrader", "benchmark_backtrader", tmp_path)


@pytest.mark.skipif(
    not _RUN_COMPARISON,
    reason="Set ML4T_RUN_COMPARISON_BENCHMARKS=1 to enable cross-framework benchmarks.",
)
def test_selected_scenario_vs_nautilus(tmp_path: Path):
    _run_framework_pair("nautilus", "benchmark_nautilus", tmp_path)


@pytest.mark.skipif(
    not _RUN_COMPARISON,
    reason="Set ML4T_RUN_COMPARISON_BENCHMARKS=1 to enable cross-framework benchmarks.",
)
def test_selected_scenario_vs_zipline_reloaded(tmp_path: Path):
    _run_framework_pair("zipline", "benchmark_zipline", tmp_path)
