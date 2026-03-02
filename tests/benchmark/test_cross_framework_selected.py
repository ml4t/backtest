"""Selected cross-framework benchmark tests.

These are opt-in and intended for comparison environments where optional
framework dependencies are installed.
"""

from __future__ import annotations

import importlib.util
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


def _run_framework_pair(framework: str, runner_name: str):
    if framework == "vectorbt":
        pytest.importorskip("vectorbt")
    elif framework == "backtrader":
        pytest.importorskip("backtrader")
    elif framework == "nautilus":
        pytest.importorskip("nautilus_trader")
    elif framework == "zipline":
        pytest.importorskip("zipline")
        pytest.importorskip("exchange_calendars")
        os.environ["ZIPLINE_ROOT"] = "/tmp/zipline-root"
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

    ml4t_result = suite.benchmark_ml4t(
        config, price_data, signals, dates, execution_mode="same_bar"
    )
    framework_result = getattr(suite, runner_name)(config, price_data, signals, dates)

    assert ml4t_result.error is None, ml4t_result.error
    assert framework_result.error is None, framework_result.error
    assert ml4t_result.runtime_sec > 0
    assert framework_result.runtime_sec > 0
    assert ml4t_result.num_trades >= 0
    assert framework_result.num_trades >= 0


@pytest.mark.skipif(
    not _RUN_COMPARISON,
    reason="Set ML4T_RUN_COMPARISON_BENCHMARKS=1 to enable cross-framework benchmarks.",
)
def test_selected_scenario_vs_vectorbt_oss():
    _run_framework_pair("vectorbt", "benchmark_vectorbt_oss")


@pytest.mark.skipif(
    not _RUN_COMPARISON,
    reason="Set ML4T_RUN_COMPARISON_BENCHMARKS=1 to enable cross-framework benchmarks.",
)
def test_selected_scenario_vs_backtrader():
    _run_framework_pair("backtrader", "benchmark_backtrader")


@pytest.mark.skipif(
    not _RUN_COMPARISON,
    reason="Set ML4T_RUN_COMPARISON_BENCHMARKS=1 to enable cross-framework benchmarks.",
)
def test_selected_scenario_vs_nautilus():
    _run_framework_pair("nautilus", "benchmark_nautilus")


@pytest.mark.skipif(
    not _RUN_COMPARISON,
    reason="Set ML4T_RUN_COMPARISON_BENCHMARKS=1 to enable cross-framework benchmarks.",
)
def test_selected_scenario_vs_zipline_reloaded():
    _run_framework_pair("zipline", "benchmark_zipline")
