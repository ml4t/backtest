"""Cross-engine validation contracts.

Runs validation scenarios against external backtesting frameworks to verify
trade-level parity. Uses the consolidated validation runner in-process.

Requires the 'comparison' optional dependency group:
    uv sync --dev --extra comparison

Set ML4T_COMPARISON_INPROC=1 to enable these tests in CI.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = PROJECT_ROOT / "validation"

# Add validation directory to path for imports
sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


FRAMEWORK_IMPORTS = {
    "vectorbt_oss": "vectorbt",
    "backtrader": "backtrader",
    "zipline": "zipline",
}

# Core scenarios to test in CI (01=long only, 05=commission, 09=trailing stop)
CI_SCENARIOS = ["01", "05", "09"]


def _framework_available(framework: str) -> bool:
    module_name = FRAMEWORK_IMPORTS[framework]
    return importlib.util.find_spec(module_name) is not None


def _run_scenario_inproc(scenario_id: str, framework: str) -> bool:
    """Run a single scenario/framework combination in-process.

    Returns True if validation passed.
    """
    from common import data_generators
    from common.comparator import compare_results
    from common.ml4t_runner import run_ml4t
    from scenarios.definitions import SCENARIOS

    scenario = SCENARIOS[scenario_id]

    if framework not in scenario.supported_frameworks:
        pytest.skip(f"Scenario {scenario_id} does not support {framework}")

    # Generate data
    gen_func = getattr(data_generators, scenario.data_generator)
    data_result = gen_func(**scenario.data_kwargs)

    if len(data_result) == 3:
        prices_df, entries, exits = data_result
    else:
        prices_df, entries = data_result
        exits = None

    # Run external framework
    fw_module_name = f"frameworks.{framework}"
    fw_module = importlib.import_module(fw_module_name)
    fw_result = fw_module.run(scenario, prices_df, entries, exits)

    # Run ml4t
    ml4t_result = run_ml4t(scenario, prices_df, entries, exits, framework=framework)

    # Compare
    result = compare_results(scenario, fw_result, ml4t_result)
    return result.passed


@pytest.mark.requires_comparison
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("framework", ["vectorbt_oss", "backtrader", "zipline"])
@pytest.mark.parametrize("scenario_id", CI_SCENARIOS)
def test_cross_engine_contract(framework: str, scenario_id: str) -> None:
    """Validate ml4t matches external framework for core scenarios."""
    if not _framework_available(framework):
        pytest.skip(f"{framework} dependencies not installed in this environment")

    if os.getenv("ML4T_COMPARISON_INPROC") != "1":
        pytest.skip("Set ML4T_COMPARISON_INPROC=1 to run cross-engine contracts")

    passed = _run_scenario_inproc(scenario_id, framework)
    assert passed, f"Scenario {scenario_id} failed against {framework}"
