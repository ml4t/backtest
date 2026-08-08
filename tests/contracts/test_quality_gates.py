"""Contracts for release-quality test signals."""

from __future__ import annotations

import importlib.util
import math
import tomllib
from pathlib import Path
from types import ModuleType

import yaml

from tests.helpers.invariants import _accounting_tolerance

_ROOT = Path(__file__).parents[2]


def _load_coverage_checker() -> ModuleType:
    path = _ROOT / "validation" / "check_coverage.py"
    spec = importlib.util.spec_from_file_location("ml4t_coverage_checker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ci_jobs() -> dict:
    payload = yaml.load(
        (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(payload, dict)
    return payload["jobs"]


def _step_commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_coverage_configuration_requires_branches_and_a_total_floor() -> None:
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    coverage = config["tool"]["coverage"]
    assert coverage["run"]["branch"] is True
    assert coverage["report"]["fail_under"] == 84

    checker = _load_coverage_checker()
    assert checker.GLOBAL_STATEMENT_MIN == 87.0
    assert checker.GLOBAL_BRANCH_MIN == 76.0
    assert len(checker.CRITICAL_THRESHOLDS) >= 10


def test_coverage_gate_rejects_missing_branch_evidence() -> None:
    checker = _load_coverage_checker()
    assert checker.coverage_failures({"meta": {"branch_coverage": False}}) == [
        "Coverage evidence was collected without branch coverage"
    ]


def test_coverage_gate_rejects_global_regressions_and_missing_critical_modules() -> None:
    checker = _load_coverage_checker()
    failures = checker.coverage_failures(
        {
            "meta": {"branch_coverage": True},
            "totals": {
                "percent_statements_covered": 86.99,
                "percent_branches_covered": 75.99,
            },
            "files": {},
        }
    )
    assert "Global statement coverage 86.99% < 87.00%" in failures
    assert "Global branch coverage 75.99% < 76.00%" in failures
    assert any("Critical module is absent" in failure for failure in failures)


def test_warning_policy_fails_new_warnings_and_has_no_global_suppression() -> None:
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    filters = config["tool"]["pytest"]["ini_options"]["filterwarnings"]
    assert filters[0] == "error"
    assert "ignore::DeprecationWarning" not in filters
    assert "ignore::PendingDeprecationWarning" not in filters
    assert len(filters) <= 5


def test_accounting_tolerance_is_an_exact_ulp_bound() -> None:
    scale = 1_000_000_000_000.0
    assert _accounting_tolerance(scale) == math.ulp(scale) * 16
    assert _accounting_tolerance(scale, operations=10) == math.ulp(scale) * 40
    assert _accounting_tolerance(1.0) == 1e-9


def test_ci_separates_and_retains_coverage_and_performance_evidence() -> None:
    jobs = _ci_jobs()
    coverage_commands = _step_commands(jobs["coverage"])
    runtime_commands = _step_commands(jobs["runtime"])
    assert "--cov-branch" in coverage_commands
    assert "validation/check_coverage.py coverage.json" in coverage_commands
    assert "--no-cov" in runtime_commands
    assert jobs["runtime"]["steps"][-1]["if"] == "always()"
    assert {"coverage", "runtime"} <= set(jobs["build"]["needs"])
