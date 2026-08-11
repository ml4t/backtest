"""Contracts for strict, non-mutating static analysis."""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import yaml

_ROOT = Path(__file__).parents[2]


def _ci_jobs() -> dict:
    payload = yaml.load(
        (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(payload, dict)
    return payload["jobs"]


def _step_commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_ruff_is_non_mutating_in_configuration_and_ci() -> None:
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert not config["tool"]["ruff"].get("fix", False)

    commands = _step_commands(_ci_jobs()["lint"])
    assert "uv sync --dev --locked" in commands
    assert "ruff check --no-fix" in commands
    assert "sha256sum" in commands
    assert "git diff --exit-code" in commands


def test_correctness_relevant_ty_rules_are_not_globally_suppressed() -> None:
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    rules = config["tool"].get("ty", {}).get("rules", {})
    prohibited_ignores = {
        "invalid-argument-type",
        "unresolved-attribute",
        "invalid-return-type",
        "unresolved-import",
    }
    assert not {rule for rule in prohibited_ignores if rules.get(rule) == "ignore"}


def test_ci_checks_source_and_an_installed_wheel_consumer() -> None:
    commands = _step_commands(_ci_jobs()["typecheck"])
    assert "xargs uv run ty check src/ < validation/release_checks.txt" in commands
    assert "uv build --wheel" in commands
    assert '--with "$GITHUB_WORKSPACE"/dist/*.whl' in commands
    assert "ty check public_api_consumer.py" in commands
    assert (_ROOT / "tests" / "typing" / "public_api_consumer.py").is_file()


def test_ci_checks_every_release_validation_script() -> None:
    jobs = _ci_jobs()
    lint_commands = _step_commands(jobs["lint"])
    typecheck_commands = _step_commands(jobs["typecheck"])

    assert "ruff check --no-fix src/ tests/ < validation/release_checks.txt" in lint_commands
    assert "ruff format --check src/ tests/ < validation/release_checks.txt" in lint_commands
    assert "ty check src/ < validation/release_checks.txt" in typecheck_commands
    assert "--force-exclude" not in lint_commands

    manifest = {
        line
        for line in (_ROOT / "validation" / "release_checks.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    }
    workflow_scripts = {
        match
        for workflow in (_ROOT / ".github" / "workflows").glob("*.yml")
        for match in re.findall(
            r"validation/[a-z][a-z0-9_]*\.py",
            workflow.read_text(encoding="utf-8"),
        )
    }
    excluded_validation_scripts = {
        # Legacy benchmark entry points run only in their named framework environments.
        "validation/backtrader/benchmark_performance.py",
        "validation/backtrader/scale_validation.py",
        "validation/vectorbt_pro/benchmark_performance.py",
        "validation/zipline/benchmark_performance.py",
        # LEAN algorithms import names supplied by the QuantConnect runtime.
        "validation/lean/scenario_01_long_only/main.py",
        "validation/lean/workspace/chapter16_etfs/main.py",
        "validation/lean/workspace/chapter16_sp500_equity_option_analytics/main.py",
        "validation/lean/workspace/chapter16_us_equities_panel/main.py",
        # Legacy manual orchestrators require optional framework environments.
        "validation/benchmark_suite.py",
        "validation/run_all_benchmarks.py",
        "validation/run_full_validation.py",
        "validation/run_scenario.py",
        # Framework adapters execute only inside their isolated correctness environments.
        "validation/frameworks/backtrader.py",
        "validation/frameworks/vectorbt_oss.py",
        "validation/frameworks/vectorbt_pro.py",
        "validation/frameworks/zipline.py",
    }
    candidate_files = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            ":(glob)validation/*.py",
            ":(glob)validation/**/*.py",
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    validation_scripts = {path for path in candidate_files if not path.endswith("/__init__.py")}

    assert workflow_scripts <= manifest
    assert not manifest & excluded_validation_scripts
    assert validation_scripts == manifest | excluded_validation_scripts
