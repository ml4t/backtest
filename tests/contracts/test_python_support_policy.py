"""Contracts for the advertised CPython and operating-system support matrix."""

from __future__ import annotations

import tomllib
from itertools import product
from pathlib import Path

import yaml

_ROOT = Path(__file__).parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_OPERATING_SYSTEMS = {"ubuntu-latest", "macos-latest", "windows-latest"}


def _workflow(name: str) -> dict:
    payload = yaml.load((_WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def _step_commands(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def test_stable_matrix_runs_every_required_check_on_all_supported_platforms() -> None:
    stable = _workflow("compatibility.yml")["jobs"]["stable"]
    matrix = stable["strategy"]["matrix"]

    combinations = set(product(matrix["os"], matrix["python-version"]))
    assert combinations == set(product(_OPERATING_SYSTEMS, {"3.12", "3.13", "3.14"}))

    commands = _step_commands(stable)
    for required_command in (
        "uv sync --dev --locked",
        "import ml4t.backtest",
        'pytest tests/ -v --tb=short --no-cov -m "not benchmark"',
        "ty check --python-version ${{ matrix.python-version }}",
        "uv build",
    ):
        assert required_command in commands


def test_python_315_prerelease_matrix_is_blocking_on_all_platforms() -> None:
    workflow = _workflow("compatibility.yml")
    prerelease = workflow["jobs"]["prerelease"]
    gate = workflow["jobs"]["gate"]

    assert set(prerelease["strategy"]["matrix"]["os"]) == _OPERATING_SYSTEMS
    setup_step = next(step for step in prerelease["steps"] if "astral-sh/setup-uv" in step["uses"])
    assert setup_step["with"]["python-version"] == "3.15"
    commands = _step_commands(prerelease)
    assert "sys.version_info[:2] == (3, 15)" in commands
    assert "{'beta', 'candidate'}" in commands
    assert "uv sync --no-dev --group test --locked" in commands
    assert 'pytest tests/ -v --tb=short --no-cov -m "not benchmark"' in commands
    assert {"stable", "prerelease"} <= set(gate["needs"])


def test_core_dependencies_are_installable_on_python_315() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = project["dependencies"]

    assert not any(dependency.startswith("pyarrow") for dependency in dependencies)
    assert "pandas>=2.3.3; python_version < '3.15'" in dependencies
    assert "pandas>=3.0.5; python_version >= '3.15'" in dependencies
    assert all(
        "python_version < '3.15'" in dependency
        for dependency in project["optional-dependencies"]["comparison"]
    )


def test_minimum_dependency_matrix_proves_declared_lower_bounds() -> None:
    workflow = _workflow("compatibility.yml")
    minimum = workflow["jobs"]["minimum"]
    gate = workflow["jobs"]["gate"]
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert set(minimum["strategy"]["matrix"]["python-version"]) == {"3.12", "3.13", "3.14"}
    commands = _step_commands(minimum)
    for required_command in (
        "uv venv --python ${{ matrix.python-version }}",
        "--resolution lowest-direct --only-binary :all:",
        "--no-binary ml4t-backtest",
        "uv pip freeze > minimum-versions-${{ matrix.python-version }}.txt",
        "import ml4t.backtest",
        'pytest tests/ -v --tb=short --no-cov -m "not benchmark"',
        "ty check --python-version ${{ matrix.python-version }}",
        "uv build",
    ):
        assert required_command in commands
    assert any("actions/upload-artifact" in step.get("uses", "") for step in minimum["steps"])
    assert set(gate["needs"]) == {"stable", "prerelease", "minimum"}
    assert set(project["dependencies"]) >= {
        "ml4t-specs>=0.1.0",
        "polars>=1.36.1",
        "pandas>=2.3.3; python_version < '3.15'",
        "numpy>=2.3.2",
        "PyYAML>=6.0.3",
        "pandas-market-calendars>=5.2.4",
    }


def test_merge_and_release_builds_depend_on_compatibility_gate() -> None:
    ci_jobs = _workflow("ci.yml")["jobs"]
    release_jobs = _workflow("release.yml")["jobs"]

    assert ci_jobs["compatibility"]["uses"] == "./.github/workflows/compatibility.yml"
    assert "compatibility" in ci_jobs["build"]["needs"]
    assert release_jobs["compatibility"]["uses"] == "./.github/workflows/compatibility.yml"
    assert release_jobs["build"]["needs"] == "compatibility"
