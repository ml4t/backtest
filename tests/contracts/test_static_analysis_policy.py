"""Contracts for strict, non-mutating static analysis."""

from __future__ import annotations

import re
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
    assert manifest == workflow_scripts
