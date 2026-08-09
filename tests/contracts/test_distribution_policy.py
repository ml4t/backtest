"""Contracts for clean, typed, reproducible distributions."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import yaml

_ROOT = Path(__file__).parents[2]


def _load_artifact_checker() -> ModuleType:
    path = _ROOT / "validation" / "check_artifacts.py"
    spec = importlib.util.spec_from_file_location("ml4t_artifact_checker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_namespace_package_contains_the_pep561_marker() -> None:
    assert (_ROOT / "src" / "ml4t" / "backtest" / "py.typed").is_file()


def test_build_configuration_excludes_internal_agent_material() -> None:
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excludes = set(config["tool"]["hatch"]["build"]["exclude"])
    assert {"**/AGENTS.md", "**/CLAUDE.md"} <= excludes


def test_distribution_declares_stable_development_status() -> None:
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    classifiers = set(config["project"]["classifiers"])
    assert "Development Status :: 5 - Production/Stable" in classifiers
    assert not any(classifier.startswith("Development Status :: 4") for classifier in classifiers)


def test_manifest_comparison_rejects_unexpected_and_missing_files() -> None:
    checker = _load_artifact_checker()
    failures = checker._manifest_diff(
        {"approved.py", "secret.env"},
        {"approved.py", "py.typed"},
        "wheel",
    )
    assert "wheel has unexpected files: ['secret.env']" in failures
    assert "wheel is missing files: ['py.typed']" in failures


def test_ci_checks_both_distribution_formats_and_reproducibility() -> None:
    payload = yaml.load(
        (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    commands = "\n".join(step.get("run", "") for step in payload["jobs"]["build"]["steps"])
    assert "uv build --out-dir dist-rebuild" in commands
    assert "validation/check_artifacts.py dist --compare dist-rebuild" in commands
    assert "uvx twine check dist/*" in commands
