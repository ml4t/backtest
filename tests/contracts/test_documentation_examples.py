"""Installed-wheel documentation example enforcement."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml

_ROOT = Path(__file__).parents[2]


def _load_checker() -> ModuleType:
    path = _ROOT / "validation" / "check_documentation_examples.py"
    spec = importlib.util.spec_from_file_location("ml4t_documentation_examples", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_required_examples_are_selected_from_primary_adoption_docs() -> None:
    checker = _load_checker()
    examples = checker.collect_examples(checker._DEFAULT_PATHS, require_all=True)

    assert {example.name for example in examples} == checker._REQUIRED_EXAMPLES
    assert {example.language for example in examples} == {"python"}


def test_explicit_documentation_path_does_not_require_the_default_example_set() -> None:
    checker = _load_checker()

    examples = checker.collect_examples([_ROOT / "docs" / "getting-started" / "quickstart.md"])

    assert {example.name for example in examples} == {"quickstart-minimal"}


def test_documentation_ci_installs_wheel_before_running_examples() -> None:
    workflow = yaml.load(
        (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["documentation"]["steps"])

    build = commands.index("uv build --wheel")
    install = commands.index("uv pip install")
    execute = commands.index("validation/check_documentation_examples.py")
    assert build < install < execute
