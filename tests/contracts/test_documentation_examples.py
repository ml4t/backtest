"""Installed-wheel documentation example enforcement."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
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
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["build"]["steps"])

    build = commands.index("uv build")
    install = commands.index("uv pip install")
    execute = commands.index("validation/check_documentation_examples.py")
    assert build < install < execute


def test_recorded_tutorial_output_mismatch_fails(tmp_path: Path) -> None:
    page = tmp_path / "example.md"
    page.write_text(
        "<!-- ml4t-doc-test: smoke-mismatch -->\n"
        "```python\nprint('actual')\n```\n\n"
        "<!-- ml4t-doc-output: smoke-mismatch -->\n"
        "```text\nexpected\n```\n",
        encoding="utf-8",
    )
    checker = _load_checker()
    examples = checker.collect_examples([page])
    with pytest.raises(AssertionError, match="output differs"):
        checker.run_examples(examples)


def test_missing_tutorial_input_fails_real_example(tmp_path: Path) -> None:
    page = tmp_path / "example.md"
    page.write_text(
        "<!-- ml4t-doc-test: smoke-missing-input -->\n"
        "```python\n"
        "from ml4t.backtest.example_data import load_example_prices\n"
        "load_example_prices('not-a-bundled-category')\n"
        "```\n\n"
        "<!-- ml4t-doc-output: smoke-missing-input -->\n"
        "```text\nunreachable\n```\n",
        encoding="utf-8",
    )
    checker = _load_checker()
    with pytest.raises(RuntimeError, match="Unknown example category"):
        checker.run_examples(checker.collect_examples([page]))
