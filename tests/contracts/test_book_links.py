"""Book companion path checks use the verified Git-tree manifest."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[2]


def _load_checker():
    path = _ROOT / "validation" / "check_book_links.py"
    spec = importlib.util.spec_from_file_location("ml4t_book_links", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_documented_companion_paths_exist_at_pinned_revision() -> None:
    checker = _load_checker()
    checker.check_links(checker.documentation_paths())


def test_nonexistent_companion_path_fails(tmp_path: Path) -> None:
    checker = _load_checker()
    manifest = checker._MANIFEST
    commit = json.loads(manifest.read_text())["commit"]
    page = tmp_path / "missing.md"
    page.write_text(
        "[Missing](https://github.com/stefan-jansen/machine-learning-for-trading/"
        f"blob/{commit}/16_strategy_simulation/not_a_notebook.ipynb)\n"
    )
    with pytest.raises(ValueError, match="does not exist"):
        checker.check_links([page], manifest)
