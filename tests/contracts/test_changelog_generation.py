"""Generated release-history policy."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import yaml

_ROOT = Path(__file__).parents[2]


def _load_generator() -> ModuleType:
    path = _ROOT / "validation" / "generate_changelog.py"
    spec = importlib.util.spec_from_file_location("ml4t_changelog", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_changelog_matches_structured_release_history() -> None:
    generator = _load_generator()
    assert (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8") == generator.render()


def test_beta_18_through_stable_candidate_have_compatibility_impact() -> None:
    source = json.loads((_ROOT / "release" / "changelog.json").read_text(encoding="utf-8"))
    releases = {release["version"]: release for release in source["releases"]}
    expected = {"0.1.0", *(f"0.1.0b{number}" for number in range(18, 23))}

    assert set(releases) == expected
    for version in expected:
        assert releases[version]["sections"]["Compatibility"]


def test_changelog_drift_is_a_non_mutating_ci_gate() -> None:
    workflow = yaml.load(
        (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["lint"]["steps"])
    assert "validation/generate_changelog.py --check" in commands
    assert "validation/generate_changelog.py --write" not in commands
