"""Contracts for frozen comparison targets and the comparison-claim inventory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parents[2]
_VALIDATION_DIR = _PROJECT_ROOT / "validation"
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

import run_all_benchmarks  # noqa: E402
from common.framework_registry import (  # noqa: E402
    ManifestError,
    load_claim_inventory,
    load_framework_manifest,
)


def test_current_framework_targets_are_exact_and_immutable() -> None:
    manifest = load_framework_manifest()

    assert manifest.freeze_date == "2026-08-13"
    assert set(manifest.targets) == {
        "backtrader",
        "lean",
        "vectorbt_oss",
        "vectorbt_pro",
        "zipline",
    }
    assert {key: target.version for key, target in manifest.targets.items()} == {
        "vectorbt_pro": "2026.6.27",
        "vectorbt_oss": "1.1.0",
        "backtrader": "1.9.78.123",
        "zipline": "3.1.1",
        "lean": "18001",
    }
    assert manifest.scenario_framework_ids == (
        "vectorbt_pro",
        "vectorbt_oss",
        "backtrader",
        "zipline",
    )
    for target in manifest.targets.values():
        assert not any(operator in target.version for operator in ("<", ">", "=", "*", "~"))
        assert target.immutable_id.startswith(("git:", "sha256:"))
        assert target.profile
        assert target.python
        assert target.license
        assert target.access in {"public", "licensed"}


def test_scenario_targets_define_reproducible_interpreters() -> None:
    manifest = load_framework_manifest()

    for framework_id in manifest.scenario_framework_ids:
        target = manifest.targets[framework_id]
        assert target.environment
        assert target.python_env_var.startswith("ML4T_")


def test_benchmark_runner_uses_frozen_target_environments() -> None:
    manifest = load_framework_manifest()

    assert set(run_all_benchmarks.FRAMEWORKS) == {"ml4t", *manifest.targets}
    for framework_id, target in manifest.targets.items():
        assert run_all_benchmarks.FRAMEWORKS[framework_id]["venv"] == target.environment


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ('version = "1.1.0"', "missing required field 'immutable_id'"),
        ('version = ">=1.1.0"\nimmutable_id = "git:abc"', "must be an exact version"),
    ],
)
def test_invalid_framework_targets_are_rejected(
    tmp_path: Path,
    replacement: str,
    message: str,
) -> None:
    manifest = tmp_path / "targets.toml"
    manifest.write_text(
        """
[metadata]
schema_version = 1
freeze_date = "2026-08-13"
scenario_frameworks = ["example"]

[framework.example]
display_name = "Example"
package = "example"
profile = "default"
python = ">=3.11"
license = "MIT"
access = "public"
source = "https://example.com"
environment = ".venv-example"
python_env_var = "ML4T_EXAMPLE_PYTHON"
scenario_matrix = true
"""
        + replacement
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match=message):
        load_framework_manifest(manifest)


def test_duplicate_framework_tables_are_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "targets.toml"
    manifest.write_text(
        """
[metadata]
schema_version = 1
freeze_date = "2026-08-13"

[framework.example]
display_name = "Example"
package = "example"
version = "1.0"
profile = "default"
python = ">=3.11"
license = "MIT"
access = "public"
source = "https://example.com"
immutable_id = "git:abc"
scenario_matrix = false

[framework.example]
version = "2.0"
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="Invalid TOML"):
        load_framework_manifest(manifest)


def test_claim_inventory_accounts_for_all_repositories_and_known_targets() -> None:
    manifest = load_framework_manifest()
    inventory = load_claim_inventory(manifest=manifest)

    assert inventory.audit_date == manifest.freeze_date
    assert {claim.repository for claim in inventory.claims} == {
        "ml4t/backtest",
        "ml4t/public",
        "ml4t/content-marketing",
    }
    assert {claim.claim_type for claim in inventory.claims} == {
        "behavioral",
        "correctness",
        "performance",
        "qualitative",
    }
    assert inventory.unresolved
    assert all(item.issue.startswith("ml4t/backtest-dev#") for item in inventory.unresolved)
    local_paths = {
        path
        for claim in inventory.claims
        if claim.repository == "ml4t/backtest"
        for path in claim.paths
    }
    assert local_paths
    assert all((_PROJECT_ROOT / path).is_file() for path in local_paths)


def test_claim_inventory_rejects_unknown_framework(tmp_path: Path) -> None:
    inventory = tmp_path / "claims.toml"
    inventory.write_text(
        """
[metadata]
schema_version = 1
audit_date = "2026-08-13"

[[claim]]
id = "unknown-target"
repository = "ml4t/backtest"
paths = ["README.md"]
type = "correctness"
frameworks = ["not-a-framework"]
required_evidence = ["canonical_trade_log"]
status = "unresolved"
issue = "ml4t/backtest-dev#9"

[[unresolved]]
id = "placeholder"
finding = "Placeholder finding."
issue = "ml4t/backtest-dev#9"
""",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="unknown framework"):
        load_claim_inventory(inventory)
