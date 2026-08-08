"""Stable API snapshots and supported beta-artifact migration contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from ml4t.backtest import ArtifactManifestError, BacktestResult, Broker

_ROOT = Path(__file__).parents[2]
_SNAPSHOT = _ROOT / "tests" / "compatibility" / "snapshots" / "v0.1.json"
_BETA_ARTIFACT = _ROOT / "tests" / "fixtures" / "artifacts" / "v0.1.0b22"


def _load_snapshot_generator() -> ModuleType:
    path = _ROOT / "validation" / "generate_compatibility_snapshot.py"
    spec = importlib.util.spec_from_file_location("ml4t_compatibility_snapshot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.no_invariant_check
def test_current_surface_exactly_matches_the_reviewed_versioned_snapshot() -> None:
    expected = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    actual = _load_snapshot_generator().build_snapshot()
    assert actual == expected


def test_snapshot_canonicalizes_interpreter_dependent_typing_qualification() -> None:
    generator = _load_snapshot_generator()
    assert generator._canonical_type_text("dict[str, typing.Any] | None") == (
        "dict[str, Any] | None"
    )


def test_beta_fixture_has_unchanged_release_generated_bytes() -> None:
    fixture = json.loads((_BETA_ARTIFACT / "fixture.json").read_text(encoding="utf-8"))
    assert fixture["source_distribution"] == "ml4t-backtest==0.1.0b22"
    for filename, expected_digest in fixture["sha256"].items():
        actual_digest = hashlib.sha256((_BETA_ARTIFACT / filename).read_bytes()).hexdigest()
        assert actual_digest == expected_digest


def test_beta_fixture_checkout_disables_platform_line_ending_conversion() -> None:
    attributes = (_ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert "/tests/fixtures/artifacts/v0.1.0b22/** -text" in attributes


def test_manifest_free_beta_artifact_requires_explicit_recovery() -> None:
    with pytest.raises(ArtifactManifestError, match="manifest is missing"):
        BacktestResult.from_parquet(_BETA_ARTIFACT)


def test_v0_1_0b22_artifact_recovers_without_semantic_change(tmp_path: Path) -> None:
    fixture = json.loads((_BETA_ARTIFACT / "fixture.json").read_text(encoding="utf-8"))
    expected = fixture["expected"]
    loaded = BacktestResult.from_parquet(_BETA_ARTIFACT, recovery=True)

    assert loaded.metrics == expected["metrics"]
    assert [value for _, value in loaded.equity_curve] == expected["equity_curve"]
    assert [state[2] for state in loaded.portfolio_state] == expected["portfolio_cash"]
    assert [fill.order_id for fill in loaded.fills] == expected["fill_order_ids"]
    assert [[item.code, item.component] for item in loaded.artifact_diagnostics] == expected[
        "diagnostics"
    ]

    trade = loaded.trades[0]
    assert {name: getattr(trade, name) for name in expected["trade"]} == expected["trade"]
    assert loaded.config is not None
    config = loaded.config.to_dict()
    assert config["cash"]["initial"] == expected["config"]["initial_cash"]
    assert config["account"]["allow_short_selling"] is expected["config"]["allow_short_selling"]
    assert config["account"]["allow_leverage"] is expected["config"]["allow_leverage"]
    assert config["execution"]["execution_price"] == expected["config"]["execution_price"]
    assert config["execution"]["execution_mode"] == expected["config"]["execution_mode"]
    assert config["position_sizing"]["share_type"] == expected["config"]["share_type"]
    assert config["orders"]["rebalance_mode"] == expected["config"]["rebalance_mode"]
    assert loaded.config.preset_name == expected["config"]["preset_name"]

    loaded.to_parquet(tmp_path)
    migrated = BacktestResult.from_parquet(tmp_path)
    assert migrated.metrics == loaded.metrics
    assert migrated.trades == loaded.trades
    assert migrated.fills == loaded.fills
    assert migrated.equity_curve == loaded.equity_curve
    assert migrated.portfolio_state == loaded.portfolio_state
    assert not migrated.artifact_diagnostics


def test_removed_account_type_keyword_is_an_explicit_correctness_exception() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'account_type'"):
        Broker(account_type="margin")  # type: ignore[call-arg]

    snapshot = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    assert "Broker(account_type=...)" in snapshot["beta_exclusions"]
    accounts_guide = (_ROOT / "docs" / "user-guide" / "accounts.md").read_text(encoding="utf-8")
    normalized_guide = " ".join(accounts_guide.split())
    assert "not part of the stable compatibility boundary" in normalized_guide
    assert "broker = Broker(account_type=" not in accounts_guide


def test_snapshot_generation_is_a_non_mutating_ci_gate() -> None:
    workflow = yaml.load(
        (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["lint"]["steps"])
    assert "validation/generate_compatibility_snapshot.py" in commands
    assert "--write" not in commands
