"""Contracts for locked framework comparison environments."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parents[2]
_VALIDATION_DIR = _PROJECT_ROOT / "validation"
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

import build_framework_env  # noqa: E402
from common.framework_registry import FrameworkTarget, load_framework_manifest  # noqa: E402


@pytest.fixture
def targets() -> dict[str, FrameworkTarget]:
    return load_framework_manifest().targets


def test_all_environment_definitions_match_frozen_targets(
    targets: dict[str, FrameworkTarget],
) -> None:
    for framework in build_framework_env.BUILDABLE_FRAMEWORKS:
        assert build_framework_env.definition_failures(framework, targets[framework]) == []


def test_public_build_uses_locked_isolated_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    targets: dict[str, FrameworkTarget],
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(command: list[str], **kwargs):
        calls.append((command, kwargs["env"]))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build_framework_env.subprocess, "run", run)

    def verify(framework: str, _target: FrameworkTarget, *, root: Path):
        assert root == tmp_path
        return {"framework": framework}

    monkeypatch.setattr(build_framework_env, "verify_environment", verify)

    result = build_framework_env.build_environment(
        "backtrader", targets["backtrader"], root=tmp_path
    )

    assert result == {"framework": "backtrader"}
    command, environment = calls[0]
    assert command[-2:] == ["--locked", "--no-dev"]
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / ".venv-backtrader")


def test_definition_rejects_wrong_public_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    targets: dict[str, FrameworkTarget],
) -> None:
    project = tmp_path / "backtrader"
    project.mkdir()
    (project / "uv.lock").write_text(targets["backtrader"].immutable_id, encoding="utf-8")
    (project / "pyproject.toml").write_text(
        '[project]\ndependencies = ["backtrader==1.9.78.122"]\n', encoding="utf-8"
    )
    monkeypatch.setattr(build_framework_env, "ENVIRONMENTS_DIR", tmp_path)

    failures = build_framework_env.definition_failures("backtrader", targets["backtrader"])

    assert any("backtrader==1.9.78.123" in failure for failure in failures)


def test_definition_rejects_wrong_private_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    targets: dict[str, FrameworkTarget],
) -> None:
    project = tmp_path / "vectorbt_pro"
    project.mkdir()
    (project / "uv.lock").write_text("wrong-commit", encoding="utf-8")
    (project / "pyproject.toml").write_text(
        '[project]\ndependencies = ["vectorbtpro @ git+https://example.invalid/repo@wrong"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(build_framework_env, "ENVIRONMENTS_DIR", tmp_path)

    failures = build_framework_env.definition_failures("vectorbt_pro", targets["vectorbt_pro"])

    assert "VectorBT Pro environment does not pin the manifest commit" in failures
    assert any("artifact identity" in failure for failure in failures)


def test_definition_rejects_private_ssh_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    targets: dict[str, FrameworkTarget],
) -> None:
    target = targets["vectorbt_pro"]
    project = tmp_path / "vectorbt_pro"
    project.mkdir()
    (project / "uv.lock").write_text(target.immutable_id.removeprefix("git:"), encoding="utf-8")
    (project / "pyproject.toml").write_text(
        "[project]\ndependencies = ["
        f'"vectorbtpro @ git+ssh://git@github.com/polakowo/vectorbt.pro.git@'
        f'{target.source_commit}"\n]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(build_framework_env, "ENVIRONMENTS_DIR", tmp_path)

    failures = build_framework_env.definition_failures("vectorbt_pro", target)

    assert "VectorBT Pro environment must use GitHub CLI authenticated HTTPS" in failures


def test_private_build_reports_missing_licensed_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    targets: dict[str, FrameworkTarget],
) -> None:
    monkeypatch.setattr(build_framework_env, "definition_failures", lambda *_: [])
    monkeypatch.setattr(
        build_framework_env.subprocess,
        "run",
        lambda *args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, args[0])),
    )

    with pytest.raises(RuntimeError, match="GitHub CLI access"):
        build_framework_env.build_environment(
            "vectorbt_pro", targets["vectorbt_pro"], root=tmp_path
        )


def test_lean_verification_separates_cli_and_engine_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    targets: dict[str, FrameworkTarget],
) -> None:
    target = targets["lean"]
    interpreter = tmp_path / ".venv-lean" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    monkeypatch.setattr(build_framework_env, "definition_failures", lambda *_: [])
    monkeypatch.setattr(
        build_framework_env,
        "_probe",
        lambda *_: {
            "version": target.cli_version,
            "ml4t_path": str(_PROJECT_ROOT / "src/ml4t/backtest/__init__.py"),
            "python": "3.12.11",
        },
    )
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(build_framework_env.subprocess, "run", run)

    evidence = build_framework_env.verify_environment("lean", target, root=tmp_path)

    assert evidence["data_preparation"]["cli_version"] == target.cli_version
    assert evidence["engine_execution"]["image_digest"] == target.immutable_id
    assert commands == [["docker", "buildx", "imagetools", "inspect", target.artifact]]


def test_private_environment_rejects_wrong_installed_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    targets: dict[str, FrameworkTarget],
) -> None:
    target = targets["vectorbt_pro"]
    interpreter = tmp_path / ".venv-vectorbt-pro" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    monkeypatch.setattr(build_framework_env, "definition_failures", lambda *_: [])
    monkeypatch.setattr(
        build_framework_env,
        "_probe",
        lambda *_: {
            "version": target.version,
            "direct_url": json.dumps({"vcs_info": {"commit_id": "wrong"}}),
            "ml4t_path": str(_PROJECT_ROOT / "src/ml4t/backtest/__init__.py"),
            "python": "3.12.11",
        },
    )

    with pytest.raises(ValueError, match="commit differs"):
        build_framework_env.verify_environment("vectorbt_pro", target, root=tmp_path)
