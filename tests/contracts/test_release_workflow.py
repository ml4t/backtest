"""Contracts for commit-bound release qualification and publication."""

from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import yaml

_ROOT = Path(__file__).parents[2]
_COMMIT = "1" * 40
_REPOSITORY = "ml4t/backtest"


def _load_release_candidate() -> ModuleType:
    path = _ROOT / "validation" / "release_candidate.py"
    spec = importlib.util.spec_from_file_location("ml4t_release_candidate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow(name: str) -> dict:
    payload = yaml.load(
        (_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(payload, dict)
    return payload


def _write_distributions(directory: Path) -> None:
    metadata = b"Metadata-Version: 2.4\nName: ml4t-backtest\nVersion: 0.1.0\n\n"
    wheel = directory / "ml4t_backtest-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("ml4t_backtest-0.1.0.dist-info/METADATA", metadata)
    source = directory / "ml4t_backtest-0.1.0.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        info = tarfile.TarInfo("ml4t_backtest-0.1.0/PKG-INFO")
        info.size = len(metadata)
        archive.addfile(info, io.BytesIO(metadata))


def _manifest(tmp_path: Path) -> tuple[ModuleType, dict[str, object]]:
    candidate = _load_release_candidate()
    _write_distributions(tmp_path)
    gates = dict.fromkeys(candidate.REQUIRED_GATES, "success")
    manifest = candidate.create_manifest(
        tmp_path,
        commit=_COMMIT,
        repository=_REPOSITORY,
        gates=gates,
    )
    return candidate, manifest


def test_candidate_rejects_missing_and_failed_release_gates(tmp_path: Path) -> None:
    candidate, manifest = _manifest(tmp_path)
    gates = manifest["gates"]
    assert isinstance(gates, dict)
    gates.pop("security")
    gates["coverage"] = "failure"

    failures = candidate.candidate_failures(
        manifest,
        tmp_path,
        expected_commit=_COMMIT,
        expected_repository=_REPOSITORY,
    )

    assert "Candidate is missing release gates: ['security']" in failures
    assert "Release gate did not pass: coverage=failure" in failures


def test_candidate_rejects_stale_commit_tag_and_changed_artifact(tmp_path: Path) -> None:
    candidate, manifest = _manifest(tmp_path)
    wheel = next(tmp_path.glob("*.whl"))
    wheel.write_bytes(wheel.read_bytes() + b"changed")

    failures = candidate.candidate_failures(
        manifest,
        tmp_path,
        expected_commit="2" * 40,
        expected_repository=_REPOSITORY,
        expected_tag="v0.1.1",
    )

    assert any("commit is stale" in failure for failure in failures)
    assert any("tag does not match" in failure for failure in failures)
    assert f"Candidate digest mismatch: {wheel.name}" in failures


def test_candidate_rejects_missing_and_undeclared_distributions(tmp_path: Path) -> None:
    candidate, manifest = _manifest(tmp_path)
    source = next(tmp_path.glob("*.tar.gz"))
    source.unlink()
    extra = tmp_path / "undeclared-0.1.0.tar.gz"
    extra.write_bytes(b"not a distribution")

    failures = candidate.candidate_failures(
        manifest,
        tmp_path,
        expected_commit=_COMMIT,
        expected_repository=_REPOSITORY,
    )

    assert any("distributions are missing" in failure for failure in failures)
    assert any("undeclared distributions" in failure for failure in failures)


def test_package_index_must_publish_only_candidate_digests(tmp_path: Path) -> None:
    candidate, manifest = _manifest(tmp_path)
    files = manifest["files"]
    assert isinstance(files, dict)
    payload = {
        "info": {"name": "ml4t-backtest", "version": "0.1.0"},
        "urls": [
            {"filename": name, "digests": {"sha256": record["sha256"]}}
            for name, record in files.items()
        ],
    }
    assert candidate.index_failures(manifest, payload) == []

    payload["urls"][0]["digests"]["sha256"] = "0" * 64
    assert candidate.index_failures(manifest, payload) == [
        f"Published digest differs from candidate: {payload['urls'][0]['filename']}"
    ]


def test_release_reuses_all_ci_gates_and_publishes_the_exact_candidate() -> None:
    ci = _workflow("ci.yml")
    release = _workflow("release.yml")
    jobs = ci["jobs"]
    release_jobs = release["jobs"]

    assert "workflow_call" in ci["on"]
    assert set(jobs["build"]["needs"]) == {
        "lint",
        "typecheck",
        "compatibility",
        "security",
        "coverage",
        "runtime",
        "public-parity",
        "documentation",
    }
    parity = jobs["public-parity"]
    assert {entry["framework"] for entry in parity["strategy"]["matrix"]["include"]} == {
        "vectorbt_oss",
        "backtrader",
        "zipline",
    }
    parity_commands = "\n".join(step.get("run", "") for step in parity["steps"])
    assert "validation/build_framework_env.py" in parity_commands
    assert "validation/native/vectorbt_behavior.py" in parity_commands
    assert "validation/native/backtrader_behavior.py" in parity_commands
    assert "validation/run_all_correctness.py" in parity_commands
    assert "--extra comparison" not in parity_commands
    build_commands = "\n".join(step.get("run", "") for step in jobs["build"]["steps"])
    assert "git rev-parse HEAD" in build_commands
    assert "validation/release_candidate.py create" in build_commands
    assert "validation/release_candidate.py verify" in build_commands
    for gate in _load_release_candidate().REQUIRED_GATES:
        assert f"--gate {gate}=" in build_commands

    assert release_jobs["qualification"]["uses"] == "./.github/workflows/ci.yml"
    assert release_jobs["private-comparisons"]["uses"] == (
        "./.github/workflows/private-comparisons.yml"
    )
    assert release_jobs["publish"]["needs"] == [
        "ecosystem-qualification",
        "private-comparisons",
        "qualification",
    ]
    assert release_jobs["publish"]["permissions"] == {
        "contents": "read",
        "id-token": "write",
    }
    publish_steps = release_jobs["publish"]["steps"]
    publish = next(
        step for step in publish_steps if "pypa/gh-action-pypi-publish" in step.get("uses", "")
    )
    assert publish["with"]["packages-dir"] == "candidate/dist/"
    assert not ({"user", "password"} & set(publish["with"]))
    assert "release_candidate.py verify" in "\n".join(step.get("run", "") for step in publish_steps)
    assert release_jobs["verify-pypi"]["needs"] == "publish"
    assert release_jobs["github-release"]["needs"] == "verify-pypi"
    assert "if" not in release_jobs["github-release"]


def test_private_comparison_workflow_pins_and_retains_evidence() -> None:
    workflow = _workflow("private-comparisons.yml")
    jobs = workflow["jobs"]

    assert workflow["on"]["workflow_call"]["secrets"]["VECTORBT_PRO_DEPLOY_KEY"]["required"]
    pro_commands = "\n".join(step.get("run", "") for step in jobs["vectorbt-pro"]["steps"])
    assert "validation/build_framework_env.py" in pro_commands
    assert "validation/native/vectorbt_behavior.py" in pro_commands
    assert "validation/run_all_correctness.py" in pro_commands
    assert "--framework vectorbt_pro" in pro_commands
    lean_commands = "\n".join(step.get("run", "") for step in jobs["lean"]["steps"])
    assert "--framework lean" in lean_commands
    assert "run_all_correctness.py" not in lean_commands
