"""Contracts for immutable LEAN engine orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ml4t.backtest._validation import lean_runner


def test_lean_backtest_requires_an_immutable_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="immutable digest"):
        lean_runner.run_lean_backtest(
            lean_cmd=["lean"],
            cwd=tmp_path,
            project_dir=tmp_path / "project",
            lean_config=tmp_path / "lean.json",
            output_dir=tmp_path / "output",
            image="quantconnect/lean:latest",
        )


def test_lean_command_resolution_does_not_install_an_unpinned_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ML4T_LEAN_COMMAND", raising=False)
    monkeypatch.setattr(lean_runner.shutil, "which", lambda _name: None)

    with pytest.raises(FileNotFoundError, match="frozen environment"):
        lean_runner.resolve_lean_command()


def test_lean_backtest_passes_the_frozen_image_to_the_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lean_runner.subprocess, "run", fake_run)
    image = "quantconnect/lean@sha256:" + "a" * 64

    lean_runner.run_lean_backtest(
        lean_cmd=["lean"],
        cwd=tmp_path,
        project_dir=tmp_path / "project",
        lean_config=tmp_path / "lean.json",
        output_dir=tmp_path / "output",
        image=image,
    )

    assert calls == [
        [
            "lean",
            "backtest",
            str(tmp_path / "project"),
            "--lean-config",
            str(tmp_path / "lean.json"),
            "--image",
            image,
            "--no-update",
            "--output",
            str(tmp_path / "output"),
        ]
    ]
