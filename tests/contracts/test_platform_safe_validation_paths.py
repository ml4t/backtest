"""Platform contracts for shipped validation helpers."""

from __future__ import annotations

from pathlib import Path

from ml4t.backtest._validation import lean_runner


def test_lean_tool_paths_follow_platform_temporary_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.setattr(lean_runner.tempfile, "gettempdir", lambda: str(tmp_path))

    env = lean_runner.make_lean_env()

    assert Path(env["UV_CACHE_DIR"]) == tmp_path / "ml4t-uv-cache"
    assert Path(env["UV_TOOL_DIR"]) == tmp_path / "ml4t-uv-tools"


def test_validation_operations_do_not_use_fixed_unix_temporary_paths() -> None:
    sources = [
        Path(lean_runner.__file__),
        Path(__file__).parents[2] / "validation" / "benchmark_suite.py",
        Path(__file__).parents[1] / "benchmark" / "test_cross_framework_selected.py",
    ]

    for source in sources:
        contents = source.read_text(encoding="utf-8")
        assert "/tmp/" not in contents
        assert "/home/" not in contents
