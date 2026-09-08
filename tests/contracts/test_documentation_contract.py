"""Contracts tying stable API documentation to the reviewed compatibility snapshot."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parents[2]
_SNAPSHOT = _ROOT / "tests" / "compatibility" / "snapshots" / "v0.1.json"


def _snapshot() -> dict[str, Any]:
    return json.loads(_SNAPSHOT.read_text(encoding="utf-8"))


def _documented_members(module: str) -> set[str]:
    api_reference = (_ROOT / "docs" / "api" / "index.md").read_text(encoding="utf-8")
    match = re.search(
        rf"^::: {re.escape(module)}\n(?P<options>(?:    .*\n|\n)*)",
        api_reference,
        flags=re.MULTILINE,
    )
    assert match is not None, f"API reference has no directive for {module}"
    return set(re.findall(r"^        - ([A-Za-z_][A-Za-z0-9_]*)$", match["options"], re.MULTILINE))


def test_reviewed_strategy_callbacks_are_in_api_and_strategy_guide() -> None:
    members = set(_snapshot()["symbols"]["ml4t.backtest:Strategy"]["members"])
    assert members <= _documented_members("ml4t.backtest.strategy.Strategy")

    strategy_guide = (_ROOT / "docs" / "user-guide" / "strategies.md").read_text(encoding="utf-8")
    missing = sorted(name for name in members if name not in strategy_guide)
    assert not missing, f"Strategy guide omits reviewed callbacks: {missing}"


def test_reviewed_broker_operations_are_in_api_reference() -> None:
    members = set(_snapshot()["symbols"]["ml4t.backtest:Broker"]["members"])
    assert members <= _documented_members("ml4t.backtest.broker.Broker")


def test_reviewed_result_operations_are_in_api_reference() -> None:
    members = set(_snapshot()["symbols"]["ml4t.backtest.result:BacktestResult"]["members"])
    members.discard("__repr__")
    assert members <= _documented_members("ml4t.backtest.result.BacktestResult")


def test_reviewed_config_fields_are_in_configuration_guide() -> None:
    fields = {item["name"] for item in _snapshot()["configuration"]["BacktestConfig"]}
    configuration = (_ROOT / "docs" / "user-guide" / "configuration.md").read_text(encoding="utf-8")
    missing = sorted(name for name in fields if f"`{name}`" not in configuration)
    assert not missing, f"Configuration guide omits reviewed fields: {missing}"


def test_readme_satisfies_the_public_entry_point_contract() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    description = (
        "Event-driven backtesting for quantitative strategies with configurable execution, "
        "accounting, risk, and framework-parity validation."
    )

    assert description in readme
    assert all(version in readme for version in ("3.12", "3.13", "3.14"))
    assert "uv add ml4t-backtest" in readme
    assert "from ml4t.backtest import" in readme
    assert "no external service or special hardware" in readme
    assert "licensed VectorBT Pro" in readme
    assert "containerized LEAN engine" in readme
    for link in (
        "https://www.ml4trading.io/docs/backtest/",
        "https://github.com/ml4t/backtest/issues",
        "https://github.com/ml4t/backtest/releases",
        "[MIT License](LICENSE)",
    ):
        assert link in readme
    assert all(
        command in readme
        for command in (
            "uv run ruff check",
            "uv run ruff format --check",
            "uv run ty check",
            "uv run pytest",
            "uv run mkdocs build --strict",
        )
    )


def test_documentation_identity_check_rejects_stale_rendered_metadata() -> None:
    path = _ROOT / "validation" / "check_documentation_identity.py"
    spec = importlib.util.spec_from_file_location("ml4t_documentation_identity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    html = (
        '<meta name="ml4t-library" content="backtest">'
        '<meta name="ml4t-version" content="0.1.6">'
        f'<meta name="ml4t-commit" content="{"1" * 40}">'
    )
    expected = {
        "expected_library": "backtest",
        "expected_version": "0.1.6",
        "expected_commit": "1" * 40,
        "source": "site/index.html",
    }
    assert module.identity_failures(html, **expected) == []

    stale = html.replace('content="0.1.6"', 'content="0.1.5"')
    assert module.identity_failures(stale, **expected) == [
        "site/index.html: ml4t-version is '0.1.5', expected '0.1.6'"
    ]
