"""Contracts tying stable API documentation to the reviewed compatibility snapshot."""

from __future__ import annotations

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
