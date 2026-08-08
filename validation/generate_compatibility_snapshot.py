"""Generate or check the reviewed 0.1 compatibility boundary."""

from __future__ import annotations

import argparse
import difflib
import importlib
import inspect
import json
import re
from dataclasses import MISSING, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

import polars as pl

import ml4t.backtest as backtest
import ml4t.backtest.profiles as profiles_module
import ml4t.backtest.result as result_module
from ml4t.backtest import BacktestConfig, BacktestResult, DataFeed, Engine, Strategy
from ml4t.backtest.analytics import EquityCurve, MAEMFEAnalyzer, TradeAnalyzer
from ml4t.backtest.config import StatsConfig

_ROOT = Path(__file__).parents[1]
_DEFAULT_SNAPSHOT = _ROOT / "tests" / "compatibility" / "snapshots" / "v0.1.json"
_DOCS = (_ROOT / "README.md", *sorted((_ROOT / "docs").rglob("*.md")))
_IMPORT = re.compile(
    r"^\s*(from\s+(ml4t\.backtest[\w.]*)\s+import\s+([^#]+)|import\s+(ml4t\.backtest[\w.]*))\s*$"
)
_TYPING_PREFIX = re.compile(r"\btyping\.")
_ENUM_MODULES = (
    "ml4t.backtest.config",
    "ml4t.backtest.execution.schedule",
    "ml4t.backtest.risk.types",
    "ml4t.backtest.types",
)
_PUBLIC_DUNDERS = {"__contains__", "__getitem__", "__iter__", "__len__", "__next__", "__repr__"}


def _signature(value: object) -> str | None:
    try:
        return _canonical_type_text(str(inspect.signature(value)))
    except (TypeError, ValueError):
        return None


def _canonical_type_text(value: str) -> str:
    """Remove interpreter-dependent qualification from type representations."""
    return _TYPING_PREFIX.sub("", value)


def _member_surface(value: type) -> dict[str, dict[str, object]]:
    members: dict[str, dict[str, object]] = {}
    for name, raw_member in value.__dict__.items():
        if name.startswith("_") and name not in _PUBLIC_DUNDERS:
            continue
        if isinstance(raw_member, property):
            members[name] = {
                "kind": "property",
                "signature": _signature(raw_member.fget) if raw_member.fget is not None else None,
            }
            continue
        if isinstance(raw_member, staticmethod | classmethod):
            member = raw_member.__func__
        else:
            member = raw_member
        if callable(member):
            members[name] = {
                "kind": "method",
                "signature": _signature(member),
            }
    return members


def _symbol_surface(value: object) -> dict[str, object]:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    if inspect.isclass(value):
        kind = "class"
    elif inspect.isfunction(value):
        kind = "function"
    else:
        kind = type(value).__name__
    surface: dict[str, object] = {
        "kind": kind,
        "module": module,
        "qualname": qualname,
        "signature": _signature(value) if callable(value) else None,
    }
    if inspect.isclass(value) and str(module).startswith("ml4t.backtest"):
        surface["members"] = _member_surface(value)
    return surface


def _documented_imports() -> tuple[list[str], dict[str, object]]:
    statements: set[str] = set()
    symbols: dict[str, object] = {}
    for path in _DOCS:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _IMPORT.match(line)
            if match is None:
                continue
            if match.group(2) is not None:
                module_name = match.group(2)
                names = [name.strip() for name in match.group(3).split(",")]
                if not names or any(not name.isidentifier() for name in names):
                    raise ValueError(f"Unsupported documented import in {path}: {line}")
                statement = f"from {module_name} import {', '.join(names)}"
                module = importlib.import_module(module_name)
                for name in names:
                    symbols[f"{module_name}:{name}"] = getattr(module, name)
            else:
                module_name = match.group(4)
                if module_name is None:
                    raise ValueError(f"Could not parse documented import in {path}: {line}")
                statement = f"import {module_name}"
                importlib.import_module(module_name)
            statements.add(statement)
    return sorted(statements), symbols


def _enum_surface(module: ModuleType) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for name, value in vars(module).items():
        if name.startswith("_") or not inspect.isclass(value) or not issubclass(value, Enum):
            continue
        if value.__module__ != module.__name__:
            continue
        result[f"{module.__name__}.{name}"] = [
            {"name": member_name, "value": member.value}
            for member_name, member in value.__members__.items()
        ]
    return result


def _annotation(value: object) -> str:
    rendered = value if isinstance(value, str) else inspect.formatannotation(value)
    return _canonical_type_text(rendered)


def _encoded_value(value: object) -> object:
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}.{value.name}",
            "value": value.value,
        }
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_encoded_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _encoded_value(item) for key, item in sorted(value.items())}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _encoded_value(getattr(value, field.name))
            for field in fields(value)
            if field.init and not field.name.startswith("_")
        }
    raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")


def _dataclass_schema(value: type) -> list[dict[str, object]]:
    schema: list[dict[str, object]] = []
    for field in fields(value):
        if not field.init or field.name.startswith("_"):
            continue
        if field.default is not MISSING:
            default: object = _encoded_value(field.default)
        elif field.default_factory is not MISSING:
            default = {
                "factory": f"{field.default_factory.__module__}.{field.default_factory.__qualname__}",
                "value": _encoded_value(field.default_factory()),
            }
        else:
            default = {"required": True}
        schema.append(
            {
                "name": field.name,
                "annotation": _annotation(field.type),
                "default": default,
                "keyword_only": field.kw_only,
            }
        )
    return schema


def _profile_surface() -> dict[str, object]:
    return {
        "canonical": {
            name: _encoded_value(profiles_module._PROFILES[name])
            for name in sorted(profiles_module._PROFILES)
        },
        "aliases": dict(sorted(profiles_module._ALIASES.items())),
        "advertised": profiles_module.list_profiles(),
    }


def _artifact_surface() -> dict[str, object]:
    schemas = {
        name: {column: str(dtype) for column, dtype in schema().items()}
        for name, schema in {
            "trades": BacktestResult._trades_schema,
            "fills": BacktestResult._fills_schema,
            "rejected_orders": BacktestResult._rejected_orders_schema,
            "equity": BacktestResult._equity_schema,
            "portfolio_state": BacktestResult._portfolio_state_schema,
        }.items()
    }
    return {
        "artifact_type": result_module._ARTIFACT_TYPE,
        "schema_version": result_module._ARTIFACT_SCHEMA_VERSION,
        "component_files": dict(result_module._COMPONENT_FILES),
        "required_components": sorted(result_module._REQUIRED_RESULT_COMPONENTS),
        "dataframe_schemas": schemas,
    }


class _NoopStrategy(Strategy):
    def on_data(
        self, timestamp: datetime, data: dict[str, dict], context: dict, broker: Any
    ) -> None:
        return None


def _metric_surface() -> dict[str, list[str]]:
    prices = pl.DataFrame(
        {
            "timestamp": [datetime(2025, 1, 2), datetime(2025, 1, 3)],
            "symbol": ["AAPL", "AAPL"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1_000.0, 1_100.0],
        }
    )
    result = Engine(DataFeed(prices_df=prices), _NoopStrategy()).run()
    return {
        "backtest_result": sorted(result.metrics),
        "equity_curve": sorted(EquityCurve().to_dict()),
        "trade_analyzer": sorted(TradeAnalyzer([]).to_dict()),
        "mae_mfe_analyzer": sorted(MAEMFEAnalyzer([]).to_dict()),
    }


def build_snapshot() -> dict[str, object]:
    documented_imports, documented_symbols = _documented_imports()
    symbols = {f"ml4t.backtest:{name}": getattr(backtest, name) for name in backtest.__all__}
    symbols.update(documented_symbols)

    analytics = importlib.import_module("ml4t.backtest.analytics")
    analytics_exports = list(analytics.__all__)
    symbols.update(
        {f"ml4t.backtest.analytics:{name}": getattr(analytics, name) for name in analytics_exports}
    )

    enum_surface: dict[str, list[dict[str, object]]] = {}
    for module_name in _ENUM_MODULES:
        enum_surface.update(_enum_surface(importlib.import_module(module_name)))

    return {
        "snapshot_schema_version": 1,
        "compatibility_version": "0.1",
        "beta_exclusions": {
            "Broker(account_type=...)": (
                "Removed before 0.1 because one string silently selected multiple shorting and "
                "leverage policies. Use explicit account-policy flags."
            )
        },
        "root_exports": list(backtest.__all__),
        "documented_imports": documented_imports,
        "symbols": {name: _symbol_surface(symbols[name]) for name in sorted(symbols)},
        "analytics_exports": analytics_exports,
        "enums": dict(sorted(enum_surface.items())),
        "configuration": {
            "BacktestConfig": _dataclass_schema(BacktestConfig),
            "StatsConfig": _dataclass_schema(StatsConfig),
            "default_serialization": BacktestConfig().to_dict(),
        },
        "profiles": _profile_surface(),
        "metrics": _metric_surface(),
        "result_artifact": _artifact_surface(),
    }


def _render(snapshot: dict[str, object]) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=_DEFAULT_SNAPSHOT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    actual = _render(build_snapshot())
    if args.write:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(actual, encoding="utf-8")
        return 0
    if not args.snapshot.is_file():
        print(f"Compatibility snapshot is missing: {args.snapshot}")
        return 1
    expected = args.snapshot.read_text(encoding="utf-8")
    if actual == expected:
        return 0
    print(
        "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile=str(args.snapshot),
                tofile="current compatibility surface",
            )
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
