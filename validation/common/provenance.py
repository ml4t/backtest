"""Content-derived provenance for cross-framework correctness evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import asdict
from decimal import ROUND_HALF_EVEN, Decimal
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import data_generators
from common.comparator import CANONICAL_QUANTUM
from common.framework_registry import DEFAULT_MANIFEST_PATH, FrameworkTarget
from common.types import FrameworkResult, ScenarioConfig

VALIDATION_DIR = Path(__file__).parents[1]
PROJECT_ROOT = VALIDATION_DIR.parent
SOURCE_DIR = PROJECT_ROOT / "src" / "ml4t" / "backtest"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return _sha256_bytes(path.read_bytes())


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256_bytes(payload.encode("utf-8"))


@cache
def _tree_digest(root: Path) -> str:
    try:
        git_root = Path(
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        relative_root = root.resolve().relative_to(git_root.resolve())
        tracked = subprocess.run(
            ["git", "-C", str(git_root), "ls-files", "-z", "--", relative_root.as_posix()],
            check=True,
            capture_output=True,
        ).stdout
        paths = sorted(
            git_root / value.decode("utf-8")
            for value in tracked.split(b"\0")
            if value and value.endswith(b".py")
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        paths = sorted(root.rglob("*.py"))

    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def generate_inputs(
    scenario: ScenarioConfig,
    framework: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None]:
    """Generate and framework-align one scenario input set."""
    generator = getattr(data_generators, scenario.data_generator)
    data_result = generator(**scenario.data_kwargs)
    if len(data_result) == 3:
        prices, entries, exits = data_result
    else:
        prices, entries = data_result
        exits = None

    if framework == "zipline":
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS")
        start = prices.index[0]
        end = prices.index[-1]
        if start.tz is not None:
            start = start.tz_convert(None)
            end = end.tz_convert(None)
        sessions = calendar.sessions_in_range(start, end)
        index = prices.index.tz_localize(None) if prices.index.tz else prices.index
        valid = index.isin(sessions)
        prices = prices[valid].copy()
        entries = entries[valid]
        if exits is not None:
            exits = exits[valid]
    return prices, np.asarray(entries), None if exits is None else np.asarray(exits)


def input_digest(
    prices: pd.DataFrame,
    entries: np.ndarray,
    exits: np.ndarray | None,
) -> str:
    """Hash generated inputs in the same fixed-point domain used for parity claims."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("validation prices must use a DatetimeIndex")

    digest = hashlib.sha256()
    frame_identity = {
        "columns": [str(column) for column in prices.columns],
        "dtypes": [str(dtype) for dtype in prices.dtypes],
        "index": {
            "dtype": "datetime64[ns]",
            "name": prices.index.name,
            "timezone": str(prices.index.tz) if prices.index.tz is not None else None,
        },
        "shape": prices.shape,
    }
    digest.update(json.dumps(frame_identity, sort_keys=True).encode("utf-8"))
    normalized_index = (
        prices.index.tz_convert("UTC").tz_localize(None)
        if prices.index.tz is not None
        else prices.index
    )
    index_values = np.ascontiguousarray(
        normalized_index.to_numpy(dtype="datetime64[ns]").view("<i8")
    )
    digest.update(index_values.tobytes())
    for column in prices.columns:
        values = np.ascontiguousarray(prices[column].to_numpy())
        digest.update(str(column).encode("utf-8"))
        digest.update(values.dtype.str.encode("utf-8"))
        digest.update(str(values.shape).encode("utf-8"))
        if np.issubdtype(values.dtype, np.floating):
            if not np.isfinite(values).all():
                raise ValueError(f"validation price column {column!r} must contain finite values")
            for value in values.flat:
                canonical = Decimal(str(value)).quantize(
                    CANONICAL_QUANTUM,
                    rounding=ROUND_HALF_EVEN,
                )
                if canonical == 0:
                    canonical = abs(canonical)
                digest.update(format(canonical, "f").encode("ascii"))
                digest.update(b"\0")
        else:
            digest.update(values.tobytes())
    for name, values in (("entries", entries), ("exits", exits)):
        digest.update(name.encode("utf-8"))
        if values is None:
            digest.update(b"none")
            continue
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def scenario_digest(scenario: ScenarioConfig) -> str:
    """Hash one declarative scenario definition."""
    return _json_digest(asdict(scenario))


def static_digests(scenario: ScenarioConfig, framework: str) -> dict[str, str]:
    """Return behavior-relevant source digests for one comparison pair."""
    adapter_path = VALIDATION_DIR / "frameworks" / f"{framework}.py"
    profile_inputs = {
        "profile_source": file_digest(SOURCE_DIR / "profiles.py"),
        "base": scenario.ml4t_config,
        "override": scenario.ml4t_overrides.get(framework, {}),
    }
    return {
        "adapter": file_digest(adapter_path),
        "canonical_records": file_digest(VALIDATION_DIR / "common" / "canonical_records.py"),
        "capabilities": file_digest(VALIDATION_DIR / "common" / "capabilities.py"),
        "comparator": file_digest(VALIDATION_DIR / "common" / "comparator.py"),
        "engine": _tree_digest(SOURCE_DIR),
        "manifest": file_digest(DEFAULT_MANIFEST_PATH),
        "ml4t_runner": file_digest(VALIDATION_DIR / "common" / "ml4t_runner.py"),
        "profile": _json_digest(profile_inputs),
        "scenario": scenario_digest(scenario),
    }


def _git_identity() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def installed_framework_version(target: FrameworkTarget) -> str:
    """Return the version imported by the isolated scenario environment."""
    return importlib.metadata.version(target.package)


def comparison_protocol_metadata(scenario: ScenarioConfig, framework: str) -> dict[str, str]:
    """Identify native and adapter-emulated portions of a comparison."""
    if framework != "zipline":
        return {"risk_rules": "framework_adapter"}
    return {
        "commission": "explicit scenario model or NoCommission",
        "execution": "custom next-session open-price slippage model",
        "fills": "native transactions with fees reconstructed from the configured model",
        "risk_rules": "adapter_emulated_daily_ohlc" if scenario.risk_rules else "none",
        "trade_records": "adapter_reconstructed_from_native_transactions",
    }


def build_record_provenance(
    *,
    scenario: ScenarioConfig,
    framework: str,
    target: FrameworkTarget,
    prices: pd.DataFrame,
    entries: np.ndarray,
    exits: np.ndarray | None,
    framework_result: FrameworkResult,
    ml4t_result: FrameworkResult,
) -> dict[str, Any]:
    """Build complete provenance for one executed comparison."""
    adapter_path = VALIDATION_DIR / "frameworks" / f"{framework}.py"
    return {
        "framework_target": {
            "version": target.version,
            "actual_version": installed_framework_version(target),
            "immutable_id": target.immutable_id,
        },
        "ml4t": _git_identity(),
        "python": {
            "version": sys.version.split()[0],
            "implementation": sys.implementation.name,
        },
        "adapter": {
            "module": f"frameworks.{framework}",
            "path": adapter_path.relative_to(PROJECT_ROOT).as_posix(),
        },
        "comparison_protocol": comparison_protocol_metadata(scenario, framework),
        "capabilities": {
            "framework": framework_result.capabilities,
            "ml4t": ml4t_result.capabilities,
        },
        "digests": static_digests(scenario, framework),
        "input_digest": input_digest(prices, entries, exits),
        "record_counts": {
            "bars": len(prices),
            "entry_signals": int(entries.sum()),
            "exit_signals": int(exits.sum()) if exits is not None else 0,
            "framework_intents": int(entries.sum())
            + (int(exits.sum()) if exits is not None else 0),
            "ml4t_intents": int(entries.sum()) + (int(exits.sum()) if exits is not None else 0),
            "framework_orders": None,
            "ml4t_orders": None,
            "framework_fills": len(framework_result.fills),
            "ml4t_fills": len(ml4t_result.fills),
            "framework_closed_trades": framework_result.num_trades,
            "ml4t_closed_trades": ml4t_result.num_trades,
        },
    }
