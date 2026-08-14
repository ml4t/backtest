#!/usr/bin/env python3
"""Generate and validate reproducible cross-framework large-scale evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from decimal import ROUND_HALF_EVEN, Decimal
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

VALIDATION_DIR = Path(__file__).parent
PROJECT_ROOT = VALIDATION_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(VALIDATION_DIR))

import benchmark_suite as suite  # noqa: E402
from common.framework_registry import FrameworkTarget, load_framework_manifest  # noqa: E402

SCHEMA_VERSION = 1
MONEY_QUANTUM = Decimal("0.000001")
FRAMEWORKS = ("vectorbt_pro", "vectorbt_oss", "backtrader", "zipline", "lean")
ACCEPTED_PATH = VALIDATION_DIR / "LARGE_SCALE_RESULTS.json"
CANDIDATE_DIR = VALIDATION_DIR / "candidates"
CANDIDATE_PATH = CANDIDATE_DIR / "LARGE_SCALE_RESULTS.candidate.json"
HISTORICAL_ACCEPTED_PATH = VALIDATION_DIR / "vectorbt_pro" / "controlled_replay-2025.12.31.json"
HISTORICAL_CANDIDATE_PATH = VALIDATION_DIR / "candidates" / "vectorbt_pro-2025.12.31.candidate.json"
HISTORICAL_VECTORBT_PRO_VERSION = "2025.12.31"
HISTORICAL_VECTORBT_PRO_COMMIT = "1305a1e1974325db9382eaeacc6452e9b075ca71"
HISTORICAL_VECTORBT_PRO_ENV = ".venv-vectorbt-pro-312"
HISTORICAL_VECTORBT_PRO_PYTHON_ENV_VAR = "ML4T_VECTORBT_PRO_HISTORICAL_PYTHON"


@dataclass(frozen=True)
class ScaleWorkload:
    """Content-addressed public workload recipe."""

    name: str = "controlled_250_assets_5040_sessions"
    seed: int = 42
    bars: int = 5_040
    assets: int = 250
    top_n: int = 25
    bottom_n: int = 25
    rebalance_frequency: int = 1
    end_session: str = "2025-12-31"
    initial_cash: float = 1_000_000.0
    long_target_shares: float = 100.0
    short_target_shares: float = -100.0

    @property
    def data_points(self) -> int:
        return self.bars * self.assets

    def benchmark_config(self) -> suite.BenchmarkConfig:
        return suite.BenchmarkConfig(
            name=self.name,
            n_bars=self.bars,
            n_assets=self.assets,
            frequency="D",
            top_n=self.top_n,
            bottom_n=self.bottom_n,
            rebalance_freq=self.rebalance_frequency,
            initial_cash=self.initial_cash,
            lean_force_zero_fee=True,
            lean_force_zero_slippage=True,
            end_session=self.end_session,
        )


WORKLOAD = ScaleWorkload()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_digest(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    )


def _update_array(digest: Any, label: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(label.encode())
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())


def _datetime_ns(index: pd.DatetimeIndex) -> np.ndarray:
    normalized = index.tz_convert("UTC").tz_localize(None) if index.tz is not None else index
    return normalized.to_numpy(dtype="datetime64[ns]").view("<i8")


def input_digest(
    price_data: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> str:
    """Hash every generated input value independently of pandas internals."""
    digest = hashlib.sha256()
    _update_array(digest, "dates", _datetime_ns(dates))
    for asset in sorted(price_data):
        frame = price_data[asset]
        digest.update(asset.encode())
        _update_array(digest, "index", _datetime_ns(pd.DatetimeIndex(frame.index)))
        for column in ("open", "high", "low", "close", "volume"):
            _update_array(digest, column, frame[column].to_numpy())
    signal_dates = pd.DatetimeIndex(pd.to_datetime(signals["timestamp"]))
    _update_array(digest, "signal_dates", _datetime_ns(signal_dates))
    digest.update("\0".join(signals["asset"].astype(str)).encode())
    _update_array(digest, "scores", signals["score"].to_numpy())
    return digest.hexdigest()


def _effective_prices(
    framework: str, price_data: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    decimals = 3 if framework == "zipline" else 4 if framework == "lean" else None
    if decimals is None:
        return price_data
    effective: dict[str, pd.DataFrame] = {}
    for asset, frame in price_data.items():
        converted = frame.copy()
        converted.loc[:, ["open", "high", "low", "close"]] = converted[
            ["open", "high", "low", "close"]
        ].round(decimals)
        effective[asset] = converted
    return effective


def _canonical_money(value: float) -> float:
    return float(Decimal(str(value)).quantize(MONEY_QUANTUM, rounding=ROUND_HALF_EVEN))


def _float_value(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Expected a numeric canonical value, got {type(value).__name__}")
    return float(value)


def _terminal_state(
    result: suite.BenchmarkResult,
    effective_prices: dict[str, pd.DataFrame],
    initial_cash: float,
) -> dict[str, object]:
    fills = suite.canonical_fill_records(result.fills_df, timestamp_domain="session_date") or []
    positions: dict[str, float] = {}
    cash = initial_cash
    for fill in fills:
        quantity = _float_value(fill["quantity"])
        signed = quantity if fill["side"] == "buy" else -quantity
        asset = str(fill["asset"])
        positions[asset] = positions.get(asset, 0.0) + signed
        cash -= signed * _float_value(fill["price"])
        cash -= _float_value(fill["commission"])
    positions = {asset: quantity for asset, quantity in sorted(positions.items()) if quantity != 0}
    holdings = sum(
        quantity * float(effective_prices[asset]["close"].iloc[-1])
        for asset, quantity in positions.items()
    )
    state = {
        "cash_from_fill_ledger": _canonical_money(cash),
        "holdings_from_fill_ledger": _canonical_money(holdings),
        "positions": {asset: _canonical_money(quantity) for asset, quantity in positions.items()},
        "reported_final_value": _canonical_money(result.final_value),
    }
    state["sha256"] = _json_digest(state)
    return state


def _source_digests() -> dict[str, str]:
    paths = {
        "benchmark_suite": VALIDATION_DIR / "benchmark_suite.py",
        "framework_manifest": VALIDATION_DIR / "framework_targets.toml",
        "large_scale_runner": Path(__file__),
        "profiles": PROJECT_ROOT / "src" / "ml4t" / "backtest" / "profiles.py",
    }
    return {name: _sha256_bytes(path.read_bytes()) for name, path in paths.items()}


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
    ).stdout.strip()
    return {"commit": commit, "dirty": bool(status)}


def _actual_target(target: FrameworkTarget) -> dict[str, object]:
    if target.package == "lean":
        actual_version = target.version
    else:
        actual_version = importlib.metadata.version(target.package)
    metadata = {**target.evidence_metadata(), "actual_version": actual_version}
    if target.package == "vectorbtpro":
        distribution = importlib.metadata.distribution(target.package)
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text is None:
            raise RuntimeError("Installed vectorbtpro lacks direct_url.json source provenance")
        direct_url = json.loads(direct_url_text)
        vcs_info = direct_url.get("vcs_info", {})
        actual_commit = vcs_info.get("commit_id")
        if actual_commit != target.source_commit:
            raise RuntimeError(
                f"Installed vectorbtpro commit {actual_commit!r} does not match declared target "
                f"{target.source_commit!r}"
            )
        metadata["actual_commit"] = actual_commit
        metadata["actual_immutable_id"] = f"git:{actual_commit}"
    return metadata


def _declared_target_metadata(target: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in target.items() if not key.startswith("actual_")}


def _framework_run(
    framework: str,
    config: suite.BenchmarkConfig,
    price_data: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[suite.BenchmarkResult, suite.BenchmarkResult]:
    if framework == "vectorbt_pro":
        external = suite.benchmark_vectorbt_pro(config, price_data, signals, dates)
        ml4t = suite.benchmark_ml4t(
            config,
            price_data,
            signals,
            dates,
            execution_mode="same_bar",
            profile_override="vectorbt_strict",
        )
    elif framework == "vectorbt_oss":
        external = suite.benchmark_vectorbt_oss(config, price_data, signals, dates)
        ml4t = suite.benchmark_ml4t(
            config,
            price_data,
            signals,
            dates,
            execution_mode="same_bar",
            profile_override="vectorbt_oss_strict",
        )
    elif framework == "backtrader":
        external = suite.benchmark_backtrader(config, price_data, signals, dates)
        ml4t = suite.benchmark_ml4t(
            config,
            price_data,
            signals,
            dates,
            execution_mode="next_bar",
            profile_override="backtrader_strict",
        )
    elif framework == "zipline":
        external = suite.benchmark_zipline(config, price_data, signals, dates)
        ml4t = suite.benchmark_ml4t(
            config,
            price_data,
            signals,
            dates,
            execution_mode="next_bar",
            profile_override="zipline_strict",
        )
    elif framework == "lean":
        external = suite.benchmark_lean(config, price_data, signals, dates)
        ml4t = suite.benchmark_ml4t(
            config,
            price_data,
            signals,
            dates,
            execution_mode="next_bar",
            profile_override="lean",
        )
    else:
        raise ValueError(f"Unknown scale framework: {framework}")
    return external, ml4t


def _use_canonical_fill_backed_trades(
    external: suite.BenchmarkResult,
    ml4t: suite.BenchmarkResult,
) -> None:
    for result in (external, ml4t):
        result.trades_df = suite.closed_trades_from_fills(result.fills_df)
        records = suite.canonical_trade_records(result.trades_df)
        result.num_trades = len(records) if records is not None else 0


def _historical_vectorbt_pro_target() -> FrameworkTarget:
    current = load_framework_manifest().targets["vectorbt_pro"]
    return replace(
        current,
        version=HISTORICAL_VECTORBT_PRO_VERSION,
        source_commit=HISTORICAL_VECTORBT_PRO_COMMIT,
        immutable_id=f"git:{HISTORICAL_VECTORBT_PRO_COMMIT}",
        environment=HISTORICAL_VECTORBT_PRO_ENV,
        python_env_var=HISTORICAL_VECTORBT_PRO_PYTHON_ENV_VAR,
    )


def run_worker(
    framework: str,
    workload: ScaleWorkload = WORKLOAD,
    *,
    target_override: FrameworkTarget | None = None,
) -> dict[str, Any]:
    """Execute one external/ML4T scale pair and retain exact digests."""
    manifest = load_framework_manifest()
    target = target_override or manifest.targets[framework]
    if target.framework_id != framework:
        raise ValueError(f"Target {target.framework_id!r} cannot identify framework {framework!r}")
    actual_target = _actual_target(target)
    if actual_target["actual_version"] != target.version:
        raise RuntimeError(
            f"Installed {target.package} version {actual_target['actual_version']} does not match "
            f"declared target {target.version}"
        )
    config = workload.benchmark_config()
    price_data, signals, dates = suite.generate_benchmark_data(config, seed=workload.seed)
    raw_digest = input_digest(price_data, signals, dates)
    effective_prices = _effective_prices(framework, price_data)
    effective_digest = input_digest(effective_prices, signals, dates)
    external, ml4t = _framework_run(framework, config, price_data, signals, dates)
    _use_canonical_fill_backed_trades(external, ml4t)
    comparison = cast(
        dict[str, Any],
        suite.compare_benchmark_results_exact(
            external,
            ml4t,
            initial_cash=config.initial_cash,
            timestamp_domain="session_date",
        ),
    )
    expected_terminal = _terminal_state(external, effective_prices, config.initial_cash)
    actual_terminal = _terminal_state(ml4t, effective_prices, config.initial_cash)
    terminal_check = {
        "name": "terminal_state",
        "passed": expected_terminal == actual_terminal,
        "expected_sha256": expected_terminal["sha256"],
        "actual_sha256": actual_terminal["sha256"],
        "expected": expected_terminal,
        "actual": actual_terminal,
    }
    comparison["checks"].append(terminal_check)
    comparison["passed"] = bool(comparison["passed"] and terminal_check["passed"])
    return {
        "framework": framework,
        "target": actual_target,
        "python": {
            "version": sys.version.split()[0],
            "implementation": sys.implementation.name,
        },
        "ml4t": _git_identity(),
        "source_digests": _source_digests(),
        "input": {
            "raw_sha256": raw_digest,
            "effective_sha256": effective_digest,
            "conversion": (
                "OHLC rounded to 3 decimals before both engines"
                if framework == "zipline"
                else "OHLC rounded to 4 decimals before both engines"
                if framework == "lean"
                else "none"
            ),
        },
        "capabilities": {
            "intents": "canonical strategy trace",
            "fills": "native",
            "closed_trades": "reconstructed from canonical native fills",
            "terminal_state": "reconstructed from native fills and final marks",
            "fill_order": "session, asset, side, quantity, price",
        },
        "runtime_seconds": {
            "framework": external.runtime_sec,
            "ml4t": ml4t.runtime_sec,
            "not_a_performance_claim": True,
        },
        "comparison": comparison,
    }


def build_report(
    records: list[dict[str, Any]], workload: ScaleWorkload = WORKLOAD
) -> dict[str, Any]:
    """Build one candidate report from isolated worker records."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "workload": {
            "recipe": asdict(workload),
            "recipe_sha256": _json_digest(asdict(workload)),
            "data_points": workload.data_points,
            "generator": "validation.benchmark_suite.generate_benchmark_data",
            "redistributable": True,
        },
        "frameworks": records,
        "release_gate_passed": len(records) == len(FRAMEWORKS)
        and all(record.get("comparison", {}).get("passed") is True for record in records),
    }


def _comparison_signature(record: dict[str, Any]) -> dict[str, Any]:
    comparison = cast(dict[str, Any], record["comparison"])
    checks = cast(list[dict[str, Any]], comparison["checks"])
    signature: dict[str, Any] = {}
    for check in checks:
        name = str(check["name"])
        if "expected_sha256" in check:
            signature[name] = {
                "count": check.get("expected_count"),
                "sha256": check["expected_sha256"],
            }
        else:
            signature[name] = check["canonical_expected"]
    return signature


def build_historical_report(
    current_record: dict[str, Any],
    historical_record: dict[str, Any],
    workload: ScaleWorkload = WORKLOAD,
) -> dict[str, Any]:
    """Build evidence that one historical target matches the accepted current target."""
    current_signature = _comparison_signature(current_record)
    historical_signature = _comparison_signature(historical_record)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "audit_trigger": {
            "ml4t_commit": "488c05d84e74156aba71c4f7ffac28df3881fea9",
            "cause": "vectorbt collateral semantics missing from the ML4T comparison profile",
            "external_fill_count": 390_369,
            "ml4t_fill_count": 418_825,
            "external_final_value": 716_785.408089,
            "ml4t_final_value": 1_162_222.354417,
        },
        "workload": {
            "recipe": asdict(workload),
            "recipe_sha256": _json_digest(asdict(workload)),
            "data_points": workload.data_points,
            "generator": "validation.benchmark_suite.generate_benchmark_data",
            "redistributable": True,
        },
        "current": current_record,
        "historical": historical_record,
        "framework_output_comparison": {
            "current": current_signature,
            "historical": historical_signature,
            "passed": current_signature == historical_signature,
        },
        "release_gate_passed": historical_record.get("comparison", {}).get("passed") is True
        and current_signature == historical_signature,
    }


@cache
def _expected_input_digest(workload: ScaleWorkload) -> str:
    config = workload.benchmark_config()
    prices, signals, dates = suite.generate_benchmark_data(config, seed=workload.seed)
    return input_digest(prices, signals, dates)


def report_failures(report: dict[str, Any], *, reconstruct_input: bool = False) -> list[str]:
    """Return every reason a large-scale candidate cannot be accepted."""
    failures: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        return [f"Unsupported large-scale schema: {report.get('schema_version')!r}"]
    raw_workload_value = report.get("workload")
    if not isinstance(raw_workload_value, dict):
        raw_workload: dict[str, Any] = {}
    else:
        raw_workload = cast(dict[str, Any], raw_workload_value)
    if raw_workload.get("recipe") != asdict(WORKLOAD):
        failures.append("Large-scale workload recipe differs")
    elif raw_workload.get("recipe_sha256") != _json_digest(asdict(WORKLOAD)):
        failures.append("Large-scale workload recipe digest differs")
    raw_records = report.get("frameworks")
    if not isinstance(raw_records, list):
        return failures + ["Large-scale framework records must be a list"]
    records: dict[str, dict[str, Any]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            continue
        record = cast(dict[str, Any], raw_record)
        framework_id = record.get("framework")
        if isinstance(framework_id, str):
            records[framework_id] = record
    if set(records) != set(FRAMEWORKS):
        failures.append("Large-scale framework coverage differs")
    manifest = load_framework_manifest()
    expected_sources = _source_digests()
    reconstructed = _expected_input_digest(WORKLOAD) if reconstruct_input else None
    raw_digests: set[str] = set()
    for framework in FRAMEWORKS:
        record = records.get(framework)
        if not isinstance(record, dict):
            continue
        target = record.get("target")
        expected_target = manifest.targets[framework].evidence_metadata()
        if (
            not isinstance(target, dict)
            or _declared_target_metadata(target) != expected_target
            or target.get("actual_version") != expected_target["version"]
        ):
            failures.append(f"{framework} target identity differs")
        elif framework == "vectorbt_pro" and (
            target.get("actual_commit") != expected_target["commit"]
            or target.get("actual_immutable_id") != expected_target["immutable_id"]
        ):
            failures.append(f"{framework} installed source identity differs")
        if record.get("source_digests") != expected_sources:
            failures.append(f"{framework} source digests differ")
        ml4t = record.get("ml4t")
        if not isinstance(ml4t, dict) or ml4t.get("dirty") is not False:
            failures.append(f"{framework} was not produced from a clean ML4T tree")
        elif not isinstance(ml4t.get("commit"), str) or len(str(ml4t["commit"])) != 40:
            failures.append(f"{framework} ML4T commit identity is missing")
        python = record.get("python")
        if not isinstance(python, dict) or not all(
            isinstance(python.get(key), str) for key in ("version", "implementation")
        ):
            failures.append(f"{framework} Python identity is missing")
        input_record = record.get("input")
        if not isinstance(input_record, dict) or not isinstance(
            input_record.get("raw_sha256"), str
        ):
            failures.append(f"{framework} input evidence is missing")
        else:
            raw_digest = str(input_record["raw_sha256"])
            raw_digests.add(raw_digest)
            if reconstructed is not None and raw_digest != reconstructed:
                failures.append(f"{framework} input digest does not reconstruct")
            effective_digest = input_record.get("effective_sha256")
            if not isinstance(effective_digest, str) or len(effective_digest) != 64:
                failures.append(f"{framework} effective-input digest is missing")
        capabilities = record.get("capabilities")
        if not isinstance(capabilities, dict) or set(capabilities) != {
            "intents",
            "fills",
            "closed_trades",
            "terminal_state",
            "fill_order",
        }:
            failures.append(f"{framework} capability declaration is incomplete")
        comparison = record.get("comparison")
        if not isinstance(comparison, dict) or comparison.get("passed") is not True:
            failures.append(f"{framework} exact comparison did not pass")
            continue
        checks = comparison.get("checks")
        if not isinstance(checks, list):
            failures.append(f"{framework} comparison checks are missing")
            continue
        names = {check.get("name") for check in checks if isinstance(check, dict)}
        expected_names = {
            "order_intents",
            "fills",
            "trades",
            "trade_count",
            "total_pnl",
            "final_value",
            "terminal_state",
        }
        if names != expected_names or any(
            not isinstance(check, dict) or check.get("passed") is not True for check in checks
        ):
            failures.append(f"{framework} comparison surface is incomplete or failed")
            continue
        for check in checks:
            assert isinstance(check, dict)
            name = check["name"]
            if name in {"order_intents", "fills", "trades", "terminal_state"}:
                expected_hash = check.get("expected_sha256")
                actual_hash = check.get("actual_sha256")
                if (
                    not isinstance(expected_hash, str)
                    or len(expected_hash) != 64
                    or expected_hash != actual_hash
                ):
                    failures.append(f"{framework} {name} digest comparison differs")
            elif check.get("canonical_expected") != check.get("canonical_actual"):
                failures.append(f"{framework} {name} canonical values differ")
    if len(raw_digests) > 1:
        failures.append("Large-scale framework inputs differ")
    if report.get("release_gate_passed") is not True:
        failures.append("Large-scale release gate did not pass")
    return failures


def historical_report_failures(
    report: dict[str, Any], *, reconstruct_input: bool = False
) -> list[str]:
    """Return every reason a historical replay candidate cannot be accepted."""
    failures: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        return [f"Unsupported historical replay schema: {report.get('schema_version')!r}"]
    workload = report.get("workload")
    if not isinstance(workload, dict) or workload.get("recipe") != asdict(WORKLOAD):
        failures.append("Historical replay workload recipe differs")
    elif workload.get("recipe_sha256") != _json_digest(asdict(WORKLOAD)):
        failures.append("Historical replay workload recipe digest differs")
    current = report.get("current")
    historical = report.get("historical")
    if not isinstance(current, dict) or not isinstance(historical, dict):
        return failures + ["Historical replay records are missing"]
    manifest = load_framework_manifest()
    current_target = manifest.targets["vectorbt_pro"].evidence_metadata()
    historical_target = _historical_vectorbt_pro_target().evidence_metadata()
    for label, record, expected_target in (
        ("current", current, current_target),
        ("historical", historical, historical_target),
    ):
        target = record.get("target")
        if (
            not isinstance(target, dict)
            or _declared_target_metadata(target) != expected_target
            or target.get("actual_version") != expected_target["version"]
        ):
            failures.append(f"VectorBT Pro {label} target identity differs")
        elif (
            target.get("actual_commit") != expected_target["commit"]
            or target.get("actual_immutable_id") != expected_target["immutable_id"]
        ):
            failures.append(f"VectorBT Pro {label} installed source identity differs")
        if record.get("source_digests") != _source_digests():
            failures.append(f"VectorBT Pro {label} source digests differ")
        ml4t = record.get("ml4t")
        if not isinstance(ml4t, dict) or ml4t.get("dirty") is not False:
            failures.append(f"VectorBT Pro {label} run used a dirty ML4T tree")
        elif not isinstance(ml4t.get("commit"), str) or len(str(ml4t["commit"])) != 40:
            failures.append(f"VectorBT Pro {label} ML4T commit identity is missing")
        comparison = record.get("comparison")
        if not isinstance(comparison, dict) or comparison.get("passed") is not True:
            failures.append(f"VectorBT Pro {label} exact comparison did not pass")
        input_record = record.get("input")
        if not isinstance(input_record, dict):
            failures.append(f"VectorBT Pro {label} input evidence is missing")
        elif reconstruct_input and input_record.get("raw_sha256") != _expected_input_digest(
            WORKLOAD
        ):
            failures.append(f"VectorBT Pro {label} input digest does not reconstruct")
    if current.get("source_digests") != historical.get("source_digests"):
        failures.append("Historical replay runs used different comparison sources")
    if current.get("input") != historical.get("input"):
        failures.append("Historical replay inputs differ")
    output_comparison = report.get("framework_output_comparison")
    expected_output_comparison = {
        "current": _comparison_signature(current),
        "historical": _comparison_signature(historical),
        "passed": _comparison_signature(current) == _comparison_signature(historical),
    }
    if output_comparison != expected_output_comparison or not expected_output_comparison["passed"]:
        failures.append("Historical and current VectorBT Pro outputs differ")
    if report.get("release_gate_passed") is not True:
        failures.append("Historical replay release gate did not pass")
    return failures


def _resolve_python(framework: str) -> Path:
    target = load_framework_manifest().targets[framework]
    override = os.getenv(target.python_env_var or "")
    if override:
        return Path(override).expanduser().resolve()
    if target.environment is None:
        raise ValueError(f"{framework} has no environment")
    return PROJECT_ROOT / target.environment / "bin" / "python"


def _resolve_historical_vectorbt_pro_python() -> Path:
    override = os.getenv(HISTORICAL_VECTORBT_PRO_PYTHON_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / HISTORICAL_VECTORBT_PRO_ENV / "bin" / "python"


def _run_isolated_worker(
    framework: str,
    interpreter: Path,
    *,
    historical_target: bool = False,
) -> dict[str, Any]:
    if not interpreter.is_file():
        raise RuntimeError(f"Missing {framework} interpreter: {interpreter}")
    with tempfile.TemporaryDirectory(prefix=f"ml4t-scale-{framework}-") as directory:
        output = Path(directory) / "record.json"
        command = [
            str(interpreter),
            str(Path(__file__).resolve()),
            "--worker",
            framework,
            "--output",
            str(output),
        ]
        if historical_target:
            command.append("--historical-target")
        environment = os.environ.copy()
        paths = [str(PROJECT_ROOT / "src"), str(VALIDATION_DIR)]
        sibling_specs = PROJECT_ROOT.parent / "ml4t-specs" / "src"
        if sibling_specs.is_dir():
            paths.append(str(sibling_specs))
        environment["PYTHONPATH"] = os.pathsep.join(
            paths + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
        )
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=3_600,
            check=False,
        )
        record: dict[str, Any] | None = None
        if output.is_file():
            try:
                loaded = json.loads(output.read_text(encoding="utf-8"))
                record = cast(dict[str, Any], loaded) if isinstance(loaded, dict) else None
            except json.JSONDecodeError:
                record = None
        if record is not None and (
            completed.returncode != 0 or record.get("comparison", {}).get("passed") is not True
        ):
            diagnostic_path = CANDIDATE_DIR / f"{framework}.large_scale.candidate.json"
            _write_json_atomic(diagnostic_path, record)
            raise RuntimeError(
                f"{framework} large-scale comparison failed; diagnostic retained at "
                f"{diagnostic_path}: " + json.dumps(record.get("comparison"), indent=2)[-20_000:]
            )
        if completed.returncode != 0 or record is None:
            details = "\n".join(
                stream.strip() for stream in (completed.stdout, completed.stderr) if stream.strip()
            )
            raise RuntimeError(
                f"{framework} scale worker exited {completed.returncode}: {details[-20_000:]}"
            )
        return record


def run_all() -> dict[str, Any]:
    """Run every framework in its isolated environment and return a candidate report."""
    records: list[dict[str, Any]] = []
    for framework in FRAMEWORKS:
        interpreter = _resolve_python(framework)
        print(f"Running {framework} large-scale pair...", flush=True)
        records.append(_run_isolated_worker(framework, interpreter))
        print(f"{framework}: PASS", flush=True)
    return build_report(records)


def run_historical_vectorbt_pro() -> dict[str, Any]:
    """Replay the historical Pro target and compare it with accepted current evidence."""
    current_report = cast(dict[str, Any], json.loads(ACCEPTED_PATH.read_text(encoding="utf-8")))
    current_failures = report_failures(current_report, reconstruct_input=True)
    if current_failures:
        raise RuntimeError(
            "Current large-scale evidence is not valid: " + "; ".join(current_failures)
        )
    current_record = next(
        record
        for record in cast(list[dict[str, Any]], current_report["frameworks"])
        if record["framework"] == "vectorbt_pro"
    )
    historical_record = _run_isolated_worker(
        "vectorbt_pro",
        _resolve_historical_vectorbt_pro_python(),
        historical_target=True,
    )
    return build_historical_report(current_record, historical_record)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Replace accepted evidence only after a complete temporary write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=FRAMEWORKS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-input", action="store_true")
    parser.add_argument("--historical-target", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--historical-vectorbt-pro", action="store_true")
    parser.add_argument("--check-historical-vectorbt-pro", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.output is None:
            parser.error("--worker requires --output")
        if args.historical_target and args.worker != "vectorbt_pro":
            parser.error("--historical-target requires --worker vectorbt_pro")
        target = _historical_vectorbt_pro_target() if args.historical_target else None
        record = run_worker(args.worker, target_override=target)
        _write_json(args.output, record)
        return 0 if record["comparison"]["passed"] else 1
    if args.historical_target:
        parser.error("--historical-target requires --worker vectorbt_pro")
    if args.check_historical_vectorbt_pro:
        report = json.loads(HISTORICAL_ACCEPTED_PATH.read_text(encoding="utf-8"))
        failures = historical_report_failures(report, reconstruct_input=args.verify_input)
        for failure in failures:
            print(f"- {failure}")
        return 1 if failures else 0
    if args.historical_vectorbt_pro:
        candidate = run_historical_vectorbt_pro()
        _write_json(HISTORICAL_CANDIDATE_PATH, candidate)
        failures = historical_report_failures(candidate, reconstruct_input=True)
        if failures:
            print("Accepted historical evidence unchanged:")
            for failure in failures:
                print(f"- {failure}")
            return 1
        _write_json_atomic(HISTORICAL_ACCEPTED_PATH, candidate)
        print(f"Accepted historical evidence: {HISTORICAL_ACCEPTED_PATH}")
        return 0
    if args.check:
        report = json.loads(ACCEPTED_PATH.read_text(encoding="utf-8"))
        failures = report_failures(report, reconstruct_input=args.verify_input)
        for failure in failures:
            print(f"- {failure}")
        return 1 if failures else 0
    candidate = run_all()
    _write_json(CANDIDATE_PATH, candidate)
    failures = report_failures(candidate, reconstruct_input=True)
    if failures:
        print("Accepted evidence unchanged:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    _write_json_atomic(ACCEPTED_PATH, candidate)
    print(f"Accepted evidence: {ACCEPTED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
