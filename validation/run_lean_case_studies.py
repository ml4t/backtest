#!/usr/bin/env python3
"""Run and retain fresh LEAN case-study parity evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import math
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
VALIDATION_ROOT = PROJECT_ROOT / "validation"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(VALIDATION_ROOT))

from common.framework_registry import load_framework_manifest  # noqa: E402

from ml4t.backtest._validation.case_study_lean import (  # noqa: E402
    RETAINED_ORDER_EVENT_ARTIFACTS,
    compare,
    lean_side,
    run_ml4t_lean,
)
from ml4t.backtest._validation.lean_runner import (  # noqa: E402
    check_lean_cli,
    make_lean_env,
    run_lean_backtest,
)

WORKSPACE = VALIDATION_ROOT / "lean" / "workspace"
DATA_DAILY = WORKSPACE / "data" / "equity" / "usa" / "daily"
SUPPORT = VALIDATION_ROOT / "lean" / "support"
CASE_STUDIES = (
    "chapter16_etfs",
    "chapter16_sp500_equity_option_analytics",
    "chapter16_us_equities_panel",
)
PROJECT_INPUTS = {
    "asset_symbols.csv",
    "config.json",
    "main.py",
    "ml4t_symbol_map.json",
    "rebalance_dates.csv",
    "weights.csv",
    "weights.csv.xz",
    "weights.csv.gz",
}


def _digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(path: Path) -> str:
    return _digest_bytes(path.read_bytes())


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _copy_project(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(PROJECT_INPUTS):
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


def _prepare_lean_root(root: Path, project: Path) -> Path:
    lean_config = root / "lean.json"
    shutil.copy2(SUPPORT / "lean.json", lean_config)
    data_root = root / "data"
    shutil.copytree(SUPPORT / "data", data_root)
    daily = data_root / "equity/usa/daily"
    maps = data_root / "equity/usa/map_files"
    factors = data_root / "equity/usa/factor_files"
    for directory in (daily, maps, factors):
        directory.mkdir(parents=True, exist_ok=True)

    symbol_map = json.loads((project / "ml4t_symbol_map.json").read_text(encoding="utf-8"))
    for ticker in sorted(symbol_map):
        ticker_lower = ticker.lower()
        source = DATA_DAILY / f"{ticker_lower}.zip"
        destination = daily / source.name
        os.link(source, destination)
        with zipfile.ZipFile(source) as archive:
            first_line = archive.read(archive.namelist()[0]).decode().splitlines()[0]
        first_date = first_line.split(",", maxsplit=1)[0].split()[0]
        (maps / f"{ticker_lower}.csv").write_text(
            f"{first_date},{ticker_lower}\n20501231,{ticker_lower}\n",
            encoding="utf-8",
        )
        (factors / f"{ticker_lower}.csv").write_text(
            f"{first_date},1,1,1\n20501231,1,1,0\n",
            encoding="utf-8",
        )
    return lean_config


def _surface_checksum(surface: Any) -> str:
    rows = surface.to_dict(orient="records")
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str).encode()
    return _digest_bytes(payload)


def _input_manifest(project: Path) -> dict[str, str]:
    inputs = {
        path.name: _digest(path)
        for path in sorted(project.iterdir())
        if path.is_file() and path.name in PROJECT_INPUTS
    }
    symbol_map = json.loads((project / "ml4t_symbol_map.json").read_text(encoding="utf-8"))
    for ticker in sorted(symbol_map):
        data_path = DATA_DAILY / f"{ticker.lower()}.zip"
        if not data_path.is_file():
            raise FileNotFoundError(data_path)
        inputs[f"data/{data_path.name}"] = _digest(data_path)
    return inputs


def _retained_order_paths(project: Path) -> list[Path]:
    names = RETAINED_ORDER_EVENT_ARTIFACTS.get(project.name)
    if names is None:
        raise ValueError(f"No retained order-event layout for {project.name}")
    paths = [project / name for name in names]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Retained order-event artifact missing for {project.name}")
    return paths


def _encode_order_parts(raw: bytes, targets: list[Path]) -> list[bytes]:
    if len(targets) == 1:
        suffix = targets[0].suffix
        if suffix == ".xz":
            return [lzma.compress(raw)]
        if suffix == ".gz":
            return [gzip.compress(raw, mtime=0)]
        return [raw]

    lines = raw.splitlines(keepends=True)
    header, rows = lines[0], lines[1:]
    chunk_size = math.ceil(len(rows) / len(targets))
    chunks = [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]
    if len(chunks) != len(targets):
        raise RuntimeError(f"Could not split order events into {len(targets)} retained parts")
    return [lzma.compress(header + b"".join(chunk)) for chunk in chunks]


def _promote_case(project: Path, order_events: bytes, daily_equity: bytes) -> dict[str, str]:
    order_paths = _retained_order_paths(project)
    encoded = _encode_order_parts(order_events, order_paths)
    for path, payload in zip(order_paths, encoded, strict=True):
        _atomic_write(path, payload)
    equity_path = project / "ml4t_daily_equity.csv"
    _atomic_write(equity_path, daily_equity)
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): _digest(path)
        for path in [*order_paths, equity_path]
    }


def run(lean_command: Path, *, promote: bool) -> dict[str, Any]:
    manifest = load_framework_manifest()
    target = manifest.targets["lean"]
    if target.artifact is None:
        raise RuntimeError("Frozen LEAN target does not define an engine image")

    lean_command = lean_command.resolve()
    env = make_lean_env()
    cli_version = check_lean_cli([str(lean_command)], PROJECT_ROOT, env)
    expected_cli = f"lean {target.cli_version}"
    if cli_version != expected_cli:
        raise RuntimeError(f"LEAN CLI differs: {cli_version} != {expected_cli}")

    results: list[dict[str, Any]] = []
    raw_artifacts: dict[str, tuple[bytes, bytes]] = {}
    for case in CASE_STUDIES:
        source = WORKSPACE / case
        with tempfile.TemporaryDirectory(
            prefix=f".ml4t-{case}-", dir=VALIDATION_ROOT / "lean"
        ) as temporary:
            temporary_root = Path(temporary)
            project = temporary_root / case
            _copy_project(source, project)
            lean_config = _prepare_lean_root(temporary_root, project)
            output = project / "backtests" / "fresh"
            runtime = run_lean_backtest(
                lean_cmd=[str(lean_command)],
                cwd=PROJECT_ROOT,
                project_dir=project,
                lean_config=lean_config,
                output_dir=output,
                image=target.artifact,
                env=env,
            )
            lean = lean_side(project)
            ml4t = run_ml4t_lean(project, DATA_DAILY)
            comparison = compare(lean, ml4t)
            order_events = (project / "ml4t_order_events.csv").read_bytes().replace(b"\r\n", b"\n")
            daily_equity = (project / "ml4t_daily_equity.csv").read_bytes()
            summaries = sorted(output.glob("*-summary.json"))
            if not summaries:
                raise FileNotFoundError(f"LEAN summary missing for {case}")
            passed = (
                comparison["sorted_fill_multiset_match"]
                and comparison["fill_gap"] == 0
                and comparison["canonical_final_value_match"]
            )
            results.append(
                {
                    "case": case,
                    "comparison": comparison,
                    "inputs": _input_manifest(source),
                    "lean_fill_surface_sha256": _surface_checksum(lean["fills"]),
                    "ml4t_fill_surface_sha256": _surface_checksum(ml4t["fills"]),
                    "raw_daily_equity_sha256": _digest_bytes(daily_equity),
                    "raw_order_events_sha256": _digest_bytes(order_events),
                    "runtime_seconds": runtime,
                    "summary_sha256": _digest(summaries[-1]),
                    "passed": passed,
                }
            )
            raw_artifacts[case] = (order_events, daily_equity)

    passed = all(result["passed"] for result in results)
    if promote and passed:
        for result in results:
            case = result["case"]
            result["retained_artifacts"] = _promote_case(
                WORKSPACE / case,
                *raw_artifacts[case],
            )

    producer_files = [
        Path(__file__),
        PROJECT_ROOT / "src/ml4t/backtest/_validation/case_study_lean.py",
        PROJECT_ROOT / "src/ml4t/backtest/_validation/lean_runner.py",
        PROJECT_ROOT / "src/ml4t/backtest/profiles.py",
    ]
    return {
        "schema_version": 1,
        "framework": target.evidence_metadata(),
        "cli_observed": cli_version,
        "comparison_protocol": {
            "account": "DefaultBrokerageModel Margin; per-project leverage 2",
            "costs": "Ml4tPercentFeeModel at project rate; ConstantSlippageModel(0)",
            "fills": "timestamp, asset, side, quantity, price rounded to 4 decimals",
            "terminal_value_quantum": "0.0001 USD",
        },
        "producer_files": {
            path.relative_to(PROJECT_ROOT).as_posix(): _digest(path) for path in producer_files
        },
        "support_files": {
            path.relative_to(SUPPORT).as_posix(): _digest(path)
            for path in sorted(SUPPORT.rglob("*"))
            if path.is_file()
        },
        "promoted": bool(promote and passed),
        "cases": results,
        "passed": passed,
    }


def _write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    _atomic_write(path, (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-command", type=Path, default=PROJECT_ROOT / ".venv-lean/bin/lean")
    parser.add_argument(
        "--output",
        type=Path,
        default=VALIDATION_ROOT / "lean/case_study_evidence.json",
    )
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    try:
        evidence = run(args.lean_command, promote=args.promote)
    except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
        print(f"LEAN case-study run failed: {error}", file=sys.stderr)
        return 2
    if not evidence["passed"]:
        candidate = args.output.with_suffix(".candidate.json")
        _write_evidence(candidate, evidence)
        failed = [case["case"] for case in evidence["cases"] if not case["passed"]]
        print(f"LEAN case-study parity differs: {failed}; candidate: {candidate}", file=sys.stderr)
        return 1
    _write_evidence(args.output, evidence)
    print(f"LEAN case-study parity passed: {len(evidence['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
