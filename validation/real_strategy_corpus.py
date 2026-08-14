#!/usr/bin/env python3
"""Inventory the current canonical real-strategy corpus without rerunning a backtest."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

VALIDATION_DIR = Path(__file__).parent
PROJECT_ROOT = VALIDATION_DIR.parent
DEFAULT_CASES_PATH = VALIDATION_DIR / "real_strategy_cases.toml"
SCHEMA_VERSION = 1


def file_digest(path: Path) -> str:
    """Return a streaming SHA-256 digest for an artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_identity(root: Path) -> dict[str, object]:
    """Return the full commit and tracked-file status for a repository."""
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def load_case_contract(path: Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    """Load and validate the bounded real-strategy corpus contract."""
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported real-strategy case contract schema")
    cases = payload.get("case")
    if not isinstance(cases, list):
        raise ValueError("Real-strategy case contract must contain [[case]] records")
    required_keys = {
        "id",
        "asset_class",
        "production_path",
        "required_backtest_artifacts",
    }
    case_ids: list[str] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not required_keys <= case.keys():
            raise ValueError(f"Invalid real-strategy case contract record {index}")
        if case["production_path"] not in {"event_driven", "vectorized"}:
            raise ValueError(f"Invalid production path for {case['id']}")
        if not isinstance(case["required_backtest_artifacts"], list):
            raise ValueError(f"Invalid artifact list for {case['id']}")
        case_ids.append(str(case["id"]))
    if not 3 <= len(case_ids) <= 5 or len(set(case_ids)) != len(case_ids):
        raise ValueError("Real-strategy corpus must contain three to five unique case studies")
    return cases


def _parquet_identity(path: Path) -> dict[str, object]:
    schema = pl.scan_parquet(path).collect_schema()
    count_frame = cast(
        pl.DataFrame,
        pl.scan_parquet(path).select(pl.len().alias("row_count")).collect(engine="streaming"),
    )
    row_count = int(count_frame["row_count"][0])
    return {
        "sha256": file_digest(path),
        "bytes": path.stat().st_size,
        "rows": int(row_count),
        "schema": {name: str(dtype) for name, dtype in schema.items()},
    }


def _artifact_identity(path: Path) -> dict[str, object]:
    identity: dict[str, object] = {
        "sha256": file_digest(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".parquet":
        identity.update(_parquet_identity(path))
    return identity


def _strategy_summary(spec: Mapping[str, Any]) -> dict[str, object]:
    strategy = spec.get("strategy")
    backtest = spec.get("backtest_config")
    if not isinstance(strategy, Mapping) or not isinstance(backtest, Mapping):
        raise ValueError("Selected backtest spec lacks strategy or backtest_config")
    execution = backtest.get("execution")
    commission = backtest.get("commission")
    slippage = backtest.get("slippage")
    sizing = backtest.get("position_sizing")
    rebalance = strategy.get("rebalance")
    risk = strategy.get("risk")
    signal = strategy.get("signal")
    allocation = strategy.get("allocation")
    return {
        "signal": signal,
        "allocation": allocation,
        "risk": risk,
        "rebalance": rebalance,
        "execution": execution,
        "commission": commission,
        "slippage": slippage,
        "position_sizing": sizing,
    }


def build_case_record(
    case: Mapping[str, Any],
    lineage: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, object]:
    """Build one immutable inventory record from a resolved canonical lineage."""
    case_id = str(case["id"])
    backtest_hash = str(lineage["val_backtest_hash"])
    prediction_hash = str(lineage["val_prediction_hash"])
    case_root = artifact_root / "case_studies" / case_id / "run_log"
    backtest_dir = case_root / "backtest" / backtest_hash
    prediction_dir = case_root / "predictions" / prediction_hash
    registry_path = case_root / "registry.db"
    prediction_path = prediction_dir / "predictions.parquet"
    required_paths = [backtest_dir / str(name) for name in case["required_backtest_artifacts"]]
    inputs = [registry_path, prediction_path, *required_paths]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        relative = [path.relative_to(artifact_root).as_posix() for path in missing]
        raise FileNotFoundError(f"{case_id} lacks required artifacts: {relative}")

    spec_path = backtest_dir / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if (
        spec.get("backtest_config", {}).get("metadata", {}).get("prediction_hash")
        != prediction_hash
    ):
        raise ValueError(f"{case_id} selected spec does not name selected prediction")
    recorded_mode = spec.get("strategy", {}).get("rebalance", {}).get("mode")
    expected_mode = "engine" if case["production_path"] == "event_driven" else "vectorized"
    if recorded_mode != expected_mode:
        raise ValueError(
            f"{case_id} production path differs: expected {expected_mode}, got {recorded_mode}"
        )

    artifacts = {
        path.name: _artifact_identity(path)
        for path in sorted(required_paths, key=lambda value: value.name)
    }
    return {
        "case_study": case_id,
        "asset_class": case["asset_class"],
        "production_path": case["production_path"],
        "selection": dict(lineage),
        "strategy": _strategy_summary(spec),
        "inputs": {
            "registry": _artifact_identity(registry_path),
            "predictions.parquet": _artifact_identity(prediction_path),
        },
        "backtest_artifacts": artifacts,
    }


def build_report(
    *,
    public_root: Path,
    artifact_root: Path,
    resolver: Callable[[str], Mapping[str, Any]],
    cases_path: Path = DEFAULT_CASES_PATH,
) -> dict[str, object]:
    """Build a complete current corpus inventory using the publication resolver."""
    cases = load_case_contract(cases_path)
    records = [
        build_case_record(case, resolver(str(case["id"])), artifact_root=artifact_root)
        for case in cases
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_contract": (
            "current canonical validation rank-1 resolved by "
            "case_studies.utils.strategy_analysis.resolve_canonical_rank1_lineage"
        ),
        "repositories": {
            "ml4t_backtest": git_identity(PROJECT_ROOT),
            "publication": git_identity(public_root),
        },
        "artifact_root": {
            "kind": "external_case_study_run_log",
            "retained_absolute_path": False,
        },
        "summary": {
            "case_studies": len(records),
            "event_driven": sum(r["production_path"] == "event_driven" for r in records),
            "vectorized": sum(r["production_path"] == "vectorized" for r in records),
        },
        "records": records,
    }


def write_report(report: Mapping[str, object], output: Path) -> None:
    """Atomically write a deterministic JSON candidate."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _load_public_resolver(public_root: Path) -> Callable[[str], Mapping[str, Any]]:
    sys.path.insert(0, str(public_root))
    os.chdir(public_root)
    module = importlib.import_module("case_studies.utils.strategy_analysis")
    return cast(Callable[[str], Mapping[str, Any]], module.resolve_canonical_rank1_lineage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=VALIDATION_DIR / "candidates" / "REAL_STRATEGY_CORPUS.candidate.json",
    )
    args = parser.parse_args()
    public_root = args.public_root.resolve()
    artifact_root = args.artifact_root.resolve()
    resolver = _load_public_resolver(public_root)
    report = build_report(
        public_root=public_root,
        artifact_root=artifact_root,
        resolver=resolver,
    )
    write_report(report, args.output)
    summary = cast(Mapping[str, object], report["summary"])
    print(f"Inventoried {summary['case_studies']} current case studies to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
