#!/usr/bin/env python3
"""Inventory the current canonical real-strategy corpus without rerunning a backtest."""

from __future__ import annotations

import argparse
import dataclasses
import enum
import gc
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
        "warmup",
        "timestamp_shift_hours",
        "target_scale",
        "contract_specs",
        "funding",
    }
    case_ids: list[str] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not required_keys <= case.keys():
            raise ValueError(f"Invalid real-strategy case contract record {index}")
        if case["production_path"] not in {"event_driven", "vectorized"}:
            raise ValueError(f"Invalid production path for {case['id']}")
        if not isinstance(case["required_backtest_artifacts"], list):
            raise ValueError(f"Invalid artifact list for {case['id']}")
        target_scale = float(case["target_scale"])
        if not 0.0 < target_scale <= 1.0:
            raise ValueError(f"Invalid target scale for {case['id']}: {target_scale}")
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
        "comparison_target_scale": float(case["target_scale"]),
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


def _json_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _json_value(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _sanitize_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(spec, default=str))
    sanitized.pop("_runtime_backtest_config", None)
    metadata = sanitized.get("backtest_config", {}).get("metadata", {})
    metadata.pop("preset_path", None)
    return sanitized


def _frame_identity(frame: pl.DataFrame) -> dict[str, object]:
    return {
        "rows": frame.height,
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
    }


def align_funding_to_market_events(funding: pl.DataFrame, market: pl.DataFrame) -> pl.DataFrame:
    """Retain funding settlements that the frozen engine timeline can present."""
    market_timestamps = market.select("timestamp").unique()
    if funding.schema["timestamp"] != market_timestamps.schema["timestamp"]:
        funding = funding.with_columns(
            pl.col("timestamp").cast(market_timestamps.schema["timestamp"])
        )
    return funding.join(market_timestamps, on="timestamp", how="semi").sort("timestamp", "symbol")


def align_targets_to_engine_schedule(targets: pl.DataFrame, schedule: pl.Series) -> pl.DataFrame:
    """Retain target rows that the production engine schedule can submit."""
    schedule_frame = schedule.alias("timestamp").to_frame().unique()
    if targets.schema["timestamp"] != schedule_frame.schema["timestamp"]:
        schedule_frame = schedule_frame.with_columns(
            pl.col("timestamp").cast(targets.schema["timestamp"])
        )
    return targets.join(schedule_frame, on="timestamp", how="semi").sort("timestamp", "symbol")


def apply_comparison_target_scale(
    targets: pl.DataFrame,
    spec: Mapping[str, Any],
    target_scale: float,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Apply declared execution headroom without changing relative model allocations."""
    scaled_spec = json.loads(json.dumps(spec, default=str))
    if target_scale == 1.0:
        return targets, scaled_spec
    scaled_spec["backtest_config"].setdefault("metadata", {})["comparison_target_scale"] = (
        target_scale
    )
    return targets.with_columns((pl.col("weight") * target_scale).alias("weight")), scaled_spec


def _bundle_digest(
    *,
    selection: Mapping[str, Any],
    source_prediction_sha256: str,
    spec: Mapping[str, Any],
    frames: Mapping[str, pl.DataFrame],
    contracts: Mapping[str, object] | None,
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "selection": selection,
        "source_prediction_sha256": source_prediction_sha256,
        "spec": spec,
        "contracts": contracts,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str).encode())
    for name, frame in sorted(frames.items()):
        digest.update(name.encode())
        digest.update(frame.serialize(format="binary"))
    return digest.hexdigest()


def write_bundle(
    *,
    case_id: str,
    selection: Mapping[str, Any],
    source_prediction_sha256: str,
    spec: Mapping[str, Any],
    market: pl.DataFrame,
    targets: pl.DataFrame,
    output_root: Path,
    funding: pl.DataFrame | None = None,
    contracts: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write one content-addressed engine-input bundle and return its manifest."""
    frames = {"market.parquet": market, "targets.parquet": targets}
    if funding is not None:
        frames["funding.parquet"] = funding
    normalized_contracts = cast(Mapping[str, object] | None, _json_value(contracts))
    sanitized_spec = _sanitize_spec(spec)
    digest = _bundle_digest(
        selection=selection,
        source_prediction_sha256=source_prediction_sha256,
        spec=sanitized_spec,
        frames=frames,
        contracts=normalized_contracts,
    )
    destination = output_root / case_id / digest
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "case_study": case_id,
        "bundle_sha256": digest,
        "selection": dict(selection),
        "source_prediction_sha256": source_prediction_sha256,
        "files": {},
    }
    if destination.is_dir():
        retained = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if retained.get("bundle_sha256") != digest:
            raise ValueError(f"Retained {case_id} bundle has the wrong identity")
        return retained

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{case_id}.", dir=destination.parent))
    try:
        for name, frame in frames.items():
            frame.write_parquet(staging / name, compression="zstd", statistics=True)
        (staging / "spec.json").write_text(
            json.dumps(sanitized_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if normalized_contracts is not None:
            (staging / "contracts.json").write_text(
                json.dumps(normalized_contracts, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        files = {
            path.name: _artifact_identity(path)
            for path in sorted(staging.iterdir())
            if path.name != "manifest.json"
        }
        manifest["files"] = files
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
    finally:
        if staging.exists():
            for path in staging.iterdir():
                path.unlink()
            staging.rmdir()
    return manifest


def materialize_bundles(
    *,
    corpus_report: Mapping[str, object],
    artifact_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Precompute engine inputs for every selected production strategy."""
    loaders = importlib.import_module("case_studies.utils.backtest_loaders")
    runner = importlib.import_module("case_studies.utils.backtest_runner")
    registry = importlib.import_module("case_studies.utils.registry")
    cv_window = importlib.import_module("case_studies.utils.cv_window")
    contract_by_id = {case["id"]: case for case in load_case_contract()}
    records = cast(list[dict[str, Any]], corpus_report["records"])
    manifests: list[dict[str, object]] = []
    for record in records:
        case_id = str(record["case_study"])
        print(f"Preparing {case_id}", flush=True)
        contract = contract_by_id[case_id]
        selection = cast(dict[str, Any], record["selection"])
        label = str(selection["label"])
        prediction_hash = str(selection["val_prediction_hash"])
        backtest_hash = str(selection["val_backtest_hash"])
        spec_path = (
            artifact_root
            / "case_studies"
            / case_id
            / "run_log"
            / "backtest"
            / backtest_hash
            / "spec.json"
        )
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        warmup = int(loaders.warmup_periods_for(case_id)) if contract["warmup"] else 0
        market = loaders.load_backtest_prices_for(
            case_id,
            label,
            split="validation",
            warmup_periods=warmup,
            max_symbols=0,
        )
        predictions = registry.read_predictions(case_id, prediction_hash)
        shift_hours = int(contract["timestamp_shift_hours"])
        if shift_hours:
            window = cv_window.canonical_window(case_id, label, split="validation")
            if window is None:
                raise ValueError(f"{case_id} lacks a canonical validation window")
            validation_end = window[1]
            market = market.with_columns(
                pl.col("timestamp") + pl.duration(hours=shift_hours)
            ).filter(pl.col("timestamp").dt.date() <= validation_end)
            predictions = predictions.with_columns(
                pl.col("timestamp") + pl.duration(hours=shift_hours)
            ).filter(pl.col("timestamp").dt.date() <= validation_end)
        targets = runner.precompute_weights(
            predictions,
            spec,
            market,
            label=label,
            case_study=case_id,
            prediction_hash=prediction_hash,
        ).sort("timestamp", "symbol")
        targets, spec = apply_comparison_target_scale(
            targets,
            spec,
            float(contract["target_scale"]),
        )
        cadence = spec["strategy"]["rebalance"]["cadence"]
        calendar = spec["backtest_config"]["calendar"]["calendar"]
        schedule = loaders.resolve_rebalance_timestamps(
            pl.Series("timestamp", predictions["timestamp"].unique().sort().to_list()),
            cadence,
            calendar,
        )
        step = int(loaders.get_rebalance_step(case_id, label))
        if step > 1:
            schedule = schedule.gather_every(step)
        targets = align_targets_to_engine_schedule(targets, schedule)
        market = market.sort("timestamp", "symbol")
        funding = None
        if contract["funding"]:
            funding_module = importlib.import_module(
                "case_studies.crypto_perps_funding.funding_data"
            )
            funding = align_funding_to_market_events(
                funding_module.load_funding_rates(symbols=market["symbol"].unique().to_list()),
                market,
            )
        contracts = None
        if contract["contract_specs"]:
            contracts = loaders.load_contract_specs_from_yaml()
        prediction_identity = cast(dict[str, Any], record["inputs"])["predictions.parquet"]
        bundle = write_bundle(
            case_id=case_id,
            selection=selection,
            source_prediction_sha256=str(prediction_identity["sha256"]),
            spec=spec,
            market=market,
            targets=targets,
            output_root=output_root,
            funding=funding,
            contracts=contracts,
        )
        manifests.append(bundle)
        print(
            f"Prepared {case_id}: {market.height:,} market rows, "
            f"{targets.height:,} targets, {str(bundle['bundle_sha256'])[:12]}",
            flush=True,
        )
        del market, predictions, targets, funding
        gc.collect()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "bundle_root": {"retained_absolute_path": False},
        "records": manifests,
    }


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
    parser.add_argument("--materialize", type=Path)
    args = parser.parse_args()
    public_root = args.public_root.resolve()
    artifact_root = args.artifact_root.resolve()
    output = args.output.resolve()
    materialize_root = args.materialize.resolve() if args.materialize is not None else None
    resolver = _load_public_resolver(public_root)
    report = build_report(
        public_root=public_root,
        artifact_root=artifact_root,
        resolver=resolver,
    )
    write_report(report, output)
    if materialize_root is not None:
        bundle_report = materialize_bundles(
            corpus_report=report,
            artifact_root=artifact_root,
            output_root=materialize_root,
        )
        bundle_output = output.with_name("REAL_STRATEGY_INPUTS.candidate.json")
        write_report(bundle_report, bundle_output)
        bundle_records = cast(list[dict[str, object]], bundle_report["records"])
        print(f"Materialized {len(bundle_records)} engine input bundles")
    summary = cast(Mapping[str, object], report["summary"])
    print(f"Inventoried {summary['case_studies']} current case studies to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
