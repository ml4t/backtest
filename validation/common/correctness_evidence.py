"""Schema validation, staleness checks, and atomic correctness-evidence promotion."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from scenarios.definitions import SCENARIOS

from common.capabilities import FRAMEWORK_CAPABILITIES, ML4T_CAPABILITIES
from common.comparator import CANONICAL_QUANTUM_TEXT, compare_results
from common.framework_registry import FrameworkManifest, load_framework_manifest
from common.provenance import generate_inputs, input_digest, static_digests
from common.types import ValidationRecord, ValidationStatus

SCHEMA_VERSION = 3


def _framework_metadata(manifest: FrameworkManifest) -> dict[str, dict[str, object]]:
    return {
        framework_id: manifest.targets[framework_id].evidence_metadata()
        for framework_id in manifest.scenario_framework_ids
    }


def build_report(
    records: list[ValidationRecord],
    *,
    manifest: FrameworkManifest | None = None,
) -> dict[str, object]:
    """Build a correctness candidate without deciding whether it can be accepted."""
    target_manifest = manifest or load_framework_manifest()
    counts = Counter(record.status.value for record in records)
    required = [record for record in records if record.required]
    release_passed = bool(required) and all(record.passed for record in required)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "comparison_policy": {
            "canonical_quantum": CANONICAL_QUANTUM_TEXT,
            "rounding": "ROUND_HALF_EVEN",
            "meaning": "equality after canonical quantization, not bit identity",
            "record_order": "significant",
            "timestamp_domain": "daily session date",
            "surfaces": ["terminal", "closed_trades", "fills"],
        },
        "frameworks": _framework_metadata(target_manifest),
        "release_gate_passed": release_passed,
        "summary": {status.value: counts[status.value] for status in ValidationStatus},
        "records": [record.to_dict() for record in records],
    }


def _mapping(value: object, *, label: str, failures: list[str]) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        failures.append(f"{label} must be an object")
        return None
    return cast(dict[str, Any], value)


def _expected_pairs(manifest: FrameworkManifest) -> dict[tuple[str, str], tuple[bool, str]]:
    return {
        (framework, scenario_id): (
            framework in scenario.supported_frameworks,
            scenario.name,
        )
        for framework in manifest.scenario_framework_ids
        for scenario_id, scenario in SCENARIOS.items()
    }


def _record_evidence_failures(
    record: ValidationRecord,
    *,
    manifest: FrameworkManifest,
) -> list[str]:
    identity = f"{record.framework}/{record.scenario_id}"
    failures: list[str] = []
    if record.duration_seconds is None or record.duration_seconds < 0:
        failures.append(f"{identity} lacks a valid runtime")
    if (
        record.provenance is None
        or record.framework_result is None
        or record.ml4t_result is None
        or record.comparison is None
    ):
        failures.append(f"{identity} lacks complete comparison evidence")
        return failures

    scenario = SCENARIOS[record.scenario_id]
    target = manifest.targets[record.framework]
    provenance = record.provenance
    framework_target = _mapping(
        provenance.get("framework_target"), label=f"{identity} framework_target", failures=failures
    )
    ml4t = _mapping(provenance.get("ml4t"), label=f"{identity} ml4t", failures=failures)
    python = _mapping(provenance.get("python"), label=f"{identity} python", failures=failures)
    adapter = _mapping(provenance.get("adapter"), label=f"{identity} adapter", failures=failures)
    digests = _mapping(provenance.get("digests"), label=f"{identity} digests", failures=failures)
    counts = _mapping(
        provenance.get("record_counts"), label=f"{identity} record_counts", failures=failures
    )
    capabilities = _mapping(
        provenance.get("capabilities"), label=f"{identity} capabilities", failures=failures
    )
    if framework_target is not None:
        expected_target = {
            "version": target.version,
            "actual_version": target.version,
            "immutable_id": target.immutable_id,
        }
        if framework_target != expected_target:
            failures.append(f"{identity} framework target identity differs from the manifest")
    if ml4t is not None:
        if ml4t.get("dirty") is not False:
            failures.append(f"{identity} was produced from a dirty ML4T tree")
        commit = ml4t.get("commit")
        if not isinstance(commit, str) or len(commit) != 40:
            failures.append(f"{identity} lacks a full ML4T commit")
    if python is not None and (
        not isinstance(python.get("version"), str)
        or not isinstance(python.get("implementation"), str)
    ):
        failures.append(f"{identity} lacks Python runtime identity")
    expected_adapter = {
        "module": f"frameworks.{record.framework}",
        "path": f"validation/frameworks/{record.framework}.py",
    }
    if adapter is not None and adapter != expected_adapter:
        failures.append(f"{identity} adapter identity differs")
    expected_capabilities = {
        "framework": FRAMEWORK_CAPABILITIES[record.framework],
        "ml4t": ML4T_CAPABILITIES,
    }
    if capabilities != expected_capabilities:
        failures.append(f"{identity} capability declaration differs")
    if record.framework_result.capabilities != FRAMEWORK_CAPABILITIES[record.framework]:
        failures.append(f"{identity} framework result capabilities differ")
    if record.ml4t_result.capabilities != ML4T_CAPABILITIES:
        failures.append(f"{identity} ML4T result capabilities differ")

    expected_digests = static_digests(scenario, record.framework)
    if digests is not None:
        for name, expected in expected_digests.items():
            if digests.get(name) != expected:
                failures.append(f"{identity} {name} digest is stale")

    prices, entries, exits = generate_inputs(scenario, record.framework)
    expected_input_digest = input_digest(prices, entries, exits)
    if provenance.get("input_digest") != expected_input_digest:
        failures.append(f"{identity} generated-input digest is stale")

    expected_counts = {
        "bars": len(prices),
        "entry_signals": int(entries.sum()),
        "exit_signals": int(exits.sum()) if exits is not None else 0,
        "framework_intents": int(entries.sum()) + (int(exits.sum()) if exits is not None else 0),
        "ml4t_intents": int(entries.sum()) + (int(exits.sum()) if exits is not None else 0),
        "framework_orders": None,
        "ml4t_orders": None,
        "framework_fills": len(record.framework_result.fills),
        "ml4t_fills": len(record.ml4t_result.fills),
        "framework_closed_trades": record.framework_result.num_trades,
        "ml4t_closed_trades": record.ml4t_result.num_trades,
    }
    if counts is not None and counts != expected_counts:
        failures.append(f"{identity} record counts differ from retained inputs or outputs")

    expected_comparison = compare_results(
        scenario,
        record.framework_result,
        record.ml4t_result,
    )
    if record.comparison.to_dict() != expected_comparison.to_dict():
        failures.append(f"{identity} comparison checks differ from retained outputs")
    expected_passed = record.status is ValidationStatus.PASS
    if record.comparison.passed is not expected_passed:
        failures.append(f"{identity} status conflicts with its comparison verdict")
    return failures


def correctness_report_failures(
    report: dict[str, object],
    *,
    manifest: FrameworkManifest | None = None,
) -> list[str]:
    """Return every reason a candidate cannot replace accepted evidence."""
    target_manifest = manifest or load_framework_manifest()
    failures: list[str] = []
    if report.get("schema_version") != SCHEMA_VERSION:
        failures.append(f"Unsupported correctness schema: {report.get('schema_version')!r}")
        return failures
    if not isinstance(report.get("generated_at"), str):
        failures.append("Correctness report lacks a generation timestamp")
    expected_policy = {
        "canonical_quantum": CANONICAL_QUANTUM_TEXT,
        "rounding": "ROUND_HALF_EVEN",
        "meaning": "equality after canonical quantization, not bit identity",
        "record_order": "significant",
        "timestamp_domain": "daily session date",
        "surfaces": ["terminal", "closed_trades", "fills"],
    }
    if report.get("comparison_policy") != expected_policy:
        failures.append("Correctness comparison policy differs")
    if report.get("frameworks") != _framework_metadata(target_manifest):
        failures.append("Correctness framework targets differ from the frozen manifest")
    if report.get("release_gate_passed") is not True:
        failures.append("Correctness release gate did not pass")

    raw_records = report.get("records")
    if not isinstance(raw_records, list):
        failures.append("Correctness records must be a list")
        return failures

    records: dict[tuple[str, str], ValidationRecord] = {}
    for index, payload in enumerate(raw_records):
        if not isinstance(payload, dict):
            failures.append(f"Correctness record {index} must be an object")
            continue
        try:
            record = ValidationRecord.from_dict(cast(dict[str, Any], payload))
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f"Correctness record {index} is invalid: {error}")
            continue
        key = (record.framework, record.scenario_id)
        if key in records:
            failures.append(
                f"Duplicate correctness record: {record.framework}/{record.scenario_id}"
            )
        records[key] = record

    expected_pairs = _expected_pairs(target_manifest)
    missing = sorted(set(expected_pairs) - set(records))
    unexpected = sorted(set(records) - set(expected_pairs))
    if missing:
        failures.append(f"Correctness matrix is missing pairs: {missing}")
    if unexpected:
        failures.append(f"Correctness matrix has unknown pairs: {unexpected}")

    for key in sorted(set(expected_pairs) & set(records)):
        required, scenario_name = expected_pairs[key]
        record = records[key]
        identity = f"{record.framework}/{record.scenario_id}"
        if record.scenario_name != scenario_name:
            failures.append(f"{identity} scenario name differs from its definition")
        if record.required is not required:
            failures.append(f"{identity} required flag differs from the scenario matrix")
        expected_status = ValidationStatus.PASS if required else ValidationStatus.UNSUPPORTED
        if record.status is not expected_status:
            failures.append(
                f"{identity} has status {record.status.value}, expected {expected_status.value}"
            )
        if required and record.status is ValidationStatus.PASS:
            failures.extend(_record_evidence_failures(record, manifest=target_manifest))
        elif not required and not record.detail:
            failures.append(f"{identity} unsupported record lacks a reason")
    expected_summary = Counter(record.status.value for record in records.values())
    summary = report.get("summary")
    if not isinstance(summary, dict) or any(
        summary.get(status.value) != expected_summary[status.value] for status in ValidationStatus
    ):
        failures.append("Correctness summary differs from retained records")
    return failures


def write_candidate(path: Path, records: list[ValidationRecord]) -> dict[str, object]:
    """Write a complete diagnostic candidate, whether it passes or fails."""
    report = build_report(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_report(path: Path) -> dict[str, object]:
    """Load a correctness report with an object root."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Correctness report must be an object: {path}")
    return cast(dict[str, object], payload)


def promote_candidate(candidate: Path, accepted: Path) -> list[str]:
    """Atomically copy a valid candidate over accepted evidence."""
    report = load_report(candidate)
    failures = correctness_report_failures(report)
    if failures:
        return failures
    accepted.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=accepted.parent,
        prefix=f".{accepted.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(candidate.read_bytes())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, accepted)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return []
