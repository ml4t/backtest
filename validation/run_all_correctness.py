#!/usr/bin/env python3
"""Run the required correctness matrix in isolated framework environments.

Every requested framework/scenario pair produces a retained terminal record. The command fails
unless every required pair executes and passes. Explicitly unsupported pairs remain visible but
do not satisfy a required count.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

VALIDATION_DIR = Path(__file__).parent
PROJECT_ROOT = VALIDATION_DIR.parent
sys.path.insert(0, str(VALIDATION_DIR))

from common.correctness_evidence import (  # noqa: E402, I001
    build_report,
    promote_candidate,
    write_candidate,
)
from common.framework_registry import load_framework_manifest  # noqa: E402
from common.types import ValidationRecord, ValidationStatus  # noqa: E402, I001
from scenarios.definitions import SCENARIOS  # noqa: E402

FRAMEWORK_MANIFEST = load_framework_manifest()
FRAMEWORK_ENVIRONMENTS = {
    framework_id: target.environment
    for framework_id, target in FRAMEWORK_MANIFEST.targets.items()
    if target.scenario_matrix and target.environment is not None
}
FRAMEWORK_PYTHON_ENV_VARS = {
    framework_id: target.python_env_var
    for framework_id, target in FRAMEWORK_MANIFEST.targets.items()
    if target.scenario_matrix and target.python_env_var is not None
}
FRAMEWORK_PINS = {
    framework_id: FRAMEWORK_MANIFEST.targets[framework_id].evidence_metadata()
    for framework_id in FRAMEWORK_MANIFEST.scenario_framework_ids
}


def _record(
    framework: str,
    scenario_id: str,
    status: ValidationStatus,
    *,
    required: bool = True,
    detail: str | None = None,
    started_at: float | None = None,
) -> ValidationRecord:
    scenario = SCENARIOS.get(scenario_id)
    return ValidationRecord(
        framework=framework,
        scenario_id=scenario_id,
        scenario_name=scenario.name if scenario else f"Scenario {scenario_id}",
        status=status,
        required=required,
        detail=detail,
        duration_seconds=time.perf_counter() - started_at if started_at is not None else None,
    )


def resolve_python(framework: str) -> Path:
    """Resolve the isolated interpreter for a framework."""
    override = os.getenv(FRAMEWORK_PYTHON_ENV_VARS[framework])
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_absolute():
            override_path = PROJECT_ROOT / override_path
        return override_path.absolute()
    return PROJECT_ROOT / FRAMEWORK_ENVIRONMENTS[framework] / "bin" / "python"


def _process_detail(result: subprocess.CompletedProcess[str]) -> str:
    output = (result.stderr or result.stdout).strip()
    if output:
        return f"Subprocess exited with code {result.returncode}: {output[-500:]}"
    return f"Subprocess exited with code {result.returncode} without a validation record"


def run_isolated(
    framework: str,
    scenario_id: str,
    *,
    python_path: Path | None = None,
    timeout: int = 180,
) -> ValidationRecord:
    """Run one pair in its framework environment and retain its exact terminal status."""
    started_at = time.perf_counter()
    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        return _record(
            framework,
            scenario_id,
            ValidationStatus.MISSING_SCENARIO,
            detail=f"Scenario {scenario_id} is not defined",
            started_at=started_at,
        )
    if framework not in scenario.supported_frameworks:
        return _record(
            framework,
            scenario_id,
            ValidationStatus.UNSUPPORTED,
            required=False,
            detail="Scenario explicitly excludes this framework",
            started_at=started_at,
        )

    interpreter = python_path or resolve_python(framework)
    if not interpreter.is_file():
        return _record(
            framework,
            scenario_id,
            ValidationStatus.UNAVAILABLE,
            detail=f"Framework interpreter not found: {interpreter}",
            started_at=started_at,
        )

    with tempfile.TemporaryDirectory(prefix="ml4t-validation-") as temporary_directory:
        result_path = Path(temporary_directory) / "result.json"
        command = [
            str(interpreter),
            str(VALIDATION_DIR / "run_scenario.py"),
            "--scenario",
            scenario_id,
            "--framework",
            framework,
            "--result-json",
            str(result_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=PROJECT_ROOT,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return _record(
                framework,
                scenario_id,
                ValidationStatus.TIMEOUT,
                detail=f"Validation subprocess timed out after {timeout} seconds",
                started_at=started_at,
            )
        except OSError as error:
            return _record(
                framework,
                scenario_id,
                ValidationStatus.SUBPROCESS_FAILURE,
                detail=f"Could not execute validation subprocess: {error}",
                started_at=started_at,
            )

        if not result_path.is_file():
            return _record(
                framework,
                scenario_id,
                ValidationStatus.SUBPROCESS_FAILURE,
                detail=_process_detail(result),
                started_at=started_at,
            )

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("single-scenario output must be a JSON object")
            record = ValidationRecord.from_dict(payload)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            return _record(
                framework,
                scenario_id,
                ValidationStatus.MALFORMED_OUTPUT,
                detail=f"Invalid validation record: {error}",
                started_at=started_at,
            )

        if record.framework != framework or record.scenario_id != scenario_id:
            return _record(
                framework,
                scenario_id,
                ValidationStatus.MALFORMED_OUTPUT,
                detail=(
                    "Validation record identity mismatch: "
                    f"received {record.framework}/{record.scenario_id}"
                ),
                started_at=started_at,
            )
        expected_returncode = 1 if record.release_blocking else 0
        if result.returncode != expected_returncode:
            return _record(
                framework,
                scenario_id,
                ValidationStatus.MALFORMED_OUTPUT,
                detail=(
                    f"Record status {record.status.value} conflicts with subprocess "
                    f"exit code {result.returncode}"
                ),
                started_at=started_at,
            )
        return record


def run_all_validations(
    frameworks: list[str] | None = None,
    scenarios: list[str] | None = None,
    *,
    timeout: int = 180,
) -> list[ValidationRecord]:
    """Run every selected pair and return records in deterministic matrix order."""
    selected_frameworks = frameworks or list(FRAMEWORK_ENVIRONMENTS)
    selected_scenarios = scenarios or list(SCENARIOS)
    records: list[ValidationRecord] = []
    for framework in selected_frameworks:
        for scenario_id in selected_scenarios:
            print(f"Running {framework}/{scenario_id}...", end=" ", flush=True)
            record = run_isolated(framework, scenario_id, timeout=timeout)
            print(record.status.value.upper())
            records.append(record)
    return records


def summarize(records: list[ValidationRecord]) -> dict[str, int]:
    """Count every status without folding unavailable or skipped work into passes."""
    counts = Counter(record.status.value for record in records)
    return {status.value: counts[status.value] for status in ValidationStatus}


def release_gate_passed(records: list[ValidationRecord]) -> bool:
    """Return whether every required validation record passed."""
    required_records = [record for record in records if record.required]
    return bool(required_records) and all(record.passed for record in required_records)


def matrix_coverage_failures(
    records: list[ValidationRecord],
    frameworks: list[str],
    scenarios: list[str],
) -> list[str]:
    """Reject omitted, duplicate, or misclassified framework/scenario pairs."""
    failures: list[str] = []
    expected_pairs = {(framework, scenario) for framework in frameworks for scenario in scenarios}
    actual_pairs = [(record.framework, record.scenario_id) for record in records]
    duplicate_pairs = sorted(pair for pair, count in Counter(actual_pairs).items() if count > 1)
    missing_pairs = sorted(expected_pairs - set(actual_pairs))
    unexpected_pairs = sorted(set(actual_pairs) - expected_pairs)
    if duplicate_pairs:
        failures.append(f"Duplicate matrix pairs: {duplicate_pairs}")
    if missing_pairs:
        failures.append(f"Missing matrix pairs: {missing_pairs}")
    if unexpected_pairs:
        failures.append(f"Unexpected matrix pairs: {unexpected_pairs}")

    if scenarios == list(SCENARIOS):
        for framework in frameworks:
            target = FRAMEWORK_MANIFEST.targets[framework]
            framework_records = [record for record in records if record.framework == framework]
            required_count = sum(record.required for record in framework_records)
            unsupported_count = sum(
                record.status is ValidationStatus.UNSUPPORTED for record in framework_records
            )
            if required_count != target.required_scenarios:
                failures.append(
                    f"{framework} required count differs: "
                    f"{required_count} != {target.required_scenarios}"
                )
            if unsupported_count != target.unsupported_scenarios:
                failures.append(
                    f"{framework} unsupported count differs: "
                    f"{unsupported_count} != {target.unsupported_scenarios}"
                )
    return failures


def write_report(path: Path, records: list[ValidationRecord]) -> None:
    """Retain the complete machine-readable release-gate result."""
    payload = build_report(records, manifest=FRAMEWORK_MANIFEST)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated correctness validations")
    parser.add_argument(
        "--framework",
        choices=tuple(FRAMEWORK_ENVIRONMENTS),
        help="Run only one required framework",
    )
    parser.add_argument(
        "--scenarios",
        help="Comma-separated scenario IDs; unknown IDs are retained as failures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VALIDATION_DIR / "candidates" / "CORRECTNESS_RESULTS.candidate.json",
        help="Diagnostic candidate path",
    )
    parser.add_argument(
        "--accepted-output",
        type=Path,
        default=VALIDATION_DIR / "CORRECTNESS_RESULTS.json",
        help="Accepted evidence path, replaced only by a complete passing matrix",
    )
    parser.add_argument("--timeout", type=int, default=180, help="Per-scenario timeout in seconds")
    args = parser.parse_args()

    frameworks = [args.framework] if args.framework else None
    scenarios = args.scenarios.split(",") if args.scenarios else None
    records = run_all_validations(frameworks, scenarios, timeout=args.timeout)
    write_candidate(args.output, records)

    nonzero = [f"{status}={count}" for status, count in summarize(records).items() if count]
    print(f"Results: {', '.join(nonzero)}")
    print(f"Candidate: {args.output}")
    selected_frameworks = frameworks or list(FRAMEWORK_MANIFEST.scenario_framework_ids)
    selected_scenarios = scenarios or list(SCENARIOS)
    coverage_failures = matrix_coverage_failures(
        records,
        selected_frameworks,
        selected_scenarios,
    )
    if coverage_failures:
        print("Matrix coverage failed:")
        for failure in coverage_failures:
            print(f"- {failure}")
        return 1
    full_matrix = frameworks is None and scenarios is None
    if full_matrix:
        promotion_failures = promote_candidate(args.output, args.accepted_output)
        if promotion_failures:
            print("Accepted evidence unchanged:")
            for failure in promotion_failures:
                print(f"- {failure}")
            return 1
        print(f"Accepted evidence: {args.accepted_output}")
    return 0 if release_gate_passed(records) else 1


if __name__ == "__main__":
    sys.exit(main())
