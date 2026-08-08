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
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

VALIDATION_DIR = Path(__file__).parent
PROJECT_ROOT = VALIDATION_DIR.parent
sys.path.insert(0, str(VALIDATION_DIR))

from common.types import ValidationRecord, ValidationStatus  # noqa: E402
from scenarios.definitions import SCENARIOS  # noqa: E402

FRAMEWORK_ENVIRONMENTS = {
    "vectorbt_pro": ".venv-vectorbt-pro",
    "vectorbt_oss": ".venv",
    "backtrader": ".venv-backtrader",
    "zipline": ".venv-zipline",
}

FRAMEWORK_PYTHON_ENV_VARS = {
    "vectorbt_pro": "ML4T_VECTORBT_PRO_PYTHON",
    "vectorbt_oss": "ML4T_VECTORBT_OSS_PYTHON",
    "backtrader": "ML4T_BACKTRADER_PYTHON",
    "zipline": "ML4T_ZIPLINE_PYTHON",
}

FRAMEWORK_PINS = {
    "vectorbt_pro": {
        "display_name": "VectorBT Pro",
        "profile": "vectorbt_strict",
        "package": "vectorbtpro",
        "version": "2025.12.31",
        "source": "https://github.com/polakowo/vectorbt.pro",
        "commit": "1305a1e1974325db9382eaeacc6452e9b075ca71",
    },
    "vectorbt_oss": {
        "display_name": "VectorBT OSS",
        "profile": "vectorbt",
        "package": "vectorbt",
        "version": "0.28.2",
        "source": "https://pypi.org/project/vectorbt/0.28.2/",
    },
    "backtrader": {
        "display_name": "Backtrader",
        "profile": "backtrader_strict",
        "package": "backtrader",
        "version": "1.9.78.123",
        "source": "https://pypi.org/project/backtrader/1.9.78.123/",
    },
    "zipline": {
        "display_name": "Zipline Reloaded",
        "profile": "zipline_strict",
        "package": "zipline-reloaded",
        "version": "3.1.1",
        "source": "https://pypi.org/project/zipline-reloaded/3.1.1/",
    },
}


def _record(
    framework: str,
    scenario_id: str,
    status: ValidationStatus,
    *,
    required: bool = True,
    detail: str | None = None,
) -> ValidationRecord:
    scenario = SCENARIOS.get(scenario_id)
    return ValidationRecord(
        framework=framework,
        scenario_id=scenario_id,
        scenario_name=scenario.name if scenario else f"Scenario {scenario_id}",
        status=status,
        required=required,
        detail=detail,
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
    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        return _record(
            framework,
            scenario_id,
            ValidationStatus.MISSING_SCENARIO,
            detail=f"Scenario {scenario_id} is not defined",
        )
    if framework not in scenario.supported_frameworks:
        return _record(
            framework,
            scenario_id,
            ValidationStatus.UNSUPPORTED,
            required=False,
            detail="Scenario explicitly excludes this framework",
        )

    interpreter = python_path or resolve_python(framework)
    if not interpreter.is_file():
        return _record(
            framework,
            scenario_id,
            ValidationStatus.UNAVAILABLE,
            detail=f"Framework interpreter not found: {interpreter}",
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
            )
        except OSError as error:
            return _record(
                framework,
                scenario_id,
                ValidationStatus.SUBPROCESS_FAILURE,
                detail=f"Could not execute validation subprocess: {error}",
            )

        if not result_path.is_file():
            return _record(
                framework,
                scenario_id,
                ValidationStatus.SUBPROCESS_FAILURE,
                detail=_process_detail(result),
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


def write_report(path: Path, records: list[ValidationRecord]) -> None:
    """Retain the complete machine-readable release-gate result."""
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "frameworks": FRAMEWORK_PINS,
        "release_gate_passed": release_gate_passed(records),
        "summary": summarize(records),
        "records": [record.to_dict() for record in records],
    }
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
        default=VALIDATION_DIR / "CORRECTNESS_RESULTS.json",
        help="Machine-readable result path",
    )
    parser.add_argument("--timeout", type=int, default=180, help="Per-scenario timeout in seconds")
    args = parser.parse_args()

    frameworks = [args.framework] if args.framework else None
    scenarios = args.scenarios.split(",") if args.scenarios else None
    records = run_all_validations(frameworks, scenarios, timeout=args.timeout)
    write_report(args.output, records)

    nonzero = [f"{status}={count}" for status, count in summarize(records).items() if count]
    print(f"Results: {', '.join(nonzero)}")
    print(f"Report: {args.output}")
    return 0 if release_gate_passed(records) else 1


if __name__ == "__main__":
    sys.exit(main())
