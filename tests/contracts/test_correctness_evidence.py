"""Contracts for complete, current, atomically accepted correctness evidence."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

_VALIDATION_DIR = Path(__file__).parents[2] / "validation"
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

from common.comparator import compare_results  # noqa: E402
from common.correctness_evidence import (  # noqa: E402
    build_report,
    correctness_report_failures,
    promote_candidate,
)
from common.framework_registry import load_framework_manifest  # noqa: E402
from common.provenance import generate_inputs, input_digest, static_digests  # noqa: E402
from common.types import FrameworkResult, ValidationRecord, ValidationStatus  # noqa: E402
from scenarios.definitions import SCENARIOS  # noqa: E402


def _complete_record(framework: str, scenario_id: str) -> ValidationRecord:
    manifest = load_framework_manifest()
    target = manifest.targets[framework]
    scenario = SCENARIOS[scenario_id]
    prices, entries, exits = generate_inputs(scenario, framework)
    extra = {"total_commission": 0.0, "exit_price": 0.0}
    expected = FrameworkResult(target.display_name, 100_000.0, 0.0, 0, extra=extra)
    actual = FrameworkResult("ml4t.backtest", 100_000.0, 0.0, 0, extra=extra)
    comparison = compare_results(scenario, expected, actual)
    provenance = {
        "framework_target": {
            "version": target.version,
            "actual_version": target.version,
            "immutable_id": target.immutable_id,
        },
        "ml4t": {"commit": "a" * 40, "dirty": False},
        "python": {"version": "3.12.11", "implementation": "cpython"},
        "adapter": {
            "module": f"frameworks.{framework}",
            "path": f"validation/frameworks/{framework}.py",
        },
        "digests": static_digests(scenario, framework),
        "input_digest": input_digest(prices, entries, exits),
        "record_counts": {
            "bars": len(prices),
            "entry_signals": int(entries.sum()),
            "exit_signals": int(exits.sum()) if exits is not None else 0,
            "framework_trades": 0,
            "ml4t_trades": 0,
        },
    }
    return ValidationRecord(
        framework=framework,
        scenario_id=scenario_id,
        scenario_name=scenario.name,
        status=ValidationStatus.PASS,
        required=True,
        duration_seconds=0.01,
        provenance=provenance,
        framework_result=expected,
        ml4t_result=actual,
        comparison=comparison,
    )


def _full_report() -> dict[str, object]:
    manifest = load_framework_manifest()
    records: list[ValidationRecord] = []
    for framework in manifest.scenario_framework_ids:
        for scenario_id, scenario in SCENARIOS.items():
            if framework in scenario.supported_frameworks:
                records.append(_complete_record(framework, scenario_id))
            else:
                records.append(
                    ValidationRecord(
                        framework=framework,
                        scenario_id=scenario_id,
                        scenario_name=scenario.name,
                        status=ValidationStatus.UNSUPPORTED,
                        required=False,
                        detail="Scenario explicitly excludes this framework",
                        duration_seconds=0.001,
                    )
                )
    return build_report(records)


def _first_required_record(report: dict[str, object]) -> dict[str, Any]:
    records = cast(list[dict[str, Any]], report["records"])
    return next(record for record in records if record["required"])


def test_complete_record_retains_outputs_checks_runtime_and_provenance() -> None:
    payload = _complete_record("backtrader", "01").to_dict()

    assert payload["duration_seconds"] > 0
    assert payload["framework_result"]["final_value"] == 100_000.0
    assert payload["ml4t_result"]["final_value"] == 100_000.0
    assert payload["provenance"]["input_digest"]
    assert payload["provenance"]["digests"]["adapter"]
    checks = payload["comparison"]["checks"]
    assert checks
    assert {check["canonical_quantum"] for check in checks} == {"1", "0.00000001"}
    assert all(
        {"name", "canonical_quantum", "expected", "actual", "difference", "message"} <= check.keys()
        for check in checks
    )


def test_complete_current_matrix_is_acceptable() -> None:
    assert correctness_report_failures(_full_report()) == []


@pytest.mark.parametrize(
    ("mutation", "expected_failure"),
    [
        ("version", "framework target identity"),
        ("input", "generated-input digest"),
        ("adapter", "adapter digest"),
        ("profile", "profile digest"),
        ("comparator", "comparator digest"),
        ("expected", "comparison checks differ"),
        ("actual", "comparison checks differ"),
        ("missing", "comparison evidence must be complete"),
    ],
)
def test_evidence_mutations_fail_closed(mutation: str, expected_failure: str) -> None:
    report = copy.deepcopy(_full_report())
    record = _first_required_record(report)
    if mutation == "version":
        record["provenance"]["framework_target"]["actual_version"] = "0.0.0"
    elif mutation == "input":
        record["provenance"]["input_digest"] = "0" * 64
    elif mutation in {"adapter", "profile", "comparator"}:
        record["provenance"]["digests"][mutation] = "0" * 64
    elif mutation in {"expected", "actual"}:
        record["comparison"]["checks"][0][mutation] = 12345
    else:
        record.pop("comparison")

    failures = correctness_report_failures(report)

    assert any(expected_failure in failure for failure in failures)


def test_source_commit_change_without_behavior_change_is_not_stale() -> None:
    report = _full_report()
    for record in cast(list[dict[str, Any]], report["records"]):
        if record["provenance"] is not None:
            record["provenance"]["ml4t"]["commit"] = "b" * 40

    assert correctness_report_failures(report) == []


def test_dirty_source_is_not_acceptable() -> None:
    report = _full_report()
    _first_required_record(report)["provenance"]["ml4t"]["dirty"] = True

    assert any("dirty ML4T tree" in failure for failure in correctness_report_failures(report))


def test_failed_candidate_does_not_replace_accepted_evidence(tmp_path: Path) -> None:
    report = _full_report()
    report["release_gate_passed"] = False
    candidate = tmp_path / "candidate.json"
    accepted = tmp_path / "accepted.json"
    candidate.write_text(json.dumps(report), encoding="utf-8")
    accepted.write_bytes(b"accepted-before\n")

    failures = promote_candidate(candidate, accepted)

    assert failures
    assert accepted.read_bytes() == b"accepted-before\n"
    assert candidate.is_file()


def test_valid_candidate_atomically_replaces_accepted_evidence(tmp_path: Path) -> None:
    report = _full_report()
    candidate = tmp_path / "candidate.json"
    accepted = tmp_path / "accepted.json"
    candidate.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    accepted.write_bytes(b"accepted-before\n")

    assert promote_candidate(candidate, accepted) == []
    assert accepted.read_bytes() == candidate.read_bytes()
    assert candidate.is_file()
