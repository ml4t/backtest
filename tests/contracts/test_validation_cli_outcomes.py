"""Release-gate outcome contracts for the cross-framework validation CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_VALIDATION_DIR = Path(__file__).parents[2] / "validation"
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

import run_all_correctness  # noqa: E402
import run_scenario  # noqa: E402
from common.types import (  # noqa: E402
    ValidationRecord,
    ValidationSkipped,
    ValidationStatus,
)


def test_explicitly_unsupported_scenario_is_visible_but_not_blocking() -> None:
    record = run_scenario.run_single("06", "vectorbt_oss")

    assert record.status is ValidationStatus.UNSUPPORTED
    assert record.required is False
    assert record.passed is False
    assert record.release_blocking is False


def test_missing_scenario_is_a_distinct_blocking_record() -> None:
    record = run_scenario.run_single("99", "vectorbt_pro")

    assert record.status is ValidationStatus.MISSING_SCENARIO
    assert record.required is True
    assert record.release_blocking is True


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (ModuleNotFoundError("adapter missing"), ValidationStatus.ADAPTER_IMPORT_FAILURE),
        (ImportError("framework package missing"), ValidationStatus.UNAVAILABLE),
        (ValidationSkipped("adapter skipped"), ValidationStatus.SKIPPED),
        (RuntimeError("adapter crashed"), ValidationStatus.ADAPTER_FAILURE),
    ],
)
def test_adapter_failures_have_distinct_records(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_status: ValidationStatus,
) -> None:
    if expected_status is ValidationStatus.ADAPTER_IMPORT_FAILURE:
        monkeypatch.setattr(
            run_scenario.importlib, "import_module", lambda _: (_ for _ in ()).throw(failure)
        )
    else:
        adapter = SimpleNamespace(run=lambda *_: (_ for _ in ()).throw(failure))
        monkeypatch.setattr(run_scenario.importlib, "import_module", lambda _: adapter)

    record = run_scenario.run_single("01", "vectorbt_pro")

    assert record.status is expected_status
    assert record.release_blocking is True
    assert str(failure) in (record.detail or "")


def test_malformed_adapter_result_is_a_distinct_blocking_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = SimpleNamespace(run=lambda *_: {"final_value": "not a framework result"})
    monkeypatch.setattr(run_scenario.importlib, "import_module", lambda _: adapter)

    record = run_scenario.run_single("01", "vectorbt_pro")

    assert record.status is ValidationStatus.MALFORMED_OUTPUT
    assert record.release_blocking is True


def test_missing_isolated_environment_is_unavailable(tmp_path: Path) -> None:
    record = run_all_correctness.run_isolated(
        "vectorbt_pro",
        "01",
        python_path=tmp_path / "missing-python",
    )

    assert record.status is ValidationStatus.UNAVAILABLE
    assert record.release_blocking is True


def test_isolated_timeout_is_retained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    python_path = tmp_path / "python"
    python_path.touch()
    monkeypatch.setattr(
        run_all_correctness.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        ),
    )

    record = run_all_correctness.run_isolated(
        "vectorbt_pro", "01", python_path=python_path, timeout=7
    )

    assert record.status is ValidationStatus.TIMEOUT
    assert "7" in (record.detail or "")


def test_isolated_subprocess_failure_without_record_is_retained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    python_path = tmp_path / "python"
    python_path.touch()
    monkeypatch.setattr(
        run_all_correctness.subprocess,
        "run",
        lambda *args, **_kwargs: subprocess.CompletedProcess(args[0], 9, "", "crash"),
    )

    record = run_all_correctness.run_isolated("vectorbt_pro", "01", python_path=python_path)

    assert record.status is ValidationStatus.SUBPROCESS_FAILURE
    assert "code 9" in (record.detail or "")


def test_isolated_malformed_output_is_retained(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    python_path = tmp_path / "python"
    python_path.touch()

    def write_malformed(command, **kwargs):
        result_path = Path(command[command.index("--result-json") + 1])
        result_path.write_text("not-json", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(run_all_correctness.subprocess, "run", write_malformed)

    record = run_all_correctness.run_isolated("vectorbt_pro", "01", python_path=python_path)

    assert record.status is ValidationStatus.MALFORMED_OUTPUT
    assert record.release_blocking is True


def test_isolated_child_failure_record_survives_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    python_path = tmp_path / "python"
    python_path.touch()
    expected = ValidationRecord(
        framework="vectorbt_pro",
        scenario_id="01",
        scenario_name="Long Only",
        status=ValidationStatus.ADAPTER_IMPORT_FAILURE,
        required=True,
        detail="adapter missing",
    )

    def write_record(command, **kwargs):
        result_path = Path(command[command.index("--result-json") + 1])
        result_path.write_text(json.dumps(expected.to_dict()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(run_all_correctness.subprocess, "run", write_record)

    record = run_all_correctness.run_isolated("vectorbt_pro", "01", python_path=python_path)

    assert record == expected


def test_release_gate_fails_for_every_non_optional_nonpass_status() -> None:
    records = [
        ValidationRecord("vectorbt_oss", "06", "Commission", ValidationStatus.UNSUPPORTED, False),
        ValidationRecord("vectorbt_pro", "01", "Long Only", ValidationStatus.UNAVAILABLE, True),
    ]

    summary = run_all_correctness.summarize(records)

    assert summary["pass"] == 0
    assert summary["unsupported"] == 1
    assert summary["unavailable"] == 1
    assert run_all_correctness.release_gate_passed(records) is False
    assert run_all_correctness.release_gate_passed(records[:1]) is False


def test_unavailable_vectorbt_pro_matrix_cannot_report_sixteen_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(framework: str, scenario_id: str, **_kwargs) -> ValidationRecord:
        return ValidationRecord(
            framework,
            scenario_id,
            f"Scenario {scenario_id}",
            ValidationStatus.UNAVAILABLE,
            True,
        )

    monkeypatch.setattr(run_all_correctness, "run_isolated", unavailable)

    records = run_all_correctness.run_all_validations(frameworks=["vectorbt_pro"])
    summary = run_all_correctness.summarize(records)

    assert len(records) == 16
    assert summary["pass"] == 0
    assert summary["unavailable"] == 16
    assert run_all_correctness.release_gate_passed(records) is False
