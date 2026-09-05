"""Contracts for the profile behavior coverage map."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_VALIDATION = _ROOT / "validation"
if str(_VALIDATION) not in sys.path:
    sys.path.insert(0, str(_VALIDATION))

from scenarios.definitions import SCENARIOS  # noqa: E402

from ml4t.backtest.profiles import get_profile_config  # noqa: E402

_COVERAGE = _VALIDATION / "behavior_coverage.toml"
_NATIVE_EVIDENCE = {
    "vectorbt": (
        _VALIDATION / "native/evidence/vectorbt_oss-1.1.0.json",
        _VALIDATION / "native/evidence/vectorbt_pro-2026.6.27.json",
    ),
    "backtrader": (_VALIDATION / "native/evidence/backtrader-1.9.78.123.json",),
    "zipline": (_VALIDATION / "native/evidence/zipline-3.1.1.json",),
    "lean": (_VALIDATION / "native/evidence/lean-18001.json",),
}
_SCENARIO_FRAMEWORKS = {
    "vectorbt": {"vectorbt_oss", "vectorbt_pro"},
    "backtrader": {"backtrader"},
    "zipline": {"zipline"},
}


def _native_checks(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    checks = payload["checks"]
    if isinstance(checks, list):
        return {check["id"] for check in checks}
    return set(checks)


def test_coverage_map_accounts_for_every_framework_profile_field() -> None:
    payload = tomllib.loads(_COVERAGE.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert set(payload["framework"]) == set(_NATIVE_EVIDENCE)

    for framework, entries in payload["framework"].items():
        expected_fields = {
            f"{section}.{field}"
            for section, values in get_profile_config(framework).items()
            for field in values
        }
        covered_fields = [field for entry in entries for field in entry["fields"]]
        assert len(covered_fields) == len(set(covered_fields))
        assert set(covered_fields) == expected_fields


def test_published_dimensions_have_native_and_cross_engine_evidence() -> None:
    payload = tomllib.loads(_COVERAGE.read_text(encoding="utf-8"))
    lean_cases = {
        case["case"]
        for case in json.loads(
            (_VALIDATION / "lean/case_study_evidence.json").read_text(encoding="utf-8")
        )["cases"]
    }

    for framework, entries in payload["framework"].items():
        evidence_sets = [_native_checks(path) for path in _NATIVE_EVIDENCE[framework]]
        for entry in entries:
            assert entry["name"]
            assert all(set(entry["native_checks"]) <= checks for checks in evidence_sets)
            if entry["published"]:
                assert entry["native_checks"]
                assert entry["scenarios"]
            else:
                assert entry["reason"]
            for scenario_id in entry["scenarios"]:
                if scenario_id.startswith("lean-case:"):
                    assert scenario_id.removeprefix("lean-case:") in lean_cases
                    continue
                assert scenario_id in SCENARIOS
                assert _SCENARIO_FRAMEWORKS[framework] <= set(
                    SCENARIOS[scenario_id].supported_frameworks
                )
