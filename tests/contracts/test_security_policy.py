"""Contracts for blocking and auditable release security checks."""

from __future__ import annotations

import importlib.util
import re
import tomllib
from datetime import date
from pathlib import Path
from types import ModuleType

import yaml

_ROOT = Path(__file__).parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"
_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _load_security_checker() -> ModuleType:
    path = _ROOT / "validation" / "check_security.py"
    spec = importlib.util.spec_from_file_location("ml4t_security_checker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow(name: str) -> dict:
    payload = yaml.load((_WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(payload, dict)
    return payload


def _policy() -> dict:
    return tomllib.loads((_ROOT / ".github" / "security-policy.toml").read_text())


def _clean_reports() -> tuple[dict, dict]:
    dependency = {
        "dependencies": [{"name": "example", "version": "1.0", "vulns": []}],
        "fixes": [],
    }
    source = {"errors": [], "results": []}
    return dependency, source


def test_every_third_party_workflow_action_is_pinned_to_a_commit() -> None:
    found = 0
    for workflow in _WORKFLOWS.glob("*.yml"):
        for line in workflow.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*(?:-\s*)?uses:\s*([^\s#]+)", line)
            if match is None or match.group(1).startswith("./"):
                continue
            found += 1
            assert _ACTION.fullmatch(match.group(1)), f"Unpinned action in {workflow}: {line}"
    assert found >= 20


def test_policy_blocks_all_dependency_findings_and_medium_source_findings() -> None:
    checker = _load_security_checker()
    policy = _policy()
    assert policy["policy"] == {
        "dependency_threshold": "any",
        "source_severity": "medium",
        "source_confidence": "medium",
        "max_exception_days": 90,
    }
    assert policy["tools"] == {"pip_audit": "2.10.1", "bandit": "1.9.4"}

    dependency = {
        "dependencies": [
            {
                "name": "affected",
                "version": "1.0",
                "vulns": [{"id": "PYSEC-2099-1"}],
            }
        ],
        "fixes": [],
    }
    source = {
        "errors": [],
        "results": [
            {
                "test_id": "B999",
                "filename": "src/example.py",
                "line_number": 7,
                "issue_severity": "MEDIUM",
                "issue_confidence": "MEDIUM",
            }
        ],
    }
    summary, failures = checker.evaluate_security(
        policy, {"exceptions": []}, dependency, source, today=date(2026, 8, 8)
    )
    assert summary["status"] == "fail"
    assert failures == [
        "Blocking security finding: dependency:affected:PYSEC-2099-1",
        "Blocking security finding: source:B999:src/example.py:7",
    ]


def test_only_a_reviewed_active_finding_specific_exception_is_accepted() -> None:
    checker = _load_security_checker()
    dependency = {
        "dependencies": [
            {
                "name": "affected",
                "version": "1.0",
                "vulns": [{"id": "PYSEC-2099-1"}],
            }
        ],
        "fixes": [],
    }
    exception = {
        "finding": "dependency:affected:PYSEC-2099-1",
        "name": "Temporary upstream remediation window",
        "reason": "No fixed release is available.",
        "reviewer": "release-owner",
        "reviewed_on": "2026-08-01",
        "expires_on": "2026-09-01",
    }
    summary, failures = checker.evaluate_security(
        _policy(),
        {"exceptions": [exception]},
        dependency,
        {"errors": [], "results": []},
        today=date(2026, 8, 8),
    )
    assert failures == []
    assert summary["status"] == "pass"
    assert summary["excepted_findings"][0]["exception"]["reviewer"] == "release-owner"


def test_expired_or_unmatched_exceptions_fail_closed() -> None:
    checker = _load_security_checker()
    dependency, source = _clean_reports()
    exception = {
        "finding": "dependency:missing:PYSEC-2099-1",
        "name": "Expired exception",
        "reason": "Test fixture.",
        "reviewer": "release-owner",
        "reviewed_on": "2026-06-01",
        "expires_on": "2026-07-01",
    }
    summary, failures = checker.evaluate_security(
        _policy(),
        {"exceptions": [exception]},
        dependency,
        source,
        today=date(2026, 8, 8),
    )
    assert summary["status"] == "fail"
    assert failures == ["exception 1 expired on 2026-07-01"]


def test_ci_and_release_both_require_the_evidence_retaining_security_workflow() -> None:
    ci_jobs = _workflow("ci.yml")["jobs"]
    release_jobs = _workflow("release.yml")["jobs"]
    security = _workflow("security.yml")

    assert ci_jobs["security"]["uses"] == "./.github/workflows/security.yml"
    assert "security" in ci_jobs["build"]["needs"]
    assert release_jobs["qualification"]["uses"] == "./.github/workflows/ci.yml"
    assert set(release_jobs["publish"]["needs"]) >= {
        "ecosystem-qualification",
        "qualification",
    }

    scan = security["jobs"]["scan"]
    commands = "\n".join(step.get("run", "") for step in scan["steps"])
    assert "uv export --locked --no-dev --no-emit-project" in commands
    assert "pip-audit==$PIP_AUDIT_VERSION" in commands
    assert "bandit==$BANDIT_VERSION" in commands
    assert "validation/check_security.py" in commands
    upload = scan["steps"][-1]
    assert upload["if"] == "always()"
    assert set(upload["with"]["path"].splitlines()) == {
        "security-requirements.txt",
        "dependency-audit.json",
        "source-security.json",
        "security-summary.json",
        ".github/security-policy.toml",
        ".github/security-exceptions.toml",
    }
