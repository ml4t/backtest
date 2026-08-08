"""Enforce dependency and source-security scan policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import date
from pathlib import Path
from typing import cast

_ROOT = Path(__file__).parents[1]
_DEFAULT_POLICY = _ROOT / ".github" / "security-policy.toml"
_DEFAULT_EXCEPTIONS = _ROOT / ".github" / "security-exceptions.toml"
_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}
_EXCEPTION_FIELDS = {
    "finding",
    "name",
    "reason",
    "reviewer",
    "reviewed_on",
    "expires_on",
}


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _date(value: object, label: str) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{label} must be an ISO date") from error
    raise ValueError(f"{label} must be an ISO date")


def _policy_values(policy: dict[str, object]) -> tuple[str, str, int, dict[str, str]]:
    settings = _mapping(policy.get("policy"), "policy")
    if settings.get("dependency_threshold") != "any":
        raise ValueError("policy.dependency_threshold must be 'any'")
    source_severity = _text(settings.get("source_severity"), "policy.source_severity").lower()
    source_confidence = _text(settings.get("source_confidence"), "policy.source_confidence").lower()
    if source_severity not in _SEVERITY_RANK:
        raise ValueError("policy.source_severity must be low, medium, or high")
    if source_confidence not in _SEVERITY_RANK:
        raise ValueError("policy.source_confidence must be low, medium, or high")
    max_days = settings.get("max_exception_days")
    if not isinstance(max_days, int) or isinstance(max_days, bool) or max_days < 1:
        raise ValueError("policy.max_exception_days must be a positive integer")

    raw_tools = _mapping(policy.get("tools"), "tools")
    tools = {
        "pip_audit": _text(raw_tools.get("pip_audit"), "tools.pip_audit"),
        "bandit": _text(raw_tools.get("bandit"), "tools.bandit"),
    }
    return source_severity, source_confidence, max_days, tools


def _dependency_findings(report: dict[str, object]) -> list[dict[str, object]]:
    dependencies = _sequence(report.get("dependencies"), "dependency report dependencies")
    if not dependencies:
        raise ValueError("dependency report contains no audited dependencies")
    findings: list[dict[str, object]] = []
    for dependency_value in dependencies:
        dependency = _mapping(dependency_value, "dependency")
        package = _text(dependency.get("name"), "dependency name")
        version = _text(dependency.get("version"), f"dependency {package} version")
        for vulnerability_value in _sequence(
            dependency.get("vulns"), f"dependency {package} vulnerabilities"
        ):
            vulnerability = _mapping(vulnerability_value, f"dependency {package} vulnerability")
            vulnerability_id = _text(vulnerability.get("id"), "vulnerability id")
            findings.append(
                {
                    "finding": f"dependency:{package}:{vulnerability_id}",
                    "kind": "dependency",
                    "package": package,
                    "version": version,
                    "vulnerability": vulnerability_id,
                }
            )
    return findings


def _source_findings(
    report: dict[str, object], severity_threshold: str, confidence_threshold: str
) -> list[dict[str, object]]:
    errors = _sequence(report.get("errors", []), "source report errors")
    if errors:
        raise ValueError(f"source scanner reported errors: {errors}")
    findings: list[dict[str, object]] = []
    for result_value in _sequence(report.get("results"), "source report results"):
        result = _mapping(result_value, "source finding")
        severity = _text(result.get("issue_severity"), "source severity").lower()
        confidence = _text(result.get("issue_confidence"), "source confidence").lower()
        if severity not in _SEVERITY_RANK or confidence not in _SEVERITY_RANK:
            raise ValueError(f"source finding has invalid severity/confidence: {result}")
        if _SEVERITY_RANK[severity] < _SEVERITY_RANK[severity_threshold]:
            continue
        if _SEVERITY_RANK[confidence] < _SEVERITY_RANK[confidence_threshold]:
            continue
        test_id = _text(result.get("test_id"), "source test id")
        filename = _text(result.get("filename"), "source filename")
        line = result.get("line_number")
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            raise ValueError("source line number must be a positive integer")
        findings.append(
            {
                "finding": f"source:{test_id}:{filename}:{line}",
                "kind": "source",
                "test_id": test_id,
                "filename": filename,
                "line": line,
                "severity": severity,
                "confidence": confidence,
            }
        )
    return findings


def _valid_exceptions(
    payload: dict[str, object], today: date, max_days: int
) -> tuple[dict[str, dict[str, str]], list[str]]:
    valid: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    try:
        entries = _sequence(payload.get("exceptions"), "exceptions")
    except ValueError as error:
        return valid, [str(error)]
    for index, entry_value in enumerate(entries, start=1):
        label = f"exception {index}"
        try:
            entry = _mapping(entry_value, label)
            fields = set(entry)
            if fields != _EXCEPTION_FIELDS:
                missing = sorted(_EXCEPTION_FIELDS - fields)
                unexpected = sorted(fields - _EXCEPTION_FIELDS)
                raise ValueError(
                    f"{label} fields invalid; missing={missing}, unexpected={unexpected}"
                )
            finding = _text(entry["finding"], f"{label} finding")
            if finding in valid:
                raise ValueError(f"duplicate exception for {finding}")
            name = _text(entry["name"], f"{label} name")
            reason = _text(entry["reason"], f"{label} reason")
            reviewer = _text(entry["reviewer"], f"{label} reviewer")
            reviewed_on = _date(entry["reviewed_on"], f"{label} reviewed_on")
            expires_on = _date(entry["expires_on"], f"{label} expires_on")
            if reviewed_on > today:
                raise ValueError(f"{label} review date is in the future")
            if expires_on <= today:
                raise ValueError(f"{label} expired on {expires_on.isoformat()}")
            if expires_on <= reviewed_on:
                raise ValueError(f"{label} must expire after its review date")
            duration = (expires_on - reviewed_on).days
            if duration > max_days:
                raise ValueError(f"{label} lasts {duration} days; maximum is {max_days}")
            valid[finding] = {
                "finding": finding,
                "name": name,
                "reason": reason,
                "reviewer": reviewer,
                "reviewed_on": reviewed_on.isoformat(),
                "expires_on": expires_on.isoformat(),
            }
        except (KeyError, ValueError) as error:
            failures.append(str(error))
    return valid, failures


def evaluate_security(
    policy: dict[str, object],
    exceptions: dict[str, object],
    dependency_report: dict[str, object],
    source_report: dict[str, object],
    *,
    today: date,
) -> tuple[dict[str, object], list[str]]:
    severity, confidence, max_days, tools = _policy_values(policy)
    findings = _dependency_findings(dependency_report)
    findings.extend(_source_findings(source_report, severity, confidence))
    valid_exceptions, failures = _valid_exceptions(exceptions, today, max_days)
    active_ids = {cast(str, finding["finding"]) for finding in findings}
    unused = sorted(set(valid_exceptions) - active_ids)
    failures.extend(
        f"Exception does not match an active blocking finding: {item}" for item in unused
    )

    excepted: list[dict[str, object]] = []
    unexcepted: list[dict[str, object]] = []
    for finding in findings:
        finding_id = cast(str, finding["finding"])
        if finding_id in valid_exceptions:
            excepted.append({**finding, "exception": valid_exceptions[finding_id]})
        else:
            unexcepted.append(finding)
            failures.append(f"Blocking security finding: {finding_id}")

    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "fail" if failures else "pass",
        "evaluated_on": today.isoformat(),
        "policy": {
            "dependency_threshold": "any",
            "source_severity": severity,
            "source_confidence": confidence,
            "max_exception_days": max_days,
        },
        "tools": tools,
        "blocking_findings": findings,
        "excepted_findings": excepted,
        "unexcepted_findings": unexcepted,
        "failures": failures,
    }
    return summary, failures


def _load_json(path: Path) -> dict[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _load_toml(path: Path) -> dict[str, object]:
    return cast(dict[str, object], tomllib.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=_DEFAULT_POLICY)
    parser.add_argument("--exceptions", type=Path, default=_DEFAULT_EXCEPTIONS)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--dependency-report", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    inputs = [
        args.policy,
        args.exceptions,
        args.requirements,
        args.dependency_report,
        args.source_report,
    ]
    try:
        summary, failures = evaluate_security(
            _load_toml(args.policy),
            _load_toml(args.exceptions),
            _load_json(args.dependency_report),
            _load_json(args.source_report),
            today=date.today(),
        )
        summary["input_sha256"] = {str(path): _sha256(path) for path in inputs}
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError, ValueError) as error:
        failures = [f"Security evidence is invalid: {error}"]
        summary = {
            "schema_version": 1,
            "status": "fail",
            "evaluated_on": date.today().isoformat(),
            "failures": failures,
        }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for failure in failures:
        print(failure)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
