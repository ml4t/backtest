"""Type definitions for the validation harness."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from common.framework_registry import load_framework_manifest


def _scenario_frameworks() -> list[str]:
    return list(load_framework_manifest().scenario_framework_ids)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a nonempty string")
    return value


def _required_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _required_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


class ValidationStatus(StrEnum):
    """Terminal status for one framework/scenario validation attempt."""

    PASS = "pass"
    COMPARISON_FAILURE = "comparison_failure"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    ADAPTER_IMPORT_FAILURE = "adapter_import_failure"
    ADAPTER_FAILURE = "adapter_failure"
    ML4T_FAILURE = "ml4t_failure"
    SUBPROCESS_FAILURE = "subprocess_failure"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"
    MALFORMED_OUTPUT = "malformed_output"
    MISSING_SCENARIO = "missing_scenario"


@dataclass(frozen=True)
class ValidationRecord:
    """Retained outcome for one framework/scenario validation attempt."""

    framework: str
    scenario_id: str
    scenario_name: str
    status: ValidationStatus
    required: bool
    detail: str | None = None
    duration_seconds: float | None = None
    provenance: dict[str, Any] | None = None
    framework_result: FrameworkResult | None = None
    ml4t_result: FrameworkResult | None = None
    comparison: ComparisonResult | None = None

    @property
    def passed(self) -> bool:
        """Whether the required validation work executed and matched."""
        return self.status is ValidationStatus.PASS

    @property
    def release_blocking(self) -> bool:
        """Whether this record must fail a release validation command."""
        return self.status not in {ValidationStatus.PASS, ValidationStatus.UNSUPPORTED}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record."""
        payload = {
            "framework": self.framework,
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "status": self.status.value,
            "required": self.required,
            "release_blocking": self.release_blocking,
            "detail": self.detail,
            "duration_seconds": self.duration_seconds,
            "provenance": _json_value(self.provenance),
            "framework_result": (
                self.framework_result.to_dict() if self.framework_result is not None else None
            ),
            "ml4t_result": self.ml4t_result.to_dict() if self.ml4t_result is not None else None,
            "comparison": self.comparison.to_dict() if self.comparison is not None else None,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValidationRecord:
        """Validate and construct a retained record payload."""
        required_fields = {"framework", "scenario_id", "scenario_name", "status", "required"}
        missing = sorted(required_fields - payload.keys())
        if missing:
            raise ValueError(f"Validation record missing fields: {', '.join(missing)}")

        framework = payload["framework"]
        scenario_id = payload["scenario_id"]
        scenario_name = payload["scenario_name"]
        required = payload["required"]
        detail = payload.get("detail")
        duration_seconds = payload.get("duration_seconds")
        provenance = payload.get("provenance")
        if not all(isinstance(value, str) for value in (framework, scenario_id, scenario_name)):
            raise TypeError("Validation record identifiers must be strings")
        if not isinstance(required, bool):
            raise TypeError("Validation record required must be a boolean")
        if detail is not None and not isinstance(detail, str):
            raise TypeError("Validation record detail must be a string or null")
        if duration_seconds is not None and (
            not isinstance(duration_seconds, (int, float)) or isinstance(duration_seconds, bool)
        ):
            raise TypeError("Validation record duration_seconds must be numeric or null")
        if provenance is not None and not isinstance(provenance, dict):
            raise TypeError("Validation record provenance must be an object or null")

        framework_result_payload = payload.get("framework_result")
        ml4t_result_payload = payload.get("ml4t_result")
        comparison_payload = payload.get("comparison")
        framework_result = (
            FrameworkResult.from_dict(framework_result_payload)
            if isinstance(framework_result_payload, dict)
            else None
        )
        ml4t_result = (
            FrameworkResult.from_dict(ml4t_result_payload)
            if isinstance(ml4t_result_payload, dict)
            else None
        )
        comparison = (
            ComparisonResult.from_dict(comparison_payload)
            if isinstance(comparison_payload, dict)
            else None
        )
        evidence_fields = (framework_result_payload, ml4t_result_payload, comparison_payload)
        if any(value is not None for value in evidence_fields) and any(
            value is None for value in evidence_fields
        ):
            raise ValueError("Validation record comparison evidence must be complete")
        status = ValidationStatus(payload["status"])
        if status in {ValidationStatus.PASS, ValidationStatus.COMPARISON_FAILURE} and (
            provenance is None
            or duration_seconds is None
            or framework_result is None
            or ml4t_result is None
            or comparison is None
        ):
            raise ValueError("Compared validation record lacks complete evidence")

        return cls(
            framework=framework,
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            status=status,
            required=required,
            detail=detail,
            duration_seconds=float(duration_seconds) if duration_seconds is not None else None,
            provenance=provenance,
            framework_result=framework_result,
            ml4t_result=ml4t_result,
            comparison=comparison,
        )


class ValidationSkipped(RuntimeError):
    """Raised by an adapter that elects not to execute a required scenario."""


CAPABILITY_KEYS = (
    "intents",
    "orders",
    "rejections",
    "fills",
    "positions",
    "cash_flows",
    "open_trades",
    "closed_trades",
    "exit_reason",
    "terminal",
)
CAPABILITY_VALUES = {
    "native",
    "native_filled_only",
    "reconstructed",
    "input_only",
    "aggregate_only",
    "unavailable",
}


def unavailable_capabilities() -> dict[str, str]:
    """Return an explicit declaration that no optional result surface is available."""
    return dict.fromkeys(CAPABILITY_KEYS, "unavailable")


@dataclass
class ScenarioConfig:
    """Declarative definition of a validation scenario."""

    id: str  # "01", "02", ..., "17"
    name: str  # Human-readable name
    description: str

    # Data generation
    data_generator: str  # Function name in data_generators module
    data_kwargs: dict[str, Any] = field(default_factory=dict)

    # Signal columns present in data
    signal_columns: list[str] = field(default_factory=lambda: ["entry", "exit"])

    # Risk rules (ml4t config names)
    risk_rules: list[dict[str, Any]] = field(default_factory=list)

    # Per-framework ml4t config overrides (beyond profile defaults)
    ml4t_config: dict[str, Any] = field(default_factory=dict)

    # Per-framework ml4t config overrides
    ml4t_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per-framework comparison tolerances
    tolerances: dict[str, Tolerance] | None = None

    # Default tolerance (used when no per-framework tolerance is specified)
    default_tolerance: Tolerance | None = None

    # Which frameworks support this scenario
    supported_frameworks: list[str] = field(default_factory=_scenario_frameworks)

    # Extra comparison checks beyond standard (trade count, final value, pnl)
    extra_checks: list[str] = field(default_factory=list)

    # Module-level constants (e.g., COMMISSION_RATE, SLIPPAGE)
    constants: dict[str, Any] = field(default_factory=dict)

    # Strategy type for ml4t (determines on_data behavior)
    strategy_type: str = "long_signal"  # long_signal, long_short, short_only, risk_entry_only

    # Shares per trade
    shares: int = 100

    # Initial cash
    initial_cash: float = 100_000.0


@dataclass
class Tolerance:
    """Diagnostic gap thresholds that never affect release pass/fail status."""

    trade_count: int = 0  # Absolute difference allowed
    value_pct: float = 0.01  # Percentage of final value
    pnl_abs: float = 1.0  # Absolute dollar amount
    exit_price_abs: float = 0.01  # Absolute price difference
    commission_abs: float = 0.01  # Absolute commission difference


@dataclass
class FrameworkResult:
    """Results from running a single framework on a scenario."""

    framework: str
    final_value: float
    total_pnl: float
    num_trades: int
    trades: list[dict[str, Any]] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    capabilities: dict[str, str] = field(default_factory=unavailable_capabilities)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "final_value": self.final_value,
            "total_pnl": self.total_pnl,
            "num_trades": self.num_trades,
            "trades": _json_value(self.trades),
            "fills": _json_value(self.fills),
            "capabilities": _json_value(self.capabilities),
            "extra": _json_value(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FrameworkResult:
        required = {
            "framework",
            "final_value",
            "total_pnl",
            "num_trades",
            "trades",
            "fills",
            "capabilities",
            "extra",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Framework result missing fields: {', '.join(missing)}")
        trades = payload["trades"]
        fills = payload["fills"]
        capabilities = payload["capabilities"]
        extra = payload["extra"]
        num_trades = payload["num_trades"]
        if not isinstance(trades, list) or not all(isinstance(trade, dict) for trade in trades):
            raise TypeError("Framework result trades must be an object array")
        if not isinstance(fills, list) or not all(isinstance(fill, dict) for fill in fills):
            raise TypeError("Framework result fills must be an object array")
        if not isinstance(capabilities, dict) or set(capabilities) != set(CAPABILITY_KEYS):
            raise TypeError("Framework result capabilities must declare every canonical surface")
        if not all(
            isinstance(value, str) and value in CAPABILITY_VALUES for value in capabilities.values()
        ):
            raise ValueError("Framework result capabilities contain an unsupported declaration")
        if not isinstance(extra, dict):
            raise TypeError("Framework result trades and extra have invalid types")
        if not isinstance(num_trades, int) or isinstance(num_trades, bool) or num_trades < 0:
            raise TypeError("Framework result num_trades must be a nonnegative integer")
        return cls(
            framework=_required_string(payload["framework"], "Framework result framework"),
            final_value=_required_number(payload["final_value"], "Framework result final_value"),
            total_pnl=_required_number(payload["total_pnl"], "Framework result total_pnl"),
            num_trades=num_trades,
            trades=trades,
            fills=fills,
            capabilities=capabilities,
            extra=extra,
        )


@dataclass
class ComparisonResult:
    """Result of comparing two framework outputs."""

    scenario_id: str
    framework: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "framework": self.framework,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ComparisonResult:
        required = {"scenario_id", "framework", "passed", "checks"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Comparison result missing fields: {', '.join(missing)}")
        checks = payload.get("checks")
        if not isinstance(checks, list) or not all(isinstance(check, dict) for check in checks):
            raise TypeError("Comparison checks must be an object array")
        return cls(
            scenario_id=_required_string(payload["scenario_id"], "Comparison scenario_id"),
            framework=_required_string(payload["framework"], "Comparison framework"),
            passed=_required_bool(payload["passed"], "Comparison passed"),
            checks=[CheckResult.from_dict(check) for check in checks],
        )

    @property
    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        failed = [c for c in self.checks if not c.passed]
        if failed:
            details = "; ".join(f"{c.name}: {c.message}" for c in failed)
            return f"{status} ({details})"
        return status


@dataclass
class CheckResult:
    """Result of a single comparison check."""

    name: str
    passed: bool
    message: str
    expected: Any = None
    actual: Any = None
    difference: Any = None
    canonical_quantum: str | None = None
    diagnostic_limit: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "canonical_quantum": self.canonical_quantum,
            "expected": _json_value(self.expected),
            "actual": _json_value(self.actual),
            "difference": _json_value(self.difference),
            "diagnostic_limit": _json_value(self.diagnostic_limit),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CheckResult:
        required = {
            "name",
            "passed",
            "canonical_quantum",
            "expected",
            "actual",
            "difference",
            "diagnostic_limit",
            "message",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Comparison check missing fields: {', '.join(missing)}")
        return cls(
            name=_required_string(payload["name"], "Comparison check name"),
            passed=_required_bool(payload["passed"], "Comparison check passed"),
            message=_required_string(payload["message"], "Comparison check message"),
            expected=payload["expected"],
            actual=payload["actual"],
            difference=payload["difference"],
            canonical_quantum=_required_string(
                payload["canonical_quantum"], "Comparison check canonical_quantum"
            ),
            diagnostic_limit=payload["diagnostic_limit"],
        )
