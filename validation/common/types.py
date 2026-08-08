"""Type definitions for the validation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


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
        return {
            "framework": self.framework,
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "status": self.status.value,
            "required": self.required,
            "release_blocking": self.release_blocking,
            "detail": self.detail,
        }

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
        if not all(isinstance(value, str) for value in (framework, scenario_id, scenario_name)):
            raise TypeError("Validation record identifiers must be strings")
        if not isinstance(required, bool):
            raise TypeError("Validation record required must be a boolean")
        if detail is not None and not isinstance(detail, str):
            raise TypeError("Validation record detail must be a string or null")

        return cls(
            framework=framework,
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            status=ValidationStatus(payload["status"]),
            required=required,
            detail=detail,
        )


class ValidationSkipped(RuntimeError):
    """Raised by an adapter that elects not to execute a required scenario."""


@dataclass
class ScenarioConfig:
    """Declarative definition of a validation scenario."""

    id: str  # "01", "02", ..., "16"
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
    supported_frameworks: list[str] = field(
        default_factory=lambda: ["vectorbt_pro", "vectorbt_oss", "backtrader", "zipline"]
    )

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
    """Comparison tolerances for a framework."""

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
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing two framework outputs."""

    scenario_id: str
    framework: str
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)

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
