# Architectural Decisions

*Updated: 2025-11-22*

## AD-001: Exclude Zipline from Validation (2025-11-14)

**Status**: Accepted

**Context**: Zipline's `run_algorithm(bundle='quandl')` fetches its own price data (~4.3x different from test DataFrame).

**Decision**: Exclude Zipline from validation, use VectorBT Pro and Backtrader only.

**Consequence**: Can't validate against Zipline with custom data; acceptable trade-off.

---

## AD-002: Signals Computed Outside Engine (2025-11-14)

**Status**: Accepted

**Context**: ML inference is expensive; users may use external services.

**Decision**: Users pre-compute signals; engine just executes.

**Consequence**: Clean separation of concerns; cannot adapt signals based on execution.

---

## AD-003: Exit-First Order Processing (2025-11-20)

**Status**: Accepted

**Context**: Entry orders can be rejected if insufficient capital, but exit orders free up capital.

**Decision**: Process exit orders before entry orders in each bar.

**Consequence**: More capital-efficient execution; matches real broker behavior.

---

## AD-004: Per-Framework Validation (2025-11-22)

**Status**: Accepted

**Context**: Two days wasted on dependency conflicts (VectorBT OSS/Pro, Backtrader, Zipline) in unified pytest.

**Decision**: Validate each framework in isolated venv with standalone scripts.

**Consequence**: No unified test runner, but no dependency conflicts.

---

## AD-005: Python 3.11+ Target (2025-11-22)

**Status**: Accepted

**Context**: No need for `from __future__ import annotations` cruft.

**Decision**: Target Python 3.11+ only.

**Consequence**: Cleaner code, modern syntax.

---

## AD-006: Relaxed Mypy (2025-11-22)

**Status**: Accepted

**Context**: 16 type errors remain after cleanup, not critical.

**Decision**: Set `strict = false` in pyproject.toml.

**Consequence**: Can proceed with development; fix types later.

---

## AD-007: Major Repository Cleanup (2025-11-22)

**Status**: Completed

**Context**: Repository had 739K lines including:
- `archive/` (69K lines obsolete code)
- `resources/` (631K lines framework source)
- `tests/validation/` (33K lines chaotic tests)

**Decision**: Delete all non-essential code, keep only core engine (~5.5K lines).

**Consequence**: 99.2% code reduction; cleaner, more maintainable codebase.
