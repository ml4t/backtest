# Architectural Decisions

*Updated: 2025-11-22*

## AD-001: Zipline Validation with Custom Bundle (2025-11-14, Updated 2025-11-23)

**Status**: Superseded - Zipline validation now works

**Original Context**: Zipline's `run_algorithm(bundle='quandl')` fetches its own price data.

**Solution Found**: Create custom bundle with test data using `register()` and `ingest()`.
See `validation/benchmark_suite.py` lines 778-857 for bundle creation pattern.

**Current Status**: Zipline validation achieves 100% match with custom bundles.
Run with: `.venv-validation/bin/python3 validation/zipline/scenario_01_long_only.py`

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

## AD-004: Validation Environment Strategy (2025-11-22, Updated 2025-11-23)

**Status**: Accepted (refined)

**Context**: VectorBT OSS and Pro cannot coexist (pandas .vbt accessor conflict).

**Decision**: Use TWO validation environments:
- `.venv-validation` - VBT OSS + Backtrader + Zipline-Reloaded (can coexist)
- `.venv-vectorbt-pro` - VBT Pro only (commercial, separate)

**Setup**:
```bash
# Create consolidated validation venv
uv venv .venv-validation --python 3.12
source .venv-validation/bin/activate
uv pip install vectorbt backtrader zipline-reloaded exchange_calendars
uv pip install -e .  # Install ml4t.backtest
```

**Consequence**: Only 2 validation venvs instead of 8. All open-source frameworks share one env.

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
