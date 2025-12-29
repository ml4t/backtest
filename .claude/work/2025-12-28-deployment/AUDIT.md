# ML4T Backtest: Release Readiness Audit

**Library**: ml4t-backtest
**Location**: /home/stefan/ml4t/software/backtest/
**Date**: 2025-12-28

---

## Pre-Audit Notes

**Known Issues for ml4t-backtest:**
- VectorBT validation confirms correctness (exact match)
- Account policies well-tested
- Risk rules have good coverage
- Python version already >=3.11 - good
- 474 tests, 73% coverage
- Most mature for release

---

## Phase 1: Current State Assessment

### 1.1 Package Configuration Audit

Read pyproject.toml and verify against STANDARD configuration:

**STANDARD pyproject.toml requirements:**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ml4t"]
namespaces = true  # Already set!

[project]
requires-python = ">=3.11"  # Already correct!

[tool.ruff]
line-length = 100  # Already correct!
target-version = "py311"  # Already correct!

[tool.mypy]
python_version = "3.11"  # Already correct!
check_untyped_defs = true  # Already set!
```

**Checklist:**
- [ ] `namespaces = true` ✓
- [ ] `requires-python = ">=3.11"` ✓
- [ ] `target-version = "py311"` ✓
- [ ] `line-length = 100` ✓
- [ ] `check_untyped_defs = true` ✓

This library appears to already meet most standards!

### 1.2 Test Suite Audit

#### 1.2.1 Test Performance

```bash
# Time the full test suite
time uv run pytest tests/ -q --tb=no

# Identify slow tests
uv run pytest tests/ --durations=20 -q --tb=no
```

**Target**: <5 minutes for full suite

#### 1.2.2 Test Coverage

```bash
uv run pytest tests/ --cov=ml4t.backtest --cov-report=term-missing -q
```

**Collect:**
1. Total tests (expect 474)
2. Current coverage (expect 73%)
3. Files with <50% coverage
4. Validation tests status

### 1.3 Type Checking Audit (BOTH mypy AND ty)

```bash
# mypy
uv run mypy src/ml4t/backtest/ --show-error-codes

# ty (catches bugs mypy misses!)
uvx ty check src/ml4t/backtest/
```

**Collect:**
1. mypy error count
2. ty error count (target: 0)

### 1.4 Linting Audit

```bash
uv run ruff check src/ tests/ --show-fixes
uv run ruff format src/ tests/ --check
```

### 1.5 Build Test

```bash
uv build
ls dist/
```

---

## Phase 2: Remediation Tasks

### P0: Blockers

- [ ] Fix any build failures
- [ ] Verify version in `__init__.py`

### P1: High Priority

- [ ] Remove `pytest>=8.4.2` from dependencies (should be in dev only!)
- [ ] Verify all validation tests pass

### P2: Medium Priority

- [ ] Add edge case tests for execution
- [ ] Improve coverage from 73% to 80%

### P3: Low Priority

- [ ] Update author info (QuantLab → Stefan Jansen)
- [ ] Clean up validation venvs documentation

---

## Phase 3: Audit Report

```markdown
# ml4t-backtest Release Readiness Report

**Date**: 2025-12-28
**Version**: [from __init__.py]

## Summary

| Category | Status | Notes |
|----------|--------|-------|
| pyproject.toml | ✅ | Mostly compliant |
| Tests | | 474 tests, 73% coverage |
| Type Checking | | |
| Linting | | |
| Build | | |

## Blockers

1. ...

## Remediation Completed

- [ ] ...
```

---

## Phase 4: Execute Remediation

**IMPORTANT**: After completing Phase 1 assessment, EXECUTE the following fixes:

### 4.1 pyproject.toml Fixes (Execute Now)

```python
# CRITICAL: Remove pytest from core dependencies!
# In [project] dependencies, remove: "pytest>=8.4.2"
# pytest should ONLY be in [project.optional-dependencies] dev

# This library is mostly compliant already
```

### 4.2 Verify Configuration

This library appears to already meet most standards. Verify:
- [ ] `namespaces = true` ✓
- [ ] `requires-python = ">=3.11"` ✓
- [ ] `target-version = "py311"` ✓
- [ ] `line-length = 100` ✓

### 4.3 After Any Changes

```bash
# Run tests to verify nothing broke
uv run pytest tests/ -q

# Verify build still works
uv build
```

### 4.4 Create Audit Report

After executing remediation, create `REPORT.md` with:
- Confirmation of compliance
- Any issues found
- Ready for TestPyPI: YES/NO

---

## Execution Commands

```bash
cd /home/stefan/ml4t/software/backtest

# Phase 1: Assessment
uv run pytest tests/ --cov=ml4t.backtest --cov-report=term-missing -q 2>&1 | head -50
uv run mypy src/ml4t/backtest/ --show-error-codes 2>&1 | tail -20
uv run ruff check src/ tests/ 2>&1 | tail -20

# Phase 4: After any fixes
uv run pytest tests/ -q
uv build
ls dist/
```
