# QEngine Testing Guide

## Quick Start

### Run Tests with Coverage
```bash
# All tests (unit + integration)
pytest tests/unit tests/integration --cov=src/qengine --cov-report=term-missing --cov-report=html

# Quick unit tests only
pytest tests/unit -v

# Specific module
pytest tests/unit/test_portfolio.py -v

# With coverage for specific module
pytest tests/unit/test_portfolio.py --cov=src/qengine/portfolio --cov-report=term-missing
```

### View Coverage Reports
```bash
# Terminal summary (always shown with --cov-report=term-missing)
pytest tests/ --cov=src/qengine --cov-report=term-missing

# HTML report (detailed, interactive)
pytest tests/ --cov=src/qengine --cov-report=html
# Then open: htmlcov/index.html

# JSON report (for CI/analysis)
pytest tests/ --cov=src/qengine --cov-report=json
```

## Coverage Configuration

Coverage is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = [
    "-ra",
    "--strict-markers",
    "--cov=qengine",
    "--cov-report=term-missing",
    "--cov-report=html",
]

[tool.coverage.run]
source = ["src/qengine"]
omit = [
    "*/tests/*",
    "*/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
    "pass",
    "@abstractmethod",
]
```

## Test Organization

```
tests/
├── unit/                      # Unit tests (fast, isolated)
│   ├── test_broker.py
│   ├── test_portfolio.py
│   ├── test_orders.py
│   └── ...
├── integration/               # Integration tests (cross-module)
│   ├── test_engine_integration.py
│   ├── test_strategy_integration.py
│   └── ...
└── validation/                # External validation (slow)
    └── test_vectorbtpro_performance.py
```

## Test Markers

Use markers to selectively run tests:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Run only specific markers
pytest -m "unit and not slow"
```

Available markers:
- `unit`: Fast unit tests
- `integration`: Integration tests
- `slow`: Slow-running tests
- `benchmark`: Performance benchmarks

## Coverage Targets

| Module Type | Minimum | Target | Excellent |
|-------------|---------|--------|-----------|
| Core | 80% | 90% | 95% |
| Execution | 75% | 85% | 90% |
| Portfolio | 80% | 90% | 95% |
| Strategy | 70% | 80% | 85% |
| Data | 70% | 80% | 85% |
| Reporting | 60% | 70% | 80% |

## Writing Good Tests

### Test Structure (AAA Pattern)
```python
def test_feature_name():
    # Arrange - Set up test data
    portfolio = Portfolio(initial_cash=10000)

    # Act - Execute the functionality
    portfolio.update_position("AAPL", 100, 150.0)

    # Assert - Verify the results
    assert portfolio.cash == 5000.0
    assert portfolio.positions["AAPL"].quantity == 100
```

### Coverage Quality
- **Focus on behavior, not lines**: Test what the code does, not just execute lines
- **Test edge cases**: Empty inputs, boundary values, error conditions
- **Test error paths**: Exception handling, validation failures
- **Avoid trivial tests**: Don't test getters/setters unless they have logic

### Example: Good vs Bad Coverage

❌ **Bad** (high coverage, low value):
```python
def test_portfolio_get_cash():
    portfolio = Portfolio(initial_cash=10000)
    assert portfolio.cash == 10000  # Just testing a getter
```

✅ **Good** (meaningful behavior):
```python
def test_portfolio_insufficient_cash_for_order():
    portfolio = Portfolio(initial_cash=1000)

    # Should raise or handle insufficient cash gracefully
    with pytest.raises(InsufficientCashError):
        portfolio.update_position("AAPL", 100, 150.0)  # Needs 15000
```

## Coverage Analysis Tools

### 1. Terminal Report
Shows missed lines directly in terminal:
```
src/qengine/portfolio/portfolio.py    95%   142, 223
```

### 2. HTML Report
Interactive, shows which lines are covered:
- Green: Covered
- Red: Not covered
- Yellow: Partially covered (branches)

### 3. Coverage Badge
Generate for README:
```bash
coverage-badge -o coverage.svg
```

## Common Issues

### Import Errors
If coverage shows 0% for a module:
```bash
# Check Python path
echo $PYTHONPATH

# Run with explicit path
PYTHONPATH=src pytest tests/ --cov=src/qengine
```

### Missing Tests
Coverage shows untested code:
1. Check the HTML report for specific lines
2. Identify untested branches/edge cases
3. Write focused tests for those areas

### Flaky Tests
Tests that fail intermittently:
1. Check for timing dependencies
2. Check for shared state
3. Use fixtures for proper isolation

## CI Integration

### GitHub Actions Example
```yaml
- name: Run tests with coverage
  run: |
    pytest tests/ --cov=src/qengine --cov-report=term-missing --cov-report=xml

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml

- name: Enforce minimum coverage
  run: |
    coverage report --fail-under=80
```

## Current Status

**Overall Coverage**: 81%
**Critical Gaps**: Engine (19%), Broker (20%), Clock (21%)
**Test Failures**: 43 (from recent refactoring)

See `.claude/reference/COVERAGE_ANALYSIS.md` for detailed analysis.

## Next Steps

1. **Fix broken tests** (43 failures from broker refactoring)
2. **Core modules to 80%** (Engine, Broker, Clock)
3. **Overall to 90%** (comprehensive test suite)
4. **Enforce in CI** (fail builds below 80%)

---

*For detailed coverage analysis, see: `.claude/reference/COVERAGE_ANALYSIS.md`*
