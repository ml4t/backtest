# Test Coverage Improvement Plan: ml4t-backtest

## Objective

Increase test coverage from **83.7%** to **92%** target.

## Current State

- **Coverage**: 83.7% (2,104 / 2,514 lines covered)
- **Tests**: 527 tests passing (154 in main suite)
- **Missing**: ~410 lines need coverage
- **Architecture**: Minimal core (~2,800 lines), event-driven

## Priority Files (Coverage Gaps)

### Tier 1: Critical Gap

| File | Coverage | Missing Lines | Priority |
|------|----------|---------------|----------|
| `analysis.py` | 22.9% | 108 | **CRITICAL** |

This is the main coverage gap - a single file with 77% of its code untested.

### Tier 2: Moderate Gaps

| File | Coverage | Missing Lines | Priority |
|------|----------|---------------|----------|
| `engine.py` | 64.7% | 30 | HIGH |
| `broker.py` | 73.6% | 100 | HIGH |
| `models.py` | 74.6% | 15 | MEDIUM |
| `config.py` | 74.8% | 31 | MEDIUM |
| `calendar.py` | 79.1% | 36 | MEDIUM |
| `analytics/equity.py` | 80.0% | 13 | LOW |

### Tier 3: Minor Gaps (>85%)

| File | Coverage | Missing Lines |
|------|----------|---------------|
| `execution/rebalancer.py` | 85.3% | 19 |
| `analytics/metrics.py` | 85.7% | 10 |
| `risk/types.py` | 90.6% | 6 |
| `accounting/account.py` | 93.5% | 3 |
| `types.py` | 94.0% | 9 |
| `analytics/trades.py` | 94.7% | 10 |
| `accounting/policy.py` | 95.0% | 5 |

## Analysis of Critical Gap: analysis.py

```bash
# First, understand what's in analysis.py
cat src/ml4t/backtest/analysis.py | head -50

# See exactly which lines are missing
uv run pytest tests/ --cov=src/ml4t/backtest/analysis.py --cov-report=term-missing -q
```

The `analysis.py` module likely contains post-backtest analysis functions that aren't exercised by the main test suite. This needs dedicated test coverage.

## Testing Strategy

### 1. Focus on analysis.py First

This single file will give you ~10% coverage improvement. Likely contains:
- Performance analysis functions
- Trade analysis utilities
- Result aggregation
- Report generation

Create `tests/test_analysis.py` with comprehensive tests.

### 2. Engine and Broker Edge Cases

The `engine.py` (64.7%) and `broker.py` (73.6%) gaps are likely:
- Error handling paths
- Edge cases (empty data, single trade, etc.)
- Cancellation/rejection scenarios
- Position limits and risk checks

### 3. Leverage Existing Validation Tests

The library has VectorBT validation tests in `validation/`. These can guide what scenarios to test:
- Check `validation/vectorbt_pro/` for reference scenarios
- Ensure edge cases from validation are covered in unit tests

## Execution Plan

### Phase 1: analysis.py (Critical)

1. Read and understand `analysis.py` structure
2. Create `tests/test_analysis.py`
3. Test all public functions
4. Cover error handling paths

**Target**: 22.9% → 80%+ coverage
**Estimated effort**: 2-3 hours

### Phase 2: Engine Edge Cases

1. Identify uncovered lines in `engine.py`
2. Add edge case tests:
   - Empty data feed
   - Strategy that generates no signals
   - Multiple fills per bar
   - Order rejection scenarios

**Target**: 64.7% → 90%+ coverage
**Estimated effort**: 1-2 hours

### Phase 3: Broker Edge Cases

1. Review `broker.py` uncovered lines
2. Add tests for:
   - Position limit violations
   - Insufficient funds scenarios
   - Partial fills
   - Cancel/modify order paths

**Target**: 73.6% → 90%+ coverage
**Estimated effort**: 1-2 hours

### Phase 4: Remaining Files

Quick wins on remaining files:
- `config.py`: Test all preset configurations
- `calendar.py`: Test holiday handling, session times
- `models.py`: Test all commission/slippage models

**Estimated effort**: 1 hour

## Test Template for Backtest Library

```python
"""Tests for analysis module."""
import pytest
import polars as pl
from datetime import date
from ml4t.backtest.analysis import (
    calculate_returns,
    analyze_trades,
    compute_drawdown,
    # ... other functions
)
from ml4t.backtest.types import Trade, Position, Fill

class TestAnalysis:
    """Test suite for analysis functions."""

    @pytest.fixture
    def sample_trades(self) -> list[Trade]:
        """Create sample trade list."""
        return [
            Trade(
                entry_date=date(2020, 1, 15),
                exit_date=date(2020, 2, 1),
                symbol="SPY",
                side="long",
                quantity=100,
                entry_price=300.0,
                exit_price=310.0,
                pnl=1000.0,
            ),
            # Add more sample trades
        ]

    @pytest.fixture
    def sample_equity_curve(self) -> pl.DataFrame:
        """Create sample equity curve."""
        return pl.DataFrame({
            "datetime": pl.date_range(date(2020, 1, 1), date(2020, 12, 31), eager=True),
            "equity": [10000 + i * 10 for i in range(366)],
        })

    def test_calculate_returns_basic(self, sample_equity_curve):
        """Test basic returns calculation."""
        returns = calculate_returns(sample_equity_curve)
        assert len(returns) == len(sample_equity_curve) - 1
        assert returns.dtypes[0] == pl.Float64

    def test_calculate_returns_empty(self):
        """Test with empty equity curve."""
        empty = pl.DataFrame({"datetime": [], "equity": []})
        returns = calculate_returns(empty)
        assert len(returns) == 0

    def test_analyze_trades_no_trades(self):
        """Test analysis with no trades."""
        result = analyze_trades([])
        assert result["total_trades"] == 0
        assert result["win_rate"] == 0.0

    def test_compute_drawdown(self, sample_equity_curve):
        """Test drawdown computation."""
        dd = compute_drawdown(sample_equity_curve)
        assert "drawdown" in dd.columns
        assert "max_drawdown" in dd.columns
        assert dd["drawdown"].min() <= 0  # Drawdowns are negative

class TestEngineEdgeCases:
    """Edge case tests for Engine."""

    def test_empty_data_feed(self):
        """Test engine with empty data."""
        # ... test implementation

    def test_no_signals_strategy(self):
        """Test strategy that never generates signals."""
        # ... test implementation

    def test_order_rejection(self):
        """Test handling of rejected orders."""
        # ... test implementation

class TestBrokerEdgeCases:
    """Edge case tests for Broker."""

    def test_insufficient_funds(self):
        """Test order when funds insufficient."""
        # ... test implementation

    def test_position_limit_exceeded(self):
        """Test when position limit would be exceeded."""
        # ... test implementation

    def test_partial_fill(self):
        """Test partial order fill scenario."""
        # ... test implementation
```

## Commands Reference

```bash
# Run all tests with coverage
uv run pytest tests/ --cov=src/ml4t/backtest --cov-report=term-missing -q

# Focus on specific module
uv run pytest tests/ --cov=src/ml4t/backtest/analysis.py --cov-report=term-missing

# Run validation tests
uv run pytest validation/ -v

# Generate HTML coverage report
uv run pytest tests/ --cov=src/ml4t/backtest --cov-report=html

# Run specific test file
uv run pytest tests/test_analysis.py -v
```

## Validation Reference

The library has validated exact matches with:
- VectorBT Pro
- VectorBT OSS
- Backtrader

Use validation test patterns as reference for expected behavior:
```bash
# Review validation tests
cat validation/vectorbt_pro/test_*.py
```

## Success Criteria

- [ ] Coverage reaches 92% (from 83.7%)
- [ ] `analysis.py` reaches 80%+ coverage
- [ ] `engine.py` reaches 90%+ coverage
- [ ] `broker.py` reaches 85%+ coverage
- [ ] No test regressions (527+ tests pass)
- [ ] Edge cases for order handling covered
- [ ] Error paths tested

## Architecture Notes

- **Event-driven**: Tests should verify event sequence
- **Exit-first processing**: Ensure exit orders process before entries
- **Point-in-time correctness**: No look-ahead bias in tests
- **Cash/Margin accounts**: Both policies need test coverage

## Estimated Total Effort

| Phase | Effort |
|-------|--------|
| Phase 1 (analysis.py) | 2-3 hours |
| Phase 2 (engine.py) | 1-2 hours |
| Phase 3 (broker.py) | 1-2 hours |
| Phase 4 (remaining) | 1 hour |
| **Total** | **5-8 hours** |

This is the easiest library to reach target coverage due to:
1. Single critical gap (analysis.py)
2. Already high baseline (83.7%)
3. Well-structured codebase
4. Existing validation tests as reference
