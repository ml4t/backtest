# Handoff: 2025-10-04_008

**Date**: 2025-10-04
**Work Unit**: 002_comprehensive_ml4t.backtest_validatio
**Phase**: Phase 0 - Infrastructure Setup

---

## Active Work

Implementing comprehensive cross-framework validation for ml4t.backtest backtesting engine against VectorBT Pro, Zipline-Reloaded, and Backtrader to prove 95%+ correctness.

**Current Sprint**: Building framework adapters and data loaders to enable end-to-end validation testing.

---

## Current State

### Progress: 5/38 Tasks Complete (13.2%)

**Completed**:
- ✅ TASK-001: VectorBT Pro installed in .venv-vectorbt (Python 3.12.3, version 2025.7.27)
- ✅ TASK-002: Zipline-Reloaded installed in .venv-zipline (version 3.1.1)
- ✅ TASK-003: Backtrader installed in .venv-backtrader (version 1.9.78.123)
- ✅ TASK-007: UniversalDataLoader created with 21/21 tests passing
- ✅ TASK-004: VectorBTProAdapter fixed and validated with 18/18 tests passing

**In Progress**: None (ready for next task)

**Next Available**:
- TASK-005: Implement Zipline-Reloaded adapter
- TASK-006: Implement Backtrader adapter

**Blocked**: TASK-008 (baseline verification) - waiting for TASK-005 and TASK-006

### Key Files

**Adapters**:
- `tests/validation/frameworks/vectorbtpro_adapter.py` - Working, 4 strategies supported
- `tests/validation/frameworks/zipline_adapter.py` - Exists, needs validation
- `tests/validation/frameworks/backtrader_adapter.py` - Exists, needs validation
- `tests/validation/frameworks/base.py` - Base classes (ValidationResult, TradeRecord, BaseFrameworkAdapter)

**Data Infrastructure**:
- `tests/validation/data_loader.py` - UniversalDataLoader (376 lines, 21/21 tests)
- `tests/validation/test_data_loader.py` - Comprehensive test suite

**Test Files**:
- `tests/validation/test_vectorbtpro_adapter.py` - 18/18 tests passing

**Work Tracking**:
- `.claude/work/current/002_comprehensive_ml4t.backtest_validatio/state.json` - 38 tasks defined
- `.claude/work/current/002_comprehensive_ml4t.backtest_validatio/implementation-plan.md` - Full plan

---

## Recent Decisions

### 1. Adapter Architecture (TASK-004)

**Decision**: VectorBTProAdapter imports vectorbtpro directly in main venv instead of subprocess execution in isolated .venv-vectorbt.

**Rationale**:
- Simpler and more efficient
- Subprocess adds complexity without benefit for this use case
- VectorBT Pro already isolated in its own installation

**Implementation**: Adapter runs in main `.venv` which has VectorBT Pro 2025.7.27 installed

### 2. Data Loader Design (TASK-007)

**Decision**: Single `UniversalDataLoader` class with framework-specific conversion methods instead of separate loaders per framework.

**API**:
```python
loader = UniversalDataLoader()
data = loader.load_simple_equity_data(ticker="AAPL", start_date="2017-01-01",
                                       end_date="2017-12-31", framework="vectorbt")
```

**Framework converters**:
- `to_vectorbt_format()` - DatetimeIndex, single-symbol OHLCV
- `to_zipline_format()` - UTC timezone required
- `to_backtrader_format()` - Lowercase column names
- `to_ml4t.backtest_format()` - Multi-asset with ticker column

### 3. Test Data Date Ranges (TASK-007)

**Decision**: Use 2017 dates for AAPL tests instead of 2020.

**Reason**: Wiki data ends 2018-03-27, tests need valid date ranges.

**Standard test params**:
```python
ticker="AAPL"
start_date="2017-01-01"
end_date="2017-12-31"
```

---

## Bug Fixes This Session

### VectorBT Pro Adapter Bugs (TASK-004)

**Bug 1: use_numba Parameter**
- **Issue**: VectorBT Pro 2025.7.27 doesn't accept `use_numba=True`
- **Error**: "Portfolio doesn't expect arguments ['use_numba']"
- **Fix**: Line 85-94 in vectorbtpro_adapter.py - removed parameter
- **Status**: ✅ Fixed

**Bug 2: Bollinger Bands API**
- **Issue**: `vbt.BBANDS.run()` signature incompatible
- **Error**: "too many arguments: expected 7, got 8"
- **Fix**: Line 238-242 - use manual pandas calculation
- **Status**: ✅ Fixed

---

## Active Challenges

### None Currently

All blockers resolved:
- ✅ VectorBT Pro `data` module missing → Fixed by copying from main venv
- ✅ VectorBT Pro `use_numba` error → Fixed by removing parameter
- ✅ Bollinger Bands API error → Fixed with manual calculation
- ✅ Test import errors → Fixed with relative imports

---

## Next Steps

### Immediate (Next `/next` Execution)

**Option A: TASK-005 - Zipline Adapter** (Recommended)
1. Read existing `tests/validation/frameworks/zipline_adapter.py`
2. Verify it extends BaseFrameworkAdapter correctly
3. Test with UniversalDataLoader data (UTC timezone required)
4. Create comprehensive unit tests (similar to TASK-004's 18 tests)
5. Handle data bundle format conversion
6. Document timezone handling quirks
7. Update state.json marking TASK-005 complete

**Option B: TASK-006 - Backtrader Adapter**
1. Read existing `tests/validation/frameworks/backtrader_adapter.py`
2. Verify it extends BaseFrameworkAdapter correctly
3. Test with UniversalDataLoader data (lowercase columns required)
4. Create comprehensive unit tests
5. Document known signal execution bugs
6. Update state.json marking TASK-006 complete

**Parallelization**: TASK-005 and TASK-006 have no interdependencies and can be done in either order.

### After Both Adapters Complete

**TASK-008: Baseline Verification Test**
- Run identical simple MA crossover across all 4 frameworks
- Compare results (target: within 10% agreement)
- Prove infrastructure end-to-end functionality
- Unblocks Phase 1 (Tier 1 validation)

---

## Session Context

### Working Directory
```
/home/stefan/ml4t/backtest
```

### Virtual Environments
- **Main**: `.venv` (Python 3.13.5) - has VectorBT Pro, pandas, pytest
- **VectorBT**: `.venv-vectorbt` (Python 3.12.3) - VectorBT Pro 2025.7.27
- **Zipline**: `.venv-zipline` (Python 3.12.3) - zipline-reloaded 3.1.1
- **Backtrader**: `.venv-backtrader` (Python 3.12.3) - backtrader 1.9.78.123

### Data Sources
Location: `~/ml4t/projects/`
- `daily_us_equities/wiki_prices.parquet` - 1980-2018 US equities
- `nasdaq100_minute_bars/2021.parquet` - Minute bars
- `crypto_futures/data/futures/BTC.parquet` - Crypto futures
- `spy_order_flow/spy_features.parquet` - Order flow microstructure

### Test Execution
```bash
# Run specific adapter tests
source .venv/bin/activate
pytest tests/validation/test_vectorbtpro_adapter.py -v

# Run data loader tests
pytest tests/validation/test_data_loader.py -v

# All validation tests
pytest tests/validation/ -v
```

### Git State
- Branch: `main`
- Last commit: "feat: Complete TASK-004 - VectorBT Pro Adapter ✅"
- Clean working directory
- All changes committed

---

## Framework-Specific Notes

### VectorBT Pro (✅ Adapter Complete)

**Strategies Supported**:
1. MovingAverageCrossover - Fast/slow MA signals
2. BollingerBandMeanReversion - BB + RSI mean reversion
3. ShortTermMomentumStrategy - EMA momentum
4. VolumeBreakoutStrategy - Volume breakouts

**Quirks**:
- `final_value` is property, not method (no parentheses)
- `use_numba` parameter not supported in 2025.7.27
- BBANDS API unstable - use manual calculation

**Performance**: ~3.7s for 249-day backtest, 31 MB memory

### Zipline-Reloaded (⏳ Adapter Pending)

**Known Requirements**:
- UTC timezone mandatory for DatetimeIndex
- Data bundle format conversion needed
- Complex timezone handling across pandas operations
- Expected in `.venv-zipline` but adapter may import directly

**Hello World Test**: `zipline_test.py` shows API works but needs data bundle

### Backtrader (⏳ Adapter Pending)

**Known Requirements**:
- Lowercase column names (open, high, low, close, volume)
- Known signal execution bugs to document
- Feed format conversion needed

**Hello World Test**: `backtrader_test.py` works (-0.05% return on 100-day test)

---

## Validation Plan Overview

**Phase 0: Infrastructure** (5/9 tasks complete)
- ✅ Framework installations
- ✅ Data loader
- ✅ VectorBT adapter
- ⏳ Zipline adapter (TASK-005)
- ⏳ Backtrader adapter (TASK-006)
- ⏳ Baseline test (TASK-008)

**Phase 1: Tier 1 Core Validation** (0/6 tasks)
- Prove 95%+ agreement across frameworks
- RSI mean reversion, Bollinger breakout, MACD+RSI, multi-asset momentum

**Phase 2-4**: Advanced execution, ML integration, performance

**Phase 5**: Documentation and production readiness

---

## Quality Standards

### Test Coverage
- **Target**: 80% minimum for all new code
- **Achieved**: 100% for UniversalDataLoader, VectorBTProAdapter

### Test Philosophy
- Write test FIRST (TDD)
- Comprehensive edge case coverage
- Data validation tests mandatory
- Performance metric tests included

### Commit Standards
```
feat: Complete TASK-XXX - [Title] ✅

[Description of work]

Acceptance criteria met:
- ✅ [Criterion 1]
- ✅ [Criterion 2]

Files: [key files]
Tests: [test status]
Progress: X/38 tasks complete (Y%)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Command Snippets

### Run Next Task
```bash
/next
```

### Test Commands
```bash
# Test specific adapter
source .venv/bin/activate && pytest tests/validation/test_vectorbtpro_adapter.py -v

# Test data loader
source .venv/bin/activate && pytest tests/validation/test_data_loader.py -v

# Quick smoke test
source .venv/bin/activate && python3 << 'EOF'
from tests.validation.data_loader import UniversalDataLoader
from tests.validation.frameworks.vectorbtpro_adapter import VectorBTProAdapter

loader = UniversalDataLoader()
data = loader.load_simple_equity_data("AAPL", "2017-01-01", "2017-12-31", "vectorbt")
adapter = VectorBTProAdapter()
result = adapter.run_backtest(data, {"name": "MovingAverageCrossover", "short_window": 10, "slow_window": 30}, 10000)
print(f"Return: {result.total_return:.2f}%, Trades: {result.num_trades}")
EOF
```

### State Management
```bash
# View work status
cat .claude/work/current/002_comprehensive_ml4t.backtest_validatio/state.json | jq '.next_available'

# Update task status
cd .claude/work/current/002_comprehensive_ml4t.backtest_validatio && python3 << 'EOF'
import json
with open('state.json', 'r') as f:
    state = json.load(f)
for task in state['tasks']:
    if task['id'] == 'TASK-XXX':
        task['status'] = 'completed'
state['next_available'] = ['TASK-YYY']
with open('state.json', 'w') as f:
    json.dump(state, f, indent=2)
EOF
```

---

## Success Metrics

### Infrastructure Phase (Current)
- ✅ All 3 frameworks installed and hello-world verified
- ✅ Data loader with all format conversions working
- ⏳ All 3 adapters implementing BaseFrameworkAdapter
- ⏳ Baseline test showing <10% variance

### Tier 1 Validation (Next Phase)
- 95%+ agreement on final portfolio value
- ±5% agreement on trade count
- Consistent trade timing
- Sharpe ratio within reasonable bounds

---

## Recovery Commands

If something goes wrong:

```bash
# Check virtual environments
ls -la .venv-*

# Verify installations
source .venv/bin/activate && python -c "import vectorbtpro as vbt; print(vbt.__version__)"
source .venv-zipline/bin/activate && python -c "import zipline; print('ok')"
source .venv-backtrader/bin/activate && python -c "import backtrader; print('ok')"

# Re-run all tests
source .venv/bin/activate && pytest tests/validation/ -v --tb=short

# Check work state
cat .claude/work/ACTIVE_WORK
cat .claude/work/current/002_comprehensive_ml4t.backtest_validatio/metadata.json
```

---

## Recommended Next Action

Run `/next` to automatically select and execute either TASK-005 (Zipline adapter) or TASK-006 (Backtrader adapter). Both are ready to start and have no blockers.

The system will:
1. Read existing adapter implementation
2. Verify BaseFrameworkAdapter compliance
3. Create comprehensive unit tests (18+ tests like TASK-004)
4. Test with UniversalDataLoader
5. Document framework-specific quirks
6. Update state.json
7. Commit with detailed message

**Estimated time**: 2-3 hours per adapter (similar to TASK-004 complexity).
