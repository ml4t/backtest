# Repository Audit - 2025-11-22

## Status: CLEANUP COMPLETE

### Summary

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Total Python LOC | 739,544 | 5,566 | 99.2% |
| Source code | 2,735 | 2,788 | - |
| Tests | 35,331 | 2,704 | 92.3% |
| Archive (removed) | 69,889 | 0 | 100% |
| Resources (removed) | 630,843 | 0 | 100% |

### Quality Checks
- ✅ Ruff: All checks passed
- ✅ Tests: 154 passed
- ✅ Code review package: `code_review_package.txt` (34,658 tokens)

---

## Previous State: CHAOS

### Line Count Breakdown

| Directory | Lines | % of Total | Status |
|-----------|-------|------------|--------|
| **src/** | 2,735 | 0.4% | KEEP - Core engine |
| **tests/accounting/** | 2,186 | 0.3% | KEEP - Unit tests |
| **tests/test_core.py** | 501 | <0.1% | KEEP - Unit tests |
| **tests/validation/** | 32,643 | 4.4% | REVIEW - Mostly bloat |
| **archive/** | 69,889 | 9.5% | DELETE - Already archived |
| **resources/** | 630,843 | 85.3% | MOVE OUT - Reference code |
| **Total** | 739,544 | 100% | |

### Core Engine (KEEP): 2,735 lines

```
src/ml4t/backtest/
├── __init__.py              (92)   - Package exports
├── broker.py                (463)  - Core broker logic
├── config.py                (439)  - Configuration
├── datafeed.py              (89)   - Data feed
├── engine.py                (263)  - Backtest engine
├── models.py                (111)  - Data models
├── strategy.py              (28)   - Strategy base
├── types.py                 (100)  - Type definitions
└── accounting/
    ├── __init__.py          (27)
    ├── account.py           (223)
    ├── gatekeeper.py        (272)
    ├── models.py            (92)
    └── policy.py            (536)
```

### Tests to KEEP: ~3,200 lines

```
tests/
├── __init__.py
├── test_core.py             (501)  - Core engine tests
├── accounting/
│   ├── test_account.py
│   ├── test_gatekeeper.py   (518)
│   ├── test_models.py
│   └── test_policy.py
└── conftest.py              (shared fixtures)
```

### Validation Directory: NEEDS COMPLETE RESTRUCTURE

Current state (32,643 lines):
- 27 markdown report files (DELETE - historical noise)
- 50+ Python files (REVIEW each)
- Multiple duplicated scripts
- Debug scripts that should never have been committed

## Proposed Clean Structure

```
backtest/
├── src/ml4t/backtest/       # Core engine (2,735 lines)
├── tests/                   # Unit/integration tests
│   ├── conftest.py
│   ├── test_core.py
│   └── accounting/
├── validation/              # Per-framework validation (SEPARATE from pytest)
│   ├── README.md            # Validation strategy document
│   ├── common/              # Shared utilities
│   │   ├── data.py          # Test data generation
│   │   ├── signals.py       # Signal generation
│   │   └── compare.py       # Trade/P&L comparison
│   ├── vectorbt_pro/        # VectorBT Pro validation
│   │   ├── venv/            # Dedicated venv (or symlink)
│   │   ├── run_validation.py
│   │   └── scenarios/       # Test scenarios
│   ├── backtrader/          # Backtrader validation
│   │   ├── venv/
│   │   ├── run_validation.py
│   │   └── scenarios/
│   └── zipline/             # Zipline validation (optional)
├── docs/                    # Documentation
├── pyproject.toml
└── CLAUDE.md
```

## Files to DELETE

### 1. archive/ directory (69,889 lines)
Already archived, should be removed entirely or moved to a separate repo.

### 2. resources/ directory (630,843 lines)
Reference framework source code. Should be:
- Moved to a separate location outside this repo
- Downloaded on-demand when needed for reference

### 3. tests/validation/ cleanup

**DELETE - Markdown reports (27 files):**
- ADAPTER_IMPLEMENTATION_GUIDE.md
- ALIGNMENT_SUCCESS_REPORT.md
- BACKTRADER_ALIGNMENT_SUCCESS.md
- COMPREHENSIVE_VALIDATION_REPORT.md
- FRAMEWORK_ALIGNMENT_RESULTS.md
- FRAMEWORK_GUIDE.md
- INTEGRATED_VALIDATION_STATUS.md
- NEXT_OPEN_FIX_DESIGN.md
- PROGRESS_REPORT.md
- RECONCILIATION_FINDINGS.md
- SIGNAL_BASED_VALIDATION.md
- SIGNAL_BASED_VALIDATION_PLAN.md
- SIGNAL_DIVERSITY_RESULTS.md
- SIGNAL_VALIDATION_ARCHITECTURE.md
- STATUS.md
- SYSTEMATIC_DIFFERENCES_ANALYSIS.md
- TASK-001_COMPLETION_REPORT.md
- TASK-003_COMPLETION_REPORT.md
- TRADE_COMPARISON_DESIGN.md
- VALIDATION_ARCHITECTURE.md
- VALIDATION_FINDINGS.md
- VALIDATION_REPORT.md
- VALIDATION_RESULTS.md
- VALIDATION_ROADMAP.md
- VALIDATION_SUMMARY.md
- VARIANCE_ANALYSIS.md
- VECTORBTPRO_INITIAL_FINDINGS.md

**DELETE - Debug/temporary scripts:**
- analyze_trade_variance.py
- analyze_variance.py
- check_final_trades.py
- check_vectorbt_34th_trade.py
- debug_final_signal.py
- debug_multi_asset.py
- debug_vectorbt_real.py
- debug_vectorbt_trades.py
- detailed_trade_comparison.py
- diagnose_differences.py
- diagnose_execution_timing.py
- expose_the_truth.py
- final_truth_test.py
- quick_test.py
- simple_trade_comparison.py
- verify_actual_agreement.py
- verify_vectorbt_execution.py
- verify_vectorbt_real.py
- zipline_bundle_free_example.py
- trade_comparison_fixed.txt
- trade_comparison_output.txt
- coverage.json

**DELETE - Redundant directories:**
- baselines/
- comparison/
- docs/
- htmlcov/
- results/
- test_cases/

## Action Plan

### Phase 1: Immediate Cleanup
1. Delete `archive/` directory
2. Move `resources/` outside repo (or delete - can be re-downloaded)
3. Delete all markdown reports in tests/validation/
4. Delete all debug scripts in tests/validation/

### Phase 2: Restructure Validation
1. Create new `validation/` directory at repo root
2. Move only essential adapters and test infrastructure
3. One directory per framework with isolated venv
4. Document validation strategy

### Phase 3: Verify Quality
1. Run pre-commit (ruff, mypy)
2. Ensure all kept tests pass
3. Prepare code review package

## Expected Final State

| Directory | Lines | Purpose |
|-----------|-------|---------|
| src/ | 2,735 | Core backtest engine |
| tests/ | ~3,000 | Unit tests (pytest) |
| validation/ | ~2,000 | Per-framework validation scripts |
| docs/ | ~500 | Essential documentation |
| **Total** | ~8,000 | Clean, focused codebase |

**Reduction: 739,544 → ~8,000 lines (99% reduction)**
