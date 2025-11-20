# Phase 2: VectorBT Exact Matching - Requirements

## Primary Objective

**Reverse engineer VectorBT Pro's backtesting logic and achieve 100% exact matching with ml4t.backtest**

This is NOT about approximation. We need **complete transparency and exact matching** to validate ml4t.backtest.

## Core Requirements

### 1. Reverse Engineering VectorBT Pro

**Must document with precision**:
- How TP (Take Profit) exits are calculated
- How SL (Stop Loss) exits are calculated
- How TSL (Trailing Stop Loss) is tracked and triggered
- Entry price determination (open next bar? close signal bar?)
- Exit price determination (close-based? high/low? intra-bar?)
- Fee calculation (applied to entry, exit, or both?)
- Slippage calculation (how applied?)
- Position sizing with `size=np.inf`
- Exit priority rules (TP vs TSL same bar)

**Sources**:
- VectorBT Pro source code analysis
- Documentation review
- Empirical testing with known edge cases
- Trade-by-trade inspection

### 2. Exact ml4t.backtest Implementation

**Must match VectorBT 100%**:
- Same entry prices
- Same exit prices
- Same exit reasons (TP vs TSL)
- Same fees calculated
- Same slippage applied
- Same position sizes
- Same PnL calculations

**No approximations. No "close enough".**

### 3. Trade-by-Trade Validation

**Success criteria**:
- Extract complete VectorBT trade log (entry/exit prices, timestamps, fees, PnL)
- Run ml4t.backtest with identical entry signals
- Compare every single trade
- Achieve 100% match on test period (Q1 2024, 87 days, ~352 trades)
- Zero tolerance for differences

### 4. Documentation

**Must produce**:
- Complete VectorBT behavior documentation
- Implementation notes for ml4t.backtest
- Trade-by-trade comparison report
- Edge case handling guide
- Verification test suite

## Non-Requirements

### What We DON'T Care About (For This Phase)

- **Signal generation**: Use pre-computed signals from Phase 1
- **Strategy logic**: Simple Donchian breakout is fine
- **ML models**: Not needed yet (Phase 3+)
- **Optimization**: Focus on correctness first, speed later

## Success Metrics

### Absolute Requirements
- ✅ 100% trade count match
- ✅ 100% entry price match (within tick precision)
- ✅ 100% exit price match (within tick precision)
- ✅ 100% fee calculation match
- ✅ 100% PnL match (within rounding precision)
- ✅ Complete VectorBT behavior documentation

### Quality Gates
- ✅ All trades verified individually
- ✅ Edge cases documented and tested
- ✅ Source code analysis complete
- ✅ Reproducible test suite
- ✅ Comprehensive implementation guide

## Test Data

**Source**: Existing VectorBT backtest from prior session
- **Period**: Q1 2024 (Jan 1 - Mar 31, 87 days)
- **Symbol**: BTC futures
- **Signal**: close > donchian_120min_upper (simple breakout)
- **Baseline trades**: 352
- **Baseline PnL**: $46,140.15
- **Exit config**: TP=2.5%, TSL=1.0%
- **Fees**: 0.02%
- **Slippage**: 0.02%

## Technical Constraints

### VectorBT Pro
- Proprietary library (limited source access)
- Vectorized architecture
- Built-in TP/SL/TSL support
- Black-box exit logic

### ml4t.backtest (ML4T)
- Event-driven architecture
- Requires manual exit implementation
- Must replicate VectorBT behavior exactly
- Located at: `/home/stefan/ml4t/software/backtest/src/ml4t.backtest/`

## Dependencies

### Required Files
- Phase 1 indicators: `projects/crypto_futures/data/indicators/BTC_indicators.parquet`
- VectorBT comparison script: `projects/crypto_futures/scripts/compare_vectorbt_vs_ml4t.backtest.py`
- ml4t.backtest library: `backtest/src/ml4t.backtest/`

### Required Knowledge
- VectorBT Pro API and behavior
- ml4t.backtest architecture (Strategy, DataFeed, Broker, Portfolio)
- Event-driven backtesting patterns
- Point-in-time correctness

## Deliverables

### 1. Documentation
- `VECTORBT_BEHAVIOR_SPEC.md` - Complete VectorBT behavior specification
- `ML4T_BACKTEST_IMPLEMENTATION_GUIDE.md` - How to replicate VectorBT in ml4t.backtest
- `TRADE_COMPARISON_REPORT.md` - Trade-by-trade validation results

### 2. Code
- ml4t.backtest strategy with exact VectorBT matching
- Trade extraction and comparison scripts
- Verification test suite

### 3. Validation
- 100% matching proof on test period
- Edge case test results
- Performance benchmarks (optional)

## Why This Matters

**We cannot validate ml4t.backtest without understanding the reference implementation**

- If we don't know how VectorBT calculates exits, we can't verify correctness
- Approximate matching hides bugs and implementation differences
- Need complete transparency to:
  - Trust ml4t.backtest for production use
  - Implement strategies correctly
  - Debug issues when they arise
  - Extend ml4t.backtest with confidence

**This is foundational validation work. Must be done right.**

## Acceptance Criteria

Phase 2 is complete when:

1. ✅ VectorBT behavior fully documented (every calculation)
2. ✅ ml4t.backtest implements identical logic (verified by code review)
3. ✅ 100% trade-by-trade matching on test period
4. ✅ Edge cases identified and tested
5. ✅ Complete implementation guide exists
6. ✅ Verification test suite passes

**Zero tolerance for "good enough" - must be exact.**

---

**Priority**: CRITICAL
**Estimated Duration**: 1-2 weeks
**Complexity**: High (reverse engineering + exact matching)
**Risk**: Medium (VectorBT is black box, may need trial-and-error)
