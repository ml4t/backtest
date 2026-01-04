# ml4t.backtest Validation Report

**Generated**: 2026-01-03T16:50:20.466205

## Summary

- **Passed**: 5
- **Failed**: 0
- **Skipped**: 0
- **Total Time**: 15.8s

## Results

| Test | Status |
|------|--------|
| Unit Tests | ✅ Pass |
| Calendar Tests | ✅ Pass |
| Calendar Scale | ✅ Pass |
| Rebalancing Scale | ✅ Pass |
| Short Selling | ✅ Pass |

## Validation Details

### Core Tests
- Unit tests: 645+ tests covering all modules
- Calendar integration: 10 tests for session enforcement
- Short selling: 4 tests including PnL correctness

### Scale Tests
- Calendar scale: 50 assets × 30 days minute data
- Rebalancing: 100-200 assets, 5-10 years
- Backtrader: 100 assets × 10 years (12,600 trades)

### Framework Matching
- VectorBT Pro: 100% exact match (all 10 scenarios)
- Backtrader: 100% exact match for entry/exit logic
- Backtrader stop-loss: Known semantic differences documented
