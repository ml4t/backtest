# ml4t.backtest - Honest Status Report

**Date**: 2025-11-19
**Context**: No marketing fluff - just facts and comparisons

---

## 1. VALIDATION STATUS

### Correctness (vs Other Frameworks)

| Framework | Test | Result | Variance | Status |
|-----------|------|--------|----------|---------|
| **VectorBT Pro** | MA crossover, 504 bars, 13 trades | Match | 0.0003% | ✅ VALIDATED |
| **Backtrader** | MA crossover, 504 bars, 13 trades | Match | 0.0003% | ✅ VALIDATED |
| **Zipline** | MA crossover, 504 bars | Different | +10.8% | ❌ EXCLUDED* |

*Zipline excluded: uses bundle data instead of custom DataFrames, requires significant rewrite

**Evidence**: `FRAMEWORK_VALIDATION_REPORT.md` (lines 1-100)
- Same signals → same trades → same returns (within 0.0003%)
- Test: AAPL 2015-2016, simple MA(10,20) crossover
- All three frameworks produce $9,517.62-$9,517.69 final value on $10K initial

**Recent Multi-Asset Test** (from trade logs):
- ml4t.backtest: 1,159 trades
- VectorBT: 81 trades
- Backtrader: 37 trades

**⚠️ PROBLEM**: 14-31x trade count discrepancy! This needs investigation before claiming correctness.

---

## 2. PERFORMANCE STATUS

### Speed Comparison (Single-Asset MA Crossover)

| Framework | Execution Time | Speed vs ml4t.backtest |
|-----------|----------------|------------------------|
| **Backtrader** | 0.15s | **6.5x FASTER** |
| **VectorBT Pro** | 0.56s | **1.7x FASTER** |
| **ml4t.backtest** | 0.97s | Baseline |

**Evidence**: `FRAMEWORK_VALIDATION_REPORT.md` line 26

**Honest Assessment**: We're the SLOWEST of the three validated frameworks.

### Throughput (Multi-Asset)

**Our measurements:**
- Empty strategy (no logic): 34,751 events/sec
- ML strategy + 3 risk rules: 11,197 events/sec
- Test: 500 stocks, 252 days, 126K events

**Context missing:**
- ❓ VectorBT throughput on same test? Unknown.
- ❓ Backtrader throughput on same test? Unknown.
- ❓ Industry standard for "good"? Unknown.

**Targets claimed** (from benchmark_integrated_system.py lines 11-18):
- Target: 10-30K events/sec with ML + risk (250 symbols)
- Achieved: 11K events/sec (500 symbols)
- Status: Within target range, but need comparison data

---

## 3. MEMORY STATUS

**Our measurement:**
- 31.5 MB for 63K events (baseline empty strategy)
- Test setup: 250 symbols, synthetic data

**Context missing:**
- ❓ Is 31.5 MB good or bad? Unknown.
- ❓ VectorBT memory usage? Unknown.
- ❓ Backtrader memory usage? Unknown.
- ❓ Target: <2GB for 250 symbols × 1 year (claimed), but no validation data

**Honest assessment**: Can't claim "memory efficient" without comparison data.

---

## 4. WHAT WE ACTUALLY KNOW

### ✅ Things We're Confident About:
1. **Correctness**: Produces identical results to VectorBT/Backtrader for simple strategies (validated 2025-11-14)
2. **Point-in-time safety**: No look-ahead bias (architectural guarantee)
3. **Feature completeness**: Supports all major order types, commission models, slippage
4. **Integration**: Data layer + risk layer + ML signals work together
5. **Throughput**: Can process ~11K events/sec for realistic ML strategy

### ❌ Things We Don't Know:
1. **Multi-asset correctness**: Trade count discrepancy (1159 vs 81 vs 37) is unexplained
2. **Relative performance**: No head-to-head comparison on same workload
3. **Memory efficiency**: No comparison data vs other frameworks
4. **Scalability limits**: Untested at 500+ symbols with real data
5. **Production readiness**: No live trading validation

### ⚠️ Known Issues:
1. **Speed**: 1.7-6.5x slower than competitors (single-asset test)
2. **Trade count variance**: Unexplained multi-asset discrepancies
3. **Test coverage**: Only 35% (target: 80%)
4. **No Zipline support**: Can't validate against Zipline with custom data

---

## 5. NEXT STEPS (User Requested)

**Goal**: Proper cross-framework validation with high turnover

### Test Specification:
- **Symbols**: 500
- **Positions**: Always 25 (high turnover)
- **Signals**: Random (reproducible seed)
- **Frameworks**: ml4t.backtest, VectorBT Pro, Backtrader
- **Metrics**: Trades, returns, execution time, memory

### What This Will Tell Us:
1. Do we produce same number of trades? (Critical correctness check)
2. Do we produce same returns? (P&L calculation validation)
3. How fast are we really? (Real performance comparison)
4. How much memory do we use? (Real memory comparison)

### Deliverable:
- Script: `tests/validation/high_turnover_comparison.py`
- Report: `CROSS_FRAMEWORK_BENCHMARK.md`
- Raw data: Trade logs, timings, memory profiles

---

## 6. CONCLUSIONS

**What we can claim:**
- ✅ Correct for simple single-asset strategies
- ✅ Architecturally sound (event-driven, no look-ahead)
- ✅ Feature-complete (orders, commission, slippage, risk)
- ✅ Integrated system works end-to-end

**What we CANNOT claim (yet):**
- ❌ "Fast" - we're actually slower than competitors
- ❌ "Memory efficient" - no comparison data
- ❌ "Production ready" - multi-asset correctness unvalidated
- ❌ "Battle tested" - coverage is 35%, not 80%

**Honest tagline:**
> ml4t.backtest: Correct for simple strategies, architecturally sound, but slower than VectorBT/Backtrader and not yet validated for complex multi-asset scenarios.

---

*This document will be updated as we complete the high-turnover validation test.*
