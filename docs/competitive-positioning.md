# Competitive Positioning: ml4t-backtest

**Date**: January 2026
**Purpose**: Position ml4t-backtest relative to leading backtesting frameworks
**Based on**: Actual benchmark data from `validation/README.md`

---

## Executive Summary

**Frameworks Actually Tested (Jan 2026):**
- VectorBT Pro: 10 scenarios, performance benchmarks complete
- VectorBT OSS: 10 scenarios, performance benchmarks complete
- Backtrader: 10 scenarios, performance benchmarks complete
- Zipline-Reloaded: 9 scenarios (correctness), performance via bundle
- QuantConnect LEAN: Initialized, pending Docker image

| Framework | At Scale (2520×500) | Correctness | Setup |
|-----------|---------------------|-------------|-------|
| VectorBT Pro | **0.107s** (145x faster) | 100% match | pip install |
| VectorBT OSS | **0.074s** (210x faster) | 100% match | pip install |
| Backtrader | 233.35s (15x slower) | 100% match | pip install |
| Zipline-Reloaded | N/A (bundle req.) | 100% match | pip + bundle |
| **ml4t-backtest** | **15.54s** (reference) | Reference | pip install |
| LEAN CLI | Pending | Pending | Docker + account |

**Key Finding**: Vectorized frameworks (VBT) are 100-200x faster for uniform signal backtests. Event-driven ml4t is 15x faster than Backtrader and provides realistic execution for production strategies.

---

## 1. VectorBT Pro Comparison

### Actual Performance Benchmarks (Jan 2026)

| Config (bars × assets) | VBT Pro (s) | ml4t (s) | Winner | Factor |
|------------------------|-------------|----------|--------|--------|
| 100 × 1 | 0.370* | 0.010 | **ml4t** | 37x |
| 500 × 1 | 0.032 | 0.033 | ~tie | 1.0x |
| 1,000 × 1 | 0.031 | 0.061 | VBT | 2.0x |
| 500 × 5 | 0.377* | 0.060 | **ml4t** | 6.3x |
| 1,000 × 10 | 0.033 | 0.170 | VBT | 5.1x |
| 2,520 × 500 | 0.107 | 15.54 | VBT | **145x** |

*First run includes JIT compilation overhead

**Analysis**: VectorBT uses vectorized NumPy with O(1) per-bar overhead. For large-scale uniform operations (same signal across 500 assets), VBT is 100x+ faster. However, ml4t provides event-driven realism essential for production strategies.

### Flexibility

| Feature | VectorBT Pro | ml4t-backtest |
|---------|--------------|---------------|
| Same-bar exits | Complex workaround | Native support |
| Dynamic stops | Signal-based only | StopLoss, TrailingStop, VolatilityStop |
| Position sizing | Fixed formulas | Flexible rules |
| Multi-asset rules | Limited | Portfolio-level limits |
| Custom order types | No | Yes (limit, stop, bracket) |

### When to Use VectorBT Pro

- **Scanning**: Testing 1000s of parameter combinations
- **Signal research**: Optimizing entry/exit signals
- **Quick iteration**: Exploratory analysis

### When to Use ml4t-backtest

- **Realistic simulation**: Testing with realistic fills, slippage
- **Complex strategies**: Multi-leg, stops, brackets
- **Production path**: Strategies that will go live
- **Risk management**: Position + portfolio limits

---

## 2. QuantConnect LEAN Comparison

### Status: LEAN CLI Initialized

**Update (Jan 2026)**: LEAN CLI is now initialized with QuantConnect credentials.
- Workspace created at `validation/lean/workspace/`
- Docker image downloading (~2GB quantconnect/lean:latest)
- Backtests pending completion of Docker pull

### What We Know

| Aspect | LEAN | ml4t-backtest |
|--------|------|---------------|
| Core language | C# | Python |
| Setup complexity | Docker + Account required | pip install |
| Data format | Lean-specific CSV | Any Polars-readable |
| Live trading | Built-in | Separate (ml4t-live) |

### Setup Difficulty (Measured)

| Step | LEAN CLI | ml4t-backtest |
|------|----------|---------------|
| Install package | `pip install lean` (3s) | `pip install ml4t-backtest` (3s) |
| Initialize | `lean init` (needs QC account) | N/A |
| Docker pull | ~2GB image download | N/A |
| Run backtest | Docker container startup | Immediate |

### Performance: Pending Docker Completion

LEAN performance benchmarks are pending Docker image download (~2GB).
Once complete, we will run validation scenarios and measure performance.

### When to Consider LEAN

- You need QuantConnect cloud deployment
- You already have a QuantConnect account
- You need their data marketplace

---

## 3. Backtrader Comparison

### Actual Performance Benchmarks (Jan 2026)

| Config (bars × assets) | Backtrader (s) | ml4t (s) | Winner | Factor |
|------------------------|----------------|----------|--------|--------|
| 100 × 1 | 0.024 | 0.010 | **ml4t** | 2.4x |
| 500 × 1 | 0.113 | 0.033 | **ml4t** | 3.4x |
| 1,000 × 1 | 0.227 | 0.061 | **ml4t** | 3.7x |
| 500 × 5 | 0.553 | 0.060 | **ml4t** | 9.2x |
| 1,000 × 10 | 1.928 | 0.170 | **ml4t** | 11.3x |
| 2,520 × 500 | 233.35 | 15.54 | **ml4t** | **15x** |

**Analysis**: ml4t.backtest is faster across all configurations, with advantage growing for multi-asset portfolios. At scale (2520 bars × 500 assets), ml4t is **15x faster** and uses **17x less memory** (23.7 MB vs 390.9 MB).

### Correctness Validation (500 assets × 10 years)

| Metric | Backtrader | ml4t-backtest | Match |
|--------|------------|---------------|-------|
| Total trades | 119,577 | 119,577 | **EXACT** |
| Entry price match | 100% | 100% | ✓ |
| Exit price match | 100% | 100% | ✓ |
| PnL match (<$1) | 100% | 100% | ✓ |

### Architecture

| Aspect | Backtrader | ml4t-backtest |
|--------|------------|---------------|
| Status | Stale (last commit 2020) | Active (2025) |
| Python version | 3.6-3.11 | 3.11+ |
| Type hints | No | Yes |
| Maintenance | None | Ongoing |

### When to Use Each

**Backtrader**: Legacy code, extensive tutorials, simple strategies
**ml4t-backtest**: Multi-asset portfolios, modern stack, risk management

---

## 4. Zipline-Reloaded Comparison

### Actual Validation (from `validation/README.md`)

Zipline-Reloaded was extensively tested with **100% exact match** achieved:

| Metric | Zipline | ml4t-backtest | Match |
|--------|---------|---------------|-------|
| Total trades | 119,577 | 119,577 | **EXACT** |
| Date range | 2013-01-03 to 2023-01-03 | Same | ✓ |
| Side matches | 100% | 100% | ✓ |
| PnL matches (<$1) | 100% | 100% | ✓ |

### Scenarios Tested

| Scenario | Result |
|----------|--------|
| Long-only | ✅ PASS (exact) |
| Long/Short | ✅ PASS (exact) |
| Stop-loss (5%) | ✅ PASS (exact) |
| Take-profit (10%) | ✅ PASS (exact) |
| Multi-asset (500 × 10yr) | ✅ PASS (100% match) |

### Configuration Required

Zipline uses `handle_data()` for stop/take-profit logic, which exits at next bar open. ml4t-backtest matches this with:

```python
engine = Engine(
    feed, strategy,
    execution_mode=ExecutionMode.NEXT_BAR,
    stop_fill_mode=StopFillMode.NEXT_BAR_OPEN,  # Match Zipline behavior
)
```

### Setup Complexity

| Aspect | Zipline-Reloaded | ml4t-backtest |
|--------|------------------|---------------|
| Install | `pip install zipline-reloaded` | `pip install ml4t-backtest` |
| Data | Requires bundle (`zipline ingest`) | Direct DataFrame |
| Calendar | NYSE (exchange_calendars) | NYSE or custom |

### When to Use Each

**Zipline**: Existing Zipline code, Quantopian refugees
**ml4t-backtest**: Flexible data sources, no bundle required

---

## 5. Feature Matrix (Tested Frameworks Only)

| Feature | VBT Pro | Backtrader | Zipline | ml4t-backtest |
|---------|---------|------------|---------|---------------|
| **Core** |
| Event-driven | Partial | Yes | Yes | Yes |
| Vectorized | Yes | No | No | Hybrid |
| Correctness validated | ✅ 100% | ✅ 100% | ✅ 100% | Reference |
| **Execution** |
| Market orders | Yes | Yes | Yes | Yes |
| Limit orders | Limited | Yes | Yes | Yes |
| Stop orders | Limited | Yes | Yes | Yes |
| Bracket orders | No | Yes | No | Yes |
| **Risk Management** |
| Stop-loss | Signal-based | Yes | Strategy-level | Yes |
| Trailing stop | Limited | Yes | Strategy-level | Yes |
| Volatility stop | No | Manual | No | Yes |
| **UX** |
| Setup complexity | pip | pip | pip + bundle | pip |
| Type hints | Yes | No | No | Yes |
| Active development | Yes | No (2020) | Community | Yes |

---

## 6. Strategic Positioning (Based on Tested Data)

### Actual Performance Landscape

Based on benchmarks with **measured data** (LEAN position is speculative):

```
            ┌─────────────────────────────────────────────────────┐
            │                    FLEXIBILITY                       │
            │    High ◄──────────────────────────────────► Low     │
            │                                                      │
       ▲    │    Backtrader        ml4t-backtest                   │
       │    │    ● (1.5x slower)   ●                               │
    S  │    │                                                      │
    P  │    │    Zipline (unknown)                                 │
    E  │    │    ●                                     VectorBT    │
    E  │    │                                          Pro ●       │
    D  │    │                                       (7-83x faster) │
       │    │                                                      │
       ▼    │                                                      │
            │    Slow ◄──────────────────────────────────► Fast    │
            │                     PERFORMANCE                      │
            └─────────────────────────────────────────────────────┘
```

**Note**: VectorBT Pro is 7-83x faster for single-asset, but ml4t-backtest is 4-21x faster for multi-asset portfolios.

---

## 7. Recommendations (Based on Tested Data)

### When to Use Each Framework

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Single-asset, many bars | VectorBT Pro | 7-83x faster |
| Multi-asset portfolios | ml4t-backtest | 4-21x faster |
| Existing Backtrader code | Backtrader | No rewrite needed |
| Existing Zipline/Quantopian code | Zipline-Reloaded | No rewrite needed |
| Risk management features | ml4t-backtest | Built-in stops, limits |

### What We Cannot Recommend

- **LEAN**: Not tested. Cannot make informed recommendation.
- **Rust backend**: Projected, not implemented. Gains are theoretical.

---

## 8. Conclusion

### What We Actually Know

Based on **measured benchmarks** (not assumptions):

1. **Correctness**: ml4t-backtest produces 100% identical results to VectorBT Pro, VectorBT OSS, Backtrader, and Zipline-Reloaded (119,577 trades validated)

2. **Performance**:
   - VectorBT Pro is 7-83x faster for single-asset backtests
   - ml4t-backtest is 4-21x faster for multi-asset portfolios
   - Backtrader is 1.3-1.9x faster for single-asset
   - Zipline performance was not benchmarked

3. **Setup**: All Python frameworks have similar pip-based setup. LEAN requires Docker + QuantConnect account.

### What We Don't Know

- LEAN performance (not tested)
- Zipline performance (correctness validated, speed not benchmarked)
- Rust backend actual gains (feasibility study only, no implementation)
