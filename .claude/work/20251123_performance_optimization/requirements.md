# Requirements: ml4t.backtest Performance Optimization

## Overview

Optimize the ml4t.backtest event-driven backtesting engine to achieve 3-5x performance improvement while maintaining API compatibility and correctness guarantees.

## Current State

### Benchmark Results (500 assets × 10yr daily = 1.26M events)
| Framework | Runtime | Memory | Notes |
|-----------|---------|--------|-------|
| VectorBT Pro | 0.9s | 98 MB | Vectorized reference |
| **ml4t.backtest** | **31.8s** | **171 MB** | Current baseline |
| Zipline | 196s | 461 MB | Event-driven |
| Backtrader | 298s | 1,635 MB | Event-driven |

### Target
- **Runtime**: 8-10s (3-4x improvement)
- **Memory**: No increase from 171 MB
- **API**: 100% backward compatible

## Profiling Results (Validated)

### Top Bottlenecks by Self-Time
| Bottleneck | Self-Time | Calls | Root Cause |
|------------|-----------|-------|------------|
| `dict.get` | 17.3s | 24M | String key hashing |
| Polars `iter_rows` | 11.0s | 2.78M | Row iteration overhead |
| `engine.run` loop | 10.6s | 2 | Python loop overhead |
| `broker._execute_fill` | 10.1s | 252K | Fill processing |
| `datafeed.__next__` | 9.2s | 5K | Per-bar yielding |
| Enum `__eq__` | 7.3s | 5.66M | OrderType comparisons |
| Polars `row_tuples` | 6.6s | 7.8K | Row→tuple conversion |
| Pandas `Series.__init__` | 4.7s | 1.26M | Series creation |
| `isinstance` | 4.6s | 66M | Type checking |

## Functional Requirements

### FR-1: Hybrid Data Architecture
Replace dictionary-based data access with NumPy array indexing:
- Map asset strings to integer IDs (0 to N-1)
- Store OHLCV as `(n_bars, n_assets)` NumPy arrays
- Access via `prices[t_idx, asset_idx]` instead of `prices_dict[asset]`

### FR-2: Pre-Materialized DataFeed
Convert data to NumPy arrays at initialization, not per-bar:
- One-time pivot from Polars to NumPy at feed creation
- Yield only `(timestamp, t_idx)` per bar
- Engine accesses arrays directly via index

### FR-3: Struct-of-Arrays (SOA) Position Storage
Replace `dict[str, Position]` with parallel NumPy arrays:
- `pos_quantities: np.ndarray` (n_assets,)
- `pos_entry_prices: np.ndarray` (n_assets,)
- Reconstruct Position objects on-demand in `get_position()`

### FR-4: Vectorized Risk Checks
Replace per-position iteration with NumPy masking:
- `hit_stops = (lows < entry_prices * 0.95) & (quantities > 0)`
- Only iterate assets that triggered (typically few per bar)

### FR-5: Reduce Enum Overhead
Replace Enum comparisons with integer flags where performance-critical:
- Define `ORDER_SIDE_BUY = 1`, `ORDER_SIDE_SELL = -1`
- Use int comparisons in hot paths
- Keep Enum for public API

## Non-Functional Requirements

### NFR-1: API Backward Compatibility
All existing public interfaces must work unchanged:
- `Engine.run()` returns same result structure
- `broker.get_position(asset)` returns Position object
- `broker.submit_order()` accepts same parameters
- Strategy `on_data()` callback unchanged

### NFR-2: Correctness Preservation
All validation tests must continue to pass:
- VectorBT Pro: EXACT MATCH
- VectorBT OSS: EXACT MATCH
- Backtrader: EXACT MATCH
- Zipline: EXACT MATCH

### NFR-3: Memory Efficiency
Memory usage should not increase significantly:
- Target: ≤200 MB for 500 assets × 10yr
- NumPy arrays typically more memory-efficient than dicts

### NFR-4: Maintainability
Keep code readable and maintainable:
- Clear separation between internal (arrays) and external (objects) representations
- Document the dual-representation pattern
- Preserve type hints where possible

## Out of Scope

- Numba JIT compilation (Phase 2)
- Cython/Rust extensions (Phase 3)
- Multi-threading (GIL limitations)
- GPU acceleration

## Dependencies

- NumPy (already installed)
- Polars (already used for data loading)
- No new dependencies required

## Acceptance Criteria

- [ ] All 206 unit tests pass
- [ ] All 4 framework validations pass (EXACT MATCH)
- [ ] Runtime < 12s for daily_baseline benchmark
- [ ] Memory ≤ 200 MB for daily_baseline benchmark
- [ ] No breaking changes to public API

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API breakage | Medium | High | Comprehensive test suite, adapter layer |
| Float precision issues | Low | Medium | Use same float64 throughout |
| Index off-by-one bugs | Medium | High | Extensive unit tests for boundaries |
| Complexity increase | Medium | Medium | Clear documentation, SOA pattern |

## References

- `.claude/work/20251123_performance_optimization/gemini_01.md` - External code review
- cProfile results from session (validated Gemini's analysis)
- VectorBT Pro as vectorized reference implementation
