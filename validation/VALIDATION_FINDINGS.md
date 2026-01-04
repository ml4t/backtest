# Cross-Implementation Validation Findings

## Date: 2026-01-02 (Updated)

## Summary

Validation of 4 backtesting implementations:
1. **VBT Pro** (ground truth) - Vectorized
2. **ml4t.backtest** (event-driven) - Python ✅ **EXACT MATCH**
3. **backtest-nb** (vectorized with Numba JIT) - Semantic differences
4. **backtest-rs** (vectorized in Rust) - Semantic differences + scaling bug

## Key Findings

### 1. Single-Asset: Near Exact Match ✅

**Status**: Signal-only mode achieves NEAR EXACT MATCH (within 1 trade difference)

Without trailing stops, all implementations produce identical trades:
- Same entry bars
- Same exit bars
- Same PnL (within floating point tolerance)

The 1-trade difference is due to:
- **VBT Pro**: Counts open positions at end as "trades" (Status=Open)
- **backtest-nb/rs**: Only counts completed round-trip trades

### 2. Trailing Stop Behavior

**ml4t.backtest ✅ MATCHES VBT Pro** when configured with `trail_hwm_source=TrailHwmSource.HIGH`:

| Implementation | Trigger Price | HWM Source | Fill Price |
|---------------|---------------|------------|------------|
| VBT Pro       | LOW price     | HIGH price | Stop level |
| ml4t.backtest | LOW price     | HIGH price (configurable) | Stop level |
| backtest-nb   | LOW price     | HIGH price | Bar low |
| backtest-rs   | LOW price     | HIGH price | Bar low |

**VBT Pro Behavior** (when OHLC data provided):
- HWM updates based on bar HIGH
- Trail triggers when bar LOW breaches stop level
- Fills at stop level (not bar low)

**ml4t.backtest** correctly implements all three behaviors with `TrailHwmSource.HIGH`.

**backtest-nb/rs Difference**:
These implementations fill at BAR_LOW instead of stop level, resulting in slightly
different exit prices (and different trade counts due to position sizing effects).

### 3. Multi-Asset Signal Architecture Difference

**Critical Finding**: VBT Pro and backtest-nb/rs have fundamentally different multi-asset
signal architectures.

| Implementation | Signal Type | Entry Evaluation |
|----------------|-------------|------------------|
| VBT Pro        | Per-asset   | Each asset has independent entry/exit signals |
| backtest-nb    | Shared      | One signal applies to ALL assets simultaneously |
| backtest-rs    | Shared      | One signal applies to ALL assets simultaneously |

**VBT Pro**:
```python
# Each column is a separate asset's signal
entries_df = pd.DataFrame({
    'asset_0': [True, False, False, True],
    'asset_1': [False, True, False, False],
    'asset_2': [True, True, False, False],
})
```

**backtest-nb/rs**:
```python
# Single signal applies to all assets
signals_df = pl.DataFrame({
    'entry': [True, False, False, True],  # All assets enter/exit together
    'exit': [False, True, False, False],
})
```

**Impact**: With per-asset signals, VBT Pro generates N× more trades (where N ≈ n_assets).
With shared signals, all assets enter/exit on the same bars.

### 4. backtest-nb and backtest-rs Match Each Other ✅

When given the same data layout and signals, backtest-nb and backtest-rs produce
**identical results**:
- Same trade counts
- Same final values
- Same PnL

This validates that the Rust implementation correctly mirrors the Numba implementation.

## Fixes Applied

### backtest-rs n_assets Parameter

**File**: `backtest-rs/crates/ml4t-python/src/lib.rs`

Added `n_assets` parameter to both `backtest()` and `sweep()` functions:
- Previously hardcoded to `n_assets = 1`
- Now computes `n_bars = total_rows / n_assets`
- Enables multi-asset backtesting

### Data Layout

Both implementations expect **bar-major layout** for multi-asset data:
```
[bar0_asset0, bar0_asset1, ..., bar1_asset0, bar1_asset1, ...]
```

Index formula: `price_idx = bar * n_assets + asset`

### 5. ml4t.backtest (Event-Driven) ✅ EXACT MATCH

**Status**: EXACT MATCH with VBT Pro when properly configured.

**Key Configuration for VBT Pro Match**:
```python
from ml4t.backtest import Broker, TrailHwmSource
from ml4t.backtest.risk.position import TrailingStop

broker = Broker(
    initial_cash=100_000.0,
    commission_model=NoCommission(),
    slippage_model=PercentageSlippage(0.0005),
    trail_hwm_source=TrailHwmSource.HIGH,  # Critical: Use HIGH for HWM, not CLOSE
)
broker.set_position_rules(TrailingStop(pct=0.03))
```

**Trailing Stop Validation Results (Updated 2026-01-02)**:

| Metric | ml4t.backtest | VBT Pro |
|--------|---------------|---------|
| Entry bar | 2 | 2 |
| Entry price | 101.0505 | 101.0505 |
| Exit bar | 9 | 9 |
| Exit price | 107.1314 | 107.1314 |
| PnL | 608.09 | 608.09 |

**Critical Settings for VBT Pro Compatibility**:
1. `trail_hwm_source=TrailHwmSource.HIGH` - VBT Pro uses bar HIGH for HWM tracking
2. Trailing stop fill uses `_risk_fill_price` mechanism to fill at trail level (not bar low)
3. Slippage applied on top of stop price

**Architectural Differences** (don't affect results when configured correctly):
- Event-driven vs vectorized execution
- Uses Strategy class with `on_data` callback vs `from_signals()`
- Prices DataFrame requires 'asset' column (not 'symbol')
- Context/signals passed via `context` parameter

### 6. Scale Test Results

**Single-asset scale test (100M data points reached)**:

| Bars | VBT Pro (trades) | VBT Pro (ms) | backtest-rs (trades) | backtest-rs (ms) | Match |
|------|------------------|--------------|---------------------|-----------------|-------|
| 10K | 74 | 121 | 74 | 1.7 | ✅ |
| 100K | 745 | 9.2 | 129 | 30.9 | ❌ |
| 1M | 7,477 | 45 | 141 | 451 | ❌ |
| 10M | 10,152 | 405 | 106 | 3,761 | ❌ |
| 100M | 10,287 | 9,339 | 129 | 39,908 | ❌ |

**Issue**: backtest-rs trade count doesn't scale with bars (stays ~130) while VBT Pro scales linearly.
- At 10K bars: EXACT MATCH and 71x faster
- At larger scales: backtest-rs signal indexing issue suspected

**Root cause**: Under investigation. Likely related to how signals are indexed for large datasets.

## Recommendations

### For Exact Match with VBT Pro

1. **Use signal-only mode** (no trailing stops) for exact trade matching
2. **For trailing stops**: Accept semantic difference or modify backtest-nb/rs to use CLOSE instead of LOW
3. **For multi-asset**:
   - Use shared signals if testing backtest-nb/rs architecture
   - Or implement per-asset signal support in backtest-nb/rs

### Future Work

1. Add per-asset signal support to backtest-nb/rs (requires API change)
2. Add `stop_trigger_price` parameter (LOW vs CLOSE) for flexibility
3. Scale testing to 100M data points
4. Trade-by-trade PnL comparison at scale

## Test Commands

```bash
# Single-asset validation
source .venv-vectorbt-pro/bin/activate
python validation/single_asset_exact_match.py

# Multi-asset validation
python validation/multi_asset_exact_match.py

# Trailing stop comparison
python validation/trailing_stop_compare.py
```
