# Performance Reporting Infrastructure

**Created**: 2025-11-16
**Status**: Implemented
**Purpose**: Robust performance reporting with trade tracking and flexible return exports

## Overview

Implemented complete performance reporting infrastructure that provides:
1. **Trade data** with entry/exit signals and reasons
2. **Portfolio returns** at configurable frequencies (daily, weekly, monthly, event-based)
3. **No metrics** - raw data only (user requirement)

## Architecture

### Data Flow

```
Strategy → Order.metadata → FillEvent.metadata → TradeRecord.metadata
                    ↓
              TradeTracker → get_trades_df()
                    ↓
         BacktestResults.export_trades()
```

```
Portfolio updates → PerformanceAnalyzer.equity_curve
                            ↓
              PerformanceAnalyzer.get_returns(frequency)
                            ↓
              BacktestResults.export_returns()
```

### Key Components

#### 1. TradeTracker Enhancements
**File**: `src/ml4t/backtest/execution/trade_tracker.py`

**Changes**:
- Added `entry_metadata` field to `OpenPosition` (line 88)
- Captures `FillEvent.metadata` on entry (line 219)
- Combines entry + exit metadata in `TradeRecord` (lines 273-277)

**Metadata Structure**:
```python
TradeRecord.metadata = {
    "entry": {
        "strategy": "momentum_top20",
        "rank": 1,
        "momentum_value": 0.0523,
        "reason": "top_1_momentum_0.0523"
    },
    "exit": {
        "strategy": "momentum_top20",
        "rank": 25,  # Dropped out of top 20
        "reason": "momentum_dropped"
    }
}
```

#### 2. PerformanceAnalyzer Enhancements
**File**: `src/ml4t/backtest/portfolio/analytics.py`

**Added Method**: `get_returns(frequency="daily")` (lines 150-202)

**Supported Frequencies**:
- `"event"` - Raw equity curve (one row per fill/update)
- `"daily"` - End-of-day equity + daily returns
- `"weekly"` - End-of-week equity + weekly returns
- `"monthly"` - End-of-month equity + monthly returns

**Implementation**:
- Uses Polars `group_by_dynamic()` for efficient resampling
- Calculates period returns: `(current - previous) / previous`
- Returns DataFrame: `[date, equity, returns]`

#### 3. BacktestResults Class
**File**: `src/ml4t/backtest/results.py` (NEW FILE - 265 lines)

**Purpose**: Unified interface for backtest results export

**Key Methods**:
```python
# Get data
results.get_trades(include_metadata=True) → pl.DataFrame
results.get_returns(frequency="daily") → pl.DataFrame
results.summary() → dict

# Export data
results.export_trades(path) → Path
results.export_returns(path, frequency="daily") → Path
results.export_all(output_dir) → dict[str, Path]
```

**Design Philosophy**:
- Raw data only (no metrics calculations)
- User-friendly API
- Flexible frequency resampling
- Polars DataFrames for performance

#### 4. BacktestEngine Integration
**File**: `src/ml4t/backtest/engine.py`

**Added Method**: `get_results()` (lines 343-363)

**Usage**:
```python
engine = BacktestEngine(...)
engine.run()
results = engine.get_results()
results.export_all("results/")
```

#### 5. Strategy Helper Updates
**File**: `src/ml4t/backtest/strategy/base.py`

**Enhanced Method**: `rebalance_to_weights()`

**Added Parameter**: `metadata_per_asset` (line 517)

**Usage**:
```python
target_weights = {"AAPL": 0.05, "MSFT": 0.05}
asset_metadata = {
    "AAPL": {"rank": 1, "signal": "momentum_high"},
    "MSFT": {"rank": 2, "signal": "momentum_mid"}
}

self.rebalance_to_weights(
    target_weights=target_weights,
    current_prices=prices,
    metadata_per_asset=asset_metadata  # Flows to Order → Fill → TradeRecord
)
```

## Export Data Schema

### trades.parquet

```
Columns:
- trade_id: int64
- asset_id: str
- entry_dt: datetime
- entry_price: float64
- entry_quantity: float64
- entry_commission: float64
- entry_slippage: float64
- entry_order_id: str
- exit_dt: datetime
- exit_price: float64
- exit_quantity: float64
- exit_commission: float64
- exit_slippage: float64
- exit_order_id: str
- pnl: float64
- return_pct: float64
- duration_bars: int64
- direction: str  # "long" or "short"
- metadata: object  # {"entry": {...}, "exit": {...}}
```

### returns_daily.parquet

```
Columns:
- date: date
- equity: float64
- returns: float64  # Daily return as decimal (0.01 = 1%)
```

### returns_event.parquet

```
Columns:
- timestamp: datetime  # Exact fill/update time
- equity: float64
- returns: float64  # Event-to-event return
- date: date
```

## Example Usage

### Momentum Strategy with Signal Tracking

```python
class MomentumStrategy(Strategy):
    def on_market_event(self, event):
        # Calculate rankings
        ranked_assets = sorted(self.momentum_signals.items(),
                              key=lambda x: x[1], reverse=True)

        # Build metadata for each asset
        asset_metadata = {}
        for rank, (asset, momentum) in enumerate(ranked_assets[:20], 1):
            asset_metadata[asset] = {
                "strategy": "momentum_top20",
                "rank": rank,
                "momentum_value": momentum,
                "reason": f"top_{rank}_momentum_{momentum:.4f}"
            }

        # Rebalance with metadata
        self.rebalance_to_weights(
            target_weights=target_weights,
            current_prices=current_prices,
            metadata_per_asset=asset_metadata
        )

# After backtest
results = engine.get_results()
exported_files = results.export_all("results/")

# Analysis in separate script
trades = pl.read_parquet("results/trades.parquet")
print(trades.select(["asset_id", "pnl", "metadata"]))

returns = pl.read_parquet("results/returns_daily.parquet")
print(f"Sharpe: {returns['returns'].mean() / returns['returns'].std() * sqrt(252)}")
```

## Validation

### Metadata Flow Verification

**Confirmed**:
1. ✅ `Order.metadata` exists (execution/order.py:96)
2. ✅ `Order.metadata` → `FillEvent.metadata` (fill_simulator.py:248)
3. ✅ `FillEvent.metadata` → `TradeRecord.metadata` (trade_tracker.py:273-277)
4. ✅ `TradeRecord.to_dict()` includes metadata (trade_tracker.py:51-72)
5. ✅ `get_trades_df()` includes metadata column (trade_tracker.py:297-335)

### Resampling Logic

**Polars group_by_dynamic()**:
- Groups timestamps into calendar periods (1d, 1w, 1mo)
- Takes last equity value in each period (end-of-period snapshot)
- Calculates period-over-period returns
- Efficient for large datasets (100k+ rows)

## Breaking Changes

**None** - All changes are backward compatible:
- TradeRecord.metadata defaults to empty dict
- metadata_per_asset defaults to None
- PerformanceAnalyzer.get_returns() is new method
- BacktestResults is new class

## Testing

### Syntax Validation
```bash
python -m py_compile src/ml4t/backtest/results.py
python -m py_compile src/ml4t/backtest/execution/trade_tracker.py
python -m py_compile src/ml4t/backtest/portfolio/analytics.py
✓ All passed
```

### Integration Test
See: `examples/momentum_top20_strategy.py` (updated to demonstrate features)

## Next Steps for Validation

1. **Run momentum example** to generate actual trade files
2. **Verify metadata** appears in exported trades.parquet
3. **Check daily returns** align with portfolio equity curve
4. **Compare event vs daily** returns to ensure resampling correct

## Future Enhancements (Not Implemented)

- [ ] Signal history tracking (all signals at each timestamp, not just entry/exit)
- [ ] Position-level returns (per-asset equity curves)
- [ ] Benchmark comparison (strategy vs SPY returns)
- [ ] Transaction cost analysis (slippage vs commission breakdown)
- [ ] Risk metrics (VaR, CVaR, max consecutive losses) - if user requests
