# Data Feeds Migration Guide

## Overview

ml4t.backtest supports multiple data feed implementations with different capabilities and performance characteristics. This guide helps you choose the right feed and migrate between them.

## Available Data Feeds

### 1. ParquetDataFeed (Legacy, Stable)

**Use When:**
- Single data source (price data only)
- Simple use cases without ML signals
- Backward compatibility is critical
- You need the most battle-tested implementation

**Features:**
- ✅ Simple API
- ✅ Stable and well-tested
- ✅ Automatic signal column extraction
- ❌ No multi-source merging
- ❌ No lazy loading (loads entire dataset)
- ❌ No FeatureProvider integration

**Example:**
```python
from ml4t.backtest.data import ParquetDataFeed
from ml4t.backtest.engine import BacktestEngine

feed = ParquetDataFeed(
    path="data/prices.parquet",
    asset_id="AAPL",
    signal_columns=["ml_pred", "confidence"],  # Optional
)

engine = BacktestEngine(
    data_feed=feed,
    strategy=my_strategy,
    initial_capital=100000.0,
)

results = engine.run()
```

### 2. PolarsDataFeed (New, Recommended)

**Use When:**
- ML-driven strategies with signals + features
- Large datasets (memory efficiency matters)
- Multi-source data (prices, signals, features from different files)
- You need indicators and market context

**Features:**
- ✅ Lazy loading (memory efficient)
- ✅ Multi-source merging (prices + signals + features)
- ✅ FeatureProvider integration (indicators + context)
- ✅ Signal timing validation (prevent look-ahead bias)
- ✅ 10-50x faster iteration (group_by optimization)
- ✅ Comprehensive data dicts (signals, indicators, context)

**Example:**
```python
from pathlib import Path
from ml4t.backtest.data import PolarsDataFeed
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider
from ml4t.backtest.engine import BacktestEngine

# Optional: Setup feature provider for indicators/context
features_df = pl.read_parquet("data/features.parquet")
feature_provider = PrecomputedFeatureProvider(features_df)

# Create feed with multi-source data
feed = PolarsDataFeed(
    price_path=Path("data/prices.parquet"),
    asset_id="AAPL",
    signals_path=Path("data/ml_signals.parquet"),
    feature_provider=feature_provider,
    validate_signal_timing=True,  # Prevent look-ahead bias
)

engine = BacktestEngine(
    data_feed=feed,
    strategy=my_strategy,
    initial_capital=100000.0,
)

results = engine.run()
```

### 3. CSVDataFeed (Legacy, Simple)

**Use When:**
- Reading from CSV files
- Quick prototyping
- Human-readable data formats

**Features:**
- ✅ CSV support
- ✅ Simple API
- ❌ Slower than Parquet
- ❌ No multi-source merging

## Migration Guide: ParquetDataFeed → PolarsDataFeed

### Step 1: Basic Migration (Price Data Only)

**Before (ParquetDataFeed):**
```python
from ml4t.backtest.data import ParquetDataFeed

feed = ParquetDataFeed(
    path="data/prices.parquet",
    asset_id="AAPL",
)
```

**After (PolarsDataFeed):**
```python
from pathlib import Path
from ml4t.backtest.data import PolarsDataFeed

feed = PolarsDataFeed(
    price_path=Path("data/prices.parquet"),  # Note: price_path instead of path
    asset_id="AAPL",
)
```

**Changes:**
- `path` → `price_path` (explicit naming)
- Use `Path` objects (recommended, not required)

### Step 2: Migration with ML Signals

**Before (ParquetDataFeed with embedded signals):**
```python
# Signals in same file as prices
feed = ParquetDataFeed(
    path="data/prices_with_signals.parquet",
    asset_id="AAPL",
    signal_columns=["ml_pred", "confidence"],
)
```

**After (PolarsDataFeed with separate signal file):**
```python
# Separate price and signal files (recommended)
feed = PolarsDataFeed(
    price_path=Path("data/prices.parquet"),
    asset_id="AAPL",
    signals_path=Path("data/ml_signals.parquet"),
    signal_columns=["ml_pred", "confidence"],  # Optional
    validate_signal_timing=True,  # NEW: Prevent look-ahead bias
)
```

**Benefits:**
- Separate concerns (prices vs signals)
- Signal timing validation
- Can use different signal sources without reprocessing prices

### Step 3: Adding Features and Context

**New capability in PolarsDataFeed:**
```python
import polars as pl
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider

# Prepare feature data (indicators per asset)
features_df = pl.DataFrame({
    "timestamp": [...],
    "asset_id": [...],
    "sma_20": [...],
    "rsi_14": [...],
    "atr_14": [...],
})

# Prepare market context data (shared across assets)
context_df = pl.DataFrame({
    "timestamp": [...],
    "VIX": [...],
    "SPY": [...],
    "market_regime": [...],
})

# Create feature provider
feature_provider = PrecomputedFeatureProvider(
    features_df=features_df,
    context_df=context_df,
)

# Create feed with features
feed = PolarsDataFeed(
    price_path=Path("data/prices.parquet"),
    asset_id="AAPL",
    signals_path=Path("data/signals.parquet"),
    feature_provider=feature_provider,  # NEW
)
```

**Strategy can now access:**
```python
class MyStrategy(Strategy):
    def on_market_event(self, event, context=None):
        # ML signals for decision making
        ml_score = event.signals.get('ml_pred', 0.0)
        confidence = event.signals.get('confidence', 0.0)

        # Per-asset indicators for risk management
        rsi = event.indicators.get('rsi_14', 50.0)
        atr = event.indicators.get('atr_14', 1.0)

        # Market-wide context for regime filtering
        vix = context.get('VIX', 0.0) if context else 0.0
        regime = context.get('market_regime', 'neutral') if context else 'neutral'

        # Trading logic with all three data sources
        if ml_score > 0.7 and vix < 20 and rsi < 70:
            self.buy_percent(event.asset_id, 0.10, event.close)
```

## Performance Comparison

Based on integration tests:

| Feed Type | Throughput (events/sec) | Memory Usage | Use Case |
|-----------|-------------------------|--------------|----------|
| ParquetDataFeed | ~50k-100k | High (full load) | Simple strategies |
| PolarsDataFeed | ~50k-150k | Low (lazy load) | ML strategies |
| CSVDataFeed | ~10k-50k | Medium | Prototyping |

**Note:** PolarsDataFeed's group_by optimization provides 10-50x speedup when processing timestamp batches in multi-asset strategies.

## Feature Flag Support

While PolarsDataFeed is production-ready, you can gradually adopt it:

```python
import os

USE_POLARS_FEED = os.getenv("USE_POLARS_FEED", "false").lower() == "true"

if USE_POLARS_FEED:
    from ml4t.backtest.data import PolarsDataFeed as DataFeed
else:
    from ml4t.backtest.data import ParquetDataFeed as DataFeed

# Rest of code uses DataFeed alias
feed = DataFeed(...)
```

## Backward Compatibility Guarantee

✅ **All existing ParquetDataFeed code continues to work unchanged.**

The BacktestEngine accepts any `DataFeed` implementation through polymorphism. Both feed types implement the same interface:

```python
class DataFeed(ABC):
    def get_next_event(self) -> Event | None: ...
    def peek_next_timestamp(self) -> datetime | None: ...
    def reset(self) -> None: ...
    def seek(self, timestamp: datetime) -> None: ...

    @property
    def is_exhausted(self) -> bool: ...
```

## Common Migration Issues

### Issue 1: Signal Timing Violations

**Problem:** Signals from the future being used in decisions (look-ahead bias)

**Solution:**
```python
feed = PolarsDataFeed(
    price_path=Path("data/prices.parquet"),
    signals_path=Path("data/signals.parquet"),
    validate_signal_timing=True,  # Enable validation
    signal_timing_mode=SignalTimingMode.NEXT_BAR,  # Enforce next-bar usage
    fail_on_timing_violation=True,  # Fail fast if violated
)
```

### Issue 2: Memory Usage with Large Datasets

**Problem:** Running out of memory with 250+ assets

**Solution:** PolarsDataFeed uses lazy loading automatically:
```python
# Lazy loading defers collection until iteration starts
feed = PolarsDataFeed(
    price_path=Path("data/prices.parquet"),  # Not loaded yet
    asset_id="AAPL",
)

# Memory usage starts here (on first get_next_event)
engine.run()
```

**Target:** <2GB for 250 assets × 252 days × daily bars

### Issue 3: Different Column Names

**Problem:** Your data uses different column names than expected

**Solution:** PolarsDataFeed supports custom column names:
```python
feed = PolarsDataFeed(
    price_path=Path("data/prices.parquet"),
    asset_id="AAPL",
    timestamp_column="datetime",  # Default: "timestamp"
    asset_column="symbol",        # Default: "asset_id"
)
```

## Testing Your Migration

### 1. Consistency Test

Verify both feeds produce identical results:

```python
def test_consistency():
    # Old feed
    old_feed = ParquetDataFeed(path="data/prices.parquet", asset_id="AAPL")
    strategy1 = MyStrategy()
    engine1 = BacktestEngine(data_feed=old_feed, strategy=strategy1)
    results1 = engine1.run()

    # New feed
    new_feed = PolarsDataFeed(price_path="data/prices.parquet", asset_id="AAPL")
    strategy2 = MyStrategy()
    engine2 = BacktestEngine(data_feed=new_feed, strategy=strategy2)
    results2 = engine2.run()

    # Compare
    assert results1["events_processed"] == results2["events_processed"]
    assert abs(results1["final_value"] - results2["final_value"]) < 0.01
```

### 2. Performance Test

Measure throughput improvement:

```python
import time

# Old feed
start = time.time()
old_results = engine_with_old_feed.run()
old_duration = time.time() - start

# New feed
start = time.time()
new_results = engine_with_new_feed.run()
new_duration = time.time() - start

# Compare
speedup = old_duration / new_duration
print(f"Speedup: {speedup:.2f}x")
print(f"Old: {old_results['events_per_second']:.0f} events/sec")
print(f"New: {new_results['events_per_second']:.0f} events/sec")
```

## Rollback Plan

If issues arise, rollback is immediate:

```python
# Change this line:
from ml4t.backtest.data import PolarsDataFeed as DataFeed

# Back to:
from ml4t.backtest.data import ParquetDataFeed as DataFeed

# Everything else stays the same
```

## Recommended Migration Path

1. **Week 1:** Test PolarsDataFeed in dev environment
   - Run integration tests
   - Verify signal timing validation
   - Measure performance

2. **Week 2:** Shadow deployment
   - Run both feeds in parallel
   - Compare results
   - Monitor memory usage

3. **Week 3:** Gradual rollout
   - Enable for 10% of strategies
   - Monitor for issues
   - Gather performance metrics

4. **Week 4:** Full migration
   - Enable for all strategies
   - Keep ParquetDataFeed as fallback
   - Update documentation

## Support

- **Documentation:** See `src/ml4t/backtest/data/polars_feed.py` docstrings
- **Examples:** See `examples/polars_feed_example.py`
- **Tests:** See `tests/integration/test_polars_engine_integration.py`
- **Issues:** Report to ml4t.backtest maintainers

## Summary

| Criterion | ParquetDataFeed | PolarsDataFeed |
|-----------|----------------|----------------|
| Stability | ✅ Battle-tested | ✅ Production-ready |
| Performance | ⚠️ Good | ✅ Excellent |
| Memory | ⚠️ High | ✅ Low (lazy) |
| ML Signals | ⚠️ Basic | ✅ Advanced |
| Features | ❌ No | ✅ Yes |
| Context | ❌ No | ✅ Yes |
| Look-ahead protection | ❌ No | ✅ Yes |
| Recommended for | Simple strategies | ML strategies |

**Bottom line:** Migrate to PolarsDataFeed for ML-driven strategies. Keep ParquetDataFeed for simple use cases or if backward compatibility is critical.
