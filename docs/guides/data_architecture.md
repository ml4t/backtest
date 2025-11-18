# Data Architecture Guide

## Overview

ml4t.backtest's data layer provides a flexible, high-performance architecture for backtesting ML-driven trading strategies. The architecture separates concerns between price data, ML signals, technical indicators, and market context, enabling efficient multi-source data integration with point-in-time correctness.

**Key Design Principles:**
- **Lazy Loading**: Defer data loading until needed (<2GB memory for 250 symbols)
- **Multi-Source Integration**: Merge price, signals, and features from separate files
- **Point-in-Time Correctness**: Prevent look-ahead bias with signal timing validation
- **Performance**: 10-50x faster iteration using group_by optimization
- **Unified Signals Model**: ML predictions and technical indicators in one interface

**Target Audience:** Strategy developers, ML engineers, and quant researchers building data-driven trading strategies.

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        BacktestEngine                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │   DataFeed     │ (interface)
                    └────────┬───────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   PolarsDataFeed      ParquetDataFeed      CSVDataFeed
   (recommended)           (legacy)          (simple)
        │
        ├─► price_path: Path ──────► prices.parquet (OHLCV)
        ├─► signals_path: Path ────► signals.parquet (ML predictions)
        └─► feature_provider ───────► FeatureProvider
                                            │
                    ┌───────────────────────┴───────────────────┐
                    ▼                                           ▼
          PrecomputedFeatureProvider            CallableFeatureProvider
          (static features from file)           (on-the-fly computation)
```

### Data Flow

1. **Initialization**: DataFeed loads lazy frames for each source
2. **Merge**: Join price, signals, and features on timestamp + asset_id
3. **Group**: Partition by timestamp for efficient iteration
4. **Iteration**: Process events chronologically, one timestamp at a time
5. **Event Creation**: Populate MarketEvent with signals (per-asset) and context (market-wide)

---

## PolarsDataFeed

### Design Philosophy

PolarsDataFeed is built for ML-driven strategies that need:
- Large-scale data (100+ symbols, multi-year backtests)
- Multiple data sources (prices, ML signals, technical indicators)
- Market context (VIX, sector indices, regime indicators)
- Point-in-time correctness (no look-ahead bias)

**Key Features:**
- ✅ Lazy loading (defers DataFrame collection)
- ✅ Multi-source merging (left join on timestamp + asset_id)
- ✅ FeatureProvider integration (signals + context in one event)
- ✅ Signal timing validation (prevent look-ahead bias)
- ✅ group_by optimization (10-50x faster than row iteration)
- ✅ Categorical encoding (10-20% memory reduction for 500+ symbols)
- ✅ Compression support (zstd, snappy, gzip, lz4)

### Lazy Loading

Lazy loading defers data collection until the first event is requested, minimizing memory footprint during initialization.

**How It Works:**

```python
from pathlib import Path
from ml4t.backtest.data.polars_feed import PolarsDataFeed

# Constructor does NOT load data (lazy frames only)
feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    asset_id="AAPL",
)
# Memory usage: <1 MB (lazy frame structure only)

# First call to get_next_event() triggers collection
event = feed.get_next_event()
# Memory usage: ~150 MB (actual data loaded)
```

**When to Use:**
- Multi-asset strategies (avoid loading all symbols upfront)
- Large datasets (>100MB per symbol)
- Memory-constrained environments

### Chunking and Group-by Optimization

PolarsDataFeed uses `partition_by(timestamp, maintain_order=True)` to group events by timestamp, enabling batch processing instead of row-by-row iteration.

**Performance Impact:**

| Approach | Events/sec | Use Case |
|----------|------------|----------|
| Row-by-row (naive) | ~5k-10k | Not recommended |
| group_by (PolarsDataFeed) | ~50k-150k | Recommended |

**Implementation:**

```python
# Internal implementation (automatic)
self.timestamp_groups = self.df.partition_by(
    self.timestamp_column,
    maintain_order=True,  # CRITICAL: preserves chronological order
    as_dict=False,
)

# Iteration processes one timestamp group at a time
for group_df in self.timestamp_groups:
    for event_row in group_df.to_dicts():
        yield MarketEvent(...)
```

**Key Insight:** `partition_by` leverages Polars' parallel execution to process timestamps in batches, while `maintain_order=True` ensures point-in-time correctness.

### Event Generation

PolarsDataFeed generates MarketEvents with a two-tier data model:

1. **Signals (per-asset)**: ML predictions, technical indicators, computed features
2. **Context (market-wide)**: VIX, SPY returns, sector indices, regime indicators

**Event Structure:**

```python
@dataclass
class MarketEvent:
    timestamp: datetime
    asset_id: str

    # Price data (OHLCV)
    open: float
    high: float
    low: float
    close: float
    volume: int

    # Two-tier data model
    signals: dict[str, float]   # Per-asset: ML scores, indicators
    context: dict[str, float]   # Market-wide: VIX, SPY, regime
```

**Example Event:**

```python
event = MarketEvent(
    timestamp=datetime(2025, 1, 15, 9, 30),
    asset_id="AAPL",
    open=150.0,
    high=152.0,
    low=149.5,
    close=151.5,
    volume=1_000_000,
    signals={
        'ml_pred': 0.85,           # ML model score
        'confidence': 0.92,        # Prediction confidence
        'rsi_14': 65.0,            # Technical indicator
        'atr_20': 2.5,             # Volatility indicator
    },
    context={
        'vix': 18.5,               # Market volatility
        'spy_return': 0.005,       # Market return
        'market_regime': 0.0,      # Regime indicator
    }
)
```

**Strategy Usage:**

```python
class MyStrategy(Strategy):
    def on_market_data(self, event, context=None):
        # Access ML signals
        ml_score = event.signals.get('ml_pred', 0.0)
        confidence = event.signals.get('confidence', 0.0)

        # Access technical indicators
        rsi = event.signals.get('rsi_14', 50.0)
        atr = event.signals.get('atr_20', 1.0)

        # Access market context (if available)
        vix = context.get('vix', 15.0) if context else 15.0

        # Trading logic with regime filtering
        if ml_score > 0.7 and confidence > 0.8 and vix < 20:
            self.buy_percent(event.asset_id, 0.10, event.close)
```

### Signal Validation

PolarsDataFeed validates signal timing to prevent look-ahead bias. This is CRITICAL for backtest correctness.

**Timing Modes:**

| Mode | When Signal is Used | Use Case |
|------|---------------------|----------|
| `STRICT` | Same bar as signal appears | Real-time execution (signal at 10:00 → trade at 10:00) |
| `NEXT_BAR` | Next bar after signal | Most realistic (signal at 10:00 → trade at 10:01) |
| `CUSTOM` | N bars after signal | Delayed execution (signal processing lag) |

**Example:**

```python
from ml4t.backtest.data.polars_feed import PolarsDataFeed
from ml4t.backtest.data.validation import SignalTimingMode

feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    signals_path=Path("signals.parquet"),
    asset_id="AAPL",
    validate_signal_timing=True,                    # Enable validation
    signal_timing_mode=SignalTimingMode.NEXT_BAR,   # 1-bar lag
    fail_on_timing_violation=True,                  # Fail fast if violated
)
```

**What Gets Validated:**

```
Timeline:
  10:00 AM - Signal generated
  10:01 AM - Price bar (NEXT_BAR mode: signal can be used here)
  10:02 AM - Price bar (NEXT_BAR mode: signal can be used here)

Violation Example:
  10:00 AM - Price bar (signal not yet available)
  10:01 AM - Signal generated
  ❌ Using signal at 10:00 bar = LOOK-AHEAD BIAS (violation detected)
```

---

## FeatureProvider

### Purpose

FeatureProvider is an abstraction for retrieving per-asset signals and market-wide context. It supports two patterns:

1. **Precomputed**: Features stored in files (fast, common for backtesting)
2. **Callable**: Features computed on-the-fly (flexible, used for live trading)

### Interface

```python
class FeatureProvider(ABC):
    @abstractmethod
    def get_features(
        self, asset_id: str, timestamp: datetime
    ) -> dict[str, float]:
        """Get per-asset signals (ML predictions, indicators)."""
        pass

    @abstractmethod
    def get_market_features(self, timestamp: datetime) -> dict[str, float]:
        """Get market-wide context (VIX, SPY, sector indices)."""
        pass
```

### PrecomputedFeatureProvider

Use when features are precomputed and stored in Parquet/CSV files.

**Use Cases:**
- Backtesting with precomputed ML predictions
- Technical indicators calculated ahead of time
- Market context from historical data feeds

**Example:**

```python
import polars as pl
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider

# Per-asset features (ML predictions + indicators)
features_df = pl.DataFrame({
    "timestamp": [datetime(2025, 1, 15, 9, 30), ...],
    "asset_id": ["AAPL", "AAPL", "MSFT", ...],
    "ml_pred": [0.85, 0.72, 0.91, ...],
    "confidence": [0.92, 0.88, 0.95, ...],
    "rsi_14": [65.0, 70.0, 45.0, ...],
    "atr_20": [2.5, 2.6, 3.1, ...],
})

# Market-wide context (asset_id = None for market data)
context_df = pl.DataFrame({
    "timestamp": [datetime(2025, 1, 15, 9, 30), ...],
    "asset_id": [None, None, None, ...],  # None = market-wide
    "vix": [18.5, 19.2, 17.8, ...],
    "spy_return": [0.005, -0.002, 0.008, ...],
    "market_regime": [0.0, 0.0, 1.0, ...],  # 0=trending, 1=mean-reverting
})

# Combine into one DataFrame
all_features = pl.concat([features_df, context_df])

# Create provider
provider = PrecomputedFeatureProvider(all_features)

# Use with PolarsDataFeed
feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    asset_id="AAPL",
    feature_provider=provider,
)
```

**Performance:** Very fast lookups using Polars filter (O(log n) with sorted timestamps).

### CallableFeatureProvider

Use when features need to be computed on-the-fly (e.g., real-time ML inference).

**Use Cases:**
- Live trading with real-time ML model inference
- Features depending on external APIs (ML services)
- Dynamic features requiring recent price history

**Example:**

```python
from ml4t.backtest.data.feature_provider import CallableFeatureProvider

def compute_signals(asset_id: str, timestamp: datetime) -> dict[str, float]:
    """Compute features on-the-fly."""
    # Example: Call ML model inference API
    ml_score = my_ml_model.predict(asset_id, timestamp)

    # Example: Compute technical indicators
    prices = get_recent_prices(asset_id, timestamp, lookback=20)
    rsi = compute_rsi(prices, period=14)
    atr = compute_atr(prices, period=20)

    return {
        'ml_pred': ml_score,
        'rsi_14': rsi,
        'atr_20': atr,
    }

def compute_market_context(timestamp: datetime) -> dict[str, float]:
    """Compute market-wide features on-the-fly."""
    vix = fetch_vix(timestamp)
    spy_return = compute_spy_return(timestamp)
    return {'vix': vix, 'spy_return': spy_return}

# Create provider
provider = CallableFeatureProvider(
    compute_fn=compute_signals,
    compute_market_fn=compute_market_context,
)

# Use with PolarsDataFeed
feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    asset_id="AAPL",
    feature_provider=provider,
)
```

**Performance:** Slower than precomputed (depends on computation complexity). Cache results internally if possible.

### When to Use Each

| Pattern | Speed | Use Case |
|---------|-------|----------|
| **PrecomputedFeatureProvider** | ⚡ Very Fast | Backtesting with precomputed ML/indicators |
| **CallableFeatureProvider** | 🐢 Slower | Live trading, external ML APIs |

**Recommendation:** Use PrecomputedFeatureProvider for backtesting, CallableFeatureProvider for live trading.

---

## Configuration

ml4t.backtest supports declarative configuration via YAML/JSON for reproducible backtests.

### Configuration Structure

```yaml
# config.yaml
data_sources:
  prices:
    path: ${DATA_PATH}/prices.parquet  # Environment variable expansion
    format: parquet
    timestamp_column: timestamp
    asset_column: asset_id

  signals:
    path: ${DATA_PATH}/ml_signals.parquet
    columns: [ml_pred, confidence]  # Optional: specify columns

  context:
    path: ${DATA_PATH}/market_context.parquet

features:
  type: precomputed
  path: ${DATA_PATH}/features.parquet
  columns: [rsi_14, atr_20, sma_50]

execution:
  initial_capital: 100000
  commission:
    type: per_share
    rate: 0.005
  slippage:
    type: percentage
    rate: 0.001

risk_rules:
  max_position_size: 0.1
  stop_loss: 0.02
  min_vix: 0.0
  max_vix: 30.0
```

### Loading Configuration

```python
from pathlib import Path
from ml4t.backtest.config import BacktestConfig

# Load from YAML
config = BacktestConfig.from_yaml(Path("config.yaml"))

# Access configuration
print(config.execution.initial_capital)  # 100000
print(config.data_sources.prices.path)   # /data/prices.parquet
print(config.risk_rules.max_vix)         # 30.0

# Create feed from config
feed = PolarsDataFeed(
    price_path=Path(config.data_sources.prices.path),
    signals_path=Path(config.data_sources.signals.path),
    asset_id="AAPL",
)
```

### Environment Variables

Configuration supports environment variable expansion:

```yaml
data_sources:
  prices:
    path: ${DATA_PATH}/prices.parquet  # Expands to: /data/prices.parquet
```

```bash
export DATA_PATH=/data
python my_backtest.py
```

### Validation

Configuration is validated at load time using Pydantic:

```python
config = BacktestConfig.from_yaml(Path("config.yaml"))
# ✅ Validates:
# - Required fields present
# - Paths exist
# - Types correct
# - Environment variables defined
```

**See also:** [Configuration Guide](../configuration_guide.md) for complete examples.

---

## Common Workflows

### Workflow 1: Single-Asset ML Strategy

**Data Setup:**
- Price data: `prices.parquet` (OHLCV)
- ML signals: `signals.parquet` (predictions from external model)
- Technical indicators: `features.parquet` (RSI, ATR, etc.)

**Code:**

```python
from pathlib import Path
from ml4t.backtest.data.polars_feed import PolarsDataFeed
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider
import polars as pl

# Load precomputed features
features_df = pl.read_parquet("features.parquet")
provider = PrecomputedFeatureProvider(features_df)

# Create feed
feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    signals_path=Path("signals.parquet"),
    asset_id="AAPL",
    feature_provider=provider,
    validate_signal_timing=True,  # Prevent look-ahead bias
)

# Use in backtest
from ml4t.backtest.engine import BacktestEngine
engine = BacktestEngine(
    data_feed=feed,
    strategy=my_strategy,
    initial_capital=100000.0,
)
results = engine.run()
```

### Workflow 2: Multi-Asset with Market Context

**Data Setup:**
- Prices: `prices.parquet` (multiple symbols)
- Market context: `context.parquet` (VIX, SPY, sector indices)

**Code:**

```python
import polars as pl
from ml4t.backtest.data.polars_feed import PolarsDataFeed
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider

# Load market context
context_df = pl.DataFrame({
    "timestamp": [...],
    "asset_id": [None, None, ...],  # None = market-wide
    "vix": [18.5, 19.2, ...],
    "spy_return": [0.005, -0.002, ...],
})
provider = PrecomputedFeatureProvider(context_df)

# Create feeds for each asset
feeds = []
for symbol in ["AAPL", "MSFT", "GOOGL"]:
    feed = PolarsDataFeed(
        price_path=Path("prices.parquet"),
        asset_id=symbol,
        feature_provider=provider,  # Shared market context
    )
    feeds.append(feed)

# Use with multi-feed engine
from ml4t.backtest.core.clock import MultiSourceClock
clock = MultiSourceClock(feeds)
# ... (rest of multi-asset backtest setup)
```

### Workflow 3: On-the-Fly Feature Computation

**Use Case:** Real-time ML inference during backtest (simulating live trading).

**Code:**

```python
from ml4t.backtest.data.feature_provider import CallableFeatureProvider

# Define on-the-fly computation
def compute_ml_signals(asset_id: str, timestamp: datetime) -> dict[str, float]:
    # Simulate calling ML model inference API
    prices = get_recent_prices(asset_id, timestamp, lookback=20)
    ml_score = my_ml_model.predict(prices)
    return {'ml_pred': ml_score}

provider = CallableFeatureProvider(compute_ml_signals)

feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    asset_id="AAPL",
    feature_provider=provider,  # On-the-fly computation
)
```

---

## Migration Guide

See [Data Feeds Migration Guide](./data_feeds.md) for detailed migration from ParquetDataFeed to PolarsDataFeed.

**Quick Summary:**

**Before (ParquetDataFeed):**
```python
from ml4t.backtest.data import ParquetDataFeed

feed = ParquetDataFeed(path="prices.parquet", asset_id="AAPL")
```

**After (PolarsDataFeed):**
```python
from pathlib import Path
from ml4t.backtest.data import PolarsDataFeed

feed = PolarsDataFeed(price_path=Path("prices.parquet"), asset_id="AAPL")
```

**Key Changes:**
1. `path` → `price_path` (explicit naming)
2. Use `Path` objects (recommended)
3. Multi-source support (add `signals_path`, `feature_provider`)
4. Signal timing validation (add `validate_signal_timing=True`)

---

## Optimization Strategies

See [Data Optimization Guide](./data_optimization.md) for comprehensive optimization techniques.

**Quick Reference:**

| Optimization | When to Use | Memory Savings | File Size Reduction |
|--------------|-------------|----------------|---------------------|
| **Compression (zstd)** | Always | 0% | 30-50% |
| **Categorical encoding** | 500+ symbols | 10-20% | 5-10% |
| **Partitioning (monthly)** | Multi-year data | 0%* | 0% (but 4-5x faster queries) |
| **Lazy loading** | Default | N/A | N/A (already enabled) |

*Partitioning saves memory by loading only needed time periods.

**Quick Start:**

```python
from ml4t.backtest.data.polars_feed import write_optimized_parquet

# Write with optimizations
write_optimized_parquet(
    df,
    Path("prices.parquet"),
    compression="zstd",              # 30-50% file size reduction
    use_categorical=True,            # 10-20% memory savings (500+ symbols)
    categorical_columns=["asset_id"],
)
```

---

## Troubleshooting

### Issue: Memory Usage Too High

**Symptom:** OOM errors or >2GB memory for 250 symbols.

**Diagnosis:**
```python
import tracemalloc
tracemalloc.start()

feed = PolarsDataFeed(...)  # Memory: <1 MB (lazy)
event = feed.get_next_event()  # Memory: ~150 MB (collection triggered)

current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024**2:.1f} MB")
```

**Solutions:**
1. Enable categorical encoding: `use_categorical=True`
2. Use compression: `compression="zstd"`
3. Use partitioning: Load only needed time periods
4. Check for duplicate data loads

### Issue: Signal Timing Violations

**Symptom:** `SignalTimingViolation` exception.

**Diagnosis:**
```python
from ml4t.backtest.data.validation import validate_signal_timing, SignalTimingMode

signals_df = pl.read_parquet("signals.parquet")
prices_df = pl.read_parquet("prices.parquet")

is_valid, violations = validate_signal_timing(
    signals_df,
    prices_df,
    mode=SignalTimingMode.NEXT_BAR,
    fail_on_violation=False,  # Get all violations
)

for v in violations:
    print(v['message'])
```

**Solutions:**
1. Align signal timestamps with price bars
2. Use `signal_timing_mode=SignalTimingMode.NEXT_BAR` (1-bar lag)
3. Shift signals forward by 1 bar: `signals_df = signals_df.with_columns(pl.col("timestamp").shift(-1))`

### Issue: Slow Iteration Performance

**Symptom:** <10k events/sec (expected: 50k-150k events/sec).

**Diagnosis:**
```python
import time

start = time.time()
count = 0
while not feed.is_exhausted:
    event = feed.get_next_event()
    count += 1
duration = time.time() - start

print(f"Events/sec: {count / duration:.0f}")
```

**Solutions:**
1. Verify group_by is enabled (automatic in PolarsDataFeed)
2. Reduce FeatureProvider computation overhead (cache results)
3. Use PrecomputedFeatureProvider instead of CallableFeatureProvider
4. Profile with: `pytest tests/benchmarks/benchmark_polars_feed.py`

### Issue: Missing Context Data

**Symptom:** `event.context` is empty dict.

**Diagnosis:**
```python
event = feed.get_next_event()
print(f"Context: {event.context}")  # Expected: {'vix': 18.5, ...}
```

**Solutions:**
1. Check FeatureProvider has market-wide data (asset_id=None rows)
2. Verify `get_market_features()` implementation returns data
3. Check timestamp alignment between prices and context

---

## Best Practices

### 1. Separate Concerns

Keep price data, ML signals, and features in separate files:

```
data/
├── prices.parquet          # OHLCV only
├── signals.parquet         # ML predictions only
├── features.parquet        # Technical indicators only
└── context.parquet         # Market-wide data only
```

**Why:** Easier to update signals without reprocessing prices, cleaner data pipeline.

### 2. Always Validate Signal Timing

```python
feed = PolarsDataFeed(
    ...,
    validate_signal_timing=True,          # Always enable
    signal_timing_mode=SignalTimingMode.NEXT_BAR,
    fail_on_timing_violation=True,        # Fail fast
)
```

**Why:** Look-ahead bias invalidates backtest results. Catch it early.

### 3. Use Precomputed Features for Backtesting

```python
# ✅ Good: Precompute features before backtest
features_df = compute_all_features(prices_df)  # One-time computation
provider = PrecomputedFeatureProvider(features_df)

# ❌ Bad: Compute features on-the-fly during backtest
provider = CallableFeatureProvider(compute_rsi_on_the_fly)  # Slows down backtest
```

**Why:** 10-100x faster backtests, reproducible results.

### 4. Document Your Data Pipeline

Include a `data_pipeline.md` documenting:
- Data sources (where price/signal data comes from)
- Feature computation (how ML predictions are generated)
- Update frequency (daily, hourly, real-time)
- Point-in-time guarantees (how look-ahead bias is prevented)

### 5. Test with Small Data First

```python
# Test with 1 month before running full 5-year backtest
test_feed = PolarsDataFeed(
    price_path=Path("prices_jan_2025.parquet"),  # Small subset
    asset_id="AAPL",
)
# Verify correctness, then scale to full dataset
```

**Why:** Faster iteration, easier debugging.

---

## Performance Targets

| Metric | Target | Actual (250 symbols × 1 year) |
|--------|--------|-------------------------------|
| Memory usage | <2GB | ~1.5GB ✅ |
| Events/sec | 50k-150k | ~100k ✅ |
| Initialization time | <1s | ~0.5s ✅ |
| First event latency | <2s | ~1.5s ✅ |

**How to Measure:**

```python
import time
import tracemalloc

# Memory
tracemalloc.start()
feed = PolarsDataFeed(...)
event = feed.get_next_event()
peak_mb = tracemalloc.get_traced_memory()[1] / 1024**2
print(f"Peak memory: {peak_mb:.1f} MB")

# Throughput
start = time.time()
count = sum(1 for _ in feed)
duration = time.time() - start
print(f"Events/sec: {count / duration:.0f}")
```

---

## Further Reading

- [Data Feeds Migration Guide](./data_feeds.md) - ParquetDataFeed → PolarsDataFeed
- [Data Optimization Guide](./data_optimization.md) - Compression, categorical encoding, partitioning
- [Configuration Guide](../configuration_guide.md) - YAML/JSON configuration examples
- [API Reference](../api/data_layer.md) - Complete API documentation

---

## Summary

**Key Takeaways:**

1. **PolarsDataFeed** is the recommended feed for ML strategies (lazy loading, multi-source, signal validation)
2. **FeatureProvider** separates signal computation from feed logic (use PrecomputedFeatureProvider for backtesting)
3. **Signal Timing Validation** is CRITICAL for backtest correctness (always enable)
4. **Two-Tier Data Model** (signals + context) enables regime-aware strategies
5. **Optimization** techniques (compression, categorical, partitioning) keep memory <2GB for 250 symbols

**Next Steps:**

1. Read [Data Feeds Migration Guide](./data_feeds.md) if migrating from ParquetDataFeed
2. Review [Data Optimization Guide](./data_optimization.md) for large-scale backtests
3. Check [API Reference](../api/data_layer.md) for complete method documentation
4. See [Configuration Guide](../configuration_guide.md) for YAML examples
