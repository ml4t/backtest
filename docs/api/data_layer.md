# Data Layer API Reference

Complete API documentation for ml4t.backtest data layer classes and methods.

**Quick Links:**
- [PolarsDataFeed](#polarsdatafeed)
- [FeatureProvider](#featureprovider)
- [PrecomputedFeatureProvider](#precomputedfeatureprovider)
- [CallableFeatureProvider](#callablefeatureprovider)
- [DataFeed Interface](#datafeed-interface)
- [Helper Functions](#helper-functions)
- [Validation Functions](#validation-functions)

---

## PolarsDataFeed

High-performance data feed with lazy loading and multi-source merging.

**Location:** `ml4t.backtest.data.polars_feed`

### Constructor

```python
PolarsDataFeed(
    price_path: Path,
    asset_id: AssetId,
    data_type: MarketDataType = MarketDataType.BAR,
    timestamp_column: str = "timestamp",
    asset_column: str = "asset_id",
    signals_path: Path | None = None,
    signal_columns: list[str] | None = None,
    feature_provider: FeatureProvider | None = None,
    filters: list[pl.Expr] | None = None,
    validate_signal_timing: bool = True,
    signal_timing_mode: SignalTimingMode = SignalTimingMode.NEXT_BAR,
    fail_on_timing_violation: bool = True,
    use_categorical: bool = False,
    compression: str | None = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `price_path` | `Path` | **Required** | Path to Parquet file with OHLCV data |
| `asset_id` | `AssetId` | **Required** | Asset identifier for this feed |
| `data_type` | `MarketDataType` | `MarketDataType.BAR` | Type of market data (BAR, TICK, QUOTE) |
| `timestamp_column` | `str` | `"timestamp"` | Name of timestamp column |
| `asset_column` | `str` | `"asset_id"` | Name of asset ID column |
| `signals_path` | `Path \| None` | `None` | Optional path to Parquet file with ML signals |
| `signal_columns` | `list[str] \| None` | `None` | Optional list of signal column names. If None, auto-detect numeric columns |
| `feature_provider` | `FeatureProvider \| None` | `None` | Optional FeatureProvider for indicators/context |
| `filters` | `list[pl.Expr] \| None` | `None` | Optional list of Polars filter expressions |
| `validate_signal_timing` | `bool` | `True` | If True, validate signals don't create look-ahead bias |
| `signal_timing_mode` | `SignalTimingMode` | `NEXT_BAR` | Timing mode for signal validation |
| `fail_on_timing_violation` | `bool` | `True` | If True, raise exception on timing violation |
| `use_categorical` | `bool` | `False` | If True, convert asset_id to categorical (10-20% memory savings for 500+ symbols) |
| `compression` | `str \| None` | `None` | Compression codec: 'zstd', 'snappy', 'gzip', 'lz4', or None |

#### Example

```python
from pathlib import Path
from ml4t.backtest.data.polars_feed import PolarsDataFeed
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider
from ml4t.backtest.data.validation import SignalTimingMode

# Basic usage (price data only)
feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    asset_id="AAPL",
)

# With ML signals and features
features_df = pl.read_parquet("features.parquet")
provider = PrecomputedFeatureProvider(features_df)

feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    signals_path=Path("signals.parquet"),
    asset_id="AAPL",
    signal_columns=["ml_pred", "confidence"],
    feature_provider=provider,
    validate_signal_timing=True,
    signal_timing_mode=SignalTimingMode.NEXT_BAR,
    use_categorical=True,  # Enable for memory savings
    compression="zstd",    # Enable for file size reduction
)
```

### Methods

#### `get_next_event()`

Get the next market event from the feed.

```python
def get_next_event(self) -> MarketEvent | None
```

**Returns:** `MarketEvent | None`
- Next MarketEvent with signals/context populated, or None if no more data

**Example:**

```python
while not feed.is_exhausted:
    event = feed.get_next_event()
    if event:
        print(f"{event.timestamp}: {event.asset_id} @ {event.close}")
        print(f"  Signals: {event.signals}")
        print(f"  Context: {event.context}")
```

#### `peek_next_timestamp()`

Peek at the timestamp of the next event without consuming it.

```python
def peek_next_timestamp(self) -> datetime | None
```

**Returns:** `datetime | None`
- Timestamp of next event or None if exhausted

**Example:**

```python
next_ts = feed.peek_next_timestamp()
if next_ts:
    print(f"Next event at: {next_ts}")
```

#### `reset()`

Reset the data feed to the beginning.

```python
def reset(self) -> None
```

**Example:**

```python
feed.reset()  # Start over from first event
```

#### `seek(timestamp)`

Seek to a specific timestamp.

```python
def seek(self, timestamp: datetime) -> None
```

**Parameters:**
- `timestamp` (`datetime`): Target timestamp to seek to

**Example:**

```python
from datetime import datetime

# Skip to January 15, 2025
feed.seek(datetime(2025, 1, 15))
```

#### `is_exhausted` (property)

Check if the data feed has no more events.

```python
@property
def is_exhausted(self) -> bool
```

**Returns:** `bool`
- True if no more events available

**Example:**

```python
if feed.is_exhausted:
    print("No more data")
```

---

## FeatureProvider

Abstract base class for signal and context computation/retrieval.

**Location:** `ml4t.backtest.data.feature_provider`

### Interface

```python
class FeatureProvider(ABC):
    @abstractmethod
    def get_features(
        self, asset_id: AssetId, timestamp: datetime
    ) -> dict[str, float]:
        """Get per-asset signals (ML predictions, indicators)."""
        pass

    @abstractmethod
    def get_market_features(self, timestamp: datetime) -> dict[str, float]:
        """Get market-wide context (VIX, SPY, sector indices)."""
        pass
```

### Methods

#### `get_features(asset_id, timestamp)`

Get per-asset signals at specific timestamp.

```python
def get_features(
    self, asset_id: AssetId, timestamp: datetime
) -> dict[str, float]
```

**Parameters:**
- `asset_id` (`AssetId`): Asset identifier (e.g., "AAPL")
- `timestamp` (`datetime`): Point in time for feature retrieval

**Returns:** `dict[str, float]`
- Dictionary of signal name → value pairs (empty dict if no signals)

**Example:**

```python
signals = provider.get_features("AAPL", datetime(2025, 1, 15, 9, 30))
# {'ml_pred': 0.85, 'confidence': 0.92, 'rsi_14': 65.0, 'atr_20': 2.5}

ml_score = signals.get('ml_pred', 0.0)
rsi = signals.get('rsi_14', 50.0)
```

#### `get_market_features(timestamp)`

Get market-wide features at specific timestamp.

```python
def get_market_features(self, timestamp: datetime) -> dict[str, float]
```

**Parameters:**
- `timestamp` (`datetime`): Point in time for feature retrieval

**Returns:** `dict[str, float]`
- Dictionary of feature name → value pairs (empty dict if no features)

**Example:**

```python
context = provider.get_market_features(datetime(2025, 1, 15, 9, 30))
# {'vix': 18.5, 'spy_return': 0.005, 'market_regime': 0.0}

vix = context.get('vix', 15.0)
spy_return = context.get('spy_return', 0.0)
```

---

## PrecomputedFeatureProvider

Feature provider for precomputed features stored in DataFrame.

**Location:** `ml4t.backtest.data.feature_provider`

### Constructor

```python
PrecomputedFeatureProvider(
    features_df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    asset_col: str = "asset_id",
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `features_df` | `pl.DataFrame` | **Required** | Polars DataFrame with precomputed features |
| `timestamp_col` | `str` | `"timestamp"` | Column name for timestamp |
| `asset_col` | `str` | `"asset_id"` | Column name for asset_id (use None for market-wide) |

#### Expected Schema

**Per-Asset Features:**
```python
pl.DataFrame({
    "timestamp": [datetime, datetime, ...],
    "asset_id": ["AAPL", "MSFT", ...],
    "ml_pred": [0.85, 0.72, ...],
    "confidence": [0.92, 0.88, ...],
    "rsi_14": [65.0, 70.0, ...],
    "atr_20": [2.5, 2.6, ...],
})
```

**Market-Wide Features:**
```python
pl.DataFrame({
    "timestamp": [datetime, datetime, ...],
    "asset_id": [None, None, ...],  # None = market-wide
    "vix": [18.5, 19.2, ...],
    "spy_return": [0.005, -0.002, ...],
})
```

#### Example

```python
import polars as pl
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider

# Load precomputed features
features_df = pl.read_parquet("features.parquet")

# Create provider
provider = PrecomputedFeatureProvider(features_df)

# Query features
signals = provider.get_features("AAPL", datetime(2025, 1, 15, 9, 30))
context = provider.get_market_features(datetime(2025, 1, 15, 9, 30))
```

### Methods

Inherits from [FeatureProvider](#featureprovider):
- `get_features(asset_id, timestamp)` - Get per-asset signals
- `get_market_features(timestamp)` - Get market-wide context

---

## CallableFeatureProvider

Feature provider for on-the-fly computation via callable.

**Location:** `ml4t.backtest.data.feature_provider`

### Constructor

```python
CallableFeatureProvider(
    compute_fn: Callable[[AssetId, datetime], dict[str, float]],
    compute_market_fn: Callable[[datetime], dict[str, float]] | None = None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compute_fn` | `Callable` | **Required** | Function taking (asset_id, timestamp) → dict[str, float] |
| `compute_market_fn` | `Callable \| None` | `None` | Optional function taking timestamp → dict[str, float] |

#### Example

```python
from ml4t.backtest.data.feature_provider import CallableFeatureProvider

# Define computation functions
def compute_signals(asset_id: str, timestamp: datetime) -> dict[str, float]:
    """Compute per-asset signals on-the-fly."""
    # Call ML model inference
    ml_score = my_ml_model.predict(asset_id, timestamp)

    # Compute technical indicators
    prices = get_recent_prices(asset_id, timestamp, lookback=20)
    rsi = compute_rsi(prices, period=14)
    atr = compute_atr(prices, period=20)

    return {
        'ml_pred': ml_score,
        'rsi_14': rsi,
        'atr_20': atr,
    }

def compute_market(timestamp: datetime) -> dict[str, float]:
    """Compute market-wide features on-the-fly."""
    vix = fetch_vix(timestamp)
    spy_return = compute_spy_return(timestamp)
    return {'vix': vix, 'spy_return': spy_return}

# Create provider
provider = CallableFeatureProvider(
    compute_fn=compute_signals,
    compute_market_fn=compute_market,
)

# Use with feed
feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    asset_id="AAPL",
    feature_provider=provider,
)
```

### Methods

Inherits from [FeatureProvider](#featureprovider):
- `get_features(asset_id, timestamp)` - Calls `compute_fn`
- `get_market_features(timestamp)` - Calls `compute_market_fn` (or returns empty dict)

**Error Handling:** Catches exceptions and returns empty dict with logged error.

---

## DataFeed Interface

Abstract base class for all data feeds.

**Location:** `ml4t.backtest.data.feed`

### Interface

```python
class DataFeed(ABC):
    @abstractmethod
    def get_next_event(self) -> Event | None:
        """Get the next event from this data feed."""
        pass

    @abstractmethod
    def peek_next_timestamp(self) -> datetime | None:
        """Peek at the timestamp of the next event."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the data feed to the beginning."""
        pass

    @abstractmethod
    def seek(self, timestamp: datetime) -> None:
        """Seek to a specific timestamp."""
        pass

    @property
    @abstractmethod
    def is_exhausted(self) -> bool:
        """Check if the data feed has no more events."""
        pass
```

### Implementations

| Class | Use Case | Performance |
|-------|----------|-------------|
| `PolarsDataFeed` | ML strategies, multi-source data | ⚡⚡⚡ Excellent |
| `ParquetDataFeed` | Simple strategies, legacy compatibility | ⚡⚡ Good |
| `CSVDataFeed` | Prototyping, human-readable data | ⚡ Moderate |

---

## Helper Functions

### `write_optimized_parquet()`

Write DataFrame to Parquet with optimizations.

```python
def write_optimized_parquet(
    df: pl.DataFrame,
    path: Path,
    compression: str = "zstd",
    use_categorical: bool = False,
    categorical_columns: list[str] | None = None,
) -> None
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | **Required** | DataFrame to write |
| `path` | `Path` | **Required** | Output path |
| `compression` | `str` | `"zstd"` | Compression codec: 'zstd', 'snappy', 'gzip', 'lz4', None |
| `use_categorical` | `bool` | `False` | If True, convert columns to categorical |
| `categorical_columns` | `list[str] \| None` | `None` | Column names to convert (default: ['asset_id']) |

#### Example

```python
from pathlib import Path
from ml4t.backtest.data.polars_feed import write_optimized_parquet
import polars as pl

df = pl.DataFrame({
    "timestamp": [...],
    "asset_id": [...],
    "close": [...],
})

# Write with optimizations
write_optimized_parquet(
    df,
    Path("prices.parquet"),
    compression="zstd",         # 30-50% file size reduction
    use_categorical=True,       # 10-20% memory savings (500+ symbols)
    categorical_columns=["asset_id"],
)
```

#### Performance Impact

| Compression | File Size Reduction | Write Speed | Read Speed |
|-------------|---------------------|-------------|------------|
| `zstd` | 30-50% | Medium | Fast |
| `snappy` | 10-20% | Fast | Fast |
| `gzip` | 40-60% | Slow | Medium |
| `lz4` | 15-25% | Very Fast | Very Fast |
| `None` | 0% | Fastest | Fastest |

---

### `create_partitioned_dataset()`

Create partitioned Parquet dataset for large data.

```python
def create_partitioned_dataset(
    df: pl.DataFrame,
    base_path: Path,
    partition_by: str = "month",
    timestamp_column: str = "timestamp",
    compression: str = "zstd",
    use_categorical: bool = False,
    categorical_columns: list[str] | None = None,
) -> dict[str, Path]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | **Required** | DataFrame to partition |
| `base_path` | `Path` | **Required** | Base directory for partitioned files |
| `partition_by` | `str` | `"month"` | Partitioning strategy: 'month', 'quarter', 'year' |
| `timestamp_column` | `str` | `"timestamp"` | Name of timestamp column |
| `compression` | `str` | `"zstd"` | Compression codec |
| `use_categorical` | `bool` | `False` | If True, convert columns to categorical |
| `categorical_columns` | `list[str] \| None` | `None` | Column names to convert |

**Returns:** `dict[str, Path]`
- Dictionary mapping partition keys to file paths

#### Example

```python
from pathlib import Path
from ml4t.backtest.data.polars_feed import create_partitioned_dataset

# Create monthly partitions
partitions = create_partitioned_dataset(
    df,
    base_path=Path("data/partitioned"),
    partition_by="month",
    compression="zstd",
    use_categorical=True,
)

# Result:
# {
#   '2025-01': Path('data/partitioned/2025-01.parquet'),
#   '2025-02': Path('data/partitioned/2025-02.parquet'),
#   ...
# }
```

#### Partition Strategies

| Strategy | Partition Key Format | Use Case |
|----------|---------------------|----------|
| `month` | `2025-01`, `2025-02`, ... | **Recommended** for multi-year backtests |
| `quarter` | `2025-Q1`, `2025-Q2`, ... | Quarterly rebalancing strategies |
| `year` | `2025`, `2026`, ... | Very long-term studies (10+ years) |

---

### `load_partitioned_dataset()`

Load partitioned Parquet dataset.

```python
def load_partitioned_dataset(
    base_path: Path,
    partitions: list[str] | None = None,
    lazy: bool = True,
) -> pl.DataFrame | pl.LazyFrame
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_path` | `Path` | **Required** | Base directory containing partitioned files |
| `partitions` | `list[str] \| None` | `None` | Optional list of partition keys to load (None = all) |
| `lazy` | `bool` | `True` | If True, return LazyFrame; otherwise collect and return DataFrame |

**Returns:** `pl.DataFrame | pl.LazyFrame`
- Combined DataFrame or LazyFrame from specified partitions

#### Example

```python
from pathlib import Path
from ml4t.backtest.data.polars_feed import load_partitioned_dataset

# Load all partitions
df_all = load_partitioned_dataset(
    Path("data/partitioned"),
    lazy=False,
)

# Load only Q1 2025 (faster)
df_q1 = load_partitioned_dataset(
    Path("data/partitioned"),
    partitions=["2025-01", "2025-02", "2025-03"],
    lazy=False,
)
```

#### Performance Impact

For selective loading of 1 month from 1 year dataset:

| Approach | Query Time | Speedup |
|----------|------------|---------|
| Single file | 0.145s | 1.0x |
| Monthly partitions | 0.032s | 4.5x ✅ |

---

## Validation Functions

### `validate_signal_timing()`

Validate signal timing to prevent look-ahead bias.

```python
def validate_signal_timing(
    signals_df: pl.DataFrame,
    prices_df: pl.DataFrame,
    mode: SignalTimingMode = SignalTimingMode.NEXT_BAR,
    custom_lag_bars: int = 1,
    timestamp_column: str = "timestamp",
    asset_column: str = "asset_id",
    fail_on_violation: bool = True,
) -> tuple[bool, list[dict[str, Any]]]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signals_df` | `pl.DataFrame` | **Required** | DataFrame with ML signals |
| `prices_df` | `pl.DataFrame` | **Required** | DataFrame with price data |
| `mode` | `SignalTimingMode` | `NEXT_BAR` | Timing validation mode |
| `custom_lag_bars` | `int` | `1` | Number of bars lag for CUSTOM mode |
| `timestamp_column` | `str` | `"timestamp"` | Name of timestamp column |
| `asset_column` | `str` | `"asset_id"` | Name of asset ID column |
| `fail_on_violation` | `bool` | `True` | If True, raise exception on violation |

**Returns:** `tuple[bool, list[dict[str, Any]]]`
- `is_valid`: True if no violations found
- `violations_list`: List of dict with violation details

**Raises:** `SignalTimingViolation` if `fail_on_violation=True` and timing violation detected

#### Signal Timing Modes

| Mode | When Signal is Used | Description |
|------|---------------------|-------------|
| `STRICT` | Same bar as signal appears | Real-time execution |
| `NEXT_BAR` | Next bar after signal | Most realistic (1-bar lag) |
| `CUSTOM` | N bars after signal | Configurable delay |

#### Example

```python
import polars as pl
from ml4t.backtest.data.validation import validate_signal_timing, SignalTimingMode

signals_df = pl.read_parquet("signals.parquet")
prices_df = pl.read_parquet("prices.parquet")

# Validate with NEXT_BAR mode
is_valid, violations = validate_signal_timing(
    signals_df,
    prices_df,
    mode=SignalTimingMode.NEXT_BAR,
    fail_on_violation=False,  # Don't raise, just report
)

if not is_valid:
    print(f"Found {len(violations)} timing violations:")
    for v in violations:
        print(f"  - {v['message']}")
```

---

### `validate_comprehensive()`

Run all validation checks on a DataFrame.

```python
def validate_comprehensive(
    df: pl.DataFrame,
    validate_duplicates: bool = True,
    validate_ohlc: bool = True,
    validate_missing: bool = True,
    validate_volume: bool = True,
    validate_price: bool = True,
    validate_gaps: bool = True,
    required_columns: list[str] | None = None,
    asset_column: str = "asset_id",
    timestamp_column: str = "timestamp",
    volume_col: str = "volume",
    expected_frequency: str | None = None,
) -> tuple[bool, dict[str, list[dict[str, Any]]]]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | **Required** | DataFrame to validate |
| `validate_duplicates` | `bool` | `True` | Check for duplicate timestamps |
| `validate_ohlc` | `bool` | `True` | Check OHLC consistency |
| `validate_missing` | `bool` | `True` | Check for missing data |
| `validate_volume` | `bool` | `True` | Check volume sanity |
| `validate_price` | `bool` | `True` | Check price sanity |
| `validate_gaps` | `bool` | `True` | Check for time series gaps |
| `required_columns` | `list[str] \| None` | `None` | List of required columns |
| `asset_column` | `str` | `"asset_id"` | Name of asset ID column |
| `timestamp_column` | `str` | `"timestamp"` | Name of timestamp column |
| `volume_col` | `str` | `"volume"` | Name of volume column |
| `expected_frequency` | `str \| None` | `None` | Expected frequency (e.g., "1d", "1h") |

**Returns:** `tuple[bool, dict[str, list[dict[str, Any]]]]`
- `is_valid`: True if all checks pass
- `violations_by_category`: Dict mapping check name to list of violations

#### Example

```python
import polars as pl
from ml4t.backtest.data.validation import validate_comprehensive

df = pl.read_parquet("prices.parquet")

# Run all validation checks
is_valid, violations = validate_comprehensive(
    df,
    expected_frequency="1d",  # Daily data
)

if not is_valid:
    print("Validation failed:")
    for check_name, check_violations in violations.items():
        print(f"\n{check_name}: {len(check_violations)} violations")
        for v in check_violations:
            print(f"  - {v['message']}")
```

#### Validation Checks

| Check | What It Validates |
|-------|-------------------|
| `missing_data` | Required columns present, no null values |
| `duplicates` | No duplicate timestamps for same asset |
| `ohlc_consistency` | high ≥ max(open, close, low), low ≤ min(open, close, high) |
| `volume_sanity` | Volume ≥ 0, no extreme outliers |
| `price_sanity` | Prices within reasonable range, no extreme changes |
| `time_series_gaps` | No missing bars (gaps > expected frequency) |

---

## Configuration Schema

### BacktestConfig

Type-safe configuration for declarative backtesting.

**Location:** `ml4t.backtest.config`

#### Example YAML

```yaml
name: "ML Strategy Backtest"
description: "ML-driven strategy with VIX filtering"

data_sources:
  prices:
    path: ${DATA_PATH}/prices.parquet
    format: parquet
  signals:
    path: ${DATA_PATH}/signals.parquet
    columns: [ml_pred, confidence]
  context:
    path: ${DATA_PATH}/market_context.parquet

features:
  type: precomputed
  path: ${DATA_PATH}/features.parquet

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
  max_vix: 30.0
```

#### Loading Configuration

```python
from pathlib import Path
from ml4t.backtest.config import BacktestConfig

# Load from YAML
config = BacktestConfig.from_yaml(Path("config.yaml"))

# Load from JSON
config = BacktestConfig.from_json(Path("config.json"))

# Access configuration
print(config.execution.initial_capital)  # 100000
print(config.data_sources.prices.path)   # /data/prices.parquet
```

#### Configuration Classes

| Class | Description |
|-------|-------------|
| `BacktestConfig` | Top-level configuration |
| `DataSourcesConfig` | Data source paths and formats |
| `DataSourceConfig` | Single data source configuration |
| `PrecomputedFeaturesConfig` | Precomputed feature provider config |
| `CallableFeaturesConfig` | Callable feature provider config |
| `ExecutionConfig` | Execution parameters (capital, commission, slippage) |
| `RiskRulesConfig` | Risk management rules |

**See:** [Configuration Guide](../configuration_guide.md) for complete examples.

---

## Type Aliases

### AssetId

```python
AssetId = str  # Asset identifier (e.g., "AAPL", "BTC-USD")
```

### MarketDataType

```python
class MarketDataType(Enum):
    BAR = "bar"      # OHLCV bars
    TICK = "tick"    # Tick-by-tick trades
    QUOTE = "quote"  # Bid/ask quotes
```

### SignalTimingMode

```python
class SignalTimingMode(Enum):
    STRICT = "strict"        # Same-bar execution
    NEXT_BAR = "next_bar"    # 1-bar lag (most realistic)
    CUSTOM = "custom"        # N-bar lag
```

---

## Error Handling

### Exceptions

| Exception | When Raised | How to Handle |
|-----------|-------------|---------------|
| `SignalTimingViolation` | Signal timing validation fails | Fix signal timestamps or adjust timing mode |
| `ConfigError` | Configuration validation fails | Check YAML/JSON syntax and required fields |
| `ValueError` | Invalid parameter values | Check API documentation for valid values |
| `FileNotFoundError` | Data file not found | Check paths and environment variables |

### Example Error Handling

```python
from ml4t.backtest.data.validation import SignalTimingViolation
from ml4t.backtest.config import ConfigError

try:
    # Load configuration
    config = BacktestConfig.from_yaml(Path("config.yaml"))

    # Create feed
    feed = PolarsDataFeed(
        price_path=Path(config.data_sources.prices.path),
        signals_path=Path(config.data_sources.signals.path),
        asset_id="AAPL",
        validate_signal_timing=True,
    )

except ConfigError as e:
    print(f"Configuration error: {e}")
    # Fix config.yaml and retry

except SignalTimingViolation as e:
    print(f"Signal timing violation: {e}")
    print(f"  Asset: {e.asset_id}")
    print(f"  Signal timestamp: {e.signal_timestamp}")
    print(f"  Use timestamp: {e.use_timestamp}")
    print(f"  Lag: {e.lag}")
    # Fix signal timestamps or adjust timing mode

except FileNotFoundError as e:
    print(f"File not found: {e}")
    # Check paths and environment variables
```

---

## Performance Considerations

### Memory Usage

| Component | Memory Cost | When to Optimize |
|-----------|-------------|------------------|
| Lazy loading | <1 MB (until first event) | Always enabled |
| Categorical encoding | 10-20% savings | Enable for 500+ symbols |
| Data collection | ~150 MB per symbol/year | Use partitioning for multi-year |

### Throughput

| Operation | Target | Actual |
|-----------|--------|--------|
| Events/sec | 50k-150k | ~100k |
| Initialization | <1s | ~0.5s |
| First event | <2s | ~1.5s |

### Optimization Tips

1. **Use PrecomputedFeatureProvider** for backtesting (10-100x faster than CallableFeatureProvider)
2. **Enable compression** for file I/O (`compression="zstd"`)
3. **Enable categorical encoding** for large symbol universes (`use_categorical=True`)
4. **Use partitioning** for multi-year data (4-5x faster selective loading)
5. **Validate once, backtest many times** (validation is expensive, cache results)

---

## Examples

### Example 1: Simple Price-Only Feed

```python
from pathlib import Path
from ml4t.backtest.data.polars_feed import PolarsDataFeed

feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    asset_id="AAPL",
)

while not feed.is_exhausted:
    event = feed.get_next_event()
    print(f"{event.timestamp}: {event.close}")
```

### Example 2: ML Signals with Validation

```python
from pathlib import Path
from ml4t.backtest.data.polars_feed import PolarsDataFeed
from ml4t.backtest.data.validation import SignalTimingMode

feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    signals_path=Path("ml_signals.parquet"),
    asset_id="AAPL",
    signal_columns=["ml_pred", "confidence"],
    validate_signal_timing=True,
    signal_timing_mode=SignalTimingMode.NEXT_BAR,
)

event = feed.get_next_event()
ml_score = event.signals.get('ml_pred', 0.0)
confidence = event.signals.get('confidence', 0.0)
```

### Example 3: Multi-Source with Context

```python
import polars as pl
from pathlib import Path
from ml4t.backtest.data.polars_feed import PolarsDataFeed
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider

# Load features and context
features_df = pl.DataFrame({
    "timestamp": [...],
    "asset_id": ["AAPL", ...],
    "rsi_14": [65.0, ...],
})

context_df = pl.DataFrame({
    "timestamp": [...],
    "asset_id": [None, ...],  # Market-wide
    "vix": [18.5, ...],
})

all_features = pl.concat([features_df, context_df])
provider = PrecomputedFeatureProvider(all_features)

feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    signals_path=Path("signals.parquet"),
    asset_id="AAPL",
    feature_provider=provider,
)

event = feed.get_next_event()
rsi = event.signals.get('rsi_14', 50.0)
vix = event.context.get('vix', 15.0)
```

### Example 4: Optimized File Writing

```python
from pathlib import Path
from ml4t.backtest.data.polars_feed import write_optimized_parquet
import polars as pl

df = pl.DataFrame({
    "timestamp": [...],
    "asset_id": [...],
    "close": [...],
})

write_optimized_parquet(
    df,
    Path("prices.parquet"),
    compression="zstd",
    use_categorical=True,
)
```

### Example 5: Comprehensive Validation

```python
import polars as pl
from ml4t.backtest.data.validation import validate_comprehensive

df = pl.read_parquet("prices.parquet")

is_valid, violations = validate_comprehensive(
    df,
    expected_frequency="1d",
)

if not is_valid:
    for check_name, check_violations in violations.items():
        print(f"{check_name}: {len(check_violations)} violations")
```

---

## See Also

- [Data Architecture Guide](../guides/data_architecture.md) - Architecture overview and design principles
- [Data Optimization Guide](../guides/data_optimization.md) - Performance optimization techniques
- [Data Feeds Migration Guide](../guides/data_feeds.md) - Migration from ParquetDataFeed
- [Configuration Guide](../configuration_guide.md) - YAML/JSON configuration examples

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-18 | Initial API reference |
