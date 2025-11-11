# Validation Data Sources

## Critical Decision: Use Real Data, Not Synthetic

**Date**: 2025-11-11
**Rationale**: User directive - "Please do not use synthetic data; we have multiple actual data sources in ~/ml4t/projects, e.g. crypto_futures"

### Available Real Data

Located in `/home/stefan/ml4t/software/projects/crypto_futures/data/`:

1. **BTC Spot** (`spot/BTC_spot.parquet`):
   - 2.1 million minute bars
   - Date range: 2021-01-01 to 2024-12-31 (4 years)
   - Columns: timestamp, open, high, low, close, volume, symbol, source
   - Source: CryptoCompare

2. **BTC Futures** (`futures/BTC_futures.parquet`):
   - 12MB of futures data
   - Same schema as spot data

### Data Loading Function

Added `load_real_crypto_data()` to `tests/validation/common/data_generator.py`:

```python
def load_real_crypto_data(
    symbol: str = "BTC",
    data_type: str = "spot",
    start_date: str | None = None,
    end_date: str | None = None,
    n_bars: int | None = None,
) -> pd.DataFrame
```

**Features**:
- Loads from projects/crypto_futures/data/{data_type}/{symbol}_{data_type}.parquet
- Optional date range filtering
- Optional n_bars limiting
- Returns pandas DataFrame with OHLCV + symbol columns
- Timestamp as index

**Usage in Tests**:
```python
from common import load_real_crypto_data

# Load 1000 bars of BTC spot data
ohlcv = load_real_crypto_data(symbol='BTC', data_type='spot', n_bars=1000)

# Load specific date range
ohlcv = load_real_crypto_data(
    symbol='BTC',
    data_type='spot',
    start_date='2021-01-01',
    end_date='2021-01-31',
)
```

### Migration Plan

**Status**: In progress

All validation tests (Test 1.1 - 6.2) should be updated to use real data:

1. ✅ Added `load_real_crypto_data()` function
2. ✅ Exported from common/__init__.py
3. ⏳ Update Test 1.1, 1.2, 1.3 to use real data
4. ⏳ Update VALIDATION_ROADMAP.md
5. ⏳ Update all remaining tests (2.1 - 6.2)

### Benefits of Real Data

1. **Realistic Price Dynamics**: Actual volatility, trends, and market behavior
2. **Edge Cases**: Real gaps, volume spikes, price extremes
3. **Reproducibility**: Fixed historical data for deterministic validation
4. **Trust**: Results reflect actual market conditions
5. **Validation Quality**: Tests against real-world scenarios

### Synthetic Data Deprecation

The `generate_ohlcv()` function remains available for unit tests or specialized scenarios where controlled data is needed (e.g., testing specific OHLC relationships), but should NOT be used for validation tests.

### Action Items

- [ ] Update Test 1.1 to use real data
- [ ] Update Test 1.2 to use real data
- [ ] Update Test 1.3 to use real data
- [ ] Update VALIDATION_ROADMAP.md to document real data usage
- [ ] Ensure all future tests use real data by default
