# Zipline Bundle for Validation Testing

This directory contains the custom Zipline bundle for cross-platform validation testing.

## Quick Setup

### 1. Prepare Bundle Data (Already Done)

The bundle data is already prepared in `validation_bundle.h5` with 2017 AAPL, MSFT, GOOGL, AMZN data.

If you need to regenerate it:

```python
from tests.validation.fixtures.market_data import prepare_zipline_bundle_data
from pathlib import Path

prepare_zipline_bundle_data(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
    start_date='2017-01-01',
    end_date='2017-12-31',
    output_dir=Path('tests/validation/bundles'),
    use_adjusted=True
)
```

### 2. Register Bundle with Zipline

**Option A: Environment Variable (Recommended for Testing)**

```bash
# Set ZIPLINE_ROOT to use this directory
export ZIPLINE_ROOT=$(pwd)/tests/validation/bundles/.zipline_root

# Copy extension to Zipline root
mkdir -p $ZIPLINE_ROOT
cp tests/validation/bundles/extension.py $ZIPLINE_ROOT/

# Ingest the bundle
zipline ingest -b validation
```

**Option B: Symlink to User Zipline Directory**

```bash
# Link extension to your ~/.zipline directory
ln -sf $(pwd)/tests/validation/bundles/extension.py ~/.zipline/

# Ingest the bundle
zipline ingest -b validation
```

### 3. Verify Bundle

```bash
# List available bundles (should include 'validation')
zipline bundles

# Check bundle data
zipline run --help  # Should accept -b validation
```

## Files

- **`validation_bundle.h5`**: HDF5 store with OHLCV data, metadata, splits, dividends
- **`validation_ingest.py`**: Ingest function that loads data into Zipline format
- **`extension.py`**: Registers bundle with Zipline
- **`README.md`**: This file

## Usage in Validation Scenarios

Once the bundle is ingested, use it in Zipline backtests:

```python
from zipline import run_algorithm

perf = run_algorithm(
    start=datetime(2017, 1, 3),
    end=datetime(2017, 12, 29),
    initialize=initialize,
    capital_base=100000,
    bundle='validation',  # ← Use the custom bundle
)
```

## Troubleshooting

### "No bundle registered with the name 'validation'"

Make sure `extension.py` is in the correct location:
- `~/.zipline/extension.py`, OR
- `$ZIPLINE_ROOT/extension.py`

### "Bundle data not found"

Run `prepare_zipline_bundle_data()` first to create `validation_bundle.h5`.

### Table warnings about object names

These are harmless warnings from PyTables about numeric SIDs. They don't affect functionality.

## Data Coverage

- **Tickers**: AAPL, MSFT, GOOGL, AMZN
- **Period**: 2017-01-03 to 2017-12-29 (249 trading days)
- **Frequency**: Daily
- **Exchange**: NYSE
- **Adjustments**: Splits and dividends included
