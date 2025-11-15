# Validation Framework Troubleshooting Guide

**Purpose**: Quick solutions to common problems when running validation tests.

**Last Updated**: 2025-11-04

---

## Table of Contents

1. [Zero Trades Extracted](#zero-trades-extracted)
2. [Timezone Errors](#timezone-errors)
3. [Bundle Issues (Zipline)](#bundle-issues-zipline)
4. [Import Errors](#import-errors)
5. [Trade Matching Errors](#trade-matching-errors)
6. [Performance Issues](#performance-issues)

---

## Zero Trades Extracted

### Symptom
```
Extracting ml4t.backtest trades...
   Found 0 trades
```

Platform runs without errors, but no trades extracted.

### Common Causes

#### 1. Timezone Mismatch (MOST COMMON)

**Problem**: Signals are timezone-naive, data is timezone-aware (or vice versa)

**Diagnosis**:
```python
# Check signal timezone
print(f"Signal: {signal.timestamp} (tz: {signal.timestamp.tzinfo})")

# Check data timezone
print(f"Data dtype: {data['timestamp'].dtype}")
print(f"First timestamp: {data['timestamp'][0]}")

# Test if signal in data
print(f"Signal in data: {signal.timestamp in data['timestamp'].to_list()}")
```

**Solution**:
```python
import pytz

# Fix signals - make UTC-aware
Signal(
    timestamp=datetime(2017, 2, 6, tzinfo=pytz.UTC),  # Add tzinfo
    asset='AAPL',
    action='BUY',
    quantity=100,
)

# Fix data if needed
df = df.with_columns(
    pl.col('timestamp').dt.replace_time_zone('UTC')
)
```

**Prevention**: Always use `pytz.UTC` for all timestamps in validation framework.

#### 2. Signal Dates Not in Dataset

**Problem**: Signal dates don't exist in market data (weekends, holidays)

**Diagnosis**:
```python
from datetime import datetime

signal_date = datetime(2017, 2, 6, tzinfo=pytz.UTC)

# Check if date exists
dates_around = data.filter(
    (data['timestamp'] >= signal_date - timedelta(days=3)) &
    (data['timestamp'] <= signal_date + timedelta(days=3))
)
print(dates_around.select(['timestamp']))
```

**Solution**:
- Use actual trading days only
- Check NYSE calendar
- Validate signal dates before creating scenario:
```python
# Validation helper
def validate_signal_dates(signals, data):
    data_dates = set(data['timestamp'].to_list())
    for sig in signals:
        if sig.timestamp not in data_dates:
            print(f"⚠️ Signal date {sig.timestamp} not in data!")
            # Find nearest trading day
            nearest = data.filter(
                data['timestamp'] >= sig.timestamp
            ).head(1)
            if len(nearest) > 0:
                print(f"   Nearest: {nearest['timestamp'][0]}")
```

#### 3. Broker Not Linked to Strategy (ml4t.backtest)

**Problem**: Strategy has no `broker` attribute

**Diagnosis**:
```python
# In strategy on_event
print(f"Has broker: {hasattr(self, 'broker')}")
print(f"Broker value: {getattr(self, 'broker', None)}")
```

**Solution**:
```python
# After creating strategy and broker
strategy.broker = broker

# Verify
assert hasattr(strategy, 'broker')
assert strategy.broker is not None
```

#### 4. Signal Format Mismatch (VectorBT)

**Problem**: VectorBT signals not aligned with data index

**Diagnosis**:
```python
# Check alignment
print(f"Data index length: {len(data.index)}")
print(f"Entries length: {len(entries)}")
print(f"Entries True count: {entries.sum()}")

# Check if signal dates in index
for sig in signals_list:
    print(f"{sig.timestamp} in index: {sig.timestamp in data.index}")
```

**Solution**:
```python
# Ensure signals match data index exactly
entries = pd.Series(False, index=data.index)
exits = pd.Series(False, index=data.index)

for signal in signals_list:
    if signal.timestamp in data.index:  # Critical check
        if signal.action == 'BUY':
            entries.loc[signal.timestamp] = True
```

---

## Timezone Errors

### Symptom
```
TypeError: can't compare offset-naive and offset-aware datetimes
```

### Common Causes

#### 1. Mixed Timezone Awareness in Matcher

**Problem**: Some trades have timezone-aware timestamps, others naive

**Diagnosis**:
```python
# Check all trade timestamps
for platform, trades in trades_by_platform.items():
    for trade in trades:
        print(f"{platform}: entry_ts.tzinfo = {trade.entry_timestamp.tzinfo}")
```

**Solution**: Normalize all timestamps in extractors

```python
# In extractor
import pytz

# After extracting timestamp
if timestamp and not timestamp.tzinfo:
    timestamp = timestamp.replace(tzinfo=pytz.UTC)
```

**Where to Fix**:
- `extractors/backtrader.py` - bt.num2date() returns naive
- `extractors/zipline.py` - Check transaction timestamps
- `scenarios/*.py` - All signal timestamps

#### 2. Pandas Timezone Localization Issues

**Problem**: `pd.to_datetime()` creating naive timestamps

**Diagnosis**:
```python
# Check timestamp conversion
ts = pd.to_datetime('2017-02-06')
print(f"Naive: {ts.tzinfo is None}")
```

**Solution**:
```python
# Explicit UTC localization
ts = pd.to_datetime('2017-02-06', utc=True)
# or
ts = pd.to_datetime('2017-02-06').tz_localize('UTC')
```

---

## Bundle Issues (Zipline)

### Symptom 1: UnknownBundle Error
```
zipline.data.bundles.core.UnknownBundle: No bundle registered with the name 'validation'
```

**Solution**:
```python
# Register bundle before running
from zipline.data.bundles import register
from validation_ingest import validation_to_bundle

register('validation', validation_to_bundle(), calendar_name='NYSE')
```

### Symptom 2: Missing Sessions
```
AssertionError: Got 249 rows for daily bars table with first day=2017-01-03,
last day=2017-12-29, expected 251 rows. Missing sessions: [Timestamp('2017-08-07')]
```

**Problem**: Data doesn't include all calendar trading days

**Solution**: Reindex with forward-fill

```python
def ingest(environ, asset_db_writer, minute_bar_writer,
           daily_bar_writer, adjustment_writer, calendar, ...):

    # Get all calendar sessions
    all_sessions = calendar.sessions_in_range(start_session, end_session)

    # Make timezone-aware if needed
    if hasattr(all_sessions, 'tz_localize') and all_sessions.tz is None:
        all_sessions = all_sessions.tz_localize('UTC')

    # Reindex to fill missing days
    df = df.reindex(all_sessions, method='ffill')

    # Now write
    daily_bar_writer.write(daily_data_generator(), show_progress=True)
```

### Symptom 3: ZIPLINE_ROOT Not Set
```
FileNotFoundError: [Errno 2] No such file or directory: '~/.zipline'
```

**Solution**:
```python
import os
from pathlib import Path

bundle_root = Path(__file__).parent / 'bundles' / '.zipline_root'
os.environ['ZIPLINE_ROOT'] = str(bundle_root)
```

---

## Import Errors

### Symptom
```
ModuleNotFoundError: No module named 'fixtures'
```

**Problem**: Python can't find validation framework modules

**Solution**: Add to path
```python
import sys
from pathlib import Path

# Add validation directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'fixtures'))
sys.path.insert(0, str(Path(__file__).parent / 'scenarios'))
```

**Prevention**: Use relative imports in package structure

```python
# In extractors/__init__.py
try:
    from .ml4t.backtest import extract_ml4t.backtest_trades
except ImportError:
    from ml4t.backtest import extract_ml4t.backtest_trades
```

---

## Trade Matching Errors

### Symptom: Can't Compare Datetimes
```
TypeError: can't compare offset-naive and offset-aware datetimes
```

**See**: [Timezone Errors](#timezone-errors) section above

### Symptom: No Matches Found
```
Matched 0 trade groups
```

**Problem**: Trades aren't matching despite existing

**Diagnosis**:
```python
# Check raw trade counts
for platform, trades in trades_by_platform.items():
    print(f"{platform}: {len(trades)} trades")
    for trade in trades:
        print(f"  Entry: {trade.entry_timestamp}")

# Check matching tolerance
print(f"Timestamp tolerance: {timestamp_tolerance_seconds} seconds")
print(f"Expected difference: ~86400 seconds (1 day) for next-bar platforms")
```

**Solution**: Increase tolerance for next-bar platforms

```python
# Allow ±1 day for next-bar vs same-bar comparison
matches = match_trades(
    trades_by_platform,
    timestamp_tolerance_seconds=86400  # 1 day = 86400 seconds
)
```

### Symptom: Too Many Match Groups
```
Matched 100 trade groups (expected 2-4)
```

**Problem**: Matching tolerance too loose

**Solution**: Tighten tolerance

```python
# Stricter matching
matches = match_trades(
    trades_by_platform,
    timestamp_tolerance_seconds=60,    # 1 minute
    price_tolerance_pct=0.1,          # 0.1%
    quantity_exact=True,              # Must match exactly
)
```

---

## Performance Issues

### Symptom: VectorBT Very Slow
```
Running VectorBT... (taking >30 seconds)
```

**Causes**:
1. Large dataset
2. Complex portfolio calculations
3. Not using vectorized operations

**Solutions**:
```python
# 1. Reduce data size for testing
data_sample = data.head(1000)  # First 1000 bars only

# 2. Disable expensive metrics
portfolio = vbt.Portfolio.from_signals(
    ...,
    freq='1d',  # Specify frequency explicitly
    init_cash='auto',  # Let VBT calculate
    cash_sharing=False,  # Disable if not needed
)

# 3. Use Numba caching
import os
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
```

### Symptom: ml4t.backtest Slow Event Processing
```
Processed 10,000 events (taking >1 minute)
```

**Causes**:
1. Too many events
2. Complex event handlers
3. Excessive logging

**Solutions**:
```python
# 1. Limit events for testing
results = engine.run(max_events=1000)

# 2. Disable verbose logging
import logging
logging.getLogger('ml4t.backtest').setLevel(logging.WARNING)

# 3. Use priority queue
engine = BacktestEngine(
    ...,
    use_priority_queue=True  # Faster event dispatch
)
```

### Symptom: Trade Extraction Slow
```
Extracting trades... (taking >10 seconds)
```

**Cause**: Inefficient DataFrame operations

**Solution**: Use vectorized operations

```python
# ❌ Slow: Row iteration
for row in df.iter_rows(named=True):
    process_row(row)

# ✅ Fast: Vectorized operations
df = df.with_columns([
    pl.when(...).then(...).otherwise(...).alias('new_col')
])
```

---

## Environment Issues

### Symptom: Package Version Conflicts
```
ImportError: cannot import name 'Portfolio' from 'vectorbtpro'
```

**Solution**: Check virtual environment

```bash
# Verify environment
uv run python -c "import vectorbtpro; print(vectorbtpro.__version__)"

# Reinstall if needed
uv sync --reinstall-package vectorbtpro
```

### Symptom: Data File Not Found
```
FileNotFoundError: data/wiki_prices.parquet
```

**Solution**: Check data location

```python
from pathlib import Path

# Find data file
data_path = Path(__file__).parent / 'fixtures' / 'data' / 'wiki_prices.parquet'
if not data_path.exists():
    print(f"❌ Data not found: {data_path}")
    print(f"Current dir: {Path.cwd()}")
    print(f"Script dir: {Path(__file__).parent}")
```

---

## Quick Diagnostic Script

Save this as `diagnose.py` and run when things go wrong:

```python
#!/usr/bin/env python3
"""Quick diagnostic script for validation framework issues."""

import sys
from pathlib import Path
from datetime import datetime
import pytz

sys.path.insert(0, str(Path(__file__).parent / 'fixtures'))
sys.path.insert(0, str(Path(__file__).parent / 'scenarios'))

def diagnose():
    print("=" * 80)
    print("VALIDATION FRAMEWORK DIAGNOSTICS")
    print("=" * 80)

    # 1. Check imports
    print("\n1. Checking imports...")
    try:
        from market_data import get_ticker_data
        print("   ✅ market_data imported")
    except Exception as e:
        print(f"   ❌ market_data: {e}")

    try:
        from scenario_001_simple_market_orders import Scenario001
        print("   ✅ scenario_001 imported")
    except Exception as e:
        print(f"   ❌ scenario_001: {e}")

    # 2. Check data
    print("\n2. Checking data...")
    try:
        data = get_ticker_data('AAPL', '2017-01-01', '2017-01-31')
        print(f"   ✅ Data loaded: {len(data)} rows")
        print(f"   Timestamp dtype: {data['timestamp'].dtype}")
        print(f"   First timestamp: {data['timestamp'][0]}")
    except Exception as e:
        print(f"   ❌ Data loading: {e}")

    # 3. Check signals
    print("\n3. Checking signals...")
    try:
        signals = Scenario001.signals
        print(f"   ✅ {len(signals)} signals loaded")
        for i, sig in enumerate(signals):
            tz_str = sig.timestamp.tzinfo if sig.timestamp.tzinfo else "NAIVE"
            print(f"   Signal {i+1}: {sig.timestamp} (tz: {tz_str})")
    except Exception as e:
        print(f"   ❌ Signals: {e}")

    # 4. Check signal alignment
    print("\n4. Checking signal alignment...")
    try:
        data = Scenario001.get_data()
        signals = Scenario001.signals
        data_dates = data['timestamp'].to_list()

        for i, sig in enumerate(signals):
            in_data = sig.timestamp in data_dates
            status = "✅" if in_data else "❌"
            print(f"   Signal {i+1}: {sig.timestamp} {status}")
    except Exception as e:
        print(f"   ❌ Alignment check: {e}")

    # 5. Check platforms
    print("\n5. Checking platform imports...")
    platforms = ['ml4t.backtest', 'vectorbtpro', 'backtrader', 'zipline']
    for platform in platforms:
        try:
            __import__(platform)
            print(f"   ✅ {platform} available")
        except Exception as e:
            print(f"   ❌ {platform}: {e}")

    print("\n" + "=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    diagnose()
```

Run with:
```bash
cd tests/validation
uv run python diagnose.py
```

---

## Getting Help

### 1. Check Existing Documentation
- `PLATFORM_EXECUTION_MODELS.md` - Platform details
- `TASK-001_COMPLETION_REPORT.md` - Timezone issue case study
- Scenario files in `scenarios/` - Working examples

### 2. Enable Debug Logging
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('validation')
logger.debug("Your debug message here")
```

### 3. Create Minimal Reproduction
```python
# Minimal test case
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'fixtures'))

from market_data import get_ticker_data
from scenario_001_simple_market_orders import Scenario001, Signal
import pytz
from datetime import datetime

# Load minimal data
data = get_ticker_data('AAPL', '2017-02-01', '2017-02-28')
print(f"Data: {len(data)} rows")

# Create one signal
signal = Signal(
    timestamp=datetime(2017, 2, 6, tzinfo=pytz.UTC),
    asset='AAPL',
    action='BUY',
    quantity=100,
)

# Test alignment
print(f"Signal in data: {signal.timestamp in data['timestamp'].to_list()}")
```

---

## Prevention Checklist

Before creating new scenarios:

- [ ] All signal timestamps use `pytz.UTC`
- [ ] Data timestamps are timezone-aware (UTC)
- [ ] Signal dates validated in dataset
- [ ] Backtrader compatibility layer included
- [ ] Commission models documented
- [ ] Expected platform differences noted
- [ ] Tolerance configuration reviewed
- [ ] Diagnostic test written first (TDD)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-04
**Maintainer**: Claude (AI Assistant)
