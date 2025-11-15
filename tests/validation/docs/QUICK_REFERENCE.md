# Validation Framework Quick Reference Card

**One-page reference for common operations and patterns.**

---

## Platform Execution Models (One-Liner)

```
VectorBT:   Same-bar @ CLOSE  (signal T → entry T close)
ml4t.backtest:    Next-bar @ CLOSE  (signal T → entry T+1 close)
Backtrader: Next-bar @ OPEN   (signal T → entry T+1 open)
Zipline:    Next-bar @ OPEN*  (signal T → entry T+1 open, *configurable)
```

---

## Critical Timezone Rule

**✅ ALWAYS**: Use UTC-aware timestamps everywhere

```python
import pytz
from datetime import datetime

# ✅ CORRECT
timestamp = datetime(2017, 2, 6, tzinfo=pytz.UTC)

# ❌ WRONG
timestamp = datetime(2017, 2, 6)  # Naive
```

---

## Zero Trades? Check These 3 Things

```python
# 1. Signal timezone
print(f"Signal tz: {signal.timestamp.tzinfo}")  # Should be UTC

# 2. Data timezone
print(f"Data dtype: {data['timestamp'].dtype}")  # Should show UTC

# 3. Signal in data
print(f"Match: {signal.timestamp in data['timestamp'].to_list()}")  # Should be True
```

---

## Platform-Specific Patterns

### ml4t.backtest
```python
# Signal creation
Signal(timestamp=datetime(2017, 2, 6, tzinfo=pytz.UTC), asset='AAPL', action='BUY', quantity=100)

# CRITICAL: Link broker to strategy
strategy.broker = broker
```

### VectorBT
```python
# Signal alignment (MUST match data.index)
entries = pd.Series(False, index=data.index)
for signal in signals:
    if signal.timestamp in data.index:  # Critical check
        entries.loc[signal.timestamp] = True
```

### Backtrader
```python
# Dual signal dictionary (naive + aware)
self.signals_naive = {sig.timestamp.replace(tzinfo=None): sig for sig in signals}
self.signals_tz = {sig.timestamp: sig for sig in signals}

# Lookup
signal = self.signals_naive.get(current_dt) or self.signals_tz.get(current_dt)

# Extractor: Add timezone
if entry_ts and not entry_ts.tzinfo:
    entry_ts = entry_ts.replace(tzinfo=pytz.UTC)
```

### Zipline
```python
# Bundle registration
os.environ['ZIPLINE_ROOT'] = str(bundle_root)
register('validation', validation_to_bundle(), calendar_name='NYSE')

# Commission (per-share, not percentage!)
set_commission(PerShare(cost=0.01))  # $0.01 per share
```

---

## Trade Matching Tolerances

```python
# Loose (for different execution models)
timestamp_tolerance_seconds = 86400  # ±1 day
price_tolerance_pct = 2.0            # ±2%

# Tight (for same execution model)
timestamp_tolerance_seconds = 60     # ±1 minute
price_tolerance_pct = 0.1            # ±0.1%
```

---

## Common Error Messages

### `can't compare offset-naive and offset-aware datetimes`
**Fix**: Add `tzinfo=pytz.UTC` to all timestamps

### `No module named 'fixtures'`
**Fix**: `sys.path.insert(0, str(Path(__file__).parent))`

### `UnknownBundle: 'validation'`
**Fix**: Call `register('validation', ...)` before `run_algorithm()`

### `AssertionError: Missing sessions`
**Fix**: Reindex data with `df.reindex(calendar.sessions, method='ffill')`

### `Found 0 trades`
**Fix**: Check timezone alignment (most common) or signal dates

---

## Running Tests

```bash
# Single platform
uv run python runner.py --scenario 001 --platforms ml4t.backtest

# Multiple platforms
uv run python runner.py --scenario 001 --platforms ml4t.backtest,vectorbt,backtrader

# All platforms with detailed report
uv run python runner.py --scenario 001 --platforms ml4t.backtest,vectorbt,backtrader,zipline --report both

# Diagnostic test
uv run python test_ml4t.backtest_signal_processing.py
```

---

## Quick Diagnostic

```python
# Check everything
from datetime import datetime
import pytz

signal_ts = datetime(2017, 2, 6, tzinfo=pytz.UTC)
print(f"1. Signal tz: {signal_ts.tzinfo}")  # Should be UTC
print(f"2. Data dtype: {data['timestamp'].dtype}")  # Should be datetime[..., UTC]
print(f"3. In data: {signal_ts in data['timestamp'].to_list()}")  # Should be True
print(f"4. Broker linked: {hasattr(strategy, 'broker') and strategy.broker is not None}")  # True
```

---

## Expected Trade Differences

For signal at 2017-02-06:

| Platform   | Entry Date | Entry Price | P&L Estimate |
|------------|-----------|-------------|--------------|
| VectorBT   | 2017-02-06 | $130.29 (C) | ~$8,600     |
| ml4t.backtest    | 2017-02-07 | $131.54 (C) | ~$900       |
| Backtrader | 2017-02-07 | $130.54 (O) | ~$1,000     |

**These differences are EXPECTED and CORRECT!**

---

## File Locations

```
tests/validation/
├── docs/
│   ├── PLATFORM_EXECUTION_MODELS.md    # Full platform details
│   ├── TROUBLESHOOTING.md              # Problem solutions
│   └── QUICK_REFERENCE.md              # This file
├── scenarios/
│   └── scenario_001_simple_market_orders.py  # Working example
├── fixtures/
│   └── market_data.py                  # Data loading
├── extractors/
│   ├── ml4t.backtest.py                      # Trade extraction
│   ├── vectorbt.py
│   ├── backtrader.py
│   └── zipline.py
└── runner.py                            # Main test runner
```

---

## Scenario Creation Template

```python
from dataclasses import dataclass
from datetime import datetime
import pytz
import polars as pl

@dataclass
class Signal:
    timestamp: datetime
    asset: str
    action: str
    quantity: float
    order_type: str = 'MARKET'

class ScenarioXXX:
    name = "XXX_scenario_name"
    description = "What this tests"

    @staticmethod
    def get_data() -> pl.DataFrame:
        from market_data import get_ticker_data
        return get_ticker_data('AAPL', '2017-01-01', '2017-12-31')

    # CRITICAL: UTC-aware timestamps!
    signals = [
        Signal(
            timestamp=datetime(2017, 2, 6, tzinfo=pytz.UTC),
            asset='AAPL',
            action='BUY',
            quantity=100,
        ),
    ]

    config = {
        'initial_capital': 100_000.0,
        'commission': 0.001,
        'slippage': 0.0,
    }

    expected = {
        'trade_count': 1,
        'execution_timing': 'next_bar',
    }

    comparison = {
        'price_tolerance_pct': 0.1,
        'pnl_tolerance': 10.0,
        'timestamp_exact': False,  # Allow day variance
    }
```

---

## Getting Help Fast

1. **Zero trades?** → Check timezone alignment (this file, top section)
2. **Import errors?** → Check `sys.path` (TROUBLESHOOTING.md)
3. **Platform differences?** → Check execution models (this file, top)
4. **Need details?** → See PLATFORM_EXECUTION_MODELS.md
5. **Weird error?** → See TROUBLESHOOTING.md

---

**Print this page and keep it handy!**

**Version**: 1.0 | **Date**: 2025-11-04
