# Platform Execution Models Reference

**Purpose**: Quick reference for understanding how each backtesting platform handles signal execution, timing, pricing, and trade lifecycle.

**Last Updated**: 2025-11-04

---

## Table of Contents

1. [Execution Model Comparison](#execution-model-comparison)
2. [qengine (ML4T Custom Engine)](#qengine)
3. [VectorBT Pro](#vectorbt-pro)
4. [Backtrader](#backtrader)
5. [Zipline-Reloaded](#zipline-reloaded)
6. [Common Gotchas](#common-gotchas)
7. [Trade Matching Guidelines](#trade-matching-guidelines)

---

## Execution Model Comparison

### Quick Reference Table

| Platform   | Timing Model | Entry Bar  | Entry Price | Exit Bar   | Exit Price | Commission Timing |
|------------|-------------|------------|-------------|------------|------------|-------------------|
| **qengine**    | Next-bar    | Signal+1   | Close       | Signal+1   | Close      | Per-trade         |
| **VectorBT**   | Same-bar    | Signal     | Close       | Signal     | Close      | Per-trade         |
| **Backtrader** | Next-bar    | Signal+1   | Open        | Signal+1   | Open       | Per-trade         |
| **Zipline**    | Next-bar    | Signal+1   | Open*       | Signal+1   | Close*     | Per-share         |

*Zipline default behavior - configurable via slippage models

### Timing Models Explained

**Same-Bar Execution** (VectorBT):
```
Signal at 2017-02-06:
├─ Entry: 2017-02-06 close ($130.29)
├─ Logic: Assumes you can execute at bar close
└─ Use case: Backtesting only, not realistic for intraday
```

**Next-Bar Execution** (qengine, Backtrader, Zipline):
```
Signal at 2017-02-06:
├─ Entry: 2017-02-07 open/close
├─ Logic: No lookahead bias - can't trade on same bar
└─ Use case: More realistic for live trading
```

### Expected Trade Date Differences

For scenario 001 with signal at 2017-02-06:

| Platform   | Entry Date | Entry Price | Difference from VBT |
|------------|-----------|-------------|---------------------|
| VectorBT   | 2017-02-06 | $130.29     | Reference           |
| qengine    | 2017-02-07 | $131.54     | +1 day, +$1.25      |
| Backtrader | 2017-02-07 | $130.54     | +1 day, +$0.25      |
| Zipline    | 2017-02-07 | $130.54*    | +1 day, +$0.25      |

*Expected - not yet validated

---

## qengine

### Overview

**Type**: Event-driven engine with vectorized hybrid execution
**Philosophy**: Point-in-time correctness, institutional-grade fidelity
**Best For**: Production trading strategies, research with realistic execution

### Execution Model

**Order Flow**:
```
1. MARKET event arrives (timestamp T)
2. Strategy receives event, generates signal
3. Strategy creates Order and submits to broker
4. Broker queues order (status: PENDING)
5. NEXT market event arrives (timestamp T+1)
6. Broker checks pending orders against new bar
7. Order filled at T+1 close price
8. FILL event published to strategy and portfolio
```

**Key Characteristics**:
- Next-bar execution (no lookahead)
- Fills at **close price** by default
- Supports intrabar execution with tick data
- Market orders fill immediately at next bar
- Limit orders may take multiple bars

### Data Requirements

**Format**: Polars DataFrame with timezone-aware timestamps (UTC)

```python
Required columns:
- timestamp: datetime[μs, UTC]  # MUST be timezone-aware
- symbol: str
- open: float
- high: float
- low: float
- close: float
- volume: float
```

**Critical**: Timestamps MUST be timezone-aware (UTC). Naive timestamps will not match signals!

```python
# ✅ CORRECT
import pytz
df = df.with_columns(
    pl.col('timestamp').dt.replace_time_zone('UTC')
)

# ❌ WRONG
df['timestamp'] = pd.to_datetime(df['date'])  # Naive
```

### Signal Handling

**Signal Format**:
```python
from datetime import datetime
import pytz

Signal(
    timestamp=datetime(2017, 2, 6, tzinfo=pytz.UTC),  # MUST be UTC-aware
    asset='AAPL',
    action='BUY',  # or 'SELL'
    quantity=100,
    order_type='MARKET',  # or 'LIMIT', 'STOP', etc.
)
```

**Order Submission**:
```python
from qengine.execution.order import Order
from qengine.core.types import OrderType, OrderSide

order = Order(
    asset_id='AAPL',
    order_type=OrderType.MARKET,
    side=OrderSide.BUY,
    quantity=100,
)
broker.submit_order(order)
```

### Trade Extraction

**Output Format**: Polars DataFrame from `broker.get_trades()`

```python
Columns:
- order_id: str
- asset_id: str
- side: str ('buy' or 'sell')
- quantity: float
- price: float
- commission: float
- status: str ('filled', 'pending', 'rejected')
- submitted_time: datetime[μs]
- filled_time: datetime[μs, UTC]
```

**Extraction Pattern**:
```python
def extract_qengine_trades(results: dict, data: pl.DataFrame) -> List[StandardTrade]:
    trades_df = results.get('trades')
    if trades_df is None or trades_df.is_empty():
        return []

    # Match BUY → SELL pairs using FIFO
    # qengine returns individual orders, not complete trades
```

### Known Quirks

1. **Returns individual fills, not complete trades**
   - Need to match BUY → SELL manually
   - Use FIFO (first in, first out) matching

2. **Timezone requirement is strict**
   - All timestamps must be timezone-aware
   - Use `pytz.UTC` consistently

3. **Event-driven means sequential processing**
   - Orders from time T can only fill at T+1
   - No same-bar execution possible

4. **Commission model is percentage-based by default**
   ```python
   PercentageCommission(rate=0.001)  # 0.1% per trade
   ```

---

## VectorBT Pro

### Overview

**Type**: Vectorized backtesting with NumPy/Numba
**Philosophy**: Fast vectorized operations, same-bar execution
**Best For**: Quick backtests, parameter optimization, portfolio analysis

### Execution Model

**Order Flow**:
```
1. Generate entry/exit boolean arrays for entire period
2. Portfolio.from_signals() processes arrays vectorized
3. Entries execute at signal bar close
4. Exits execute at signal bar close
5. Returns complete Trade objects
```

**Key Characteristics**:
- Same-bar execution (signal bar close)
- **Lookahead bias** - knows future prices when entering
- Extremely fast (vectorized NumPy operations)
- Not realistic for live trading simulation
- Best for strategy research and optimization

### Data Requirements

**Format**: Pandas DataFrame or Series with DatetimeIndex (UTC preferred)

```python
Required:
- index: DatetimeIndex (UTC timezone preferred)
- close: float (required for from_signals)
- open, high, low, volume: float (optional but recommended)

# VectorBT handles both timezone-aware and naive timestamps
# but UTC-aware is recommended for consistency
```

### Signal Handling

**Signal Format**: Boolean Series matching data index

```python
import pandas as pd

# Create entry/exit signals
entries = pd.Series(False, index=data.index)
exits = pd.Series(False, index=data.index)

# Set signals (must match data index exactly)
for signal in signals_list:
    if signal.timestamp in data.index:
        if signal.action == 'BUY':
            entries.loc[signal.timestamp] = True
        elif signal.action == 'SELL':
            exits.loc[signal.timestamp] = True
```

**Portfolio Creation**:
```python
import vectorbtpro as vbt

portfolio = vbt.Portfolio.from_signals(
    close=data['close'],
    entries=entries,
    exits=exits,
    init_cash=100000,
    fees=0.001,  # 0.1% commission
    slippage=0.0,
)
```

### Trade Extraction

**Output Format**: VectorBT Trades object

```python
# Access trades
trades = portfolio.trades.records_readable

# Key fields:
# - Entry Idx: Index in data where trade entered
# - Exit Idx: Index in data where trade exited
# - Entry Price: Price at entry (close)
# - Exit Price: Price at exit (close)
# - PnL: Net profit/loss
# - Return: Percentage return
# - Size: Position size
# - Fees: Total commission paid
```

**Extraction Pattern**:
```python
def extract_vectorbt_trades(portfolio, data: pd.DataFrame) -> List[StandardTrade]:
    trades_df = portfolio.trades.records_readable

    for idx, trade in trades_df.iterrows():
        entry_timestamp = data.index[trade['Entry Idx']]
        exit_timestamp = data.index[trade['Exit Idx']]
        # VectorBT provides complete trade information
```

### Known Quirks

1. **Same-bar execution creates lookahead bias**
   - Signal at T → entry at T close
   - Not realistic for live trading
   - Expect different results from next-bar platforms

2. **Trades always at close price**
   - No option to use open price
   - Can't simulate next-bar open execution

3. **Commission is per-trade percentage**
   - Different from Zipline's per-share model
   - Need to convert for comparison

4. **Index alignment is critical**
   ```python
   # ✅ CORRECT: Signal timestamp in data.index
   signal.timestamp in data.index  # True

   # ❌ WRONG: Mismatched timezones
   datetime(2017, 2, 6) in data.index  # False if data is UTC-aware
   ```

5. **Returns complete trades, not individual orders**
   - Already matched entry → exit
   - No need for manual FIFO matching

---

## Backtrader

### Overview

**Type**: Event-driven with strategy-based architecture
**Philosophy**: Realistic next-bar execution at open price
**Best For**: Strategy development with realistic fills, live trading simulation

### Execution Model

**Order Flow**:
```
1. Strategy.next() called for each bar
2. Strategy checks for signals at current bar
3. Strategy calls self.buy() or self.sell()
4. Order queued in broker
5. Next bar arrives
6. Broker fills order at OPEN price
7. Strategy.notify_trade() called when trade closes
```

**Key Characteristics**:
- Next-bar execution (no lookahead)
- Fills at **open price** by default
- Most realistic for live trading
- Slower than VectorBT (event-driven)
- Complex order types supported

### Data Requirements

**Format**: Pandas DataFrame with DatetimeIndex (timezone-naive)

```python
Required columns:
- index: DatetimeIndex (Backtrader expects NAIVE timestamps)
- open: float
- high: float
- low: float
- close: float
- volume: float (optional)

# CRITICAL: Backtrader works with timezone-NAIVE datetimes
data.index.tz is None  # Should be True
```

**Data Feed Setup**:
```python
import backtrader as bt

class PandasData(bt.feeds.PandasData):
    params = (
        ('datetime', None),  # Use index
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )

bt_data = PandasData(dataname=data)
cerebro.adddata(bt_data)
```

### Signal Handling

**Strategy Pattern**:
```python
import backtrader as bt

class SignalStrategy(bt.Strategy):
    def __init__(self):
        # Store signals with BOTH timezone-aware and naive versions
        self.signals_tz = {sig.timestamp: sig for sig in signals_list}
        self.signals_naive = {
            sig.timestamp.replace(tzinfo=None): sig
            for sig in signals_list
        }

    def next(self):
        # Backtrader returns NAIVE datetime
        current_dt = self.datas[0].datetime.datetime(0)

        # Lookup signal (try naive first, then aware)
        signal = self.signals_naive.get(current_dt) or self.signals_tz.get(current_dt)

        if signal and signal.action == 'BUY':
            self.buy(size=signal.quantity)
        elif signal and signal.action == 'SELL':
            self.sell(size=signal.quantity)
```

### Trade Extraction

**Trade Notification**:
```python
def notify_trade(self, trade):
    """Called when a trade closes."""
    if trade.isclosed:
        self.trades_list.append({
            'entry_time': bt.num2date(trade.dtopen),    # NAIVE datetime
            'exit_time': bt.num2date(trade.dtclose),    # NAIVE datetime
            'entry_price': trade.price,
            'exit_price': trade.pnlcomm / trade.size + trade.price,
            'size': trade.size,
            'pnl': trade.pnlcomm,  # Includes commission
            'commission': trade.commission,
        })
```

**Extraction Pattern**:
```python
def extract_backtrader_trades(trades_list: List[dict], data: pd.DataFrame):
    # CRITICAL: Backtrader returns NAIVE datetimes
    # Must convert to UTC-aware for comparison with other platforms

    for trade_dict in trades_list:
        entry_ts = trade_dict['entry_time']

        # Convert to UTC-aware
        if entry_ts and not entry_ts.tzinfo:
            entry_ts = entry_ts.replace(tzinfo=pytz.UTC)
```

### Known Quirks

1. **Returns timezone-NAIVE datetimes**
   - `bt.num2date()` strips timezone information
   - Must add `tzinfo=pytz.UTC` manually for matching
   ```python
   # ❌ From Backtrader
   datetime(2017, 2, 7, 0, 0)  # Naive

   # ✅ After normalization
   datetime(2017, 2, 7, 0, 0, tzinfo=pytz.UTC)
   ```

2. **Strategy.datetime.datetime(0) is also naive**
   - Need dual signal dictionary (naive + aware)
   - Try naive lookup first, fall back to aware

3. **Fills at OPEN price by default**
   - Different from qengine (close) and VectorBT (close)
   - More realistic for market orders
   - Causes price variance in comparisons (0.5-1%)

4. **Trade.price is entry price, not exit**
   - Must calculate exit price from PnL:
   ```python
   exit_price = trade.pnlcomm / trade.size + trade.price
   ```

5. **Commission included in pnlcomm**
   - `trade.pnl` = gross P&L (doesn't exist)
   - `trade.pnlcomm` = net P&L (includes commission)
   - `trade.commission` = total commission

6. **Data feed expects timezone-naive index**
   - Must strip timezone from data before feeding:
   ```python
   data.index = data.index.tz_localize(None)
   ```

---

## Zipline-Reloaded

### Overview

**Type**: Event-driven with pipeline architecture
**Philosophy**: Research-to-production pipeline, institutional-grade
**Best For**: Factor research, production deployment, institutional backtests

### Execution Model

**Order Flow**:
```
1. handle_data() called for each bar
2. Strategy calls order() or order_target_percent()
3. Order queued in broker
4. Next bar arrives
5. Broker fills order (configurable via slippage model)
6. Default: Market orders fill at open, limit orders at specified price
```

**Key Characteristics**:
- Next-bar execution (no lookahead)
- Highly configurable execution (slippage models)
- Designed for large-scale research
- Requires data bundles (not ad-hoc DataFrames)
- Pipeline API for factor research

### Data Requirements

**Format**: Zipline Bundle (HDF5-based)

```python
# Bundle structure
validation_bundle/
├── daily_equities.bcolz/        # OHLCV data
├── adjustments.sqlite           # Splits, dividends, mergers
└── assets.sqlite                # Symbol metadata

# Data must be:
- Aligned to trading calendar (NYSE, etc.)
- Timezone-aware (UTC)
- Forward-filled for missing sessions
- Includes adjustment data
```

**Bundle Creation**:
```python
def validation_to_bundle(interval='1d'):
    def ingest(environ, asset_db_writer, minute_bar_writer,
               daily_bar_writer, adjustment_writer, calendar, ...):

        # Critical: Reindex to fill missing sessions
        all_sessions = calendar.sessions_in_range(start_session, end_session)
        if hasattr(all_sessions, 'tz_localize') and all_sessions.tz is None:
            all_sessions = all_sessions.tz_localize('UTC')

        df = df.reindex(all_sessions, method='ffill')

        # Write daily bars
        daily_bar_writer.write(daily_data_generator(), show_progress=True)

    return ingest
```

### Signal Handling

**Algorithm Pattern**:
```python
from zipline.api import order, symbol

def initialize(context):
    """Setup - called once at start."""
    context.signals = signals_by_date
    context.asset = symbol('AAPL')

def handle_data(context, data):
    """Called for each bar."""
    current_dt = data.current_dt

    # Check for signals
    if current_dt not in context.signals:
        return

    for signal in context.signals[current_dt]:
        if signal.action == 'BUY':
            order(context.asset, signal.quantity)
        elif signal.action == 'SELL':
            order(context.asset, -signal.quantity)
```

### Trade Extraction

**Output Format**: Performance DataFrame from `run_algorithm()`

```python
# Performance columns include:
perf.columns:
- orders: List of order objects
- transactions: List of transaction objects
- positions: Current position quantities
- portfolio_value: Total portfolio value
- pnl: Profit and loss
- returns: Period returns

# Extract transactions (actual fills)
transactions = perf['transactions'].explode()
```

**Extraction Pattern**:
```python
def extract_zipline_trades(perf: pd.DataFrame, data: pd.DataFrame):
    # Zipline returns transactions (fills), not complete trades
    # Must match BUY → SELL transactions manually

    transactions = []
    for idx, row in perf.iterrows():
        if row['transactions']:
            for txn in row['transactions']:
                transactions.append({
                    'timestamp': idx,
                    'price': txn['price'],
                    'amount': txn['amount'],
                    'sid': txn['sid'],
                })
```

### Known Quirks

1. **Bundle-based data only**
   - Can't use ad-hoc DataFrames like qengine/VectorBT
   - Must create custom bundle
   - Bundle ingestion can be tricky

2. **Calendar alignment required**
   - Data must include all trading calendar sessions
   - Missing sessions cause errors:
   ```
   AssertionError: Missing sessions: [Timestamp('2017-08-07')]
   ```
   - Solution: Reindex with `method='ffill'`

3. **Commission is PER-SHARE, not percentage**
   - Different from all other platforms
   ```python
   # Zipline
   PerShare(cost=0.01)  # $0.01 per share

   # Other platforms
   PercentageCommission(rate=0.001)  # 0.1% of trade value
   ```
   - Must convert for comparison

4. **Returns transactions, not trades**
   - Transaction = individual fill
   - Trade = entry + exit pair
   - Must match manually like qengine

5. **Requires ZIPLINE_ROOT environment variable**
   ```python
   import os
   os.environ['ZIPLINE_ROOT'] = '/path/to/bundle/root'
   ```

6. **Bundle registration needed**
   ```python
   from zipline.data.bundles import register
   register('validation', validation_to_bundle(), calendar_name='NYSE')
   ```

---

## Common Gotchas

### 1. Timezone Mismatches

**Problem**: Signals don't execute because timestamps don't match

**Symptoms**:
- Platform runs without errors
- Zero trades extracted
- Signals appear valid

**Diagnosis**:
```python
# Check signal timezone
signal.timestamp.tzinfo  # None = naive, UTC = aware

# Check data timezone
data['timestamp'].dtype  # Should show timezone
data.index.tz  # Should be 'UTC' or None consistently
```

**Solution**:
```python
import pytz

# Make ALL timestamps UTC-aware
signal_ts = datetime(2017, 2, 6, tzinfo=pytz.UTC)

# For data
df = df.with_columns(
    pl.col('timestamp').dt.replace_time_zone('UTC')
)

# For Backtrader (needs naive)
data.index = data.index.tz_localize(None)
```

### 2. Execution Timing Confusion

**Problem**: Trades on different days across platforms

**This is EXPECTED!**

| Platform   | Entry Date for Signal 2017-02-06 |
|------------|----------------------------------|
| VectorBT   | 2017-02-06 (same-bar)           |
| Others     | 2017-02-07 (next-bar)           |

**Validation Strategy**:
- Group trades by "intent" not timestamp
- Allow ±1 day tolerance for next-bar platforms
- Focus on P&L consistency, not exact timestamps

### 3. Price Component Differences

**Problem**: Entry/exit prices vary by 0.5-1% across platforms

**Root Cause**: Different price components used

| Platform   | Entry  | Exit   |
|------------|--------|--------|
| qengine    | Close  | Close  |
| VectorBT   | Close  | Close  |
| Backtrader | Open   | Open   |
| Zipline    | Open*  | Close* |

*Configurable via slippage model

**Validation Strategy**:
- Accept 0-2% price variance as "minor difference"
- Verify price is within OHLC range (no impossible prices)
- Document expected variance in scenario

### 4. Commission Model Differences

**Problem**: Different commission calculation methods

| Platform   | Commission Type | Example            |
|------------|----------------|-------------------|
| qengine    | Percentage     | 0.1% of trade value |
| VectorBT   | Percentage     | 0.1% of trade value |
| Backtrader | Percentage     | 0.1% of trade value |
| Zipline    | Per-share      | $0.01 per share     |

**Conversion**:
```python
# Percentage → Per-share
per_share = (price * quantity * pct_rate) / quantity
per_share = price * pct_rate

# Per-share → Percentage
pct_rate = per_share / price
```

---

## Trade Matching Guidelines

### Tolerance Configuration

```python
comparison = {
    'price_tolerance_pct': 0.1,     # 0.1% price variance OK
    'pnl_tolerance': 10.0,          # $10 total PnL difference OK
    'timestamp_tolerance_sec': 86400,  # ±1 day OK for next-bar platforms
    'timestamp_exact': False,       # Don't require exact timestamp match
    'trade_count_exact': True,      # Require same trade count
}
```

### Severity Classification

**Perfect Match**: All fields within tolerance
- Timestamp: ±1 second
- Price: ±0.01%
- P&L: ±$0.10

**Minor Difference**: Expected variance
- Timestamp: ±1 day (same-bar vs next-bar)
- Price: 0.01-2% (open vs close)
- P&L: $0.10-$50 (commission differences)

**Major Difference**: Investigation needed
- Price: 2-5% variance
- P&L: $50-$200 difference
- Quantity mismatch (but not zero)

**Critical Difference**: Likely bug
- Price: >5% variance or outside OHLC
- P&L: >$200 difference
- Trade missing from one platform
- Quantity completely wrong

### Example Validation

```python
# Trade Group: Entry 2017-02-07
trades = {
    'qengine': {
        'entry_date': '2017-02-07',
        'entry_price': 131.54,
        'exit_date': '2017-04-18',
        'exit_price': 141.19,
        'net_pnl': 937.00,
    },
    'backtrader': {
        'entry_date': '2017-02-07',
        'entry_price': 130.54,  # Open vs Close
        'exit_date': '2017-04-18',
        'exit_price': 140.54,   # Estimated
        'net_pnl': 1032.61,
    }
}

# Analysis
price_diff_pct = abs(131.54 - 130.54) / 131.54 * 100  # 0.76%
pnl_diff = abs(937.00 - 1032.61)  # $95.61

# Verdict: Minor Difference ⚠️
# - Price variance: 0.76% (within 2% tolerance)
# - Different price components: close vs open (expected)
# - P&L variance: $95.61 (within $200 tolerance)
```

---

## Quick Reference: Scenario Creation Checklist

When creating a new scenario:

- [ ] Signals use UTC-aware timestamps: `datetime(..., tzinfo=pytz.UTC)`
- [ ] Data has UTC-aware timestamps for qengine/VectorBT
- [ ] Expected results document execution model differences
- [ ] Tolerance configuration accounts for platform variance
- [ ] Test data includes splits/dividends if testing adjustments
- [ ] Commission model consistent across platforms (or documented)
- [ ] Signal dates validated in actual dataset
- [ ] Backtrader compatibility layer included in runner

---

## Maintenance Notes

**When to Update This Document**:
1. New platform added
2. Execution model changed in any platform
3. New quirk discovered during testing
4. Commission or pricing logic changes
5. Data format requirements change

**Related Documentation**:
- `TASK-001_COMPLETION_REPORT.md` - Timezone issue case study
- `scenarios/scenario_001_simple_market_orders.py` - Reference implementation
- `extractors/*.py` - Trade extraction patterns
- `comparison/matcher.py` - Trade matching logic

---

**Document Version**: 1.0
**Contributors**: Claude (AI Assistant), Stefan (Human)
**Based On**: Work Unit 005, TASK-001 findings
**Status**: Living document - update as we learn more
