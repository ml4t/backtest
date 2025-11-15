# Multi-Source Data Integration: Context & Conditional Logic

**Created:** 2025-11-14
**Status:** Architectural Analysis Complete
**Related:** `ml_signal_architecture.md`

## Executive Summary

Real-world trading strategies need to combine multiple data sources:
- **Asset-specific signals:** ML predictions per stock (AAPL ml_pred, MSFT ml_pred)
- **Market context:** VIX, SPY trend, market breadth, regime indicators
- **Conditional logic:** "If VIX > 30, ignore ML predictions"

This document defines how ml4t.backtest will efficiently handle multi-source data with context-dependent rules while maintaining performance and clean API design.

---

## 1. The Multi-Source Problem

### Real-World Scenario

```python
# Asset-specific: Different ML prediction for each stock
aapl_ml_pred = 0.85  # Strong buy signal
msft_ml_pred = 0.90  # Stronger buy signal

# Market context: Same for all assets
vix = 35             # High volatility (risk-off)
spy_trend = -0.02    # Market down 2%
breadth = 0.3        # Only 30% stocks above MA

# Conditional logic
if vix > 30:
    # High volatility regime - ignore ML signals entirely
    action = None
elif breadth < 0.4:
    # Weak market - reduce position size by 50%
    action = buy_half_size(aapl_ml_pred)
else:
    # Normal regime - trust ML fully
    action = buy_full_size(aapl_ml_pred)
```

**Challenge:** How to efficiently provide both asset-specific and context data to strategy logic?

---

## 2. Data Source Taxonomy

### Type 1: Asset-Specific Time Series (Primary)
- **Examples:** AAPL OHLCV + ml_pred, MSFT OHLCV + ml_pred
- **Frequency:** Same as trading (e.g., daily bars)
- **Scope:** Different for each asset
- **Storage:** Separate DataFrame per asset

### Type 2: Market Context Time Series (Secondary)
- **Examples:** VIX, SPY, DXY (dollar index)
- **Frequency:** May differ from Type 1 (e.g., intraday VIX for daily stocks)
- **Scope:** Global (same value for all assets at timestamp)
- **Storage:** Single Series/DataFrame shared across all assets

### Type 3: Derived Cross-Asset Context (Tertiary)
- **Examples:** Market breadth (% stocks above 200 MA), sector rotation
- **Frequency:** Same as trading
- **Scope:** Computed across all assets
- **Storage:** Pre-computed Series or computed on-the-fly

### Type 4: Exogenous Events (Asynchronous)
- **Examples:** Fed announcements, earnings calendars, economic data
- **Frequency:** Irregular, event-driven
- **Scope:** Can affect all or specific assets
- **Storage:** Event log (future work)

**Focus for Phase 1-2:** Types 1-3 (time series data)

---

## 3. Architectural Options Analysis

### Option 1: Global Context Object

**API:**
```python
class Strategy:
    def on_market_data(self, event: MarketEvent, context: Context):
        # Asset-specific
        aapl_pred = event.data.signals['ml_pred']

        # Market context
        vix = context.get('VIX', event.timestamp)
        spy_trend = context.get('SPY_trend', event.timestamp)

        if vix > 30:
            return  # Don't trade
```

**Implementation:**
```python
class Context:
    def __init__(self, context_feeds: dict[str, pd.Series]):
        self._data = context_feeds
        self._cache = {}
        self._cache_timestamp = None

    def get(self, key: str, timestamp: datetime) -> float:
        # Forward-fill lookup with caching
        if timestamp != self._cache_timestamp:
            self._update_cache(timestamp)
        return self._cache.get(key)
```

**Pros:**
- Clean API
- Centralized caching (compute once per timestamp)
- Easy to add new context sources

**Cons:**
- Need to pass context dict at initialization
- Manual forward-fill for different frequencies

### Option 2: Clock-Level Context Synchronization

**API:**
```python
class Clock:
    def add_context_feed(self, name: str, feed: DataFeed):
        self._context_feeds.append((name, feed))

    def next_event(self):
        event = super().next_event()
        # Update global context to event timestamp
        self._update_context(event.timestamp)
        return event, self._current_context
```

**Pros:**
- Clock handles synchronization automatically
- Context always in sync with events

**Cons:**
- Clock becomes more complex
- Tight coupling between clock and context

### Option 3: Pre-Join Everything (DataFrame Approach)

**API:**
```python
# User joins all data upfront
aapl = aapl.join(vix, how='left').ffill()
aapl = aapl.join(spy, how='left').ffill()

feed = HistoricalDataFeed(aapl, signals=['ml_pred', 'VIX', 'SPY'])

# Strategy treats everything as signals
def on_market_data(self, event):
    pred = event.data.signals['ml_pred']
    vix = event.data.signals['VIX']
```

**Pros:**
- Simplest implementation (no engine changes)
- User has full control over joins
- Works with existing signals dict

**Cons:**
- **Memory inefficient for multi-asset** (VIX duplicated 500x)
- Users must manually handle joins
- No distinction between asset vs context data

### Option 4: Hybrid Approach ⭐ RECOMMENDED

**API:**
```python
# Single-asset: Embed context in signals (simple)
aapl_df = aapl_df.join(vix).ffill()
engine = Engine(
    feed=HistoricalDataFeed(aapl_df, signals=['ml_pred', 'VIX']),
    strategy=MyStrategy(),
)

# Multi-asset: Separate context (memory efficient)
engine = Engine(
    feeds={
        'AAPL': HistoricalDataFeed(aapl_df, signals=['ml_pred']),
        'MSFT': HistoricalDataFeed(msft_df, signals=['ml_pred']),
    },
    context=Context({
        'VIX': vix_series,
        'SPY': spy_series,
    }),
    strategy=MyStrategy(),
)

# Strategy API is unified
def on_market_data(self, event, context):
    pred = event.data.signals['ml_pred']
    vix = context.get('VIX')  # Works in both modes
```

**Pros:**
- Best of both worlds
- Simple for single-asset, efficient for multi-asset
- Clear separation: signals (asset) vs context (market)

**Cons:**
- Two initialization patterns (minor)

**Decision: Implement Option 4**

---

## 4. Memory & Performance Analysis

### Memory Comparison (500 stocks, 252 days, 10 context indicators)

**Option 3 (Pre-join):**
- VIX series: 252 values * 8 bytes = 2 KB
- Duplicated 500x: 2 KB * 500 = 1 MB per indicator
- 10 indicators: **10 MB overhead**
- At scale (5000 stocks, 10 years): **5 GB overhead!**

**Option 4 (Separate context):**
- VIX series: 252 * 10 years * 8 bytes = 20 KB per indicator
- 10 indicators: **200 KB total**
- Shared across all stocks: **No duplication**

**Speedup: 250x less memory for context data**

### Performance: Context Caching

**Naive approach (no caching):**
```python
for event in events:  # 500 stocks * 252 days = 126,000 events
    vix = vix_df.loc[event.timestamp]  # O(log n) lookup
    spy = spy_df.loc[event.timestamp]  # O(log n) lookup
```
Cost: 126,000 events * 2 lookups = **252,000 DataFrame lookups**

**Cached approach:**
```python
class Context:
    def at(self, timestamp):
        if timestamp == self._cache_timestamp:
            return self._cache  # Cache hit!

        # Cache miss - compute once
        self._cache = {k: series.asof(timestamp) for k, series in self._sources.items()}
        self._cache_timestamp = timestamp
        return self._cache
```

With events sorted by timestamp (which Clock guarantees):
- 500 stocks at timestamp T: 1 cache miss + 499 cache hits
- Cost: 252 timestamps * 2 lookups = **504 DataFrame lookups**

**Speedup: 500x fewer lookups with caching!**

---

## 5. Data Synchronization: Different Frequencies

### Challenge: Intraday Context with Daily Trading

```
Stock bars (daily):
2023-01-03 EOD: AAPL close = $150

VIX bars (intraday):
2023-01-03 09:30: VIX = 18.5
2023-01-03 10:00: VIX = 19.2
2023-01-03 15:45: VIX = 21.3
2023-01-03 16:00: VIX = 20.8

Question: Which VIX value to use for AAPL EOD bar?
```

### Solution: Forward-Fill with .asof()

```python
class Context:
    def at(self, timestamp: datetime) -> dict:
        """Get context at timestamp using forward-fill (asof)"""
        context = {}
        for key, series in self._sources.items():
            # .asof() returns last known value <= timestamp
            value = series.asof(timestamp)
            context[key] = value if not pd.isna(value) else None
        return context
```

**For daily AAPL bar at 2023-01-03 16:00:**
- `vix_series.asof('2023-01-03 16:00')` returns VIX at 16:00 = 20.8 ✓

**For intraday AAPL bar at 2023-01-03 10:30:**
- `vix_series.asof('2023-01-03 10:30')` returns VIX at 10:00 = 19.2 ✓

**Point-in-time correctness maintained!**

### Advanced: Alignment Specifications (Future)

For complex use cases:
```python
context = Context({
    'VIX_close': {
        'data': vix_intraday,
        'alignment': 'close',  # Use value at market close
    },
    'VIX_max': {
        'data': vix_intraday,
        'aggregation': 'max',  # Max VIX during day
    }
})
```

**Phase 1: Use simple .asof() forward-fill**
**Phase 2: Add alignment/aggregation if users need it**

---

## 6. Practical Examples

### Example 1: VIX-Filtered ML Strategy (Single Asset)

```python
import pandas as pd
from ml4t.backtest import Engine, Strategy
from ml4t.backtest.data import HistoricalDataFeed

# Load AAPL data
aapl = pd.read_parquet('aapl_daily.parquet')

# Add ML predictions
aapl['ml_pred'] = my_model.predict(aapl)

# Add VIX context (simple join)
vix = pd.read_parquet('vix_daily.parquet')['close'].rename('VIX')
aapl = aapl.join(vix, how='left').ffill()

# Strategy
class VIXFilteredMLStrategy(Strategy):
    def on_market_data(self, event, context):
        pred = event.data.signals['ml_pred']
        vix = event.data.signals['VIX']  # VIX is just another signal

        position = self.get_position(event.asset_id)

        # VIX filter
        if vix > 30:
            if position > 0:
                self.close_position(event.asset_id)  # Exit in high vol
            return  # Don't enter new positions

        # ML logic
        if pred > 0.8 and position == 0:
            self.buy_percent(event.asset_id, 0.95)
        elif pred < 0.4 and position > 0:
            self.close_position(event.asset_id)

# Run
feed = HistoricalDataFeed(aapl, signals=['ml_pred', 'VIX'])
engine = Engine(feed=feed, strategy=VIXFilteredMLStrategy())
results = engine.run()
```

**Note:** For single asset, context can be embedded in signals (Option 3). No separate Context object needed!

### Example 2: Multi-Asset with Shared Context

```python
from ml4t.backtest import Engine, Strategy
from ml4t.backtest.data import HistoricalDataFeed, Context

# Load multiple stocks
stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
asset_feeds = {}

for symbol in stocks:
    df = load_stock(symbol)
    df['ml_pred'] = my_model.predict(symbol, df)
    asset_feeds[symbol] = HistoricalDataFeed(df, signals=['ml_pred'])

# Load context (NOT duplicated per stock)
vix = load_vix()['close']
spy_trend = load_spy()['trend_5d']
breadth = compute_market_breadth(stocks)

# Strategy
class MultiAssetMLStrategy(Strategy):
    def on_market_data(self, event, context):
        # Asset-specific
        pred = event.data.signals['ml_pred']
        position = self.get_position(event.asset_id)

        # Market context (shared across all assets)
        vix = context.get('VIX')
        breadth = context.get('breadth')

        # Regime-dependent logic
        if vix and vix > 30:
            # High vol: close 50% of all positions
            if position > 0:
                self.reduce_position(event.asset_id, 0.5)
            return

        if breadth and breadth < 0.3:
            return  # Weak market, don't enter

        # ML logic
        if pred > 0.8 and position == 0:
            # Position size depends on VIX
            size = 0.02 if vix < 20 else 0.01
            self.buy_percent(event.asset_id, size)

# Run with separate context
engine = Engine(
    feeds=asset_feeds,
    context=Context({
        'VIX': vix,
        'SPY_trend': spy_trend,
        'breadth': breadth,
    }),
    strategy=MultiAssetMLStrategy(),
)

results = engine.run()
```

**Memory savings:** VIX stored once (2 KB) instead of 4x (8 KB). At 500 stocks: 1 MB vs 500 MB!

### Example 3: Declarative Rules with Context (Phase 3)

```python
from ml4t.backtest.strategies import SignalStrategy, Rule

strategy = SignalStrategy(
    entry_rules=[
        # Asset-specific
        Rule.signal_above('ml_pred', 0.8),
        Rule.not_in_position(),

        # Context filters
        Rule.context_below('VIX', 30),          # Only trade if VIX < 30
        Rule.context_above('breadth', 0.4),     # Only if market healthy
    ],
    exit_rules=[
        Rule.signal_below('ml_pred', 0.4),
        Rule.stop_loss(0.05),
        Rule.take_profit(0.15),

        # Emergency exit on VIX spike
        Rule.context_above('VIX', 35),
    ],
    position_sizer=PercentSizer(0.95),
)
```

**Benefits:**
- Fully declarative (Numba-compilable)
- Context rules work exactly like signal rules
- 10-100x faster than imperative Python

---

## 7. Edge Cases & Error Handling

### Missing Context Data

**Scenario:** Stock data 2020-2023, but VIX only available 2021-2023

**Solution: Return None for missing values**
```python
class Context:
    def at(self, timestamp):
        context = {}
        for key, series in self._sources.items():
            value = series.asof(timestamp)
            context[key] = value if not pd.isna(value) else None
        return context

# Strategy checks explicitly
def on_market_data(self, event, context):
    vix = context.get('VIX')
    if vix is None:
        return  # No VIX data, skip this bar

    if vix > 30:
        # ... normal logic
```

**Rationale:** Explicit is better than implicit. Strategy controls how to handle missing data.

### Context Timestamp Precision

**Challenge:** Stock timestamp = 2023-01-03 (date only), VIX timestamp = 2023-01-03 16:00 (datetime)

**Solution:** .asof() handles this automatically
```python
# Stock event at 2023-01-03 (midnight)
vix_value = vix_series.asof(pd.Timestamp('2023-01-03'))
# Returns VIX <= 2023-01-03, which is end of previous day

# Better: Normalize stock timestamps to market close
stock_timestamps = pd.DatetimeIndex(dates) + pd.Timedelta(hours=16)
```

**Recommendation:** Users should normalize timestamps to avoid confusion. Document best practices.

---

## 8. Live Trading Implications

### Asynchronous Context Updates

In backtesting: all data pre-loaded
In live trading: context arrives asynchronously

```python
class LiveDataFeed:
    def __init__(self):
        self.context_buffer = {}  # Current context state

    def __iter__(self):
        while True:
            message = self.websocket.recv()

            if message.symbol in ['VIX', 'SPY']:
                # Update context buffer
                self.context_buffer[message.symbol] = message.price
            else:
                # Asset tick - create market event
                event = self._create_market_event(message)

                # Attach current context snapshot
                context = self.context_buffer.copy()

                yield event, context
```

**Key insight:** Strategy code is IDENTICAL in backtest and live:
```python
def on_market_data(self, event, context):
    vix = context.get('VIX')
    # ... same logic in both modes
```

---

## 9. Testing Strategy

### Test 1: Context Synchronization
```python
def test_vix_synchronized_to_stock_bars():
    # Daily stock bars
    aapl = pd.DataFrame({
        'close': [100, 101, 102],
    }, index=pd.date_range('2023-01-01', periods=3, freq='D'))

    # Intraday VIX (hourly)
    vix_hourly = pd.Series(
        range(72),  # Values 0-71
        index=pd.date_range('2023-01-01', periods=72, freq='H')
    )

    # Expected: Stock bar at day N should use VIX at hour 24*N
    # Day 0 (2023-01-01): VIX hour 0 = 0
    # Day 1 (2023-01-02): VIX hour 24 = 24
    # Day 2 (2023-01-03): VIX hour 48 = 48

    engine = Engine(
        feed=HistoricalDataFeed(aapl, signals=[]),
        context=Context({'VIX': vix_hourly}),
        strategy=RecordingStrategy(),
    )

    result = engine.run()
    assert result.recorded_context[0]['VIX'] == 0
    assert result.recorded_context[1]['VIX'] == 24
    assert result.recorded_context[2]['VIX'] == 48
```

### Test 2: VIX Filter Prevents Trading
```python
def test_vix_spike_stops_trading():
    aapl = pd.DataFrame({
        'ml_pred': [0.9, 0.9, 0.9, 0.9],  # Always bullish
        'close': [100, 101, 102, 103],
    }, index=pd.date_range('2023-01-01', periods=4))

    vix = pd.Series([15, 15, 35, 15], index=aapl.index)  # Spike on day 3

    class VIXFilterStrategy(Strategy):
        def on_market_data(self, event, context):
            if context.get('VIX', 0) > 30:
                return
            if event.data.signals['ml_pred'] > 0.8:
                self.buy_percent(event.asset_id, 1.0)

    engine = Engine(
        feed=HistoricalDataFeed(aapl, signals=['ml_pred']),
        context=Context({'VIX': vix}),
        strategy=VIXFilterStrategy(),
    )

    result = engine.run()

    # Should trade on days 0, 1, 3 but NOT day 2 (VIX spike)
    trade_dates = [t.timestamp.date() for t in result.trades]
    assert pd.Timestamp('2023-01-03').date() not in trade_dates
```

### Test 3: Multi-Asset Context Sharing
```python
def test_context_shared_across_assets():
    """Verify VIX is computed once, used for all assets"""

    feeds = {
        'AAPL': HistoricalDataFeed(aapl_df, signals=[]),
        'MSFT': HistoricalDataFeed(msft_df, signals=[]),
        'GOOGL': HistoricalDataFeed(googl_df, signals=[]),
    }

    # Track context lookups
    lookup_count = 0
    original_asof = pd.Series.asof

    def tracked_asof(self, timestamp):
        nonlocal lookup_count
        lookup_count += 1
        return original_asof(self, timestamp)

    with patch.object(pd.Series, 'asof', tracked_asof):
        engine = Engine(
            feeds=feeds,
            context=Context({'VIX': vix_series}),
            strategy=DummyStrategy(),
        )
        engine.run()

    # Should be ~252 lookups (once per timestamp)
    # NOT 252 * 3 = 756 (once per asset-event)
    assert lookup_count < 300  # Allow some overhead
```

---

## 10. Implementation Roadmap

### Phase 1a: Basic Signals (Week 1) ✅ PRIORITY
**Goal:** Asset-specific ML signals work

- [ ] Add `signals: dict[str, float]` to MarketData
- [ ] Extend HistoricalDataFeed with signal_columns parameter
- [ ] Update event creation to populate signals dict
- [ ] Add Strategy helper methods (get_position, buy_percent, etc.)
- [ ] Write basic signal tests
- [ ] Create example ML strategy notebook

**Deliverable:** Users can backtest ML strategies (no context yet)

### Phase 1b: Basic Context (Week 2)
**Goal:** Single-asset + context filtering works

- [ ] Implement Context class with caching
- [ ] Add context parameter to Strategy.on_market_data()
- [ ] Support embedding context in signals (for single-asset)
- [ ] Add tests for context synchronization
- [ ] Document VIX filtering pattern

**Deliverable:** "If VIX > 30, don't trade" works

### Phase 2: Multi-Asset Context (Week 3-4) - ONLY IF NEEDED
**Goal:** Memory-efficient multi-asset with shared context

- [ ] Extend Engine to accept feeds + context separately
- [ ] Implement efficient context caching
- [ ] Add support for different data frequencies
- [ ] Test with 100+ stock universe
- [ ] Benchmark memory usage

**Deliverable:** Production-ready multi-asset strategies

### Phase 3: Declarative Context Rules (Week 5) - FUTURE
**Goal:** Fast declarative strategies with regime filters

- [ ] Add Rule.context_above(), Rule.context_below()
- [ ] Extend SignalStrategy to support context rules
- [ ] Numba-compile rule evaluator with context
- [ ] Benchmark performance (target: 100k bars/sec)

**Deliverable:** Declarative strategies 10-100x faster

---

## 11. API Summary

### Single-Asset Pattern (Simple)
```python
# Join context to asset data
df = df.join(vix).join(spy).ffill()

engine = Engine(
    feed=HistoricalDataFeed(df, signals=['ml_pred', 'VIX', 'SPY']),
    strategy=MyStrategy(),
)

def on_market_data(self, event, context):
    vix = event.data.signals['VIX']  # Context via signals
```

### Multi-Asset Pattern (Efficient)
```python
engine = Engine(
    feeds={
        'AAPL': HistoricalDataFeed(aapl_df, signals=['ml_pred']),
        'MSFT': HistoricalDataFeed(msft_df, signals=['ml_pred']),
    },
    context=Context({'VIX': vix_series, 'SPY': spy_series}),
    strategy=MyStrategy(),
)

def on_market_data(self, event, context):
    pred = event.data.signals['ml_pred']  # Asset-specific
    vix = context.get('VIX')              # Market-wide
```

### Declarative Pattern (Fast)
```python
strategy = SignalStrategy(
    entry_rules=[
        Rule.signal_above('ml_pred', 0.8),
        Rule.context_below('VIX', 30),
    ],
    exit_rules=[
        Rule.stop_loss(0.05),
        Rule.context_above('VIX', 35),
    ],
)
```

---

## 12. Framework Comparison

| Feature | Backtrader | Zipline | ml4t.backtest |
|---------|-----------|---------|---------|
| Multi-feed support | ✅ self.datas[N] | ✅ data.current() | ✅ feeds + context |
| Primary/context distinction | ❌ All feeds equal | ❌ All symbols equal | ✅ Clear separation |
| Memory efficient | ❌ Duplicates data | ❌ Duplicates data | ✅ Shared context |
| Context caching | ❌ No caching | ❌ No caching | ✅ Automatic caching |
| ML-first design | ❌ Indicator-focused | ❌ Pipeline-focused | ✅ Signal-focused |
| Different frequencies | ⚠️ Manual handling | ⚠️ Manual handling | ✅ Auto forward-fill |
| Live trading | ✅ Supported | ✅ Supported | ✅ Same API |

**ml4t.backtest advantage:** Designed from ground-up for ML workflows with context-dependent logic.

---

## 13. Open Questions

### Q1: Should context be mandatory or optional?
**A: Optional - default to empty dict**
- Single-asset strategies don't need it
- Multi-asset strategies with regime filtering need it
- Making it mandatory breaks backward compatibility

### Q2: How to handle derived context (market breadth)?
**A: Pre-compute before backtest (Phase 1)**
- Computed fresh at each timestamp (too slow)
- Pre-compute in preparation step (recommended)
- Future: Engine can compute if provided function

### Q3: Support for event-driven context (Fed announcements)?
**A: Phase 4 (future work)**
- Type 4 data sources need different architecture
- Can be modeled as sparse context series for now
- Full event calendar support is post-MVP

---

## 14. Success Criteria

**Phase 1a (Signals):**
- [ ] ML strategy in <30 lines of code
- [ ] No look-ahead bias by design
- [ ] Signals accessible via event.data.signals dict

**Phase 1b (Basic Context):**
- [ ] VIX filtering works correctly
- [ ] Context synchronized to event timestamp
- [ ] Missing context handled gracefully

**Phase 2 (Multi-Asset):**
- [ ] 500 stocks + 10 context indicators < 100 MB memory
- [ ] Context cached efficiently (< 1000 DataFrame lookups)
- [ ] Same API for single and multi-asset

**Phase 3 (Declarative):**
- [ ] Rule.context_below('VIX', 30) works
- [ ] Numba compilation successful
- [ ] 10-100x faster than imperative Python

---

## 15. Related Documents

- `ml_signal_architecture.md` - ML signal integration (base design)
- Current `data/feed.py` - DataFeed implementation
- Current `engine.py` - Event loop and dispatch
- Current `core/clock.py` - Multi-feed synchronization

---

**Last Updated:** 2025-11-14
**Next Review:** After Phase 1a implementation
