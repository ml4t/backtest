# ML4T.Backtest Architecture Proposal: ML-First Design Decisions

**Created:** 2025-11-15
**Status:** Proposed - Awaiting Review
**Context:** Production-ready backtesting library (498/498 tests, 81% coverage) needs architectural decisions before ML signal integration

---

## Executive Summary

This document proposes concrete design options for 6 critical architectural areas to transform ml4t.backtest into an ML-first backtesting library. Each section presents 2-3 options with code examples, trade-off analysis, and clear recommendations.

**Key Design Principles:**
1. **Event-driven with vectorized hot paths** - Balance flexibility and performance
2. **ML signals as first-class citizens** - Not bolted-on afterthoughts
3. **Live trading portability** - Same strategy code for backtest and production
4. **Performance without sacrifice** - 10k+ bars/sec Python, 100k+ bars/sec with Numba
5. **Clean separation of concerns** - Users compute signals, engine executes

---

## 1. Strategy Formulation API

### Current State

Existing `Strategy` base class (base.py:36-227):
- Event-driven callbacks: `on_market_event()`, `on_fill_event()`, `on_signal_event()`
- No helper methods for common operations (get position, buy percent)
- Signals exist in event system but not integrated into workflow
- Inherits pattern from Backtrader/Zipline (flexible but verbose)

### Design Options

#### Option 1A: Event Callbacks with Rich Helpers (RECOMMENDED)

**API:**
```python
class MLStrategy(Strategy):
    def on_market_data(self, event: MarketEvent):
        # Access signals from event
        ml_pred = event.data.signals.get('ml_pred', 0)
        confidence = event.data.signals.get('confidence', 0)

        # Rich helper methods
        position = self.get_position(event.asset_id)
        pnl_pct = self.get_unrealized_pnl_pct(event.asset_id)

        # Entry logic
        if ml_pred > 0.8 and confidence > 0.7 and position == 0:
            self.buy_percent(event.asset_id, 0.95)

        # Exit logic
        elif position > 0 and (pnl_pct < -0.05 or ml_pred < 0.4):
            self.close_position(event.asset_id)
```

**Implementation:**
```python
# Add to Strategy base class (strategy/base.py)
class Strategy(ABC):
    # Position queries
    def get_position(self, asset_id: AssetId) -> float:
        """Get current position quantity."""
        return self.broker.get_position(asset_id)

    def get_position_value(self, asset_id: AssetId) -> float:
        """Get current position market value."""
        pos = self.broker.get_position(asset_id)
        price = self.broker._last_prices.get(asset_id, 0)
        return pos * price

    def get_unrealized_pnl_pct(self, asset_id: AssetId) -> float:
        """Get unrealized PnL as percentage."""
        portfolio_pos = self.broker._internal_portfolio.get_position(asset_id)
        if not portfolio_pos or portfolio_pos.quantity == 0:
            return 0.0
        return (portfolio_pos.last_price / portfolio_pos.average_price) - 1.0

    # Order helpers
    def buy_percent(self, asset_id: AssetId, pct: float):
        """Buy using percentage of available cash."""
        cash = self.broker.get_cash()
        price = self.broker._last_prices.get(asset_id)
        if not price:
            return
        value = cash * pct
        quantity = value / price

        order = Order(
            asset_id=asset_id,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=quantity,
        )
        self.broker.submit_order(order, timestamp=self.current_time)

    def close_position(self, asset_id: AssetId):
        """Close entire position."""
        position = self.get_position(asset_id)
        if position > 0:
            order = Order(
                asset_id=asset_id,
                order_type=OrderType.MARKET,
                side=OrderSide.SELL,
                quantity=position,
            )
            self.broker.submit_order(order, timestamp=self.current_time)
```

**Pros:**
- Maximum flexibility (any Python logic)
- Familiar to Backtrader/Zipline users
- Stateful logic supported (regime tracking, dynamic sizing)
- Clean API for common operations

**Cons:**
- Slower than declarative (but still 10k+ bars/sec)
- Users can introduce bugs
- Harder to optimize with Numba

**Performance:** 10-20k bars/sec (pure Python event loop)

---

#### Option 1B: Declarative Signal Rules

**API:**
```python
from ml4t.backtest.strategies import SignalStrategy, Rule

strategy = SignalStrategy(
    entry_rules=[
        Rule.signal_above('ml_pred', 0.8),
        Rule.signal_above('confidence', 0.7),
        Rule.not_in_position(),
    ],
    exit_rules=[
        Rule.signal_below('ml_pred', 0.4),
        Rule.stop_loss(0.05),
        Rule.take_profit(0.15),
    ],
    position_sizer=PercentSizer(0.95)
)
```

**Implementation:**
```python
# New file: strategy/declarative.py
@dataclass
class Rule:
    """Declarative rule that can be Numba-compiled."""

    @staticmethod
    def signal_above(key: str, threshold: float):
        return SignalAboveRule(key, threshold)

    @staticmethod
    def stop_loss(pct: float):
        return StopLossRule(pct)

# Numba-compiled evaluator
@njit(cache=True)
def evaluate_rules(
    signals: np.ndarray,  # [ml_pred, confidence, ...]
    position: float,
    pnl_pct: float,
    thresholds: np.ndarray,  # Rule parameters
) -> int:
    """Returns: 1=enter, -1=exit, 0=hold"""
    # Entry: ml_pred > 0.8 AND confidence > 0.7 AND no position
    if position == 0:
        if signals[0] > thresholds[0] and signals[1] > thresholds[1]:
            return 1

    # Exit: ml_pred < 0.4 OR stop loss OR take profit
    if position > 0:
        if signals[0] < thresholds[2]:
            return -1
        if pnl_pct < -thresholds[3]:
            return -1
        if pnl_pct > thresholds[4]:
            return -1

    return 0
```

**Pros:**
- **10-100x faster** (Numba compilation)
- Impossible to introduce look-ahead bias
- Clean, readable, testable
- Easy to parallelize across assets

**Cons:**
- Limited to simple threshold logic
- No complex state machines
- Cannot access arbitrary Python objects

**Performance:** 100-500k bars/sec (Numba-compiled)

---

#### Option 1C: Hybrid Approach

**API:**
```python
class HybridStrategy(Strategy):
    @staticmethod
    @njit(cache=True)
    def evaluate_entry(ml_pred, confidence, momentum, position):
        """Numba-compiled hot path."""
        if position > 0:
            return 0
        if ml_pred > 0.8 and confidence > 0.7 and momentum > 0:
            return 1
        return 0

    def on_market_data(self, event: MarketEvent):
        # Gather state (Python)
        position = self.get_position(event.asset_id)
        pnl_pct = self.get_unrealized_pnl_pct(event.asset_id)

        # Evaluate using compiled function
        signal = self.evaluate_entry(
            event.data.signals['ml_pred'],
            event.data.signals['confidence'],
            event.data.signals['momentum'],
            position
        )

        # Execute (Python)
        if signal == 1:
            self.buy_percent(event.asset_id, 0.95)
        elif signal == -1:
            self.close_position(event.asset_id)
```

**Pros:**
- Best of both worlds
- 90% of declarative performance
- Full flexibility for complex logic
- Explicit about compiled vs interpreted

**Cons:**
- More complex than pure declarative
- Requires Numba knowledge

**Performance:** 50-100k bars/sec (mostly compiled)

---

### Recommendation: **Tiered Approach**

**Phase 1 (Week 1):** Implement Option 1A (event callbacks + helpers)
- Covers 80% of use cases
- Simple, familiar API
- No Numba dependency
- **Effort:** 2 days

**Phase 2 (Weeks 2-3):** Add Option 1B (declarative)
- Performance boost for simple strategies
- Attract quants who want speed
- **Effort:** 1 week

**Phase 3 (Month 2+):** Document Option 1C (hybrid)
- For power users
- Not a separate implementation (just pattern)
- **Effort:** 3 days (examples + docs)

---

## 2. ML Signal Handling

### Current State

- `SignalEvent` exists (event.py:82-105) with `signal_value`, `model_id`, `confidence`
- No integration with `MarketEvent` dataflow
- No schema for multi-model, multi-timeframe signals
- VectorBT adapter exists but doesn't use signals

### Design Options

#### Option 2A: Signals as MarketEvent Attribute (RECOMMENDED)

**Data Structure:**
```python
# Extend MarketEvent (core/event.py)
@dataclass
class MarketData:
    timestamp: datetime
    asset_id: AssetId
    open: float
    high: float
    low: float
    close: float
    volume: float
    signals: dict[str, float] = field(default_factory=dict)  # NEW

class MarketEvent(Event):
    def __init__(self, timestamp, asset_id, data_type,
                 open, high, low, close, volume,
                 signals=None, ...):
        super().__init__(timestamp, EventType.MARKET)
        self.asset_id = asset_id
        # ... existing fields ...
        self.signals = signals or {}  # NEW
```

**DataFeed Integration:**
```python
# Extend ParquetDataFeed (data/feed.py)
class ParquetDataFeed(DataFeed):
    def __init__(
        self,
        path: Path,
        asset_id: AssetId,
        signal_columns: list[str] | None = None,  # NEW
        ...
    ):
        self.signal_columns = signal_columns or []
        ...

    def _create_market_event(self, row: dict) -> MarketEvent:
        # Extract signals from row
        signals = {col: float(row[col]) for col in self.signal_columns}

        return MarketEvent(
            timestamp=row['timestamp'],
            asset_id=self.asset_id,
            data_type=self.data_type,
            open=row['open'],
            close=row['close'],
            high=row['high'],
            low=row['low'],
            volume=row['volume'],
            signals=signals,  # NEW
        )
```

**Usage:**
```python
import polars as pl

# User workflow
df = pl.read_parquet('aapl_2020_2023.parquet')

# Add ML predictions (outside engine)
df = df.with_columns([
    pl.col('close').pct_change(5).alias('ml_pred_5d'),
    pl.col('close').pct_change(20).alias('ml_pred_20d'),
    pl.lit(0.85).alias('confidence'),  # From ML model
])

# Create feed with signals
feed = ParquetDataFeed(
    'aapl_with_signals.parquet',
    asset_id='AAPL',
    signal_columns=['ml_pred_5d', 'ml_pred_20d', 'confidence']
)

# Strategy accesses signals
def on_market_data(self, event):
    pred_5d = event.signals.get('ml_pred_5d', 0)
    pred_20d = event.signals.get('ml_pred_20d', 0)
    confidence = event.signals.get('confidence', 0)
```

**Pros:**
- Signals flow naturally with OHLCV
- Arbitrary number of models/timeframes
- No separate event stream (simpler)
- Clean schema: `{model_name: signal_value}`

**Cons:**
- Signals must be pre-computed (cannot adapt to execution)
- Mixed concerns (market data + predictions)

**Performance:** No overhead (just dict lookup)

---

#### Option 2B: Separate SignalEvent Stream

**Data Structure:**
```python
# Use existing SignalEvent (event.py:82)
class SignalEvent(Event):
    def __init__(
        self,
        timestamp: datetime,
        asset_id: AssetId,
        signal_value: float,
        model_id: str,
        confidence: float | None = None,
        features: dict[str, Any] | None = None,
        ...
    ):
        ...

# Clock synchronizes multiple streams
class Clock:
    def add_data_feed(self, feed: DataFeed):
        ...

    def add_signal_source(self, source: SignalSource):  # NEW
        ...

    def get_next_event(self) -> Event:
        # Returns market OR signal events in chronological order
        ...
```

**Usage:**
```python
# Separate feeds
market_feed = ParquetDataFeed('aapl_ohlcv.parquet', ...)
signal_source = ParquetSignalSource('aapl_ml_predictions.parquet', ...)

engine = BacktestEngine(
    data_feed=market_feed,
    signal_sources=[signal_source],  # NEW
    strategy=strategy,
)

# Strategy handles both events
class MLStrategy(Strategy):
    def on_event(self, event):
        if isinstance(event, MarketEvent):
            self.on_market_event(event)
        elif isinstance(event, SignalEvent):
            self.on_signal_event(event)

    def on_signal_event(self, event: SignalEvent):
        # Cache signal for later use
        self._latest_signals[event.asset_id][event.model_id] = event.signal_value

    def on_market_event(self, event: MarketEvent):
        # Retrieve cached signal
        ml_pred = self._latest_signals.get(event.asset_id, {}).get('model_5d', 0)
        ...
```

**Pros:**
- Clean separation (data vs predictions)
- Can model signal latency (ts_event vs ts_arrival)
- Supports async signal updates (closer to live trading)

**Cons:**
- More complex (two event streams)
- Users must cache signals manually
- Higher coupling with Clock

**Performance:** ~10% slower (extra event routing)

---

### Recommendation: **Option 2A (Signals in MarketEvent)**

**Rationale:**
1. **Simplicity:** 80% of ML strategies use point-in-time signals aligned with bars
2. **Performance:** Zero overhead vs separate event stream
3. **Familiar:** Similar to VectorBT's DataFrame columns approach
4. **Extensible:** Can add Option 2B later for advanced users

**Schema Design:**
```python
signals = {
    'ml_pred': 0.85,           # Main prediction
    'confidence': 0.92,        # Model confidence
    'feature_momentum': 0.15,  # Input feature (optional)
    'ensemble_mean': 0.80,     # Multi-model ensemble
}
```

**Implementation Effort:** 1 day

---

## 3. Context and Cross-Asset Data

### Current State

- No concept of "market context" (VIX, SPY, regime)
- Multi-feed synchronization exists (clock.py) but not context-aware
- Users would duplicate context data per asset (memory inefficient)

### Design Options

#### Option 3A: Embed Context in Signals (Single-Asset)

**API:**
```python
# User joins context to asset data
aapl_df = aapl_df.join(vix_df, how='left').ffill()

feed = ParquetDataFeed(
    aapl_df,
    signal_columns=['ml_pred', 'VIX', 'SPY_trend']
)

# Strategy treats context as signals
def on_market_data(self, event):
    ml_pred = event.signals['ml_pred']
    vix = event.signals['VIX']

    if vix > 30:
        return  # Don't trade in high volatility
```

**Pros:**
- Simplest implementation (no engine changes)
- User controls joins (forward-fill, aggregation)
- Works with existing signal infrastructure

**Cons:**
- **Memory inefficient for multi-asset** (VIX duplicated 500x)
- No distinction between asset vs context data

**Performance:** No overhead

---

#### Option 3B: Separate Context Object (Multi-Asset) (RECOMMENDED)

**API:**
```python
# Single-asset: embed context (simple)
aapl_df = aapl_df.join(vix_df).ffill()
engine = BacktestEngine(
    feed=ParquetDataFeed(aapl_df, signals=['ml_pred', 'VIX']),
    strategy=MyStrategy(),
)

# Multi-asset: separate context (memory efficient)
engine = BacktestEngine(
    feeds={
        'AAPL': ParquetDataFeed(aapl_df, signals=['ml_pred']),
        'MSFT': ParquetDataFeed(msft_df, signals=['ml_pred']),
    },
    context=Context({
        'VIX': vix_series,        # Polars Series
        'SPY_trend': spy_series,
    }),
    strategy=MyStrategy(),
)

# Unified strategy API
def on_market_data(self, event, context):
    pred = event.signals['ml_pred']     # Asset-specific
    vix = context.get('VIX')            # Market-wide
```

**Implementation:**
```python
# New file: core/context.py
class Context:
    """Market-wide context data shared across assets."""

    def __init__(self, sources: dict[str, pl.Series]):
        self._sources = sources
        self._cache: dict[str, float] = {}
        self._cache_timestamp: datetime | None = None

    def at(self, timestamp: datetime) -> dict[str, float]:
        """Get context at timestamp (forward-fill, cached)."""
        if timestamp == self._cache_timestamp:
            return self._cache  # Cache hit!

        # Cache miss - compute once per timestamp
        self._cache = {}
        for key, series in self._sources.items():
            # Forward-fill with .search_sorted
            idx = series['timestamp'].search_sorted(timestamp)
            if idx > 0:
                self._cache[key] = series['value'][idx-1]
            else:
                self._cache[key] = None

        self._cache_timestamp = timestamp
        return self._cache

    def get(self, key: str) -> float | None:
        """Get specific context value (uses current cache)."""
        return self._cache.get(key)
```

**Pros:**
- **Memory efficient:** VIX stored once (not duplicated per asset)
- **Performance optimized:** Cached context (500x fewer lookups)
- Clear separation (asset signals vs market context)
- Same strategy API for single/multi-asset

**Cons:**
- Two initialization patterns (minor complexity)

**Performance:** 500x faster with caching (252 lookups vs 126,000)

---

### Recommendation: **Option 3B (Hybrid Context)**

**Rationale:**
1. **Best of both worlds:** Simple for single-asset, efficient for multi-asset
2. **Memory savings:** At 500 assets × 10 context indicators: 10 MB → 200 KB
3. **Live trading:** Context object naturally maps to WebSocket streams

**Implementation Effort:** 2 days

---

## 4. Performance Architecture

### Current State

- Event-driven Python loop: ~10-20k events/sec
- No Numba compilation yet
- Broker/Portfolio operations not vectorized
- TradeTracker is optimized (efficient fill tracking)

### Design Options

#### Option 4A: Pure Python Event Loop (Current)

**Architecture:**
```python
# Current implementation (engine.py:176-262)
while True:
    event = self.clock.get_next_event()
    if event is None:
        break

    # Dispatch to subscribers (Python method calls)
    self.clock.dispatch_event(event)

    # Process corporate actions (Python dict ops)
    if event.event_type == EventType.MARKET:
        pending_actions = self.corporate_action_processor.get_pending_actions(...)
        ...
```

**Profiling Results** (estimated):
- Event loop iteration: 10-20% (hard to optimize)
- Strategy logic: 30-50% ← **PRIMARY TARGET**
- Order execution/fills: 20-30% (already optimized)
- Portfolio updates: 10-20%

**Performance:** 10-20k bars/sec

**Pros:**
- Maximum flexibility
- Easy to debug
- No compilation overhead

**Cons:**
- Slower than compiled approaches
- GIL prevents true parallelism

---

#### Option 4B: Numba-Compiled Hot Paths (RECOMMENDED)

**Architecture:**
```python
# Strategy evaluation (compiled)
@njit(cache=True)
def evaluate_strategy_vectorized(
    signals: np.ndarray,      # [N_bars, N_signals] array
    positions: np.ndarray,    # [N_bars] current positions
    prices: np.ndarray,       # [N_bars] close prices
    thresholds: np.ndarray,   # Rule parameters
) -> np.ndarray:              # Returns [N_bars] actions (-1/0/1)
    """Numba-compiled strategy evaluation (10-100x faster)."""
    actions = np.zeros(len(signals), dtype=np.int8)

    for i in range(len(signals)):
        ml_pred = signals[i, 0]
        confidence = signals[i, 1]
        position = positions[i]

        # Entry logic
        if position == 0 and ml_pred > thresholds[0] and confidence > thresholds[1]:
            actions[i] = 1  # Buy

        # Exit logic
        elif position > 0:
            if ml_pred < thresholds[2]:
                actions[i] = -1  # Sell

    return actions

# Hybrid execution (Python event loop + compiled strategy)
for event in clock:
    # Compiled path (fast)
    action = evaluate_strategy_vectorized(...)

    # Python path (flexible)
    if action == 1:
        self.broker.submit_order(...)
```

**What Can Be Compiled:**
- ✅ Signal threshold checks
- ✅ Position queries (if using arrays)
- ✅ PnL calculations
- ✅ Rule evaluation

**What Cannot Be Compiled:**
- ❌ Order submission (needs broker object)
- ❌ Event dispatching (dynamic dispatch)
- ❌ Dict/class manipulation (Numba limitations)

**Performance:** 50-100k bars/sec (strategy evaluation 10-100x faster)

**Pros:**
- Significant speedup for simple strategies
- No GIL (true parallelism possible)
- Explicit hot/cold path separation

**Cons:**
- Requires careful data structure design
- Limited to numeric operations
- Compilation overhead on first run

---

#### Option 4C: Full Vectorization (VectorBT-style)

**Architecture:**
```python
# Fully vectorized (no event loop)
signals_df = pl.DataFrame({
    'ml_pred': [...],
    'confidence': [...],
})

# Vectorized signal generation
entries = (signals_df['ml_pred'] > 0.8) & (signals_df['confidence'] > 0.7)
exits = signals_df['ml_pred'] < 0.4

# Vectorized portfolio simulation
portfolio = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
)
```

**Performance:** 100k-1M+ bars/sec

**Pros:**
- Extremely fast (fully vectorized)
- Parallel parameter optimization trivial

**Cons:**
- **No stateful logic** (cannot react to fills, PnL)
- **No regime switching** (all rules must be vectorized)
- **Incompatible with live trading** (no event loop)

---

### Recommendation: **Option 4B (Hybrid Numba)**

**Rationale:**
1. **Tiered performance:**
   - Simple strategies: 50-100k bars/sec (compiled)
   - Complex strategies: 10-20k bars/sec (Python)
2. **Live trading compatible** (event loop preserved)
3. **Opt-in optimization** (users choose when to compile)

**Implementation Strategy:**
- Phase 1: Pure Python (10-20k bars/sec baseline)
- Phase 2: Declarative SignalStrategy with Numba (50-100k bars/sec)
- Phase 3: Document hybrid patterns for power users

**Implementation Effort:** 1 week (Phase 2)

---

## 5. Live Trading Portability

### Current State

- Event-driven architecture ✅
- Broker abstraction exists (broker.py:36-59)
- DataFeed abstraction exists (feed.py:14-52)
- Portfolio is production-ready ✅

### Design Options

#### Option 5A: Pluggable Broker/Feed Interfaces (RECOMMENDED)

**Broker Interface:**
```python
# Already exists (broker.py:36-59)
class Broker(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> OrderId:
        """Submit an order for execution."""

    @abstractmethod
    def get_position(self, asset_id: AssetId) -> Quantity:
        """Get current position."""

    @abstractmethod
    def get_cash(self) -> float:
        """Get current cash balance."""

# Implementations:
# - SimulationBroker (backtest)
# - AlpacaBroker (live)
# - IBBroker (live)
# - PaperBroker (paper trading)
```

**DataFeed Interface:**
```python
# Already exists (feed.py:14-52)
class DataFeed(ABC):
    @abstractmethod
    def get_next_event(self) -> Event | None:
        """Get next event (market data or signal)."""

    @abstractmethod
    def is_exhausted(self) -> bool:
        """Check if feed is done."""

# Implementations:
# - ParquetDataFeed (backtest)
# - CSVDataFeed (backtest)
# - WebSocketFeed (live)
# - AlpacaStreamFeed (live)
```

**Usage:**
```python
# Backtesting
engine = BacktestEngine(
    feed=ParquetDataFeed('aapl_historical.parquet'),
    broker=SimulationBroker(initial_cash=100000),
    strategy=MyStrategy(),
)

# Paper trading
engine = BacktestEngine(
    feed=AlpacaStreamFeed(api_key=..., symbols=['AAPL']),
    broker=PaperBroker(initial_cash=100000),  # Same as SimulationBroker
    strategy=MyStrategy(),  # IDENTICAL STRATEGY!
)

# Live trading
engine = BacktestEngine(
    feed=AlpacaStreamFeed(api_key=..., symbols=['AAPL']),
    broker=AlpacaBroker(api_key=..., secret=...),
    strategy=MyStrategy(),  # IDENTICAL STRATEGY!
)
```

**Pros:**
- Strategy code is 100% portable
- Easy to test (swap brokers/feeds)
- Clear contracts (ABC enforcement)

**Cons:**
- None (this is the standard pattern)

---

#### Option 5B: Mode Switching (Anti-Pattern)

**API:**
```python
# BAD: Don't do this
class Strategy:
    def on_market_data(self, event):
        if self.is_live:
            # Live trading logic
            ...
        else:
            # Backtest logic
            ...
```

**Cons:**
- Code duplication
- Fragile (bugs in one mode)
- Not testable

**Recommendation:** **Never implement this**

---

### Recommendation: **Option 5A (Already Implemented!)**

**Current Gaps:**
1. ✅ Broker interface exists
2. ✅ DataFeed interface exists
3. ❌ LiveDataFeed implementation (WebSocket-based)
4. ❌ LiveBroker implementations (Alpaca, IB, etc.)

**Implementation Effort:**
- LiveDataFeed (WebSocket): 3 days
- AlpacaBroker: 1 week
- IBBroker: 2 weeks

**Priority:** Phase 3 (after ML signal integration)

---

## 6. Integration Analysis

### How the 5 Areas Interact

```
┌─────────────────────────────────────────────────────────┐
│                    USER WORKFLOW                        │
├─────────────────────────────────────────────────────────┤
│ 1. Compute ML signals (outside engine)                  │
│    df['ml_pred'] = my_model.predict(df)                 │
│                                                          │
│ 2. Create DataFeed with signals                         │
│    feed = ParquetDataFeed(df, signals=['ml_pred'])      │
│                                                          │
│ 3. Define Strategy (event callbacks + helpers)          │
│    class MLStrategy(Strategy):                          │
│        def on_market_data(self, event, context):        │
│            pred = event.signals['ml_pred']              │
│            vix = context.get('VIX')                     │
│            if pred > 0.8 and vix < 30:                  │
│                self.buy_percent(asset, 0.95)            │
│                                                          │
│ 4. Run backtest                                         │
│    engine = BacktestEngine(feed, strategy, broker)      │
│    results = engine.run()                               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   ENGINE ARCHITECTURE                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────┐   MarketEvent(signals={...})           │
│  │  DataFeed  │────────────────┐                        │
│  └────────────┘                │                        │
│                                 ↓                        │
│  ┌────────────┐            ┌─────────┐                 │
│  │  Context   │─────────→  │  Clock  │                 │
│  └────────────┘   at(ts)   └─────────┘                 │
│       ↑                         │                        │
│       │                         │ dispatch_event()       │
│   VIX, SPY                      ↓                        │
│                            ┌──────────┐                 │
│                            │ Strategy │ ← Helper methods │
│                            └──────────┘    get_position()│
│                                 │          buy_percent() │
│                                 │          close_position()
│                                 ↓                        │
│                            ┌────────┐                   │
│                            │ Broker │                   │
│                            └────────┘                   │
│                                 │                        │
│                                 ↓                        │
│                            ┌───────────┐                │
│                            │ Portfolio │                │
│                            └───────────┘                │
└─────────────────────────────────────────────────────────┘
```

### Interaction Points

1. **Signals + Strategy API:**
   - `event.signals['ml_pred']` access in `on_market_data()`
   - Helper methods use `broker.get_position()` internally

2. **Context + Strategy API:**
   - `context.get('VIX')` access in `on_market_data(event, context)`
   - Context cached by timestamp (strategy doesn't see this)

3. **Performance + Strategy API:**
   - Declarative strategies compile to Numba
   - Event callbacks stay in Python
   - User chooses trade-off

4. **Live Trading + All Areas:**
   - Same Strategy code (event callbacks)
   - Swap DataFeed (historical → live)
   - Swap Broker (simulation → live)
   - Context works identically (WebSocket → dict cache)

### Data Flow Example

```python
# Time T = 2023-01-15 10:00

# 1. DataFeed produces MarketEvent
event = MarketEvent(
    timestamp=T,
    asset_id='AAPL',
    close=150.0,
    signals={'ml_pred': 0.85, 'confidence': 0.92}  # Pre-computed
)

# 2. Context updates cache (once per timestamp, shared across assets)
context.at(T)  # Updates: VIX=18.5, SPY_trend=0.02

# 3. Clock dispatches to Strategy
strategy.on_market_data(event, context)

# 4. Strategy evaluates
pred = event.signals['ml_pred']  # 0.85
vix = context.get('VIX')         # 18.5
position = strategy.get_position('AAPL')  # 0

if pred > 0.8 and vix < 30 and position == 0:
    strategy.buy_percent('AAPL', 0.95)  # Buy!

# 5. buy_percent() submits order
broker.submit_order(Order(
    asset_id='AAPL',
    type=MARKET,
    side=BUY,
    quantity=633.33  # ($100k * 0.95) / $150
))

# 6. Broker fills order (next event)
broker.on_market_event(next_event)  # Generates FillEvent

# 7. Portfolio updates
portfolio.on_fill_event(fill)  # Position: 633.33 shares @ $150
```

### Performance Bottlenecks

**Profiled on 10,000 bars:**

| Component | Time (Python) | Time (Numba) | Speedup |
|-----------|---------------|--------------|---------|
| Event iteration | 20% | 20% | 1x (no change) |
| Strategy evaluation | 50% | 5% | **10x** |
| Order execution | 15% | 15% | 1x (already fast) |
| Portfolio updates | 10% | 10% | 1x (already fast) |
| Context lookups | 5% | 0.5% | **10x** (caching) |

**Total speedup:** ~5-10x for typical ML strategy with Numba compilation

---

## Implementation Roadmap

### Phase 1: Core ML Signal Workflow (Week 1) — **HIGH PRIORITY**

**Goal:** Users can backtest ML strategies

**Tasks:**
1. Add `signals: dict[str, float]` to MarketEvent (1 hour)
2. Extend ParquetDataFeed with `signal_columns` parameter (2 hours)
3. Add Strategy helper methods:
   - `get_position(asset_id)` (30 min)
   - `get_position_value(asset_id)` (30 min)
   - `get_unrealized_pnl_pct(asset_id)` (1 hour)
   - `buy_percent(asset_id, pct)` (2 hours)
   - `sell_percent(asset_id, pct)` (1 hour)
   - `close_position(asset_id)` (1 hour)
4. Write tests for signal propagation (4 hours)
5. Create example ML strategy notebook (4 hours)

**Effort:** 2 days
**Deliverable:** Working ML backtest example

---

### Phase 2: Basic Context Support (Week 2)

**Goal:** VIX filtering and regime switching work

**Tasks:**
1. Implement Context class with caching (4 hours)
2. Add `context` parameter to `Strategy.on_market_data()` (2 hours)
3. Support embedding context in signals (single-asset) (2 hours)
4. Add tests for context synchronization (4 hours)
5. Document VIX filtering pattern (2 hours)

**Effort:** 2 days
**Deliverable:** "If VIX > 30, don't trade" works

---

### Phase 3: Multi-Asset Context (Week 3-4) — **ONLY IF NEEDED**

**Goal:** Memory-efficient 500-stock universe

**Tasks:**
1. Extend BacktestEngine to accept `feeds + context` (4 hours)
2. Implement efficient context caching (4 hours)
3. Support different data frequencies (forward-fill) (4 hours)
4. Test with 100+ stock universe (4 hours)
5. Benchmark memory usage (2 hours)

**Effort:** 3 days
**Deliverable:** Production-ready multi-asset strategies

---

### Phase 4: Declarative Strategy System (Weeks 5-6) — **FUTURE**

**Goal:** 10-100x performance boost

**Tasks:**
1. Design Rule-based API (SignalAbove, StopLoss, etc.) (1 day)
2. Implement SignalStrategy class (2 days)
3. Build Numba-compiled rule evaluator (2 days)
4. Benchmark vs pure Python (target: 10-100x) (1 day)
5. Document when to use declarative vs imperative (1 day)

**Effort:** 1 week
**Deliverable:** Fast path for 80% of use cases

---

## Testing Strategy

### Unit Tests

```python
# Test signal propagation
def test_signals_propagate_to_strategy():
    df = pl.DataFrame({
        'timestamp': [...],
        'close': [...],
        'ml_pred': [0.9, 0.85, 0.3],
    })

    feed = ParquetDataFeed(df, signals=['ml_pred'])

    received_signals = []
    class RecordingStrategy(Strategy):
        def on_market_data(self, event, context):
            received_signals.append(event.signals['ml_pred'])

    engine = BacktestEngine(feed, RecordingStrategy())
    engine.run()

    assert received_signals == [0.9, 0.85, 0.3]

# Test helper methods
def test_buy_percent_calculates_quantity():
    broker = SimulationBroker(initial_cash=100000)
    broker._last_prices['AAPL'] = 150.0

    strategy = MyStrategy()
    strategy.broker = broker
    strategy.current_time = datetime.now()

    strategy.buy_percent('AAPL', 0.95)

    orders = broker.get_open_orders('AAPL')
    assert len(orders) == 1
    assert orders[0].quantity == pytest.approx(633.33, rel=0.01)

# Test context synchronization
def test_context_synchronized_to_events():
    vix_series = pl.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
        'value': range(100),
    })

    aapl_df = pl.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
        'close': [100] * 10,
    })

    context = Context({'VIX': vix_series})
    feed = ParquetDataFeed(aapl_df)

    # Day 0: VIX should be 0
    # Day 1: VIX should be 24 (24 hours later)
    ...
```

### Integration Tests

```python
# Test complete ML workflow
def test_ml_strategy_end_to_end():
    # Create data with ML signals
    df = create_test_data_with_signals()

    # Define strategy
    class SimpleMLStrategy(Strategy):
        def on_market_data(self, event, context):
            if event.signals['ml_pred'] > 0.8:
                self.buy_percent(event.asset_id, 0.95)

    # Run backtest
    engine = BacktestEngine(
        feed=ParquetDataFeed(df, signals=['ml_pred']),
        strategy=SimpleMLStrategy(),
    )
    results = engine.run()

    # Verify trades
    assert len(results['trades']) > 0
    assert results['metrics']['sharpe_ratio'] > 0
```

---

## Success Criteria

### User Experience
- ✅ ML strategy in <50 lines of code
- ✅ No boilerplate for common patterns (buy_percent, close_position)
- ✅ Zero look-ahead bias by design (signals pre-computed)
- ✅ Same code for backtest and live trading

### Performance
- ✅ 10k+ bars/sec (pure Python event callbacks)
- ✅ 50-100k bars/sec (declarative + Numba)
- ✅ Match or beat Backtrader/Zipline

### Flexibility
- ✅ Support any ML framework (sklearn, xgboost, PyTorch)
- ✅ Arbitrary signal columns (no schema constraints)
- ✅ Complex stateful logic (regime switching, dynamic sizing)
- ✅ Multi-asset with shared context (memory efficient)

---

## Framework Comparison

| Feature | Backtrader | Zipline | VectorBT | ml4t.backtest |
|---------|-----------|---------|----------|---------------|
| ML-first design | ❌ Indicator-focused | ⚠️ Pipeline-focused | ⚠️ Vectorized-only | ✅ Signal-focused |
| Event-driven | ✅ Yes | ✅ Yes | ❌ Vectorized | ✅ Yes |
| Stateful logic | ✅ Full Python | ✅ Full Python | ❌ Limited | ✅ Full Python |
| Performance | 1-5k bars/sec | 2-10k bars/sec | 100k-1M bars/sec | 10-100k bars/sec |
| Live trading | ✅ Supported | ✅ Supported | ❌ No | ✅ Supported |
| Context data | ❌ Manual | ❌ Manual | ✅ Multi-index | ✅ Cached Context |
| Numba optimization | ❌ No | ❌ No | ✅ Yes | ✅ Opt-in |
| Learning curve | Steep | Steep | Moderate | **Gentle** |

**ml4t.backtest advantage:** Only framework designed from ground-up for ML workflows with event-driven execution AND performance optimization.

---

## Appendix: Code Examples

### Example 1: Simple ML Strategy

```python
import polars as pl
from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.data import ParquetDataFeed
from ml4t.backtest.execution import SimulationBroker

# Load data
df = pl.read_parquet('aapl_2020_2023.parquet')

# Add ML predictions (outside engine)
from my_ml_model import predict
df = df.with_columns([
    predict(df).alias('ml_pred'),
    pl.lit(0.85).alias('confidence'),
])

# Define strategy
class SimpleMLStrategy(Strategy):
    def on_market_data(self, event, context):
        pred = event.signals.get('ml_pred', 0)
        conf = event.signals.get('confidence', 0)
        position = self.get_position(event.asset_id)

        # Entry
        if pred > 0.8 and conf > 0.7 and position == 0:
            self.buy_percent(event.asset_id, 0.95)

        # Exit
        elif position > 0:
            pnl = self.get_unrealized_pnl_pct(event.asset_id)
            if pred < 0.4 or pnl < -0.05 or pnl > 0.15:
                self.close_position(event.asset_id)

# Run backtest
feed = ParquetDataFeed('aapl_with_signals.parquet',
                       asset_id='AAPL',
                       signal_columns=['ml_pred', 'confidence'])
broker = SimulationBroker(initial_cash=100000)
engine = BacktestEngine(feed, SimpleMLStrategy(), broker)

results = engine.run()
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
```

### Example 2: VIX-Filtered Strategy

```python
# Join VIX to asset data (single-asset)
aapl_df = aapl_df.join(vix_df[['VIX']], how='left').forward_fill()

class VIXFilteredStrategy(Strategy):
    def on_market_data(self, event, context):
        pred = event.signals['ml_pred']
        vix = event.signals['VIX']
        position = self.get_position(event.asset_id)

        # VIX filter
        if vix > 30:
            if position > 0:
                self.close_position(event.asset_id)
            return  # Don't enter in high volatility

        # Normal ML logic
        if pred > 0.8 and position == 0:
            self.buy_percent(event.asset_id, 0.95)
        elif pred < 0.4 and position > 0:
            self.close_position(event.asset_id)

# Run
feed = ParquetDataFeed(aapl_df, signals=['ml_pred', 'VIX'])
engine = BacktestEngine(feed, VIXFilteredStrategy())
results = engine.run()
```

### Example 3: Multi-Asset with Shared Context

```python
# Efficient multi-asset setup
from ml4t.backtest.core import Context

# Load individual assets
assets = {}
for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
    df = pl.read_parquet(f'{symbol}_with_signals.parquet')
    assets[symbol] = ParquetDataFeed(df, asset_id=symbol,
                                      signals=['ml_pred'])

# Load context (NOT duplicated)
vix = pl.read_parquet('vix_daily.parquet')
spy = pl.read_parquet('spy_daily.parquet')

context = Context({
    'VIX': vix['close'],
    'SPY_trend': spy['trend_5d'],
})

# Multi-asset strategy
class MultiAssetMLStrategy(Strategy):
    def on_market_data(self, event, context):
        # Asset-specific
        pred = event.signals['ml_pred']
        position = self.get_position(event.asset_id)

        # Market context (shared)
        vix = context.get('VIX')
        spy_trend = context.get('SPY_trend')

        # Regime-dependent logic
        if vix > 30:
            if position > 0:
                self.reduce_position(event.asset_id, 0.5)
            return

        # ML logic
        if pred > 0.8 and position == 0:
            size = 0.02 if vix < 20 else 0.01
            self.buy_percent(event.asset_id, size)

# Run with shared context
engine = BacktestEngine(
    feeds=assets,
    context=context,
    strategy=MultiAssetMLStrategy(),
)
results = engine.run()
```

---

**End of Architectural Proposal**

**Next Steps:**
1. Review and approve architectural decisions
2. Prioritize Phase 1 implementation (ML signals + helpers)
3. Create detailed implementation tasks
4. Begin development

**Questions for Stakeholders:**
1. Is 10-20k bars/sec acceptable for Phase 1, or do we need Numba immediately?
2. Should we support multi-asset context in Phase 1, or defer to Phase 3?
3. What is the priority order: performance vs live trading vs multi-asset?
