# ML4T.Backtest Architectural Review Request

**Date:** 2025-11-15
**Reviewer Instructions:** This document requests architectural review and recommendations for a Python backtesting
library focused on machine learning trading strategies. Please provide feedback on API design, flexibility, user
experience, and performance trade-offs.

---

## Executive Summary

We are designing **ml4t.backtest**, a backtesting library for the "Machine Learning for Trading" book series. The
library is production-ready (498/498 tests passing, 81% coverage) but needs architectural decisions before adding ML
signal integration.

**We request your review on 6 critical design areas**, focusing on:

1. **Flexibility** - Supporting diverse ML workflows
2. **User Experience** - Simple, transparent, reproducible
3. **Performance** - Fast enough for practical use, not maximum speed
4. **Live Trading Portability** - Same code for backtest and production

**Key Question:** Which architectural options best balance these goals for an ML-first backtesting library?

---

## Project Context

### What is ml4t.backtest?

An **event-driven backtesting library** designed specifically for machine learning trading strategies, part of a suite
of ML4T libraries (data, features, evaluation, backtest).

**Current State:**

- **Production-ready**: 498/498 tests passing, 81% code coverage
- **Event-driven architecture**: Clock-driven event loop with broker/portfolio abstractions
- **Tested**: Single-asset strategies validated against VectorBT/Backtrader
- **Missing**: ML signal integration, multi-asset context, performance optimization

**Repository:** `/home/stefan/ml4t/software/backtest/`

### Design Philosophy

**Primary Goals (in priority order):**

1. **ML-First Design**
    - Strategies driven by ML predictions (not just technical indicators)
    - Multiple ML models per asset (entry, exit, different horizons)
    - Signal values used directly (confidence-based position sizing, etc.)
    - Hybrid approaches (ML + quantitative indicators combined)

2. **Transparency and Reproducibility**
    - Clear how results are computed
    - No hidden optimizations that obscure logic
    - Easy to debug and understand
    - Point-in-time correctness (no look-ahead bias)

3. **Stateful Strategy Support**
    - Regime switching (react to market conditions)
    - Dynamic position sizing (based on portfolio state)
    - Complex entry/exit logic (not just vectorized rules)
    - State machines and conditional logic

4. **Live Trading Portability**
    - Same strategy code for backtest and live
    - Swap data sources (historical → WebSocket/API)
    - Swap brokers (simulation → Alpaca/IB)
    - Minimal code changes for production deployment

5. **Performance (Pragmatic)**
    - Fast enough for typical use cases (daily rebalancing, 10-100 assets)
    - NOT optimized for million-parameter sweeps (like VectorBT)
    - Willing to trade speed for flexibility/transparency
    - Target: Multi-year backtest on 100 assets in minutes, not hours

**Explicit Non-Goals:**

- ❌ Competing with VectorBT on parameter sweep speed
- ❌ Incorporating MLOps (model training, deployment)
- ❌ Built-in ML models (users bring their own)

---

## Use Cases

### Typical ML Trading Workflow

```python
# 1. User trains ML model (outside library)
model = train_xgboost_model(features)

# 2. Generate predictions (outside library)
df['ml_pred_5d'] = model.predict(features)
df['confidence'] = model.predict_proba(features)[:, 1]

# 3. Backtest with predictions
strategy = MLStrategy(
    entry_threshold=0.8,
    confidence_threshold=0.7,
)
engine = BacktestEngine(data=df, strategy=strategy)
results = engine.run()

# 4. Deploy to live trading (SAME STRATEGY CODE)
engine = BacktestEngine(
    data=AlpacaStreamFeed(api_key=...),
    broker=AlpacaBroker(api_key=...),
    strategy=strategy,  # IDENTICAL
)
engine.run_live()
```

### Expected ML Signal Patterns

**Base case:**

- 1 ML signal per asset (e.g., `ml_pred_5d`)

**Common cases:**

- Multiple models per asset (entry model, exit model, different horizons)
- Confidence/uncertainty scores (for position sizing)
- Hybrid signals (ML + technical indicators combined)
- Cross-asset context (VIX, market regime, sector trends)

**Example signal structure:**

```python
# Per-asset signals
signals = {
    'ml_pred_5d': 0.85,  # 5-day return prediction
    'ml_pred_20d': 0.72,  # 20-day return prediction
    'confidence': 0.92,  # Model confidence
    'exit_signal': 0.15,  # Separate exit model
    'momentum_80pct': True,  # Quantitative filter
}

# Cross-asset context
context = {
    'VIX': 18.5,  # Market volatility
    'SPY_trend': 0.02,  # Market trend
    'sector_momentum': 0.15,  # Sector strength
}
```

### Critical Requirements

1. **Signal values matter** - Not just binary entry/exit, but magnitudes
    - Position size scales with confidence
    - Risk adjusts based on uncertainty
    - Different actions for different signal strengths

2. **Multiple signals per asset** - Not just one prediction
    - Entry model vs exit model
    - Different horizons (5-day, 20-day)
    - Ensembles (multiple models combined)

3. **Hybrid logic** - ML + traditional quant
    - "Only trade if momentum > 80th percentile AND ml_pred > 0.8"
    - "Use ML for entry, technical indicators for exit"
    - "Filter ML signals by VIX regime"

4. **Live trading signals** - Prepared for real-time delivery
    - Signals may arrive via API calls
    - Signals may stream via WebSocket
    - Model inference may be external service
    - Architecture should not prevent async signal updates

---

## Current Architecture

### Key Components

```
┌─────────────────────────────────────────────────────────┐
│                   Current Architecture                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  DataFeed → Clock → Strategy → Broker → Portfolio       │
│                                                          │
│  [Event Loop]                                           │
│  while event := clock.get_next_event():                 │
│      strategy.on_market_event(event)                    │
│      broker.process_orders()                            │
│      portfolio.update_state()                           │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Existing Abstractions (Already Implemented)

**Broker Interface** (`execution/broker.py`):

```python
class Broker(ABC):
    @abstractmethod
    def submit_order(self, order: Order) -> OrderId:
        """Submit order for execution."""

    @abstractmethod
    def get_position(self, asset_id: AssetId) -> Quantity:
        """Get current position."""

    @abstractmethod
    def get_cash(self) -> float:
        """Get available cash."""

# Implementations:
# - SimulationBroker (for backtest) ✅
# - AlpacaBroker (for live) ❌ Not implemented yet
# - IBBroker (for live) ❌ Not implemented yet
```

**DataFeed Interface** (`data/feed.py`):

```python
class DataFeed(ABC):
    @abstractmethod
    def get_next_event(self) -> Event | None:
        """Get next market event."""

    @abstractmethod
    def is_exhausted(self) -> bool:
        """Check if feed is done."""

# Implementations:
# - ParquetDataFeed (historical data) ✅
# - CSVDataFeed (historical data) ✅
# - WebSocketFeed (live streaming) ❌ Not implemented yet
```

**Strategy Base Class** (`strategy/base.py`):

```python
class Strategy(ABC):
    @abstractmethod
    def on_market_event(self, event: MarketEvent):
        """React to new market data."""

    def on_fill_event(self, event: FillEvent):
        """React to order fills (optional)."""
```

**Event System** (`core/event.py`):

```python
@dataclass
class MarketEvent(Event):
    timestamp: datetime
    asset_id: AssetId
    open: float
    high: float
    low: float
    close: float
    volume: float
    # signals: ??? <- MISSING


class SignalEvent(Event):  # EXISTS but not integrated
    signal_value: float
    model_id: str
    confidence: float | None
```

### What Works Well

✅ **Event-driven architecture** - Clean separation of concerns
✅ **Clock synchronization** - Multi-feed event ordering
✅ **Broker abstraction** - Ready for live trading
✅ **Portfolio tracking** - Single source of truth (no dual state)
✅ **Test coverage** - 81% coverage, comprehensive validation
✅ **Point-in-time correctness** - No look-ahead bias

### What's Missing

❌ **ML signal integration** - No schema for predictions
❌ **Helper methods** - Boilerplate for common operations (buy_percent, etc.)
❌ **Cross-asset context** - No VIX/SPY/regime data handling
❌ **Multi-asset scaling** - Only tested with single assets
❌ **Performance profiling** - Unknown performance at scale
❌ **Live trading implementations** - Only simulation broker exists

---

## Architectural Questions for Review

We have identified **6 critical design areas** that need architectural decisions. For each area, we present 2-3 options
with trade-offs and request your recommendation.

---

## Question 1: Strategy Formulation API

**How should users write trading rules that access ML signals, portfolio state, and context data?**

### Option 1A: Event Callbacks with Helper Methods

**API:**

```python
class MLStrategy(Strategy):
    def on_market_data(self, event: MarketEvent):
        # Access signals
        ml_pred = event.signals.get('ml_pred', 0)
        confidence = event.signals.get('confidence', 0)

        # Access state (via helpers)
        position = self.get_position(event.asset_id)
        pnl_pct = self.get_unrealized_pnl_pct(event.asset_id)

        # Logic
        if ml_pred > 0.8 and confidence > 0.7 and position == 0:
            self.buy_percent(event.asset_id, 0.95)
        elif position > 0 and (pnl_pct < -0.05 or ml_pred < 0.4):
            self.close_position(event.asset_id)
```

**Pros:**

- Maximum flexibility (arbitrary Python logic)
- Familiar to Backtrader/Zipline users
- Stateful logic fully supported
- Easy to debug (step through code)

**Cons:**

- Slower than declarative/vectorized
- Users can introduce bugs
- Cannot auto-optimize with Numba

**Estimated Performance:** Unknown (needs profiling)

---

### Option 1B: Declarative Signal Rules

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

**Pros:**

- Potentially faster (can compile with Numba)
- Impossible to introduce look-ahead bias
- Clean, testable, composable
- Easy to parallelize

**Cons:**

- Limited to simple threshold logic
- No complex state machines
- Cannot express all strategies
- New abstraction to learn

**Estimated Performance:** Unknown (Numba compilation potential)

---

### Option 1C: Hybrid (Event Callbacks + Compiled Hot Path)

**API:**

```python
class HybridStrategy(Strategy):
    @staticmethod
    @njit(cache=True)
    def evaluate_signal(ml_pred, confidence, momentum, position):
        """Numba-compiled decision logic."""
        if position > 0:
            return 0
        if ml_pred > 0.8 and confidence > 0.7 and momentum > 0:
            return 1
        return 0

    def on_market_data(self, event: MarketEvent):
        # Gather state (Python)
        position = self.get_position(event.asset_id)

        # Evaluate (compiled)
        action = self.evaluate_signal(
            event.signals['ml_pred'],
            event.signals['confidence'],
            event.signals['momentum'],
            position
        )

        # Execute (Python)
        if action == 1:
            self.buy_percent(event.asset_id, 0.95)
```

**Pros:**

- Best of both worlds
- Explicit about hot/cold paths
- Full flexibility where needed
- Performance where it matters

**Cons:**

- More complex than pure options
- Requires Numba knowledge
- Not all users will use optimization

**Estimated Performance:** Unknown (partial compilation)

---

### Question 1: Your Recommendation?

**Which option best balances flexibility, usability, and performance for ML strategies?**

Consider:

- Users range from beginners to experts
- Strategies vary from simple thresholds to complex state machines
- Live trading requires debugging capability
- Performance requirements are unknown (needs profiling)

**Should we:**

- Start with 1A (simple) and add 1B later?
- Implement 1A and 1B simultaneously?
- Only do 1C (hybrid) for advanced users?
- Different recommendation entirely?

---

## Question 2: ML Signal Data Structure

**How should ML predictions flow through the system?**

### Option 2A: Signals as MarketEvent Attributes (Embedded)

**Data Structure:**

```python
@dataclass
class MarketEvent(Event):
    timestamp: datetime
    asset_id: AssetId
    open: float
    high: float
    low: float
    close: float
    volume: float
    signals: dict[str, float] = field(default_factory=dict)  # NEW
```

**Usage:**

```python
# User workflow
df = pl.read_parquet('aapl_ohlcv.parquet')
df = df.with_columns([
    model.predict(df).alias('ml_pred_5d'),
    model.predict_proba(df)[:, 1].alias('confidence'),
])

feed = ParquetDataFeed(
    df,
    signal_columns=['ml_pred_5d', 'confidence']
)


# Strategy access
def on_market_data(self, event):
    pred = event.signals['ml_pred_5d']
    conf = event.signals['confidence']
```

**Pros:**

- Simplest (no separate event stream)
- Zero overhead (just dict access)
- Familiar (like VectorBT DataFrame columns)
- Signals naturally aligned with OHLCV

**Cons:**

- Signals must be pre-computed
- Mixed concerns (market data + predictions)
- Cannot model signal latency separately

**Memory:** Minimal overhead

---

### Option 2B: Separate SignalEvent Stream

**Data Structure:**

```python
class SignalEvent(Event):
    timestamp: datetime
    asset_id: AssetId
    signal_value: float
    model_id: str
    confidence: float | None


# Clock synchronizes two streams
clock.add_data_feed(market_feed)
clock.add_signal_source(ml_feed)


# Strategy caches signals
class MLStrategy(Strategy):
    def on_signal_event(self, event: SignalEvent):
        self._signals[event.asset_id][event.model_id] = event.signal_value

    def on_market_event(self, event: MarketEvent):
        pred = self._signals[event.asset_id]['ml_5d']
```

**Pros:**

- Clean separation (data vs predictions)
- Can model signal latency (event time vs arrival time)
- Closer to live trading (signals may arrive async)
- Supports multiple model updates

**Cons:**

- More complex (two event streams to manage)
- Users must cache signals manually
- Higher coupling with Clock
- Potentially slower (more events to process)

**Memory:** Additional event stream overhead

---

### Question 2: Your Recommendation?

**Which approach better serves ML workflows?**

Consider:

- Most ML models generate predictions in batch (pre-computed)
- Live trading may receive signals asynchronously (WebSocket, API)
- Multiple models per asset are common
- Users may want different update frequencies (daily predictions, hourly context)

**Should we:**

- Start with 2A (embedded) for simplicity?
- Implement 2B for live trading compatibility?
- Support both patterns?
- Different approach entirely?

---

## Question 3: Context and Cross-Asset Data

**How should market-wide context (VIX, SPY, regime) be handled alongside per-asset signals?**

### Option 3A: Embed Context in Signals (Single-Asset)

**API:**

```python
# User joins context to asset data
aapl_df = aapl_df.join(vix_df, how='left').forward_fill()

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

- Simplest (no engine changes)
- User controls joins (forward-fill, aggregation)
- Works with existing infrastructure

**Cons:**

- **Memory inefficient for multi-asset**
    - VIX duplicated 500 times (one per asset)
    - 500 assets × 10 context variables × 252 days = huge waste
- No distinction between asset vs context data

**Memory:** ~10 MB for 500 assets with 10 context variables

---

### Option 3B: Separate Context Object (Multi-Asset)

**API:**

```python
# Single-asset: embed context (simple)
aapl_df = aapl_df.join(vix_df).forward_fill()
feed = ParquetDataFeed(aapl_df, signals=['ml_pred', 'VIX'])

# Multi-asset: separate context (efficient)
engine = BacktestEngine(
    feeds={
        'AAPL': ParquetDataFeed(aapl_df, signals=['ml_pred']),
        'MSFT': ParquetDataFeed(msft_df, signals=['ml_pred']),
        # ... 500 assets
    },
    context=Context({
        'VIX': vix_series,
        'SPY_trend': spy_series,
    }),
)


# Unified strategy API
def on_market_data(self, event, context):
    pred = event.signals['ml_pred']  # Asset-specific
    vix = context.get('VIX')  # Market-wide (shared)
```

**Implementation:**

```python
class Context:
    """Market-wide context shared across assets."""

    def __init__(self, sources: dict[str, pl.Series]):
        self._sources = sources
        self._cache: dict[str, float] = {}
        self._cache_timestamp: datetime | None = None

    def at(self, timestamp: datetime) -> dict[str, float]:
        """Get context at timestamp (cached)."""
        if timestamp == self._cache_timestamp:
            return self._cache  # Hit cache!

        # Update cache (happens once per timestamp, not per asset)
        self._cache = {
            key: series.at(timestamp)
            for key, series in self._sources.items()
        }
        self._cache_timestamp = timestamp
        return self._cache
```

**Pros:**

- Memory efficient (VIX stored once, not 500 times)
- Performance optimized (cached lookups)
- Clear separation (asset signals vs market context)
- Same API for single/multi-asset

**Cons:**

- Two initialization patterns (slight complexity)
- New abstraction to learn

**Memory:** ~200 KB for 500 assets (50x reduction)

---

### Question 3: Your Recommendation?

**How should context data be handled?**

Consider:

- Most strategies trade 10-100 assets, not 1000
- Context variables are low-frequency (daily VIX, not tick data)
- Live trading context arrives separately (WebSocket feeds)
- Memory efficiency matters for large universes

**Should we:**

- Start with 3A (embedded) and add 3B later?
- Implement 3B (hybrid) from the start?
- Only support 3A (embedded)?
- Different approach?

---

## Question 4: Performance Architecture

**Where should we optimize, and how much performance do we need?**

### Current Reality

**We do not have performance data.** The library has never been profiled with realistic workloads.

**Unknown:**

- How fast is the current event loop?
- What are the bottlenecks?
- How does it scale with asset count?
- How does it scale with data frequency?

**Hypothetical scenarios to consider:**

| Scenario                   | Events      | Target Time   |
|----------------------------|-------------|---------------|
| Daily, 1 asset, 5 years    | ~1,250      | < 1 second    |
| Daily, 100 assets, 5 years | ~125,000    | < 1 minute    |
| Minute, 1 asset, 1 year    | ~100,000    | < 10 seconds  |
| Minute, 100 assets, 1 year | ~10,000,000 | < 10 minutes? |

**Question:** Are these reasonable targets?

---

### Option 4A: Pure Python Event Loop (Current)

**Architecture:**

```python
while event := clock.get_next_event():
    strategy.on_market_event(event)
    broker.process_orders()
    portfolio.update_state()
```

**Pros:**

- Maximum flexibility
- Easy to debug
- No compilation overhead
- Works with any Python code

**Cons:**

- Unknown performance ceiling
- GIL prevents parallelism
- May be too slow for HFT scenarios

**Performance:** Unknown (needs profiling)

---

### Option 4B: Numba-Compiled Hot Paths

**Architecture:**

```python
@njit(cache=True)
def evaluate_strategy(signals, positions, thresholds):
    """Numba-compiled strategy evaluation."""
    actions = np.zeros(len(signals))
    for i in range(len(signals)):
        if signals[i] > thresholds[0] and positions[i] == 0:
            actions[i] = 1  # Buy
    return actions


# Hybrid execution
for event in clock:
    action = evaluate_strategy(...)  # Compiled
    if action == 1:
        broker.submit_order(...)  # Python
```

**Pros:**

- Potentially much faster
- Can compile strategy logic
- No GIL (parallelism possible)

**Cons:**

- Cannot compile all code (dicts, classes)
- Compilation overhead
- Requires NumPy-compatible data
- Limits user flexibility

**Performance:** Unknown (needs benchmarking)

---

### Option 4C: Delay Optimization

**Approach:**

1. Profile current architecture
2. Identify actual bottlenecks
3. Optimize only if needed
4. Start with simplest implementation

**Pros:**

- Don't optimize prematurely
- Focus on correctness first
- Add speed when proven necessary

**Cons:**

- May discover performance issues late
- Harder to fix after API is public

---

### Question 4: Your Recommendation?

**What is the right performance strategy?**

Consider:

- We don't have performance data yet
- Most ML strategies rebalance daily/weekly (not HFT)
- Live trading needs responsiveness (seconds), not throughput
- Transparency/debuggability matters more than raw speed

**Should we:**

- Profile first, optimize later (4C)?
- Build Numba support from start (4B)?
- Accept Python speed limits (4A)?
- Different approach?

**What are realistic performance targets** for our use cases?

---

## Question 5: Live Trading Portability

**How do we design for backtest → live trading with minimal code changes?**

### Current State

**Already implemented:**

- ✅ Broker interface (abstract base class)
- ✅ DataFeed interface (abstract base class)
- ✅ Event-driven architecture (not vectorized)
- ✅ Portfolio state management

**Missing implementations:**

- ❌ LiveDataFeed (WebSocket/API streaming)
- ❌ LiveBroker (Alpaca, Interactive Brokers)
- ❌ Signal streaming (async ML predictions)

---

### Option 5A: Pluggable Broker/Feed (Current Design)

**Architecture:**

```python
# Backtesting
engine = BacktestEngine(
    feed=ParquetDataFeed('aapl_historical.parquet'),
    broker=SimulationBroker(initial_cash=100000),
    strategy=MyStrategy(),
)

# Live trading (SAME STRATEGY CODE)
engine = BacktestEngine(
    feed=AlpacaStreamFeed(api_key=..., symbols=['AAPL']),
    broker=AlpacaBroker(api_key=...),
    strategy=MyStrategy(),  # IDENTICAL
)
```

**What changes between backtest and live:**

- DataFeed implementation (historical → streaming)
- Broker implementation (simulation → API)
- **Strategy code: NO CHANGE**

**Pros:**

- Strategy is 100% portable
- Easy to test (swap implementations)
- Clear contracts (ABC enforcement)

**Cons:**

- None (this is the standard pattern)

**Gap:** Need to implement live versions

---

### Question 5: Your Recommendation?

**Is the current abstraction sufficient for live trading?**

Consider:

- Live signals may arrive asynchronously (WebSocket)
- Live trading may need signal buffering
- Model inference may be external service (API calls)
- Error handling/reconnection logic needed

**Should we:**

- Keep current design (implement LiveBroker/LiveFeed later)?
- Add signal streaming abstractions now?
- Add async event loop support?
- Different approach?

**Does the architecture prevent live trading patterns?**

---

## Question 6: Integration and Trade-offs

**How do the 5 design areas interact, and what trade-offs should we make?**

### Synthesis

The decisions are interconnected:

1. **Strategy API** determines what signals look like
2. **Signal structure** affects live trading integration
3. **Context handling** impacts multi-asset performance
4. **Performance** constrains API flexibility
5. **Live trading** requires async-compatible design

### Example Integration

```
User Workflow:
1. Train ML model (outside library)
2. Generate predictions (Polars DataFrame)
3. Define strategy (event callbacks or declarative)
4. Run backtest (SimulationBroker)
5. Deploy live (AlpacaBroker, same strategy)

Data Flow:
DataFeed(signals) → Clock → Strategy(helpers) → Broker → Portfolio
                ↓
            Context(VIX, SPY)
```

---

### Question 6: Your Recommendation?

**What is the right balance between:**

1. **Simplicity vs Power**
    - Simple API (Option 1A) vs Advanced features (Option 1C)
    - Embedded signals (2A) vs Separate streams (2B)

2. **Performance vs Flexibility**
    - Python flexibility vs Numba speed
    - Event-driven vs Vectorized

3. **Present vs Future**
    - Single-asset first vs Multi-asset from start
    - Backtest only vs Live trading ready

**Specific trade-offs to consider:**

- **If we embed signals in MarketEvent (2A)**, does that prevent live signal streaming?
- **If we optimize with Numba (4B)**, does that limit strategy expressiveness?
- **If we support multi-asset context (3B)**, is the added complexity worth it for 10-asset portfolios?
- **If we add helper methods (1A)**, does that prevent future optimizations?

**What would you prioritize if implementing this library?**

---

## Request for Specific Feedback

### 1. Architectural Recommendations

For each of the 6 questions:

- Which option do you recommend?
- Why is it the best balance of goals?
- What are the risks/limitations?
- What alternatives should we consider?

### 2. Implementation Priorities

Given limited development time:

- What should be implemented first?
- What can be deferred?
- What should be avoided entirely?

Proposed phases:

- **Phase 1 (Week 1):** Basic ML signal support
- **Phase 2 (Week 2):** Context data handling
- **Phase 3 (Weeks 3-4):** Multi-asset optimization (if needed)
- **Phase 4 (Future):** Performance optimization (if needed)

**Does this sequencing make sense?**

### 3. Performance Expectations

Without profiling data:

- What are realistic performance targets?
- When should we profile?
- What workloads should we test?
- Is pure Python event loop acceptable?

### 4. API Design Critique

Review the proposed APIs:

- Are they intuitive for ML practitioners?
- Do they follow Python best practices?
- Are there hidden footguns?
- What would you change?

### 5. Live Trading Concerns

From a live trading perspective:

- Does the architecture support async signal delivery?
- Are there race conditions to worry about?
- How should we handle signal buffering?
- Should we support async/await patterns?

### 6. Alternative Approaches

Are there better architectures we haven't considered?

- Different event systems?
- Different signal structures?
- Different abstraction layers?
- Lessons from other frameworks?

---

## Additional Context

### Comparison with Other Frameworks

| Framework         | Approach              | Pros                             | Cons                               |
|-------------------|-----------------------|----------------------------------|------------------------------------|
| **VectorBT Pro**  | Fully vectorized      | Very fast, parallel optimization | No stateful logic, no live trading |
| **Backtrader**    | Event-driven Python   | Flexible, live trading           | Slow, complex API, dated           |
| **Zipline**       | Event-driven Pipeline | Clean API, Quantopian legacy     | Pipeline overhead, discontinued    |
| **ml4t.backtest** | Event-driven ML-first | ???                              | ???                                |

**Question:** What niche should ml4t.backtest fill?

### Technology Constraints

**Must use:**

- Python 3.9+
- Polars (DataFrames)
- Existing event-driven architecture

**Can use:**

- Numba (JIT compilation)
- NumPy (arrays)
- Asyncio (async patterns)

**Cannot change:**

- Event-driven architecture (not vectorized)
- 498 existing tests must keep passing
- Broker/Portfolio abstractions (production-ready)

---

## How to Provide Feedback

**Format:**
Please structure your response as:

```markdown
## Question 1: Strategy API

**Recommendation:** Option [X]
**Reasoning:** [Why this balances goals]
**Risks:** [What could go wrong]
**Alternatives:** [Other approaches to consider]

## Question 2: ML Signals

...

## Overall Assessment

- Priority 1: [What to implement first]
- Priority 2: [What to implement second]
- Avoid: [What not to do]
- Unknown: [What needs more investigation]

## Performance Expectations

[Realistic targets and when to optimize]

## API Critique

[Specific feedback on proposed APIs]
```

**Focus areas:**

- Practical advice over theory
- Specific code examples appreciated
- Flag hidden complexity
- Suggest simpler alternatives
- Point out non-obvious trade-offs

---

## Summary of Key Questions

1. **Strategy API:** Event callbacks (1A), Declarative (1B), or Hybrid (1C)?
2. **Signal Structure:** Embedded in MarketEvent (2A) or Separate stream (2B)?
3. **Context Data:** Embedded (3A) or Separate Context object (3B)?
4. **Performance:** Pure Python (4A), Numba (4B), or Delay (4C)?
5. **Live Trading:** Is current abstraction sufficient?
6. **Trade-offs:** What to prioritize given interconnected decisions?

**The meta-question:** What is the right balance between simplicity, flexibility, performance, and live trading
portability for an ML-first backtesting library?

---

**Thank you for your review!** Your expert perspective will help us make better architectural decisions before
committing to an API.

**Contact:** [Your contact info]
**Repository:** `/home/stefan/ml4t/software/backtest/`
**Full Proposal:** `.claude/memory/ml_architecture_proposal.md` (77 pages)
