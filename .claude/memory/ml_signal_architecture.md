# ML Signal Integration & Event-Driven Strategy Architecture

**Created:** 2025-11-14
**Status:** Architectural Analysis Complete - Ready for Implementation

## Executive Summary

This document defines how ml4t.backtest will support ML-first trading strategies with event-driven execution, balancing Python flexibility with performance optimization, and creating a clean path to live trading.

**Key Decision:** Offer three API levels to serve different user needs:
1. **Declarative Rules** (fast, simple) - 80% of use cases
2. **Event Callbacks** (flexible, powerful) - 20% of use cases
3. **Hybrid Numba** (optimal performance) - power users

## 1. The Core Problem

Users need to:
- Import ML predictions alongside OHLCV data
- Write stateful trading logic (react to positions, PnL, etc.)
- Achieve good performance (10k+ bars/second)
- Transition to live trading without code changes

Current backtesting frameworks fall short:
- **Backtrader/Zipline:** Flexible but slow (pure Python event loops)
- **VectorBT:** Fast but inflexible (vectorized, no stateful logic)
- **ml4t.backtest (current):** Event-driven foundation but missing ML workflow

## 2. Data Model: Signals as First-Class Citizens

### Extend MarketData with Signal Dictionary

```python
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
```

**Rationale:**
- Signals flow naturally with market events
- Arbitrary number of signals (ml_pred, confidence, regime, etc.)
- No special "signal" type - just float64 data
- Clean separation: users compute signals, engine executes

### DataFeed Integration

```python
class HistoricalDataFeed:
    def __init__(self, data: pd.DataFrame, signal_columns: list[str] = None):
        """
        Args:
            data: DataFrame with OHLCV + any extra columns
            signal_columns: List of column names to treat as signals
        """
        self.data = data
        self.signal_columns = signal_columns or []

    def _create_market_event(self, row) -> MarketEvent:
        # Extract signals from DataFrame row
        signals = {col: row[col] for col in self.signal_columns}

        return MarketEvent(
            timestamp=row.name,
            asset_id=self.asset_id,
            data=MarketData(
                open=row['open'], close=row['close'],
                high=row['high'], low=row['low'],
                volume=row['volume'],
                signals=signals  # NEW
            )
        )
```

**Implementation Effort:** 1 day

## 3. Strategy API: Three Levels of Abstraction

### Level 1: Declarative Rules (Recommended for Most Users)

**Target:** 80% of ML strategies (simple threshold logic)

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
        Rule.stop_loss(0.05),      # 5% stop
        Rule.take_profit(0.15),    # 15% target
    ],
    position_sizer=PercentSizer(0.95)
)
```

**Benefits:**
- Declarative = fully Numba-compilable
- 10-100x faster than imperative Python
- Impossible to introduce look-ahead bias
- Clean, readable, testable

**Limitations:**
- Cannot express complex state machines
- No dynamic position sizing based on volatility
- No multi-asset dependencies

**Implementation Effort:** 1 week (Phase 3)

### Level 2: Event Callbacks (Maximum Flexibility)

**Target:** 20% of strategies (complex stateful logic)

```python
class MLStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.regime = None  # Stateful tracking

    def on_market_data(self, event: MarketEvent):
        # Access signals directly
        ml_pred = event.data.signals['ml_pred']
        confidence = event.data.signals['confidence']

        # Access portfolio state via helper methods
        position = self.get_position(event.asset_id)
        pnl_pct = self.get_unrealized_pnl_pct(event.asset_id)

        # Complex stateful logic
        if self._detect_regime_change(event):
            self.regime = 'high_vol'
            if position > 0:
                self.reduce_position(event.asset_id, 0.5)

        # Entry logic
        if ml_pred > 0.8 and confidence > 0.7 and position == 0:
            size = self._dynamic_size(confidence)
            self.buy_percent(event.asset_id, size)

        # Exit logic
        elif position > 0 and (pnl_pct < -0.05 or ml_pred < 0.4):
            self.close_position(event.asset_id)

    def _detect_regime_change(self, event) -> bool:
        # Custom logic using multiple signals
        volatility = event.data.signals.get('volatility', 0)
        return volatility > self.vol_threshold

    def _dynamic_size(self, confidence: float) -> float:
        # Position size based on model confidence
        return min(0.95, confidence * 1.2)
```

**Benefits:**
- Full Python flexibility
- Stateful logic (regime tracking, complex stops)
- Multi-asset dependencies
- Custom analytics

**Limitations:**
- Slower than declarative (but still fast enough)
- Easier to introduce bugs

**Implementation Effort:** 2 days (Phase 2 - helper methods)

### Level 3: Hybrid Numba (Best of Both Worlds)

**Target:** Power users wanting 90% performance with full flexibility

```python
class HybridStrategy(Strategy):
    @staticmethod
    @njit(cache=True)
    def evaluate_entry(ml_pred, confidence, momentum, position):
        """Numba-compiled hot path"""
        if position > 0:
            return 0  # Already in position
        if ml_pred > 0.8 and confidence > 0.7 and momentum > 0:
            return 1  # Enter long
        return 0  # Hold

    @staticmethod
    @njit(cache=True)
    def evaluate_exit(ml_pred, position, pnl_pct):
        """Numba-compiled exit logic"""
        if position == 0:
            return 0
        if pnl_pct < -0.05 or pnl_pct > 0.15 or ml_pred < 0.4:
            return -1  # Exit
        return 0  # Hold

    def on_market_data(self, event: MarketEvent):
        # Gather state (Python)
        position = self.get_position(event.asset_id)
        pnl_pct = self.get_unrealized_pnl_pct(event.asset_id)

        # Evaluate using compiled functions
        entry_signal = self.evaluate_entry(
            event.data.signals['ml_pred'],
            event.data.signals['confidence'],
            event.data.signals['momentum'],
            position
        )

        exit_signal = self.evaluate_exit(
            event.data.signals['ml_pred'],
            position,
            pnl_pct
        )

        # Execute (Python)
        if entry_signal == 1:
            self.buy_percent(event.asset_id, 0.95)
        elif exit_signal == -1:
            self.close_position(event.asset_id)
```

**Benefits:**
- 90% of declarative performance
- Full flexibility for edge cases
- Explicit about what's compiled vs interpreted

**Implementation Effort:** 3 days (Phase 3)

## 4. Required Strategy Helper Methods

Add to `Strategy` base class:

```python
class Strategy:
    # Position queries
    def get_position(self, asset_id: AssetId) -> float:
        """Get current position size (quantity)"""
        return self.broker.get_position(asset_id).quantity

    def get_position_value(self, asset_id: AssetId) -> float:
        """Get current position market value"""
        position = self.broker.get_position(asset_id)
        return position.quantity * position.current_price

    def get_unrealized_pnl(self, asset_id: AssetId) -> float:
        """Get unrealized PnL in dollars"""
        position = self.broker.get_position(asset_id)
        return (position.current_price - position.avg_price) * position.quantity

    def get_unrealized_pnl_pct(self, asset_id: AssetId) -> float:
        """Get unrealized PnL as percentage"""
        position = self.broker.get_position(asset_id)
        if position.quantity == 0:
            return 0.0
        return (position.current_price / position.avg_price) - 1.0

    # Order helpers
    def buy_percent(self, asset_id: AssetId, pct: float, **kwargs):
        """Buy using percentage of available cash"""
        cash = self.broker.get_cash()
        value = cash * pct
        # Submit market order for this value

    def sell_percent(self, asset_id: AssetId, pct: float, **kwargs):
        """Sell percentage of current position"""
        position = self.get_position(asset_id)
        quantity = position * pct
        # Submit market sell order

    def close_position(self, asset_id: AssetId, **kwargs):
        """Close entire position"""
        self.sell_percent(asset_id, 1.0, **kwargs)

    def reduce_position(self, asset_id: AssetId, pct: float, **kwargs):
        """Reduce position by percentage"""
        self.sell_percent(asset_id, pct, **kwargs)
```

**Implementation Effort:** 1 day

## 5. User Journey: Typical ML Strategy Workflow

```python
import pandas as pd
from ml4t.backtest import Engine, Strategy
from ml4t.backtest.data import HistoricalDataFeed
from ml4t.backtest.execution import SimulationBroker

# 1. Load historical OHLCV data
df = pd.read_parquet('aapl_daily_2020_2023.parquet')

# 2. Compute features (outside engine)
from my_features import compute_technicals, compute_ml_signals

df = compute_technicals(df)  # RSI, MACD, etc. (could use VectorBT)
df['ml_pred'] = compute_ml_signals(df)  # Your ML model
df['confidence'] = compute_confidence(df)

# 3. Define strategy
class MLStrategy(Strategy):
    def on_market_data(self, event):
        pred = event.data.signals['ml_pred']
        conf = event.data.signals['confidence']
        position = self.get_position(event.asset_id)

        if pred > 0.8 and conf > 0.7 and position == 0:
            self.buy_percent(event.asset_id, 0.95)
        elif position > 0:
            pnl = self.get_unrealized_pnl_pct(event.asset_id)
            if pred < 0.4 or pnl < -0.05 or pnl > 0.15:
                self.close_position(event.asset_id)

# 4. Run backtest
feed = HistoricalDataFeed(df, signal_columns=['ml_pred', 'confidence'])
broker = SimulationBroker(initial_cash=100000)
engine = Engine(feed, MLStrategy(), broker)

results = engine.run()
print(results.summary())
```

**Key Observations:**
- Signal computation happens OUTSIDE engine (flexibility)
- Strategy code is clean, readable, stateful
- No look-ahead bias (signals are pre-computed correctly)
- Easy transition to live trading (swap HistoricalDataFeed → LiveDataFeed)

## 6. Performance Strategy

### Where Time is Actually Spent

Profiling typical event-driven backtests:

1. **Event loop iteration:** 10-20% (already fast, hard to optimize)
2. **Strategy logic:** 30-50% ← **PRIMARY OPTIMIZATION TARGET**
3. **Order execution/fills:** 20-30% (already vectorized)
4. **Portfolio updates:** 10-20% (simple arithmetic)

### Pragmatic Optimization Approach

**Phase 1: Make it work (Weeks 1-2)**
- Pure Python implementation
- Focus on API ergonomics
- Measure baseline performance

**Phase 2: Make it right (Weeks 3-4)**
- Add declarative SignalStrategy
- Numba-compile rule evaluators
- Target: 10-100x speedup for simple strategies

**Phase 3: Make it fast (Month 2+)**
- Profile real-world strategies
- Optimize hot paths identified by profiling
- Hybrid Numba approach for complex strategies

**Expectation:** Most users won't need Phase 3. Event-driven Python is "fast enough" for daily/hourly strategies. Numba optimization matters for minute/tick data.

## 7. VectorBT Integration Pattern

**VectorBT is not a competitor - it's a complement:**

```python
import vectorbt as vbt
import pandas as pd

# Step 1: Use VectorBT for fast indicator computation
close = df['close']
rsi = vbt.RSI.run(close, window=14).rsi
bb_bands = vbt.BBANDS.run(close, window=20)

# Step 2: Add to DataFrame
df['rsi'] = rsi
df['bb_upper'] = bb_bands.upper
df['bb_lower'] = bb_bands.lower

# Step 3: Compute ML predictions
from my_ml_model import predict
df['ml_pred'] = predict(df[['rsi', 'bb_upper', 'bb_lower']])

# Step 4: Use ml4t.backtest for stateful execution
strategy = ComplexMLStrategy()
result = engine.run(df, strategy, signal_columns=['ml_pred', 'rsi'])
```

**Division of Labor:**
- **VectorBT:** Fast preprocessing (indicators, filters)
- **ml4t.backtest:** Stateful execution (event-driven logic)

## 8. Live Trading Architecture

The beauty of event-driven design: swap data sources without changing strategy code.

### Interface Abstraction

```python
class DataFeed(ABC):
    """Base class for all data feeds"""

    @abstractmethod
    def __iter__(self):
        """Yield events one at a time"""
        pass

class HistoricalDataFeed(DataFeed):
    """Read from DataFrame (backtesting)"""

    def __iter__(self):
        for idx, row in self.data.iterrows():
            yield self._create_market_event(row)

class LiveDataFeed(DataFeed):
    """Stream from WebSocket (live trading)"""

    def __init__(self, websocket_url: str, signal_computer):
        self.ws = connect(websocket_url)
        self.signal_computer = signal_computer

    def __iter__(self):
        for message in self.ws:
            bar = parse_message(message)
            # Compute signals in real-time
            signals = self.signal_computer.compute(bar)
            yield self._create_market_event(bar, signals)
```

### Same Strategy, Different Environment

```python
# Backtesting
feed = HistoricalDataFeed(df, signal_columns=['ml_pred'])
broker = SimulationBroker(initial_cash=100000)
engine = Engine(feed, MyStrategy(), broker)

# Live Trading (paper)
feed = LiveDataFeed("wss://api.exchange.com", signal_computer=MyMLModel())
broker = PaperBroker(initial_cash=100000)
engine = Engine(feed, MyStrategy(), broker)  # SAME STRATEGY!

# Live Trading (real money)
feed = LiveDataFeed("wss://api.exchange.com", signal_computer=MyMLModel())
broker = LiveBroker(api_key="...", secret="...")
engine = Engine(feed, MyStrategy(), broker)  # SAME STRATEGY!
```

**Critical Requirement:** Strategy code is 100% identical. No `if is_live` conditionals.

## 9. Implementation Roadmap

### Phase 1: Core ML Signal Workflow (Week 1)
**Goal:** Users can backtest ML strategies

- [ ] Add `signals: dict[str, float]` to MarketData dataclass
- [ ] Extend HistoricalDataFeed.__init__() with signal_columns parameter
- [ ] Update _create_market_event() to populate signals dict
- [ ] Add Strategy helper methods (get_position, buy_percent, etc.)
- [ ] Write tests for signal propagation
- [ ] Create example ML strategy notebook

**Deliverable:** Working ML backtest example

### Phase 2: Declarative Strategy System (Week 2-3)
**Goal:** Fast path for simple strategies

- [ ] Design Rule-based API (SignalAbove, StopLoss, etc.)
- [ ] Implement SignalStrategy class
- [ ] Build rule evaluator with Numba compilation
- [ ] Benchmark vs pure Python (target: 10-100x)
- [ ] Document when to use declarative vs imperative

**Deliverable:** Performance boost for 80% of use cases

### Phase 3: Production Readiness (Month 2)
**Goal:** Path to live trading

- [ ] Define LiveDataFeed interface
- [ ] Implement WebSocket-based live feeds
- [ ] Create PaperBroker for paper trading
- [ ] Add comprehensive logging and monitoring
- [ ] Write live trading deployment guide

**Deliverable:** Production-ready event-driven framework

## 10. Open Questions & Decisions Needed

### Q1: Should signals be computed inside or outside the engine?

**Decision: OUTSIDE (pre-computed)**

Rationale:
- ML inference is expensive (could be seconds per prediction)
- Users may use external services (cloud APIs, GPU servers)
- Separation of concerns: engine executes, user computes
- Easier to test (deterministic signals)

Trade-off: Cannot adapt signals based on execution results (but this is rare)

### Q2: How to handle multi-asset strategies with different signals?

**Decision: Separate DataFeed per asset, OR multi-index columns**

Option A: One feed per asset
```python
feeds = {
    'AAPL': HistoricalDataFeed(aapl_df, signals=['ml_pred']),
    'MSFT': HistoricalDataFeed(msft_df, signals=['ml_pred']),
}
```

Option B: Multi-index columns
```python
df.columns = pd.MultiIndex.from_tuples([
    ('AAPL', 'close'), ('AAPL', 'ml_pred'),
    ('MSFT', 'close'), ('MSFT', 'ml_pred'),
])
```

**Recommendation:** Start with Option A (simpler), add Option B later if needed

### Q3: Performance target for event-driven execution?

**Decision: 10,000 bars/second minimum (pure Python)**

Rationale:
- 10k bars/sec = 252 days in 25ms (fast enough for daily/hourly strategies)
- Numba optimization can reach 100k+ bars/sec for simple strategies
- Tick data (millions of events) requires Numba, but that's Phase 2

Benchmark against Backtrader (typically 1-5k bars/sec).

## 11. Success Criteria

### User Experience
- [ ] ML strategy in <50 lines of code
- [ ] No boilerplate for common patterns
- [ ] Zero look-ahead bias by design
- [ ] Same code for backtest and live trading

### Performance
- [ ] 10k+ bars/sec (pure Python)
- [ ] 100k+ bars/sec (declarative + Numba)
- [ ] Match or beat Backtrader/Zipline

### Flexibility
- [ ] Support any ML framework (sklearn, xgboost, PyTorch, etc.)
- [ ] Arbitrary signal columns (no schema constraints)
- [ ] Complex stateful logic (regime switching, pairs trading)

### Production Ready
- [ ] Clean path to live trading
- [ ] Comprehensive logging and monitoring
- [ ] Paper trading mode for validation
- [ ] Error handling and recovery

## 12. References

- Backtrader docs: https://www.backtrader.com/docu/strategy/
- Zipline API: https://zipline.ml4trading.io/api-reference.html
- VectorBT signals: https://vectorbt.pro/pvt_bc80e00e/api/signals/
- Numba performance tips: https://numba.pydata.org/numba-doc/dev/user/performance-tips.html

## Appendix: Example Strategies

### A1: Simple ML Threshold Strategy

```python
class SimpleMLStrategy(Strategy):
    """Enter when ML prediction > threshold, exit on stop/target"""

    def on_market_data(self, event):
        pred = event.data.signals['ml_pred']
        position = self.get_position(event.asset_id)

        if pred > 0.8 and position == 0:
            self.buy_percent(event.asset_id, 0.95)
        elif position > 0:
            pnl = self.get_unrealized_pnl_pct(event.asset_id)
            if pnl < -0.05 or pnl > 0.15:
                self.close_position(event.asset_id)
```

### A2: Regime-Switching Strategy

```python
class RegimeSwitchingStrategy(Strategy):
    """Adapt position sizing based on volatility regime"""

    def __init__(self):
        super().__init__()
        self.regime = 'low_vol'

    def on_market_data(self, event):
        # Detect regime
        vol = event.data.signals['volatility']
        new_regime = 'high_vol' if vol > 0.02 else 'low_vol'

        # Regime change: reduce positions
        if new_regime != self.regime and self.regime == 'high_vol':
            for asset_id in self.get_positions():
                self.reduce_position(asset_id, 0.5)

        self.regime = new_regime

        # Entry logic depends on regime
        pred = event.data.signals['ml_pred']
        position = self.get_position(event.asset_id)

        if pred > 0.8 and position == 0:
            size = 0.95 if self.regime == 'low_vol' else 0.50
            self.buy_percent(event.asset_id, size)
```

### A3: Declarative Equivalent (Phase 2)

```python
# Same as A1 but declarative (10-100x faster)
strategy = SignalStrategy(
    entry_rules=[
        Rule.signal_above('ml_pred', 0.8),
        Rule.not_in_position(),
    ],
    exit_rules=[
        Rule.stop_loss(0.05),
        Rule.take_profit(0.15),
    ],
    position_sizer=PercentSizer(0.95)
)
```

---

**Last Updated:** 2025-11-14
**Next Review:** After Phase 1 implementation
