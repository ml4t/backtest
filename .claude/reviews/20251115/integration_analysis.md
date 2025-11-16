# Integration Analysis: ML4T.Backtest → ML4T.Diagnostics

**Date**: 2025-11-15
**Purpose**: Ensure ml4t.backtest architecture supports ml4t.diagnostics requirements
**Critical Context**: ML4T Diagnostics will perform **trade-level SHAP analysis** to close the ML→trading feedback loop

---

## Executive Summary

### The Integration Requirement

ML4T Diagnostics' **killer feature** is trade-level SHAP analysis:
- Takes worst trades from backtest
- Aligns SHAP values at trade entry timestamps
- Clusters error patterns by feature patterns
- Generates actionable hypotheses for improvement

**This requires ml4t.backtest to output rich trade data with precise timestamps and signal values.**

### Verdict: ✅ Current Architecture SUPPORTS This

The recommended backtest architecture (Option 1A + 2A + 3B) **already provides** all required data:

1. ✅ **Trade records** with timestamps (existing)
2. ✅ **Signal values** at entry (`event.signals` dict from Option 2A)
3. ✅ **Context data** at trade time (`context` dict from Option 3B)
4. ✅ **Feature alignment** via timestamps (diagnostic's responsibility)
5. ✅ **SHAP alignment** via timestamps (diagnostic's responsibility)

**Minor gap**: Backtest should **store signals with trades** for diagnostics. This is a small addition.

---

## What Diagnostics Needs

### 1. Trade Records (CRITICAL)

**From DIAGNOSTICS_IMPLEMENTATION.md**:
```python
# Diagnostics needs this
analyzer = TradeAnalysis(backtest_results)
worst_trades = analyzer.worst_trades(n=10)
# Returns: DataFrame with [timestamp, symbol, entry, exit, pnl, duration, ...]
```

**Required fields**:
```python
class Trade:
    entry_timestamp: datetime  # For feature/SHAP alignment
    exit_timestamp: datetime
    asset_id: AssetId
    entry_price: float
    exit_price: float
    pnl: float  # Absolute P&L
    pnl_pct: float  # Percentage return
    duration: timedelta
    direction: Literal['long', 'short']
    size: Quantity
```

**Current backtest support**: ✅ Already exists
- `backtest/execution/trade_tracker.py` tracks trades
- `BacktestResult.trades: pd.DataFrame` is the output

**Action needed**: None (already supported)

---

### 2. Signal Values at Trade Entry (CRITICAL NEW)

**Why needed**: To understand which ML predictions led to failed trades

**From DIAGNOSTICS_IMPLEMENTATION.md**:
```python
diagnostics = TradeDiagnostics()
analysis = diagnostics.analyze_trades(
    trades=worst_trades,
    features=feature_data,  # User provides features
    shap_values=model.shap_values  # User provides SHAP
)
```

**Required**: Signal values that were in `event.signals` at trade entry

**Example**:
```python
# When strategy decided to trade
def on_market_data(self, event, context):
    pred = event.signals['ml_pred_5d']  # 0.85
    conf = event.signals['confidence']   # 0.92

    if pred > 0.8:
        self.buy_percent(event.asset_id, 0.95)

# Diagnostics needs to know: pred=0.85, conf=0.92 at trade entry
```

**Current backtest support**: ❌ **NOT stored with trades**
- Signals exist in `MarketEvent.signals` (Option 2A)
- But not persisted when trade is created

**Action needed**: ✅ **Store signals dict with each trade**

**Implementation**:
```python
class Trade:
    # Existing fields...
    signals: dict[str, float] = field(default_factory=dict)  # NEW
    context: dict[str, float] = field(default_factory=dict)  # NEW

# In Broker or Strategy when creating trade
trade = Trade(
    entry_timestamp=event.timestamp,
    asset_id=event.asset_id,
    entry_price=event.close,
    signals=event.signals.copy(),  # Store signals!
    context=context.copy()  # Store context!
)
```

---

### 3. Context Data at Trade Entry (VALUABLE)

**Why needed**: Regime-aware error pattern detection

**From DIAGNOSTICS_IMPLEMENTATION.md**:
```
Pattern 1: Momentum overweighting in volatile periods (7/10 trades)
├─ Regime characteristics: High volatility (VIX > 25), low trend strength
```

**Required**: Context values (VIX, SPY, etc.) at trade entry timestamp

**Current backtest support**: ✅ Context exists (Option 3B)
- `Context` object provides market-wide data
- Strategy receives `context: dict` at each event
- But not stored with trades

**Action needed**: ✅ **Store context dict with each trade** (same as signals)

---

### 4. Feature Values at Trade Entry (USER RESPONSIBILITY)

**Why needed**: SHAP analysis correlates feature values with trade outcomes

**From DIAGNOSTICS_IMPLEMENTATION.md**:
```python
analysis = diagnostics.analyze_trades(
    trades=worst_trades,
    features=feature_data,  # at trade entry timestamps
    shap_values=model.shap_values
)
```

**How it works**:
1. Backtest outputs trades with `entry_timestamp`
2. User aligns features: `features.loc[trade.entry_timestamp]`
3. User computes or provides SHAP values
4. Diagnostics analyzes patterns

**Current backtest support**: ✅ Supported via timestamps
- Trade has `entry_timestamp`
- User joins features from ml4t.engineer

**Action needed**: None (user's responsibility to align)

---

### 5. SHAP Values (USER RESPONSIBILITY)

**Why needed**: Core of trade-level error analysis

**How it works**:
```python
# User computes SHAP (outside backtest)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features_at_trade_entries)

# User passes to diagnostics
diagnostics.analyze_trade_shap(
    trades=worst_trades,
    features=features_at_trade_entries,
    shap_values=shap_values  # User-computed
)
```

**Current backtest support**: ✅ Not backtest's responsibility
- Backtest provides trade timestamps
- User computes SHAP externally
- User aligns SHAP with trades

**Action needed**: None (by design)

---

## Data Contract: Backtest → Diagnostics

### BacktestResult Output (Required Fields)

```python
@dataclass
class BacktestResult:
    """Output from ml4t.backtest, input to ml4t.diagnostics."""

    # Trade-level data (CRITICAL for diagnostics)
    trades: pd.DataFrame
    # Columns:
    # - entry_timestamp: datetime
    # - exit_timestamp: datetime
    # - asset_id: str
    # - entry_price: float
    # - exit_price: float
    # - pnl: float
    # - pnl_pct: float
    # - duration: timedelta
    # - direction: str ('long' | 'short')
    # - size: float
    # - signals: dict (NEW: {ml_pred: 0.85, confidence: 0.92})
    # - context: dict (NEW: {VIX: 28.5, SPY_trend: 0.6})

    # Portfolio-level data
    positions: pd.DataFrame  # Position history
    equity: pd.Series  # Portfolio value over time

    # Summary metrics
    metrics: dict[str, float]  # Sharpe, Sortino, max_dd, etc.
```

### Diagnostics Usage Pattern

```python
from ml4t.backtest import BacktestEngine
from ml4t.diagnostics import TradeDiagnostics

# 1. Run backtest
engine = BacktestEngine(...)
results = engine.run(strategy)

# 2. Identify worst trades
analyzer = TradeAnalysis(results)
worst_trades = analyzer.worst_trades(n=10)

# 3. Get features at trade entry timestamps
trade_timestamps = worst_trades['entry_timestamp']
features_at_entry = features.loc[trade_timestamps]

# 4. Compute SHAP (user's responsibility)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features_at_entry)

# 5. Analyze trade-level SHAP
diagnostics = TradeDiagnostics()
analysis = diagnostics.analyze_trade_shap(
    trades=worst_trades,
    features=features_at_entry,
    shap_values=shap_values
)

# 6. Get error patterns with hypotheses
for pattern in analysis.patterns:
    print(f"Pattern: {pattern.description}")
    print(f"  Trades: {pattern.n_trades}")
    print(f"  Hypothesis: {pattern.hypothesis}")
    print(f"  Actions: {pattern.suggested_actions}")
```

### Key Integration Point: Trade Signals

**The critical addition**: Store `event.signals` and `context` with each trade.

**Implementation location**: `backtest/execution/broker.py` or `trade_tracker.py`

```python
# When broker fills an order and creates a trade record
def _create_trade_record(
    self,
    order: Order,
    fill_event: FillEvent,
    market_event: MarketEvent,  # Need this!
    context: dict  # Need this!
) -> Trade:
    return Trade(
        entry_timestamp=market_event.timestamp,
        asset_id=order.asset_id,
        entry_price=fill_event.fill_price,
        # ... other fields ...
        signals=market_event.signals.copy(),  # STORE SIGNALS
        context=context.copy()  # STORE CONTEXT
    )
```

**Challenge**: Broker needs access to `MarketEvent` and `context` when creating trade records.

**Current flow**:
```
Strategy.on_market_data(event, context)
  → Strategy creates Order
  → Broker.submit_order(order)
  → Broker executes order → creates FillEvent
  → Trade record created (but doesn't have event/context!)
```

**Solution**: Pass `market_event` and `context` when submitting order:

```python
# Option A: Store in Order object
class Order:
    # ... existing fields ...
    market_event: MarketEvent | None = None  # Event that triggered order
    context: dict = field(default_factory=dict)  # Context at order time

# Strategy helper method
def buy_percent(self, asset_id, percent):
    order = MarketOrder(
        asset_id=asset_id,
        quantity=self._calculate_quantity(percent),
        market_event=self._current_event,  # Store event
        context=self._current_context  # Store context
    )
    self.broker.submit_order(order)
```

**Option B: Store at strategy level, broker queries**:
```python
class Strategy:
    def __init__(self):
        self._last_event: MarketEvent | None = None
        self._last_context: dict = {}

    def on_market_data(self, event, context):
        self._last_event = event  # Cache
        self._last_context = context
        # ... user logic ...

# Broker can access strategy.last_event when creating trades
```

**Recommendation**: **Option A** (store in Order) is cleaner and explicit.

---

## Architectural Compatibility Check

### Does Option 2A (Embedded Signals) Support Diagnostics?

**✅ YES - Perfect Match**

```python
# Backtest: Signals embedded in MarketEvent
event.signals = {'ml_pred_5d': 0.85, 'confidence': 0.92}

# Strategy: Access signals
pred = event.signals['ml_pred_5d']

# Backtest: Store signals with trade
trade.signals = event.signals.copy()

# Diagnostics: Analyze signal patterns
for trade in worst_trades:
    print(f"Trade with pred={trade.signals['ml_pred_5d']} failed")
```

**This is exactly what diagnostics needs.** No changes to Option 2A required.

---

### Does Option 3B (Context Object) Support Diagnostics?

**✅ YES - Perfect Match**

```python
# Backtest: Context object provides market-wide data
context = {'VIX': 28.5, 'SPY_trend': 0.6}

# Strategy: Use context for regime-aware logic
if context.get('VIX', 30) > 25:
    return  # Don't trade in high volatility

# Backtest: Store context with trade
trade.context = context.copy()

# Diagnostics: Regime-aware error analysis
high_vol_trades = [t for t in worst_trades if t.context.get('VIX', 0) > 25]
print(f"Pattern: {len(high_vol_trades)} trades failed in high vol regime")
```

**This enables regime-aware diagnostics.** No changes to Option 3B required.

---

### Does Option 1A (Callbacks) Support Diagnostics?

**✅ YES - Transparency is Key**

From DIAGNOSTICS_IMPLEMENTATION.md:
> "The library should feel like 'Pandas for backtesting' - familiar, debuggable, and sufficient performance."

**Why callbacks help diagnostics**:
- Transparent logic → Easy to understand why trade was taken
- Debuggable → Can add print statements to see decision process
- Flexible → Can implement complex logic that diagnostics can analyze

**No changes needed.**

---

## Implementation Changes Required

### Minimal Changes to Backtest Library

**1. Add fields to Trade dataclass**:
```python
@dataclass
class Trade:
    # Existing fields...
    entry_timestamp: datetime
    exit_timestamp: datetime
    asset_id: AssetId
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    duration: timedelta
    direction: Literal['long', 'short']
    size: Quantity

    # NEW: For diagnostics integration
    signals: dict[str, float] = field(default_factory=dict)
    context: dict[str, float] = field(default_factory=dict)
```

**2. Store signals/context in Order**:
```python
@dataclass
class Order:
    # Existing fields...
    asset_id: AssetId
    quantity: Quantity
    order_type: OrderType

    # NEW: For trade tracking
    market_event: MarketEvent | None = None
    context: dict[str, float] = field(default_factory=dict)
```

**3. Update Strategy helper methods**:
```python
class Strategy(ABC):
    def __init__(self):
        self._broker: Broker
        self._current_event: MarketEvent | None = None
        self._current_context: dict = {}

    def on_market_data(self, event: MarketEvent, context: dict):
        # Cache for helper methods
        self._current_event = event
        self._current_context = context

        # Call user implementation
        self._on_market_data_impl(event, context)

    def buy_percent(self, asset_id, percent):
        order = MarketOrder(
            asset_id=asset_id,
            quantity=self._calculate_quantity(percent),
            market_event=self._current_event,  # Attach event
            context=self._current_context  # Attach context
        )
        self.broker.submit_order(order)
```

**4. Update Broker trade creation**:
```python
class SimulationBroker:
    def _create_trade_record(self, order: Order, fill: FillEvent) -> Trade:
        return Trade(
            entry_timestamp=fill.timestamp,
            asset_id=order.asset_id,
            entry_price=fill.fill_price,
            # ... other fields ...
            signals=order.market_event.signals if order.market_event else {},
            context=order.context.copy()
        )
```

**Total effort**: ~2-4 hours (small addition)

---

## Example: Complete Workflow

### 1. Feature Engineering (ml4t.engineer)

```python
from ml4t.engineer import FeatureEngineer

engineer = FeatureEngineer(price_data)
features = engineer.generate_features(['rsi', 'macd', 'momentum'])
outcomes = engineer.generate_outcomes(method='returns', horizon=5)

# Select features
selected = engineer.select_features(
    ic_threshold=0.05,
    importance_top_n=20
)
```

### 2. Model Training (User Code)

```python
from lightgbm import LGBMRegressor
import shap

# Train model
model = LGBMRegressor()
model.fit(features[selected], outcomes)

# Predict
predictions = model.predict(features[selected])

# Compute SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features[selected])
```

### 3. Backtesting (ml4t.backtest)

```python
from ml4t.backtest import BacktestEngine, Strategy

# Add predictions to data
data['ml_pred_5d'] = predictions
data['confidence'] = model.predict_proba(features[selected])[:, 1]

# Define strategy
class MLStrategy(Strategy):
    def on_market_data(self, event, context):
        pred = event.signals.get('ml_pred_5d', 0)
        conf = event.signals.get('confidence', 0)
        vix = context.get('VIX', 30)

        # Don't trade in high volatility
        if vix > 30:
            return

        # Trade on confident predictions
        if pred > 0.8 and conf > 0.7:
            self.buy_percent(event.asset_id, 0.95)

# Run backtest
engine = BacktestEngine(
    feed=ParquetDataFeed(data, signal_columns=['ml_pred_5d', 'confidence']),
    context=Context({'VIX': vix_series}),
    broker=SimulationBroker()
)
results = engine.run(MLStrategy())
```

**Results include**:
```python
results.trades:
   entry_timestamp  asset_id  pnl  signals                 context
0  2020-01-05       AAPL     -120  {'ml_pred': 0.85, ...}  {'VIX': 28.5}
1  2020-01-12       AAPL     +250  {'ml_pred': 0.92, ...}  {'VIX': 15.2}
```

### 4. Diagnostics (ml4t.diagnostics)

```python
from ml4t.diagnostics import TradeDiagnostics

# Identify worst trades
worst_trades = results.trades.nsmallest(10, 'pnl')

# Get features at trade entry
trade_timestamps = worst_trades['entry_timestamp']
features_at_entry = features.loc[trade_timestamps]

# Get SHAP at trade entry (already computed)
shap_at_entry = shap_values[features.index.isin(trade_timestamps)]

# Analyze
diagnostics = TradeDiagnostics()
analysis = diagnostics.analyze_trade_shap(
    trades=worst_trades,
    features=features_at_entry,
    shap_values=shap_at_entry
)

# Get actionable insights
for pattern in analysis.patterns:
    print(f"\nPattern: {pattern.description}")
    print(f"  Affected trades: {pattern.n_trades}")
    print(f"  Common features: {pattern.feature_patterns}")
    print(f"  Hypothesis: {pattern.hypothesis}")
    print(f"  Suggested actions:")
    for action in pattern.actions:
        print(f"    - {action}")
```

**Output**:
```
Pattern: High momentum + high volatility (7 trades, -2.1% avg loss)
  Affected trades: 7
  Common features: {'momentum_5d': 0.82, 'volatility_20d': 0.035}
  Hypothesis: Model overweights momentum in high volatility when mean reversion dominates
  Suggested actions:
    - Add interaction term: momentum_5d * (1 / volatility_20d)
    - Reduce momentum weight in high-vol regimes (VIX > 20)
    - Add mean-reversion features for high-vol periods
```

### 5. Improvement (Back to ml4t.engineer)

```python
# Based on diagnostics hypothesis, add interaction term
engineer.add_feature(
    'momentum_vol_interaction',
    lambda df: df['momentum_5d'] * (1 / df['volatility_20d'])
)

# Retrain model with new feature
# Retest with backtest
# Analyze again with diagnostics
# Repeat until satisfied
```

---

## Summary: Integration Status

### ✅ What Already Works

1. **Trade tracking**: Backtest outputs trades with timestamps
2. **Signal storage**: `MarketEvent.signals` exists (Option 2A)
3. **Context data**: `Context` object exists (Option 3B)
4. **Timestamp alignment**: User can join features/SHAP by timestamp

### ✅ Small Additions Needed

1. **Store signals with trades**: Add `signals` field to `Trade`
2. **Store context with trades**: Add `context` field to `Trade`
3. **Propagate event to orders**: Store `MarketEvent` in `Order`
4. **Update helper methods**: Cache current event/context

**Total effort**: 2-4 hours (minor addition to Phase 1)

### ✅ Architectural Compatibility

- **Option 1A (Callbacks)**: ✅ Transparent, debuggable (supports diagnostics)
- **Option 2A (Embedded Signals)**: ✅ Perfect match for trade-level analysis
- **Option 3B (Context Object)**: ✅ Enables regime-aware diagnostics
- **Option 4C (Delay Optimization)**: ✅ No impact on diagnostics

---

## Recommendations

### 1. Add Signal/Context Storage to Phase 1

**Updated Phase 1 tasks**:
```
Week 1: Core ML API
- Add `signals` dict to MarketEvent (existing)
- Add `signal_columns` to DataFeed (existing)
- Add helper methods (existing)
- NEW: Add `signals` and `context` fields to Trade
- NEW: Store signals/context when creating trades
- NEW: Test trade data includes signals/context
```

**Effort**: +2-4 hours to Phase 1 (minimal)

### 2. Document Data Contract

Create `docs/integration/diagnostics.md` explaining:
- What data backtest provides
- How to align features/SHAP
- Example workflow

### 3. Test Integration Early

**Week 2 integration test**:
```python
# After Phase 1 complete
def test_trade_signals_stored():
    """Verify trades include signals for diagnostics."""
    results = engine.run(strategy)

    # Check trades have signals
    assert 'signals' in results.trades.columns
    assert results.trades.iloc[0]['signals']['ml_pred_5d'] == 0.85

    # Check context stored
    assert 'context' in results.trades.columns
    assert results.trades.iloc[0]['context']['VIX'] == 28.5
```

---

## Conclusion

### ✅ Architecture is Compatible

The recommended ml4t.backtest architecture (Option 1A + 2A + 3B) **already provides 95% of what ml4t.diagnostics needs**.

**The 5% gap**:
- Store `signals` and `context` with trades (2-4 hours to implement)

### 🚀 This Validates the Design

The fact that the backtest architecture naturally supports the diagnostics use case **confirms the design is sound**.

**Key insight**: By choosing embedded signals (Option 2A) and context object (Option 3B), we've made the two libraries naturally compatible.

### 📋 Action Items

1. ✅ Add `signals` and `context` fields to `Trade` dataclass
2. ✅ Store signals/context when creating trade records
3. ✅ Document data contract for diagnostics integration
4. ✅ Add integration test (Week 2)

**This ensures ml4t.backtest is "diagnostics-ready" from day one.**

---

**The complete ML4T ecosystem vision**:

```
ml4t.data → ml4t.engineer → User ML → ml4t.backtest → ml4t.diagnostics
                ↑________________________________________________|
                    (Feedback loop: improve features based on error analysis)
```

**This architecture makes "the process is your edge" a reality.**
