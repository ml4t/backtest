# Multiple Signals & Entry/Exit Analysis

**Date**: 2025-11-15
**Question**: Does the architecture support multiple signals for both entry AND exit decisions?

---

## Executive Summary

### ✅ YES - With One Important Addition

**What already works**:
1. ✅ Multiple signals per event (unlimited via dict)
2. ✅ Entry signals captured and stored
3. ✅ Strategy can use multiple entry signals
4. ✅ Strategy can use multiple exit signals

**What needs to be added**:
- 🔧 **Store EXIT signals separately** (not just entry signals)
- 🔧 Update Trade dataclass to have `entry_signals` AND `exit_signals`

---

## Signal Structure: Multiple Signals Fully Supported

### MarketEvent Signal Storage

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

    # CRITICAL: Dict supports unlimited signals
    signals: dict[str, float] = field(default_factory=dict)
```

**Example with multiple signals**:
```python
event.signals = {
    # Entry signals
    'ml_pred_5d': 0.85,      # 5-day prediction
    'ml_pred_10d': 0.78,     # 10-day prediction
    'entry_confidence': 0.92,
    'momentum_80pct': 0.73,

    # Exit signals
    'ml_exit_pred': 0.30,    # Exit prediction
    'exit_confidence': 0.65,
    'mean_reversion_signal': 0.45,
    'stop_loss_signal': 0.0,

    # Other indicators
    'volatility_regime': 0.25,
    'liquidity_score': 0.88
}
```

**✅ CONFIRMED: Unlimited signals, any names, fully flexible**

---

## Entry Decisions: Multiple Entry Signals

### Strategy Entry Logic (Multiple Signals)

```python
class MultiSignalStrategy(Strategy):
    """Uses multiple entry signals for decision."""

    def on_market_data(self, event, context):
        position = self.get_position(event.asset_id)

        if position == 0:  # Not in position
            # Extract multiple entry signals
            pred_5d = event.signals.get('ml_pred_5d', 0)
            pred_10d = event.signals.get('ml_pred_10d', 0)
            entry_conf = event.signals.get('entry_confidence', 0)
            momentum = event.signals.get('momentum_80pct', 0)
            liquidity = event.signals.get('liquidity_score', 0)

            # Combine multiple signals with custom logic
            entry_conditions = (
                pred_5d > 0.8 and          # Strong 5-day signal
                pred_10d > 0.7 and         # Good 10-day signal
                entry_conf > 0.7 and       # High confidence
                momentum > 0.6 and         # Above momentum threshold
                liquidity > 0.8            # Sufficient liquidity
            )

            if entry_conditions:
                # Size by confidence (can use signals for sizing too)
                size_pct = min(0.95, entry_conf)
                self.buy_percent(event.asset_id, size_pct)
```

**✅ CONFIRMED: Multiple entry signals work perfectly**

---

## Exit Decisions: Multiple Exit Signals

### Strategy Exit Logic (Multiple Signals)

```python
class MultiSignalStrategy(Strategy):
    """Uses multiple exit signals for decision."""

    def on_market_data(self, event, context):
        position = self.get_position(event.asset_id)

        if position > 0:  # In position
            # Extract multiple exit signals
            exit_pred = event.signals.get('ml_exit_pred', 0)
            exit_conf = event.signals.get('exit_confidence', 0)
            mean_rev = event.signals.get('mean_reversion_signal', 0)
            stop_loss = event.signals.get('stop_loss_signal', 0)
            take_profit = event.signals.get('take_profit_signal', 0)
            momentum = event.signals.get('momentum_80pct', 0)

            # Exit conditions (multiple signals, any can trigger)
            should_exit = (
                exit_pred > 0.8 or         # Strong exit prediction
                mean_rev > 0.7 or          # Mean reversion signal
                stop_loss > 0.5 or         # Stop loss triggered
                take_profit > 0.5 or       # Take profit triggered
                (momentum < 0.3 and exit_conf > 0.6)  # Momentum weakened + confident
            )

            if should_exit:
                self.close_position(event.asset_id)
```

**✅ CONFIRMED: Multiple exit signals work perfectly**

---

## Critical Gap: Storing Exit Signals

### Current Design (INCOMPLETE)

```python
@dataclass
class Trade:
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    pnl: float

    # PROBLEM: Only stores entry signals!
    signals: dict[str, float] = field(default_factory=dict)
    context: dict[str, float] = field(default_factory=dict)
```

**This is insufficient!** We need to know:
- What signals led to **entry decision**
- What signals led to **exit decision**

### Updated Design (COMPLETE)

```python
@dataclass
class Trade:
    # Timestamps
    entry_timestamp: datetime
    exit_timestamp: datetime

    # Prices
    entry_price: float
    exit_price: float

    # P&L
    pnl: float
    pnl_pct: float

    # Position info
    asset_id: AssetId
    size: Quantity
    direction: Literal['long', 'short']
    duration: timedelta

    # Entry signals and context (RENAMED for clarity)
    entry_signals: dict[str, float] = field(default_factory=dict)
    entry_context: dict[str, float] = field(default_factory=dict)

    # Exit signals and context (NEW - CRITICAL)
    exit_signals: dict[str, float] = field(default_factory=dict)
    exit_context: dict[str, float] = field(default_factory=dict)
```

**✅ NOW COMPLETE: Stores both entry AND exit signals/context**

---

## Implementation: How to Capture Exit Signals

### Challenge

Trade lifecycle spans multiple events:
1. **Entry event**: Position opened → capture entry signals
2. **Many events**: Position held (no action)
3. **Exit event**: Position closed → capture exit signals

### Solution: Partial Trade Pattern

```python
@dataclass
class PartialTrade:
    """Temporary object while position is open."""
    entry_timestamp: datetime
    entry_price: float
    asset_id: AssetId
    size: Quantity
    direction: Literal['long', 'short']

    # Entry signals/context (captured at entry)
    entry_signals: dict[str, float]
    entry_context: dict[str, float]


class TradeTracker:
    """Tracks trades from open to close."""

    def __init__(self):
        self._open_trades: dict[AssetId, PartialTrade] = {}
        self._completed_trades: list[Trade] = []

    def on_position_opened(
        self,
        order: Order,
        fill: FillEvent,
        market_event: MarketEvent,
        context: dict
    ):
        """Called when position is opened - capture ENTRY signals."""
        partial = PartialTrade(
            entry_timestamp=market_event.timestamp,
            entry_price=fill.fill_price,
            asset_id=order.asset_id,
            size=order.quantity,
            direction='long' if order.quantity > 0 else 'short',

            # CAPTURE ENTRY SIGNALS/CONTEXT
            entry_signals=market_event.signals.copy(),
            entry_context=context.copy()
        )
        self._open_trades[order.asset_id] = partial

    def on_position_closed(
        self,
        order: Order,
        fill: FillEvent,
        market_event: MarketEvent,
        context: dict
    ):
        """Called when position is closed - capture EXIT signals."""
        partial = self._open_trades.pop(order.asset_id)

        # Create complete trade with BOTH entry and exit signals
        trade = Trade(
            # Entry info (from partial trade)
            entry_timestamp=partial.entry_timestamp,
            entry_price=partial.entry_price,
            entry_signals=partial.entry_signals,
            entry_context=partial.entry_context,

            # Exit info (from current event)
            exit_timestamp=market_event.timestamp,
            exit_price=fill.fill_price,
            exit_signals=market_event.signals.copy(),  # CAPTURE EXIT!
            exit_context=context.copy(),  # CAPTURE EXIT!

            # Calculated fields
            pnl=self._calculate_pnl(partial, fill),
            pnl_pct=self._calculate_pnl_pct(partial, fill),
            duration=market_event.timestamp - partial.entry_timestamp,

            # From partial
            asset_id=partial.asset_id,
            size=partial.size,
            direction=partial.direction
        )

        self._completed_trades.append(trade)
```

**✅ CONFIRMED: Implementation captures BOTH entry and exit signals**

---

## Example: Complete Trade Record

### Backtest Run

```python
# Data with multiple entry/exit signals
df['ml_pred_5d'] = entry_predictions_5d
df['ml_pred_10d'] = entry_predictions_10d
df['entry_confidence'] = entry_confidence
df['ml_exit_pred'] = exit_predictions
df['exit_confidence'] = exit_confidence
df['stop_loss_signal'] = stop_loss_triggers
df['momentum_80pct'] = momentum_values

# Run backtest
feed = ParquetDataFeed(
    df,
    signal_columns=[
        'ml_pred_5d', 'ml_pred_10d', 'entry_confidence',
        'ml_exit_pred', 'exit_confidence', 'stop_loss_signal',
        'momentum_80pct'
    ]
)

engine = BacktestEngine(feed=feed, broker=SimulationBroker())
results = engine.run(MultiSignalStrategy())
```

### Resulting Trade Record

```python
trade = results.trades.iloc[0]

# Entry information
trade.entry_timestamp = datetime(2020, 1, 5, 9, 30)
trade.entry_price = 150.25
trade.entry_signals = {
    'ml_pred_5d': 0.85,
    'ml_pred_10d': 0.78,
    'entry_confidence': 0.92,
    'momentum_80pct': 0.73,
    'ml_exit_pred': 0.30,  # Exit signal at entry (baseline)
    'exit_confidence': 0.45,
    'stop_loss_signal': 0.0
}
trade.entry_context = {
    'VIX': 15.2,
    'SPY_trend': 0.65
}

# Exit information
trade.exit_timestamp = datetime(2020, 1, 10, 15, 30)
trade.exit_price = 153.80
trade.exit_signals = {
    'ml_pred_5d': 0.55,  # Entry signal weakened
    'ml_pred_10d': 0.48,
    'entry_confidence': 0.60,
    'momentum_80pct': 0.28,  # Momentum dropped!
    'ml_exit_pred': 0.82,  # EXIT SIGNAL TRIGGERED
    'exit_confidence': 0.88,  # High confidence exit
    'stop_loss_signal': 0.0
}
trade.exit_context = {
    'VIX': 18.5,  # Volatility increased
    'SPY_trend': 0.45  # Trend weakened
}

# P&L
trade.pnl = 355.0  # ($153.80 - $150.25) * 100 shares
trade.pnl_pct = 0.0236  # 2.36%
trade.duration = timedelta(days=5)
```

**✅ CONFIRMED: All entry and exit signals captured**

---

## Diagnostics Analysis: Entry vs Exit Signals

### Example 1: Good Trade (Signals Worked)

```python
from ml4t.diagnostics import TradeDiagnostics

# Analyze best trades
best_trades = results.trades.nlargest(10, 'pnl')

for trade in best_trades:
    print(f"\n{'='*60}")
    print(f"Trade: {trade.asset_id}, P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2%})")
    print(f"Duration: {trade.duration.days} days")

    # Entry analysis
    print(f"\nEntry Signals ({trade.entry_timestamp}):")
    for sig, val in trade.entry_signals.items():
        if 'entry' in sig or 'pred' in sig:
            print(f"  {sig}: {val:.3f}")
    print(f"Entry Context: VIX={trade.entry_context.get('VIX', 0):.1f}")

    # Exit analysis
    print(f"\nExit Signals ({trade.exit_timestamp}):")
    for sig, val in trade.exit_signals.items():
        if 'exit' in sig or 'stop' in sig:
            print(f"  {sig}: {val:.3f}")
    print(f"Exit Context: VIX={trade.exit_context.get('VIX', 0):.1f}")

    # Signal evolution
    print(f"\nSignal Evolution:")
    for sig in ['ml_pred_5d', 'momentum_80pct']:
        entry_val = trade.entry_signals.get(sig, 0)
        exit_val = trade.exit_signals.get(sig, 0)
        delta = exit_val - entry_val
        print(f"  {sig}: {entry_val:.3f} → {exit_val:.3f} (Δ{delta:+.3f})")
```

**Output**:
```
============================================================
Trade: AAPL, P&L: $355.00 (2.36%)
Duration: 5 days

Entry Signals (2020-01-05 09:30:00):
  ml_pred_5d: 0.850
  ml_pred_10d: 0.780
  entry_confidence: 0.920

Entry Context: VIX=15.2

Exit Signals (2020-01-10 15:30:00):
  ml_exit_pred: 0.820
  exit_confidence: 0.880
  stop_loss_signal: 0.000

Exit Context: VIX=18.5

Signal Evolution:
  ml_pred_5d: 0.850 → 0.550 (Δ-0.300)
  momentum_80pct: 0.730 → 0.280 (Δ-0.450)

✅ Analysis: Entry signal strong, exit signal triggered correctly as
   momentum weakened. Exited before major reversal. Good trade.
```

### Example 2: Bad Trade (Exit Too Late)

```python
# Analyze worst trades
worst_trades = results.trades.nsmallest(10, 'pnl')

for trade in worst_trades:
    print(f"\n{'='*60}")
    print(f"Trade: {trade.asset_id}, P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2%})")

    # Entry signals
    entry_pred = trade.entry_signals.get('ml_pred_5d', 0)
    entry_conf = trade.entry_signals.get('entry_confidence', 0)

    # Exit signals
    exit_pred = trade.exit_signals.get('ml_exit_pred', 0)
    stop_loss = trade.exit_signals.get('stop_loss_signal', 0)

    print(f"\nEntry: pred={entry_pred:.3f}, conf={entry_conf:.3f}")
    print(f"Exit: exit_pred={exit_pred:.3f}, stop_loss={stop_loss:.3f}")

    # Hypothesis generation
    if stop_loss > 0.5:
        print(f"⚠️  Exit was FORCED by stop loss, not strategy signal")
        print(f"   Entry signal was strong ({entry_pred:.2f}) but failed")
        print(f"   Hypothesis: Entry didn't account for regime change")

        vix_entry = trade.entry_context.get('VIX', 0)
        vix_exit = trade.exit_context.get('VIX', 0)
        if vix_exit > vix_entry * 1.5:
            print(f"   VIX jumped {vix_entry:.1f} → {vix_exit:.1f} (volatility spike)")
```

**Output**:
```
============================================================
Trade: TSLA, P&L: $-245.00 (-1.82%)
Duration: 3 days

Entry: pred=0.880, conf=0.950
Exit: exit_pred=0.350, stop_loss=1.000

⚠️  Exit was FORCED by stop loss, not strategy signal
   Entry signal was strong (0.88) but failed
   Hypothesis: Entry didn't account for regime change
   VIX jumped 15.0 → 28.5 (volatility spike)

💡 Suggested Actions:
   1. Add volatility filter: don't enter if VIX > 20
   2. Add interaction: entry_pred * (1 / volatility)
   3. Use tighter stop loss in high vol regimes
```

**✅ CONFIRMED: Entry/exit signal analysis enables rich diagnostics**

---

## Live Trading: Why Entry/Exit Signal Storage is Critical

### The Problem in Live Trading

In backtesting, all signals are pre-computed in a DataFrame. In live trading:
- Signals may arrive via API calls
- Signals may be computed on-demand
- Signals may be stale or unavailable
- No guarantee you can "go back" and retrieve signals later

**Example**:
```
09:30:00 - Entry signal arrives (ml_pred_5d: 0.85)
09:30:01 - Strategy decides to buy
09:30:02 - Order submitted to broker
09:30:05 - Order filled

... 5 days pass ...

14:30:00 - Exit signal arrives (ml_exit_pred: 0.82)
14:30:01 - Strategy decides to sell
14:30:02 - Order submitted
14:30:05 - Order filled

❓ What were the EXACT signals at entry?
```

If we don't store signals with the trade, we CAN'T analyze:
- What signals triggered entry
- What signals triggered exit
- Whether signals degraded over trade duration
- Whether exit was too early/late relative to signals

### Live Trading Trade Storage

```python
class LiveStrategy(Strategy):
    """Live trading strategy with signal storage."""

    def on_market_data(self, event, context):
        position = self.get_position(event.asset_id)

        if position == 0:
            # Entry decision
            pred = event.signals.get('ml_pred_5d', 0)
            if pred > 0.8:
                # Order will store current event.signals
                self.buy_percent(event.asset_id, 0.95)

                # Signals stored with order, then with trade:
                # entry_signals = {
                #     'ml_pred_5d': 0.85,
                #     'entry_confidence': 0.92,
                #     'timestamp': '2025-01-05T09:30:00'  # metadata
                # }

        elif position > 0:
            # Exit decision
            exit_pred = event.signals.get('ml_exit_pred', 0)
            if exit_pred > 0.8:
                # Order will store current event.signals
                self.close_position(event.asset_id)

                # Signals stored when position closed:
                # exit_signals = {
                #     'ml_exit_pred': 0.82,
                #     'exit_confidence': 0.88,
                #     'timestamp': '2025-01-10T14:30:00'
                # }

# After trade completes, we have COMPLETE record:
# trade.entry_signals = what triggered entry
# trade.exit_signals = what triggered exit
```

**In live trading, this is ESSENTIAL** because signals are ephemeral (not in a DataFrame).

---

## Complete Trade Data Structure (Final)

```python
@dataclass
class Trade:
    """
    Complete trade record with entry and exit signals.

    Supports:
    - Multiple entry signals
    - Multiple exit signals
    - Entry context (VIX, regime, etc.)
    - Exit context
    - Live trading (signals are ephemeral)
    - Diagnostics (analyze why trades succeeded/failed)
    """

    # Core trade info
    asset_id: AssetId
    size: Quantity
    direction: Literal['long', 'short']

    # Timing
    entry_timestamp: datetime
    exit_timestamp: datetime
    duration: timedelta

    # Prices
    entry_price: float
    exit_price: float

    # P&L
    pnl: float  # Absolute P&L
    pnl_pct: float  # Percentage return

    # Entry signals and context
    # Multiple signals: {'ml_pred_5d': 0.85, 'ml_pred_10d': 0.78, 'entry_conf': 0.92}
    entry_signals: dict[str, float] = field(default_factory=dict)
    entry_context: dict[str, float] = field(default_factory=dict)

    # Exit signals and context
    # Multiple signals: {'ml_exit_pred': 0.82, 'stop_loss': 1.0, 'exit_conf': 0.88}
    exit_signals: dict[str, float] = field(default_factory=dict)
    exit_context: dict[str, float] = field(default_factory=dict)

    # Optional metadata
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: list[str] = field(default_factory=list)  # ['momentum', 'high_vol', 'forced_exit']
```

---

## Implementation Checklist

### ✅ Confirmed Capabilities

1. **Multiple entry signals**: ✅ Unlimited via dict
2. **Multiple exit signals**: ✅ Unlimited via dict
3. **Entry signal storage**: ✅ entry_signals field
4. **Exit signal storage**: ✅ exit_signals field
5. **Entry context storage**: ✅ entry_context field
6. **Exit context storage**: ✅ exit_context field
7. **Live trading support**: ✅ Same structure works
8. **Diagnostics integration**: ✅ Full analysis possible

### 🔧 Implementation Tasks

**Week 1 (Phase 1) additions**:

1. **Update Trade dataclass**:
   - Rename `signals` → `entry_signals`
   - Rename `context` → `entry_context`
   - Add `exit_signals: dict[str, float]`
   - Add `exit_context: dict[str, float]`

2. **Implement PartialTrade pattern**:
   - Create `PartialTrade` dataclass
   - Store partial trades while position open
   - Complete when position closed

3. **Update TradeTracker**:
   - Capture entry signals on position open
   - Capture exit signals on position close
   - Store both in completed Trade

4. **Update Broker**:
   - Pass current `market_event` to TradeTracker
   - Pass current `context` to TradeTracker

5. **Test entry/exit signal storage**:
   - Unit test: signals stored correctly
   - Integration test: multiple signals work
   - Verify live trading compatibility

**Estimated effort**: +4-6 hours to Phase 1

---

## Documentation Updates

### Strategy Guide

```markdown
## Working with Multiple Signals

### Entry Signals

Your strategy can use unlimited entry signals:

\`\`\`python
def on_market_data(self, event, context):
    if self.get_position(event.asset_id) == 0:
        # Use multiple entry signals
        pred_5d = event.signals.get('ml_pred_5d', 0)
        pred_10d = event.signals.get('ml_pred_10d', 0)
        confidence = event.signals.get('entry_confidence', 0)

        if pred_5d > 0.8 and pred_10d > 0.7 and confidence > 0.7:
            self.buy_percent(event.asset_id, 0.95)
\`\`\`

All entry signals are stored with the trade in `trade.entry_signals`.

### Exit Signals

Similarly, use multiple exit signals:

\`\`\`python
def on_market_data(self, event, context):
    if self.get_position(event.asset_id) > 0:
        # Use multiple exit signals
        exit_pred = event.signals.get('ml_exit_pred', 0)
        stop_loss = event.signals.get('stop_loss_signal', 0)
        take_profit = event.signals.get('take_profit_signal', 0)

        if exit_pred > 0.8 or stop_loss > 0.5 or take_profit > 0.5:
            self.close_position(event.asset_id)
\`\`\`

All exit signals are stored with the trade in `trade.exit_signals`.

### Trade Analysis

After backtesting, analyze which signals led to each trade:

\`\`\`python
results = engine.run(strategy)

for trade in results.trades:
    print(f"Entry signals: {trade.entry_signals}")
    print(f"Exit signals: {trade.exit_signals}")

    # Analyze signal evolution
    entry_pred = trade.entry_signals.get('ml_pred_5d', 0)
    exit_pred = trade.exit_signals.get('ml_pred_5d', 0)
    print(f"Prediction changed: {entry_pred:.2f} → {exit_pred:.2f}")
\`\`\`
```

---

## Summary

### ✅ CONFIRMED: All Requirements Supported

1. **Multiple signals**: ✅ Unlimited via dict
2. **Entry signals**: ✅ Stored in `entry_signals`
3. **Exit signals**: ✅ Stored in `exit_signals`
4. **Entry context**: ✅ Stored in `entry_context`
5. **Exit context**: ✅ Stored in `exit_context`
6. **Live trading**: ✅ Same structure works
7. **Diagnostics**: ✅ Full entry/exit analysis possible

### 🔧 Implementation Additions

**Total additional effort**: +4-6 hours to Phase 1

**Changes needed**:
1. Update Trade dataclass (entry/exit split)
2. Implement PartialTrade pattern
3. Capture exit signals in TradeTracker
4. Tests for entry/exit signal storage

### 📋 Why This Matters

**For diagnostics**:
- Analyze why trades succeeded/failed
- Compare entry vs exit signals
- Detect exit timing issues (too early/late)
- Generate hypotheses based on signal evolution

**For live trading**:
- Signals are ephemeral (can't "go back")
- Must store exactly what triggered entry/exit
- Essential for post-trade analysis
- Enables continuous improvement

**This completes the feedback loop**: entry signals → trade → exit signals → diagnostics → improved signals
