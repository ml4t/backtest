# ml4t.backtest Differentiators

What ml4t.backtest can do that vectorized frameworks (VectorBT) cannot, and what it does better than other event-driven frameworks (Backtrader, Zipline).

## Quick Comparison

| Feature | VectorBT | Backtrader | ml4t.backtest |
|---------|----------|------------|---------------|
| Time-based exit (after N bars) | Hard (manual tracking) | Manual | Native (`TimeExit`) |
| Dynamic stop tightening | Impossible | Manual | Native in `on_data()` |
| ML confidence → position size | Manual post-processing | Manual | Native via `signals_df` |
| Cash vs Margin accounts | N/A | Limited | Policy-based |
| Futures multipliers | Manual | Manual | `ContractSpec` |
| Short selling constraints | N/A | Limited | Account policy |
| Position-aware rules | N/A | Manual | `PositionRules` |

---

## Differentiator 1: Time-Based Exits (Exit After N Bars)

**The Problem**: You want to exit a position after holding it for N bars, regardless of price action.

### VectorBT Approach (Complex)

```python
import vectorbt as vbt
import numpy as np

# VectorBT is vectorized - you need to track entry bars manually
def calculate_time_exits(entries, max_bars):
    """Manual time exit logic for VectorBT."""
    exits = np.zeros_like(entries, dtype=bool)
    in_position = False
    entry_bar = 0

    for i in range(len(entries)):
        if entries[i] and not in_position:
            in_position = True
            entry_bar = i
        elif in_position:
            if i - entry_bar >= max_bars:
                exits[i] = True
                in_position = False

    return exits

# This breaks VectorBT's vectorized optimization
entries = signals > 0
time_exits = calculate_time_exits(entries.values, max_bars=5)
exits = (signals < 0) | time_exits

pf = vbt.Portfolio.from_signals(
    close=prices,
    entries=entries,
    exits=exits,
)
```

**Problems:**
- Breaks vectorization (defeats VectorBT's speed advantage)
- Complex manual logic
- Hard to combine with other exit conditions
- Doesn't handle partial exits

### ml4t.backtest Approach (Native)

```python
from ml4t.backtest import Strategy, Engine, DataFeed
from ml4t.backtest.risk import TimeExit, StopLoss, RuleChain

class MomentumStrategy(Strategy):
    def on_start(self, broker):
        # Define exit rules: stop-loss OR time-based exit
        rules = RuleChain([
            StopLoss(pct=0.02),       # 2% stop-loss
            TimeExit(max_bars=5),      # Exit after 5 bars
        ])
        broker.set_position_rules(rules)

    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            signal = bar['signals'].get('momentum', 0)
            pos = broker.get_position(asset)

            if signal > 0 and pos is None:
                qty = int(broker.get_cash() * 0.95 / bar['close'])
                if qty > 0:
                    broker.submit_order(asset, qty)

            # Exits handled automatically by rules!

# Run backtest
feed = DataFeed(prices_df=prices_df, signals_df=signals_df)
engine = Engine(feed, MomentumStrategy(), initial_cash=100000)
result = engine.run()
```

**Advantages:**
- Native support, no manual tracking
- Combines naturally with other rules
- Position-aware (tracks `bars_held` per position)
- Works with partial exits

---

## Differentiator 2: Dynamic Stop Tightening

**The Problem**: You want to tighten your stop-loss as profit increases. For example:
- Initial stop: 2% below entry
- If profit > 3%: tighten stop to breakeven
- If profit > 5%: tighten stop to 2% above entry

### VectorBT Approach (Impossible with Standard API)

VectorBT's vectorized approach fundamentally cannot do this:

```python
# This CANNOT be done in VectorBT's from_signals()
# because exit signals are computed before knowing entry prices
# and cannot be dynamically adjusted during the backtest

# You would need to:
# 1. Run backtest to get entry prices
# 2. Calculate dynamic stops post-hoc
# 3. Re-run with new exit signals
# 4. Iterate until convergence (if it converges at all!)

# This is not practical and loses VectorBT's speed advantage
```

### ml4t.backtest Approach (Natural in Event Loop)

```python
from ml4t.backtest import Strategy, Engine, DataFeed

class DynamicStopStrategy(Strategy):
    def __init__(self, initial_stop: float = 0.02):
        self.initial_stop = initial_stop
        self.stop_prices = {}  # Track stop per asset

    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            pos = broker.get_position(asset)
            price = bar['close']
            signal = bar['signals'].get('entry_signal', 0)

            if pos is None:
                # Entry logic
                if signal > 0:
                    qty = int(broker.get_cash() * 0.95 / price)
                    if qty > 0:
                        broker.submit_order(asset, qty)
                        # Set initial stop
                        self.stop_prices[asset] = price * (1 - self.initial_stop)
            else:
                # Dynamic stop tightening
                entry_price = pos.entry_price
                profit_pct = (price - entry_price) / entry_price

                # Tighten stop based on profit
                if profit_pct > 0.05:
                    # Lock in 2% profit
                    new_stop = entry_price * 1.02
                elif profit_pct > 0.03:
                    # Move to breakeven
                    new_stop = entry_price
                else:
                    # Keep initial stop
                    new_stop = entry_price * (1 - self.initial_stop)

                # Only tighten, never loosen
                current_stop = self.stop_prices.get(asset, 0)
                self.stop_prices[asset] = max(new_stop, current_stop)

                # Check if stop hit
                if price <= self.stop_prices[asset]:
                    broker.close_position(asset)
                    del self.stop_prices[asset]

feed = DataFeed(prices_df=prices_df, signals_df=signals_df)
engine = Engine(feed, DynamicStopStrategy(), initial_cash=100000)
result = engine.run()
```

**Advantages:**
- Natural expression of the logic
- Position-aware (uses actual entry price)
- Can implement any stop logic (ATR-based, percentage, time-decay, etc.)
- No post-hoc iteration needed

---

## Differentiator 3: ML Signal Confidence → Position Sizing

**The Problem**: Your ML model outputs not just direction but confidence (probability). You want:
- High confidence → larger position
- Low confidence → smaller position
- Below threshold → no trade

### VectorBT Approach (Manual Post-Processing)

```python
import vectorbt as vbt
import numpy as np

# VectorBT from_signals() uses fixed sizing
# For dynamic sizing, you need from_orders() which is much more complex

# Option 1: Binned sizing (loses granularity)
high_conf_entries = (signals > 0) & (confidence > 0.8)
med_conf_entries = (signals > 0) & (confidence > 0.6) & (confidence <= 0.8)
low_conf_entries = (signals > 0) & (confidence <= 0.6)

# Run separate backtests and combine... messy!

# Option 2: from_orders() - much more complex
# Need to generate order sizes for every bar
sizes = np.where(signals > 0, confidence * account_value / price, 0)

# But this doesn't account for:
# - Current positions
# - Cash constraints
# - Partial fills
# - Rebalancing
```

### ml4t.backtest Approach (Native via signals_df)

```python
from ml4t.backtest import Strategy, Engine, DataFeed

class ConfidenceWeightedStrategy(Strategy):
    def __init__(self, confidence_threshold: float = 0.5, max_position: float = 0.2):
        self.confidence_threshold = confidence_threshold
        self.max_position = max_position  # Max 20% per position

    def on_data(self, timestamp, data, context, broker):
        account_value = broker.get_account_value()

        for asset, bar in data.items():
            signal = bar['signals'].get('ml_direction', 0)  # +1, -1, 0
            confidence = bar['signals'].get('ml_confidence', 0)  # 0.0 to 1.0

            pos = broker.get_position(asset)
            price = bar['close']

            if signal != 0 and confidence > self.confidence_threshold:
                # Scale position size by confidence
                # confidence 0.5 → 0% of max, confidence 1.0 → 100% of max
                scale = (confidence - self.confidence_threshold) / (1 - self.confidence_threshold)
                target_allocation = scale * self.max_position
                target_value = account_value * target_allocation
                target_qty = int(target_value / price) * signal  # Sign determines direction

                current_qty = pos.quantity if pos else 0
                delta = target_qty - current_qty

                if abs(delta) > 0:
                    broker.submit_order(asset, delta)

            elif signal == 0 and pos is not None:
                # No signal, close position
                broker.close_position(asset)

# Prepare signals with ML predictions
signals_df = prices_df.select(['timestamp', 'asset']).with_columns([
    pl.Series('ml_direction', model.predict(X)),           # -1, 0, +1
    pl.Series('ml_confidence', model.predict_proba(X).max(axis=1)),  # 0.0 to 1.0
])

feed = DataFeed(prices_df=prices_df, signals_df=signals_df)
engine = Engine(feed, ConfidenceWeightedStrategy(), initial_cash=100000)
result = engine.run()
```

**Advantages:**
- Confidence flows naturally through `signals_df`
- Position sizing logic is explicit
- Easy to experiment with different scaling functions
- Works with any ML model output

---

## Differentiator 4: Cash vs Margin Account Policies

**The Problem**: Different account types have different rules:
- **Cash account**: Can't short, must have full cash for purchases
- **Margin account**: Can short, can use leverage, margin requirements apply

### VectorBT Approach (No Native Support)

```python
# VectorBT has no concept of account type
# You need to manually filter signals to prevent shorts in "cash mode"
# And manually track margin... defeating the purpose of vectorization

entries = signals > 0  # Long only for "cash account"
# shorts = signals < 0  # Can't do this in cash mode

# No way to enforce cash constraints or margin requirements
```

### Backtrader Approach (Limited)

```python
# Backtrader has some broker settings but limited
cerebro.broker.set_shortcash(False)  # Disable shorting
# But margin calculations are basic
```

### ml4t.backtest Approach (Policy-Based)

```python
from ml4t.backtest import Engine, DataFeed, Strategy

# Cash Account - no shorting, full cash required
engine_cash = Engine(
    feed, strategy,
    initial_cash=100000,
    account_type='cash',  # Rejects short orders automatically
)

# Margin Account - shorting allowed, margin requirements
engine_margin = Engine(
    feed, strategy,
    initial_cash=100000,
    account_type='margin',
    initial_margin=0.5,       # 50% initial margin
    maintenance_margin=0.25,  # 25% maintenance margin
)

# Orders are validated through Gatekeeper:
# - Cash account: Rejects shorts, validates sufficient cash
# - Margin account: Validates margin requirements

class LongShortStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            signal = bar['signals'].get('direction', 0)
            pos = broker.get_position(asset)

            if signal > 0 and (pos is None or pos.quantity < 0):
                # Go long
                if pos and pos.quantity < 0:
                    broker.close_position(asset)  # Close short first
                qty = int(broker.get_cash() * 0.5 / bar['close'])
                if qty > 0:
                    broker.submit_order(asset, qty)

            elif signal < 0 and (pos is None or pos.quantity > 0):
                # Go short - will be rejected in cash account!
                if pos and pos.quantity > 0:
                    broker.close_position(asset)  # Close long first
                qty = int(broker.get_cash() * 0.5 / bar['close'])
                if qty > 0:
                    broker.submit_order(asset, -qty)  # Negative = short
```

**Advantages:**
- Account type is a configuration, not code changes
- Orders validated automatically
- Same strategy code works for both account types
- Realistic margin calculations

---

## Differentiator 5: Futures with Contract Multipliers

**The Problem**: Futures have contract multipliers (ES = $50 per point). P&L must account for this.

### VectorBT Approach (Manual)

```python
# VectorBT doesn't know about multipliers
# You need to manually adjust returns
multiplier = 50  # ES futures

pf = vbt.Portfolio.from_signals(close=es_prices, entries=entries, exits=exits)

# Returns are wrong! Need to adjust
# But Portfolio doesn't expose easy way to do this
# Would need to use from_orders() with custom sizing
```

### ml4t.backtest Approach (ContractSpec)

```python
from ml4t.backtest import Engine, Strategy, ContractSpec, AssetClass

# Define contract specification
es_spec = ContractSpec(
    symbol='ES',
    asset_class=AssetClass.FUTURE,
    multiplier=50.0,      # $50 per point
    tick_size=0.25,       # Minimum price increment
    margin_requirement=0.05,  # 5% margin
    exchange='CME',
)

# Engine uses multiplier automatically
engine = Engine(
    feed, strategy,
    initial_cash=100000,
    account_type='margin',
    contract_specs={'ES': es_spec},
)

# Position P&L calculated correctly:
# P&L = (exit_price - entry_price) * quantity * multiplier
```

**Or use the built-in registry:**

```python
from ml4t.data import load_contract_specs

# Load standard futures specs
specs = load_contract_specs(['ES', 'NQ', 'CL', 'GC'])

engine = Engine(
    feed, strategy,
    initial_cash=100000,
    account_type='margin',
    contract_specs=specs,
)
```

**Advantages:**
- P&L calculations are correct automatically
- Tick size can be used for price validation
- Exchange metadata available
- 34 common futures pre-defined in registry

---

## Differentiator 6: Position-Aware Exit Rules

**The Problem**: Exit logic that depends on position state:
- Water marks (highest/lowest price since entry)
- Maximum adverse excursion (worst drawdown)
- Entry time (time-based exits)

### VectorBT Approach (Not Possible)

```python
# VectorBT computes exits before knowing position details
# Cannot reference:
# - Actual entry price (only signal price)
# - Bars held
# - High water mark since entry
# - Entry time
```

### ml4t.backtest Approach (Position State)

```python
from ml4t.backtest import Strategy

class AdvancedExitStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            pos = broker.get_position(asset)
            if pos is None:
                # Entry logic...
                continue

            # Position provides rich state
            entry_price = pos.entry_price
            entry_time = pos.entry_time
            bars_held = pos.bars_held
            high_water_mark = pos.high_water_mark  # Highest price since entry
            low_water_mark = pos.low_water_mark    # Lowest price since entry

            price = bar['close']

            # Advanced exit conditions

            # 1. Trailing stop from high water mark
            trailing_stop = high_water_mark * 0.95
            if price < trailing_stop:
                broker.close_position(asset)
                continue

            # 2. Maximum adverse excursion limit
            max_drawdown = (high_water_mark - low_water_mark) / high_water_mark
            if max_drawdown > 0.10:  # 10% drawdown from peak
                broker.close_position(asset)
                continue

            # 3. Time decay exit
            if bars_held > 20:
                # Tighten stop as time passes
                time_adjusted_stop = entry_price * (1 + 0.01 * (bars_held - 20))
                if price < time_adjusted_stop:
                    broker.close_position(asset)
                    continue
```

**Advantages:**
- Full position history available
- Can implement sophisticated exit logic
- Water marks tracked automatically
- Entry time enables time-based logic

---

## Summary

ml4t.backtest excels at scenarios that require:

1. **Event-driven logic** - Decisions that depend on position state
2. **Dynamic behavior** - Rules that change based on market conditions
3. **ML integration** - Natural flow of model predictions to execution
4. **Account realism** - Proper handling of cash, margin, and constraints
5. **Derivatives** - Futures with multipliers and contract specifications

For simple vectorized backtests with fixed rules, VectorBT is faster. But for anything involving:
- Position-dependent exits
- Dynamic sizing
- Complex rule combinations
- Realistic account simulation

...ml4t.backtest provides the flexibility and correctness you need.
