# Migration Guide: From Basic Setup to Integrated ML + Risk Architecture

**Version**: 1.0.0
**Last Updated**: November 2025
**Target Audience**: Existing ml4t.backtest users migrating to integrated architecture
**Prerequisite Reading**: `risk_management_quickstart.md`, `data_architecture.md`, `integrated_ml_risk.md`

---

## Table of Contents

1. [Overview](#overview)
2. [Migration Path Options](#migration-path-options)
3. [Migration 1: Data Layer (Parquet/CSV → PolarsDataFeed)](#migration-1-data-layer-parquetcsv--polarsdatafeed)
4. [Migration 2: Strategy Pattern (Per-Event → Batch Mode)](#migration-2-strategy-pattern-per-event--batch-mode)
5. [Migration 3: Risk Management (Bracket Orders → RiskManager)](#migration-3-risk-management-bracket-orders--riskmanager)
6. [Migration 4: Feature Integration (In-Strategy → FeatureProvider)](#migration-4-feature-integration-in-strategy--featureprovider)
7. [Performance Tuning Checklist](#performance-tuning-checklist)
8. [Common Migration Issues](#common-migration-issues)
9. [Backward Compatibility](#backward-compatibility)
10. [Success Stories](#success-stories)

---

## Overview

### What's Changing?

ml4t.backtest's integrated ML + Risk architecture introduces a unified data model where:

**Old Approach (Basic Setup)**:
- Custom CSV/DataFrame loading
- Features calculated in strategy `on_market_data()`
- Bracket orders for risk management
- Per-event processing (5-10k events/sec)
- Memory inefficient for 100+ assets

**New Approach (Integrated)**:
- `PolarsDataFeed` with lazy loading
- `FeatureProvider` serves features to both strategy AND risk
- Declarative `RiskManager` with composable rules
- Batch processing for multi-asset (50-150k events/sec)
- Shared context for 500+ assets (250x memory reduction)

### Should You Migrate?

**Migrate if you need:**
- ✅ Multi-asset strategies (100+ symbols)
- ✅ ML predictions + technical indicators
- ✅ Declarative risk management (stop-loss, trailing, time-based)
- ✅ Performance (10-100x faster data loading)
- ✅ Memory efficiency (trading 500+ stocks)
- ✅ Production path (backtest → live trading)

**Stay with basic setup if:**
- ❌ Single-asset, simple strategies
- ❌ No ML or complex features
- ❌ Small datasets (<1000 bars)
- ❌ Learning/prototyping only

### Migration Time Estimate

| Scenario | Time | Complexity |
|----------|------|------------|
| Data layer only (CSV → Polars) | 1-2 hours | Low |
| Add FeatureProvider (no ML) | 2-4 hours | Medium |
| Add RiskManager (replace brackets) | 2-3 hours | Medium |
| Full migration (all components) | 1-2 days | High |
| Multi-asset batch mode | 3-5 days | High |

### Incremental Adoption

**You don't need to migrate everything at once!** The architecture supports incremental adoption:

1. **Phase 1**: Migrate data layer (PolarsDataFeed) → 10x faster loading
2. **Phase 2**: Add FeatureProvider → share features across strategy/risk
3. **Phase 3**: Add RiskManager → declarative risk rules
4. **Phase 4**: Batch mode → multi-asset efficiency

Each phase provides value independently.

---

## Migration Path Options

### Path A: Incremental (Recommended)

**Best for**: Production strategies, gradual rollout, risk-averse teams

```
Week 1: Data Layer
  ├─ Replace CSV loading with PolarsDataFeed
  ├─ Validate output matches old approach
  └─ Measure performance improvement

Week 2: FeatureProvider
  ├─ Move feature calculation to PrecomputedFeatureProvider
  ├─ Access features via event.signals
  └─ Validate results unchanged

Week 3: RiskManager
  ├─ Replace bracket orders with declarative rules
  ├─ Test rule behavior matches old logic
  └─ Monitor exit attribution

Week 4: Batch Mode (if multi-asset)
  ├─ Refactor strategy to process timestamp groups
  └─ Optimize for 500+ assets
```

**Pros**: Low risk, incremental validation, learn as you go
**Cons**: Takes longer, requires multiple testing cycles

### Path B: Big Bang

**Best for**: New strategies, greenfield projects, small codebases

```
Day 1: Architecture Design
  ├─ Design FeatureProvider schema
  ├─ Identify all features (ML + indicators)
  └─ Plan risk rules

Day 2-3: Implementation
  ├─ Create PolarsDataFeed with FeatureProvider
  ├─ Implement RiskManager with rules
  ├─ Refactor strategy to batch mode
  └─ Wire everything together

Day 4: Validation
  ├─ Compare results to old implementation
  └─ Performance testing
```

**Pros**: Fastest path to full benefits, clean architecture
**Cons**: Higher risk, harder to debug if issues arise

### Path C: Hybrid (Pragmatic)

**Best for**: Complex strategies with tight deadlines

```
Phase 1 (Week 1): Critical Path
  ├─ Migrate data layer (biggest performance win)
  ├─ Add FeatureProvider (enables risk integration)
  └─ Keep old strategy logic temporarily

Phase 2 (Week 2-3): Risk Layer
  ├─ Add RiskManager (parallel with old bracket logic)
  ├─ Shadow mode: compare decisions
  └─ Switch over when validated

Phase 3 (Future): Batch Mode
  ├─ Defer multi-asset optimization
  └─ Migrate when scaling needs arise
```

**Pros**: Balanced risk/reward, can defer complex parts
**Cons**: May carry technical debt longer

---

## Migration 1: Data Layer (Parquet/CSV → PolarsDataFeed)

### Before: Custom CSV Loading

```python
# OLD APPROACH: Manual CSV loading
import pandas as pd
from ml4t.backtest import BacktestEngine, Strategy

# Load data manually
df = pd.read_csv("market_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Custom data feed wrapper
class PandasDataFeed:
    def __init__(self, df):
        self.df = df
        self.index = 0

    def __iter__(self):
        for _, row in self.df.iterrows():  # SLOW: 5-10k rows/sec
            yield MarketEvent(
                timestamp=row.name,
                asset_id='AAPL',
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
            )

# Run backtest
feed = PandasDataFeed(df)
engine = BacktestEngine(data_feed=feed, strategy=MyStrategy(), ...)
results = engine.run()
```

**Performance**: ~5-10k events/sec
**Memory**: Entire DataFrame in memory (~500MB for 1M rows)
**Issues**: No lazy loading, slow iteration, no feature integration

### After: PolarsDataFeed

```python
# NEW APPROACH: PolarsDataFeed with lazy loading
from pathlib import Path
from ml4t.backtest import BacktestEngine
from ml4t.backtest.data import PolarsDataFeed

# Create data feed (lazy - no data loaded yet!)
feed = PolarsDataFeed(
    price_path=Path("market_data.parquet"),  # Or .csv
    asset_id='AAPL',
)
# Memory usage: <1 MB (just schema)

# Run backtest (data loaded on first iteration)
engine = BacktestEngine(data_feed=feed, strategy=MyStrategy(), ...)
results = engine.run()
```

**Performance**: ~50-150k events/sec (10-15x faster)
**Memory**: Lazy loading (~50MB initial, grows as needed)
**Benefits**: Automatic optimization, compression support, no code changes to strategy

### Migration Steps

**Step 1: Convert CSV to Parquet (Optional but Recommended)**

```python
import polars as pl

# One-time conversion
df = pl.read_csv("market_data.csv")
df.write_parquet("market_data.parquet", compression="zstd")
# Result: 5-10x smaller file, 10-50x faster reads
```

**Step 2: Replace DataFeed**

```python
# Before
feed = PandasDataFeed(df)

# After
feed = PolarsDataFeed(price_path=Path("market_data.parquet"), asset_id='AAPL')
```

**Step 3: Validate Output**

```python
# Run both old and new, compare results
old_results = run_with_pandas_feed()
new_results = run_with_polars_feed()

assert abs(old_results['final_equity'] - new_results['final_equity']) < 0.01
print("✅ Migration validated - results match!")
```

### Data Schema Requirements

PolarsDataFeed expects this schema:

```python
# Required columns
price_df = pl.DataFrame({
    'timestamp': [...],  # datetime64[ns] or datetime
    'asset_id': [...],   # str (if multi-asset, else use constructor arg)
    'open': [...],       # float64
    'high': [...],       # float64
    'low': [...],        # float64
    'close': [...],      # float64
    'volume': [...],     # int64 or float64
})
```

**If your CSV has different column names:**

```python
# Rename during load
df = pl.read_csv("data.csv").rename({
    'Date': 'timestamp',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume',
})
df.write_parquet("data.parquet")
```

### Expected Performance Gains

| Dataset | Old (Pandas) | New (Polars) | Speedup |
|---------|-------------|--------------|---------|
| 10k rows | 2 sec | 0.2 sec | 10x |
| 100k rows | 20 sec | 1 sec | 20x |
| 1M rows | 200 sec | 5 sec | 40x |
| 10 symbols, 252 days | 5 sec | 0.3 sec | 16x |

---

## Migration 2: Strategy Pattern (Per-Event → Batch Mode)

### Before: Per-Event Processing

```python
# OLD APPROACH: Process one event at a time
class SimpleStrategy(Strategy):
    def on_market_data(self, event, context=None):
        # Calculate features inline
        if len(self.price_history) < 20:
            self.price_history.append(event.close)
            return

        # Compute indicator
        sma_20 = sum(self.price_history[-20:]) / 20

        # Entry logic
        position = self.get_position(event.asset_id)
        if position == 0 and event.close > sma_20:
            self.buy_percent(event.asset_id, 0.95)

        # Exit logic
        elif position > 0 and event.close < sma_20:
            self.close_position(event.asset_id)
```

**Issues**:
- Features calculated per-event (wasteful if also used by risk)
- Hard to share context across multiple assets
- Sequential processing (can't batch)

### After: Batch Mode with FeatureProvider

```python
# NEW APPROACH: Features from provider, batch processing
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider

class BatchStrategy(Strategy):
    def on_market_data(self, event, context=None):
        # Access precomputed features (no calculation needed!)
        sma_20 = event.signals.get('sma_20', event.close)

        # Entry logic (same as before)
        position = self.get_position(event.asset_id)
        if position == 0 and event.close > sma_20:
            self.buy_percent(event.asset_id, 0.95)

        # Exit logic handled by RiskManager (see Migration 3)
```

**Benefits**:
- Features computed once, used by strategy AND risk
- Same API (event.signals), cleaner code
- Enables batch processing for multi-asset

### Migration Steps

**Step 1: Extract Feature Calculation**

Move feature computation from strategy to separate Polars pipeline:

```python
# Before: In strategy __init__ or on_market_data()
self.price_history = []
sma_20 = sum(self.price_history[-20:]) / 20

# After: Precompute in data preparation
import polars as pl

price_df = pl.read_parquet("prices.parquet")
features_df = price_df.with_columns([
    # Compute SMA using Polars (vectorized, fast!)
    pl.col('close').rolling_mean(window_size=20).alias('sma_20'),
])
features_df.write_parquet("features.parquet")
```

**Step 2: Create FeatureProvider**

```python
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider

# Load features
features_df = pl.read_parquet("features.parquet")

# Create provider
provider = PrecomputedFeatureProvider(
    features_df=features_df,
    timestamp_col='timestamp',
    asset_col='asset_id',
)
```

**Step 3: Wire to DataFeed**

```python
# Attach provider to data feed
feed = PolarsDataFeed(
    price_path=Path("prices.parquet"),
    asset_id='AAPL',
    feature_provider=provider,  # Features auto-populated in event.signals
)
```

**Step 4: Update Strategy**

```python
# Before
sma_20 = sum(self.price_history[-20:]) / 20

# After
sma_20 = event.signals.get('sma_20', event.close)  # Fallback to close if missing
```

### Multi-Asset Batch Processing

If trading 100+ symbols, process all events at each timestamp together:

```python
# OLD: Per-event (inefficient for multi-asset)
def on_market_data(self, event, context=None):
    # Process one asset at a time
    pass

# NEW: Batch mode (process all assets per timestamp)
def process_timestamp_batch(self, events, broker, portfolio):
    """Process all events at one timestamp together.

    Args:
        events: List[MarketEvent] - All events at this timestamp
        broker: SimulationBroker
        portfolio: Portfolio
    """
    # Get market context once (shared across all assets)
    vix = events[0].context.get('vix', 0.0) if events else 0.0

    # VIX filter (skip entire timestamp)
    if vix > 30:
        return

    # Rank all assets by ML score
    asset_scores = [
        (event.asset_id, event.signals['ml_score'], event.close)
        for event in events
    ]
    asset_scores.sort(key=lambda x: x[1], reverse=True)

    # Select top N
    top_n = asset_scores[:25]

    # Equal-weight allocation
    for asset_id, ml_score, price in top_n:
        self.set_position_percent(asset_id, 0.04)  # 4% per position
```

**When to use batch mode:**
- Trading 10+ assets
- Ranking/selection strategies (top-N by ML score)
- Market regime filters (VIX, breadth)
- Portfolio rebalancing

---

## Migration 3: Risk Management (Bracket Orders → RiskManager)

### Before: Bracket Orders in Strategy

```python
# OLD APPROACH: Manual bracket order management
class StrategyWithBrackets(Strategy):
    def on_market_data(self, event, context=None):
        position = self.get_position(event.asset_id)

        # Entry with bracket orders
        if position == 0 and self.entry_condition(event):
            entry_price = event.close
            stop_loss = entry_price * 0.95  # 5% stop
            take_profit = entry_price * 1.10  # 10% profit

            # Submit main order
            self.buy_percent(event.asset_id, 0.95)

            # Submit bracket orders (attached to fill)
            # NOTE: This is pseudocode - actual bracket order API varies
            self.submit_bracket_orders(
                asset_id=event.asset_id,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        # Manual exit logic (if no brackets)
        elif position > 0:
            pnl_pct = self.get_unrealized_pnl_pct(event.asset_id)
            if pnl_pct <= -0.05 or pnl_pct >= 0.10:
                self.close_position(event.asset_id)
```

**Issues**:
- Risk logic mixed with strategy logic (hard to test/reuse)
- Manual P&L tracking
- No support for volatility-scaled stops (ATR-based)
- Can't easily combine multiple risk rules

### After: Declarative RiskManager

```python
# NEW APPROACH: Separate risk management
from ml4t.backtest.risk import RiskManager, VolatilityScaledStopLoss, PercentageBasedExit

# Create risk manager
risk_manager = RiskManager(feature_provider=provider)

# Add rules (composable!)
risk_manager.add_rule(
    VolatilityScaledStopLoss(
        atr_multiplier=2.0,
        volatility_key='atr_20',  # From FeatureProvider
    )
)
risk_manager.add_rule(
    PercentageBasedExit(
        take_profit_pct=0.10,  # 10% profit target
    )
)

# Wire to engine
engine = BacktestEngine(
    data_feed=feed,
    strategy=CleanStrategy(),  # No exit logic needed!
    broker=broker,
    risk_manager=risk_manager,
    ...
)

# Strategy focuses on ENTRY only
class CleanStrategy(Strategy):
    def on_market_data(self, event, context=None):
        ml_score = event.signals.get('ml_score', 0.0)

        position = self.get_position(event.asset_id)
        if position == 0 and ml_score > 0.8:
            self.buy_percent(event.asset_id, 0.95)

        # Exit handled automatically by RiskManager!
```

**Benefits**:
- Separation of concerns (strategy = entry, risk = exit)
- Reusable rules across strategies
- Declarative (easy to test/audit)
- Combines multiple rules with automatic conflict resolution
- Volatility-scaled stops (adapts to market conditions)

### Migration Steps

**Step 1: Identify Exit Logic**

Extract all exit conditions from your strategy:

```python
# Find patterns like:
if pnl_pct <= -0.05:          # → PriceBasedStopLoss or PercentageBasedExit
if pnl_pct >= 0.10:           # → PercentageBasedExit(take_profit_pct=0.10)
if bars_held > 60:            # → TimeBasedExit(max_bars=60)
if event.close < stop_price:  # → PriceBasedStopLoss or VolatilityScaledStopLoss
```

**Step 2: Map to Built-In Rules**

| Old Pattern | New Rule | Example |
|-------------|----------|---------|
| Fixed % stop | `PercentageBasedExit` | `PercentageBasedExit(stop_loss_pct=-0.05)` |
| Fixed price stop | `PriceBasedStopLoss` | `PriceBasedStopLoss(stop_loss_price=95.0)` |
| Time-based exit | `TimeBasedExit` | `TimeBasedExit(max_bars=60)` |
| ATR-based stop | `VolatilityScaledStopLoss` | `VolatilityScaledStopLoss(atr_multiplier=2.0)` |
| Trailing stop | `DynamicTrailingStop` | `DynamicTrailingStop(initial_trail_pct=0.05)` |

**Step 3: Create RiskManager**

```python
from ml4t.backtest.risk import RiskManager

# Initialize with feature provider (for ATR, volatility, etc.)
risk_manager = RiskManager(feature_provider=provider)

# Add rules (order doesn't matter - priority determines execution)
risk_manager.add_rule(VolatilityScaledStopLoss(atr_multiplier=2.0, volatility_key='atr_20'))
risk_manager.add_rule(TimeBasedExit(max_bars=60))
```

**Step 4: Remove Exit Logic from Strategy**

```python
# Before
def on_market_data(self, event, context=None):
    # Entry
    if position == 0 and self.entry_condition(event):
        self.buy_percent(event.asset_id, 0.95)

    # Exit (DELETE THIS)
    # elif position > 0:
    #     pnl_pct = self.get_unrealized_pnl_pct(event.asset_id)
    #     if pnl_pct <= -0.05:
    #         self.close_position(event.asset_id)

# After
def on_market_data(self, event, context=None):
    # Entry only
    if position == 0 and self.entry_condition(event):
        self.buy_percent(event.asset_id, 0.95)

    # Exit handled by RiskManager!
```

**Step 5: Validate Exit Behavior**

```python
# Compare old vs new results
old_trades = run_with_bracket_orders()
new_trades = run_with_risk_manager()

# Check exit counts match
assert len(old_trades) == len(new_trades)

# Check exit reasons are similar
for old_trade, new_trade in zip(old_trades, new_trades):
    assert abs(old_trade['exit_price'] - new_trade['exit_price']) < 0.01
```

### Custom Risk Rules

If built-in rules don't cover your logic, create a custom rule:

```python
from ml4t.backtest.risk.base import RiskRule, RiskDecision, RiskContext

class VIXBasedExit(RiskRule):
    """Exit positions when VIX spikes above threshold."""

    def __init__(self, vix_threshold: float = 30.0):
        super().__init__()
        self.vix_threshold = vix_threshold

    def evaluate(self, context: RiskContext) -> RiskDecision:
        # Access market context
        vix = context.market_features.get('vix', 0.0)

        # Exit if VIX too high
        if vix > self.vix_threshold:
            return RiskDecision.exit_now(
                exit_type=ExitType.REGIME_CHANGE,
                reason=f"VIX spike: {vix:.1f} > {self.vix_threshold}",
                priority=50,
            )

        return RiskDecision.no_action()

# Use it
risk_manager.add_rule(VIXBasedExit(vix_threshold=35.0))
```

---

## Migration 4: Feature Integration (In-Strategy → FeatureProvider)

### Before: Features in Strategy

```python
# OLD APPROACH: Features calculated in strategy
import talib

class IndicatorStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.price_history = []

    def on_market_data(self, event, context=None):
        # Track price history
        self.price_history.append(event.close)

        # Compute indicators inline
        if len(self.price_history) >= 20:
            prices = np.array(self.price_history[-20:])

            # ATR (for position sizing)
            atr = talib.ATR(
                np.array(self.high_history[-20:]),
                np.array(self.low_history[-20:]),
                prices,
                timeperiod=14
            )[-1]

            # RSI (for entry filter)
            rsi = talib.RSI(prices, timeperiod=14)[-1]

            # Entry logic
            if rsi < 30 and not self.has_position():
                # Size by ATR (risk 1% of portfolio per ATR)
                size = self.calculate_atr_size(atr)
                self.buy_shares(event.asset_id, size)
```

**Issues**:
- Features computed per-event (slow)
- Not shared with RiskManager (duplication if risk needs ATR)
- Manual state management (price_history)
- Hard to test feature calculation separately

### After: Precomputed Features via FeatureProvider

```python
# NEW APPROACH: Features precomputed and served via provider

# Step 1: Compute features offline (data prep)
import polars as pl
import talib

def prepare_features(price_df: pl.DataFrame) -> pl.DataFrame:
    """Precompute all features using Polars + talib."""

    # Convert to numpy for talib
    close = price_df['close'].to_numpy()
    high = price_df['high'].to_numpy()
    low = price_df['low'].to_numpy()

    # Compute indicators
    atr = talib.ATR(high, low, close, timeperiod=14)
    rsi = talib.RSI(close, timeperiod=14)

    # Add to DataFrame
    features_df = price_df.with_columns([
        pl.Series('atr_14', atr),
        pl.Series('rsi_14', rsi),
    ])

    return features_df

# Prepare once, save
price_df = pl.read_parquet("prices.parquet")
features_df = prepare_features(price_df)
features_df.write_parquet("features.parquet")

# Step 2: Create FeatureProvider
from ml4t.backtest.data.feature_provider import PrecomputedFeatureProvider

features_df = pl.read_parquet("features.parquet")
provider = PrecomputedFeatureProvider(features_df)

# Step 3: Strategy uses features from event.signals
class CleanIndicatorStrategy(Strategy):
    def on_market_data(self, event, context=None):
        # Access precomputed features (no calculation!)
        atr = event.signals.get('atr_14', 1.0)
        rsi = event.signals.get('rsi_14', 50.0)

        # Entry logic (same as before)
        if rsi < 30 and not self.has_position():
            size = self.calculate_atr_size(atr)
            self.buy_shares(event.asset_id, size)

# Step 4: RiskManager uses SAME features
risk_manager = RiskManager(feature_provider=provider)
risk_manager.add_rule(
    VolatilityScaledStopLoss(
        atr_multiplier=2.0,
        volatility_key='atr_14',  # Same ATR as strategy!
    )
)
```

**Benefits**:
- Features computed once (10-100x faster)
- Shared between strategy and risk (no duplication)
- Easy to test (validate features separately)
- Point-in-time correct (timestamps enforced)

### Migration Steps

**Step 1: Audit Feature Calculations**

Identify all features calculated in your strategy:

```python
# Find patterns like:
np.mean(prices[-20:])         # → SMA
talib.RSI(prices, 14)          # → RSI
talib.ATR(high, low, close, 14) # → ATR
ml_model.predict(features)     # → ML predictions
```

**Step 2: Create Feature Computation Script**

```python
# features_pipeline.py
import polars as pl
import talib

def compute_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Compute all technical indicators."""

    # Convert to numpy for talib
    close = df['close'].to_numpy()
    high = df['high'].to_numpy()
    low = df['low'].to_numpy()
    volume = df['volume'].to_numpy()

    # Compute indicators
    indicators = {
        'sma_20': talib.SMA(close, timeperiod=20),
        'rsi_14': talib.RSI(close, timeperiod=14),
        'atr_20': talib.ATR(high, low, close, timeperiod=20),
        'macd': talib.MACD(close)[0],  # Just MACD line
    }

    # Add to DataFrame
    for name, values in indicators.items():
        df = df.with_columns(pl.Series(name, values))

    return df

# Run pipeline
price_df = pl.read_parquet("prices.parquet")
features_df = compute_technical_indicators(price_df)
features_df.write_parquet("features.parquet")
```

**Step 3: Add ML Predictions (If Applicable)**

```python
def add_ml_predictions(df: pl.DataFrame, model) -> pl.DataFrame:
    """Add ML model predictions as features."""

    # Extract features for ML model
    feature_cols = ['sma_20', 'rsi_14', 'atr_20', 'macd']
    X = df.select(feature_cols).to_numpy()

    # Predict
    predictions = model.predict(X)
    confidences = model.predict_proba(X)[:, 1]  # Probability of class 1

    # Add to DataFrame
    df = df.with_columns([
        pl.Series('ml_pred', predictions),
        pl.Series('ml_confidence', confidences),
    ])

    return df

# Run with ML
features_df = add_ml_predictions(features_df, trained_model)
features_df.write_parquet("features_with_ml.parquet")
```

**Step 4: Replace In-Strategy Calculations**

```python
# Before
atr = talib.ATR(self.high_history, self.low_history, self.close_history, 20)[-1]

# After
atr = event.signals.get('atr_20', 1.0)  # Precomputed!
```

**Step 5: Share with RiskManager**

```python
# Both strategy and risk use same provider
provider = PrecomputedFeatureProvider(features_df)

feed = PolarsDataFeed(price_path, feature_provider=provider)
risk_manager = RiskManager(feature_provider=provider)

# Now both access identical features!
```

### Market Context Integration

If you use market-wide indicators (VIX, SPY, breadth):

```python
# Prepare market features separately
market_df = pl.DataFrame({
    'timestamp': [...],
    'asset_id': [None, None, ...],  # CRITICAL: None for market-wide
    'vix': [...],
    'spy_return': [...],
})

# Combine with per-asset features
# (Need matching schemas - add null columns)
per_asset_df = per_asset_df.with_columns([
    pl.lit(None).cast(pl.Float64).alias('vix'),
    pl.lit(None).cast(pl.Float64).alias('spy_return'),
])

market_df = market_df.with_columns([
    pl.lit(None).cast(pl.Float64).alias('atr_20'),
    pl.lit(None).cast(pl.Float64).alias('rsi_14'),
])

# Concatenate
combined = pl.concat([per_asset_df, market_df])

# Create provider
provider = PrecomputedFeatureProvider(combined)

# Access in strategy
def on_market_data(self, event, context=None):
    # Per-asset features
    atr = event.signals.get('atr_20', 1.0)

    # Market context
    vix = event.context.get('vix', 15.0)

    # VIX filter
    if vix > 30:
        return  # Don't trade in high volatility
```

---

## Performance Tuning Checklist

After migration, optimize for production:

### Data Layer

- [ ] **Use Parquet instead of CSV** (5-10x smaller, 10-50x faster reads)
- [ ] **Enable compression** (`write_parquet(..., compression='zstd')`)
- [ ] **Use categorical encoding** for asset_id if 100+ symbols
- [ ] **Lazy loading enabled** (default in PolarsDataFeed)
- [ ] **Chunk size tuned** for your dataset (default usually optimal)

### Feature Computation

- [ ] **Precompute all features** (don't calculate in strategy)
- [ ] **Use Polars for indicators** when possible (faster than talib)
- [ ] **Cache ML predictions** (inference is expensive)
- [ ] **Vectorize feature calculation** (avoid loops)
- [ ] **Profile feature pipeline** (identify bottlenecks)

### RiskManager

- [ ] **Context caching enabled** (automatic, but verify)
- [ ] **Clear cache periodically** (every day or N bars)
- [ ] **Minimize rule count** (3-5 rules is typical)
- [ ] **Use priorities** to avoid redundant checks
- [ ] **Profile rule evaluation** (ensure <1ms per position)

### Multi-Asset Strategies

- [ ] **Batch process timestamps** (process all assets together)
- [ ] **Share market context** (don't duplicate VIX 500 times)
- [ ] **Use group_by optimization** (automatic in PolarsDataFeed)
- [ ] **Limit universe size** (top 500-1000 stocks max)
- [ ] **Profile memory usage** (should be <2GB for 250 symbols)

### Expected Performance

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Data loading | 20 sec | 1 sec | <2 sec |
| Event throughput | 10k/sec | 100k/sec | >50k/sec |
| Memory (100 symbols) | 5 GB | 500 MB | <1 GB |
| Backtest (1 year, 1 symbol) | 30 sec | 3 sec | <5 sec |
| Backtest (1 year, 500 symbols) | 1 hour | 5 min | <10 min |

---

## Common Migration Issues

### Issue 1: Features Missing in MarketEvent

**Symptom**:
```python
KeyError: 'atr_20'
# or
event.signals.get('rsi_14') returns None
```

**Root Cause**: FeatureProvider not configured or features missing for timestamp.

**Solution**:
```python
# Debug: Check feature availability
timestamp = event.timestamp
asset_id = event.asset_id

features = provider.get_features(asset_id, timestamp)
print(f"Available features: {list(features.keys())}")
print(f"ATR: {features.get('atr_20')}")

# Fix: Use .get() with defaults
atr = event.signals.get('atr_20', 1.0)  # Safe
```

### Issue 2: Results Don't Match After Migration

**Symptom**: Different P&L, trade counts, or final equity.

**Possible Causes**:
1. Look-ahead bias in old implementation
2. Feature calculation differences
3. Different fill prices or commission

**Debugging**:
```python
# Compare trade-by-trade
old_trades = old_results['trades']
new_trades = new_results['trades']

for i, (old, new) in enumerate(zip(old_trades, new_trades)):
    if abs(old['entry_price'] - new['entry_price']) > 0.01:
        print(f"Trade {i}: Entry price mismatch")
        print(f"  Old: {old['entry_price']:.2f}")
        print(f"  New: {new['entry_price']:.2f}")
```

**Common Fixes**:
- Ensure features are aligned by timestamp (no look-ahead)
- Use same commission model
- Verify slippage settings match

### Issue 3: Performance Regression

**Symptom**: New implementation slower than old.

**Possible Causes**:
1. Not using Polars lazy loading
2. Redundant feature computation
3. Cache not enabled

**Solution**:
```python
# Profile to find bottleneck
import cProfile

profiler = cProfile.Profile()
profiler.enable()
results = engine.run()
profiler.disable()

import pstats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Issue 4: Memory Leak

**Symptom**: Memory grows unbounded during backtest.

**Possible Causes**:
1. RiskManager cache not cleared
2. Storing full event history in strategy
3. Large feature DataFrames not garbage collected

**Solution**:
```python
# Clear risk manager cache periodically
if event.timestamp.hour == 16:  # Market close
    risk_manager.clear_cache()

# Don't store full event history
# Before (BAD)
self.event_history.append(event)  # Grows unbounded!

# After (GOOD)
self.last_event = event  # Only store latest
```

### Issue 5: Context Not Shared Across Assets

**Symptom**: VIX different for each asset at same timestamp.

**Root Cause**: Market features not marked with `asset_id=None`.

**Solution**:
```python
# WRONG
market_df = pl.DataFrame({
    'timestamp': [ts1],
    'asset_id': ['SPY'],  # WRONG - should be None
    'vix': [18.5],
})

# CORRECT
market_df = pl.DataFrame({
    'timestamp': [ts1],
    'asset_id': [None],  # Correct - market-wide
    'vix': [18.5],
})
```

---

## Backward Compatibility

### What's Still Supported?

The integrated architecture maintains backward compatibility with:

- ✅ **Old DataFeed interface** - Custom feeds still work
- ✅ **Strategy.on_market_data()** - No API changes
- ✅ **Manual order submission** - `broker.submit_order()` unchanged
- ✅ **Portfolio API** - `get_position()`, `get_equity()` same
- ✅ **CSV/Pandas** - Can still use CSVDataFeed

### What's Deprecated (But Still Works)?

- ⚠️ **ParquetDataFeed** - Use PolarsDataFeed instead (10x faster)
- ⚠️ **In-strategy feature calculation** - Use FeatureProvider
- ⚠️ **Manual bracket orders** - Use RiskManager rules

### What's Not Supported?

- ❌ **Mixing old and new risk patterns** - Don't use both bracket orders AND RiskManager
- ❌ **DataFrame mutation** - FeatureProvider expects immutable data
- ❌ **Non-timestamp indexed data** - Must have timestamp column

### Version Compatibility

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| Python | 3.9 | 3.11+ |
| Polars | 0.19.0 | Latest |
| ml4t.backtest | 0.9.0 | Latest |
| Pandas (optional) | 1.5.0 | 2.0+ |

---

## Success Stories

### Case Study 1: Single-Asset ML Strategy

**Before**:
- Custom CSV loading with pandas
- Features calculated per-event
- Manual stop-loss in strategy
- 30 seconds per backtest

**After**:
- PolarsDataFeed with PrecomputedFeatureProvider
- RiskManager with VolatilityScaledStopLoss
- 3 seconds per backtest (10x faster)
- Cleaner code (50% fewer lines)

**Migration Time**: 4 hours

### Case Study 2: Top-25 Momentum Strategy (500 Stocks)

**Before**:
- Pandas DataFrame for 500 stocks
- VIX embedded in each stock's row (memory waste)
- Per-event processing (5k events/sec)
- 5 GB memory, 45 minutes per backtest

**After**:
- PolarsDataFeed with batch mode
- Shared market context (VIX once per timestamp)
- Batch processing (100k events/sec)
- 500 MB memory, 3 minutes per backtest (15x faster!)

**Migration Time**: 2 days (rewrote for batch mode)

### Case Study 3: Multi-Indicator Strategy

**Before**:
- talib indicators computed per-event
- 10 indicators × 252 days × 50 symbols = 126k calculations
- 15 seconds per backtest

**After**:
- Precomputed indicators in Polars
- 10 indicators × 252 days = 2.5k calculations (once!)
- 1.5 seconds per backtest (10x faster)
- Same indicators used by RiskManager (no duplication)

**Migration Time**: 3 hours

---

## Next Steps

### After Successful Migration

1. **Monitor Performance** - Profile your backtest, ensure targets met
2. **Test Edge Cases** - Validate behavior matches old implementation
3. **Optimize Further** - Use profiling to identify bottlenecks
4. **Share Features** - Leverage FeatureProvider across multiple strategies
5. **Build Risk Library** - Create reusable custom rules for your use cases

### Resources

- **Integration Guide**: `docs/guides/integrated_ml_risk.md` - Full architecture reference
- **Risk Quickstart**: `docs/guides/risk_management_quickstart.md` - Risk rules deep dive
- **Data Architecture**: `docs/guides/data_architecture.md` - Data layer details
- **Reference Example**: `examples/integrated/top25_ml_strategy_complete.py` - Complete working example

### Getting Help

If you encounter migration issues:

1. **Check Common Issues** above
2. **Review reference example** (`top25_ml_strategy_complete.py`)
3. **File GitHub issue** with:
   - Old code snippet
   - New code attempt
   - Error message or unexpected behavior
   - Expected vs actual results

---

## Summary

**Key Migration Benefits**:

| Area | Old Approach | New Approach | Benefit |
|------|-------------|--------------|---------|
| Data Loading | Pandas CSV | Polars Parquet | 10-50x faster |
| Features | Per-event calc | Precomputed | 10-100x faster |
| Risk Management | Manual/Brackets | RiskManager | Reusable, testable |
| Multi-Asset | Sequential | Batch mode | 500x less memory |
| Code Quality | Mixed concerns | Separated | Easier to maintain |

**Migration Effort vs Reward**:

- **Low effort** (1-2 hours): Data layer migration → 10x faster loading
- **Medium effort** (1 day): Add FeatureProvider + RiskManager → Clean architecture
- **High effort** (2-5 days): Full batch mode for multi-asset → 10-100x total speedup

**The integrated architecture is designed for incremental adoption - you don't need to migrate everything at once. Start with the data layer for quick wins, then add components as needed.**

---

**Document Version**: 1.0.0
**Last Updated**: November 2025
**Maintainer**: ml4t.backtest team
**Feedback**: File issues or improvements via project repository
