# Migration Guide to QEngine

## Philosophy

QEngine does **not** provide drop-in compatibility with other frameworks. Instead, we offer:
- Clear migration patterns
- Helper utilities for common tasks
- Excellent documentation
- A superior developer experience that makes the migration worthwhile

## Quick Comparison

| Feature | Zipline | Backtrader | QEngine |
|---------|---------|------------|---------|
| Strategy Method | `handle_data()` | `next()` | `on_event()` |
| Data Access | `data.current()` | `self.data[0]` | `pit_data.get_latest()` |
| Order Submission | `order_target_percent()` | `self.buy()` | `broker.submit_order()` |
| Indicators | Pipeline factors | Built-in indicators | Pluggable indicators |
| Performance | Moderate | Slow | Fast (Polars + Numba) |

## Migrating from Zipline

### Basic Strategy Structure

**Zipline:**
```python
def initialize(context):
    context.asset = symbol('AAPL')
    context.short_ma = 50
    context.long_ma = 200

def handle_data(context, data):
    short_mavg = data.history(
        context.asset, 'price',
        context.short_ma, '1d'
    ).mean()

    long_mavg = data.history(
        context.asset, 'price',
        context.long_ma, '1d'
    ).mean()

    if short_mavg > long_mavg:
        order_target_percent(context.asset, 1.0)
    else:
        order_target_percent(context.asset, 0.0)
```

**QEngine:**
```python
class MovingAverageCrossover(Strategy):
    def on_start(self):
        self.asset = AssetId('AAPL')
        self.short_ma = SMA(50)
        self.long_ma = SMA(200)
        self.subscribe(asset=self.asset, event_type=EventType.MARKET)

    def on_event(self, event: Event, pit_data: PITData):
        if isinstance(event, MarketEvent):
            # Update indicators
            price = event.close
            short_val = self.short_ma.update(price)
            long_val = self.long_ma.update(price)

            # Generate signals
            if short_val > long_val:
                self.broker.target_position(self.asset, 1.0)
            else:
                self.broker.target_position(self.asset, 0.0)
```

### Key Differences

1. **No context object** - State is managed in class attributes
2. **Explicit event handling** - Clear what type of data you're processing
3. **PIT data access** - All historical data through `pit_data` object
4. **Type safety** - Full type hints for better IDE support

## Migrating from Backtrader

### Basic Strategy Structure

**Backtrader:**
```python
class SmaCross(bt.Strategy):
    params = (
        ('period1', 50),
        ('period2', 200),
    )

    def __init__(self):
        self.sma1 = bt.indicators.SMA(period=self.params.period1)
        self.sma2 = bt.indicators.SMA(period=self.params.period2)
        self.crossover = bt.indicators.CrossOver(self.sma1, self.sma2)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()
```

**QEngine:**
```python
class SmaCross(Strategy):
    def __init__(self, period1: int = 50, period2: int = 200):
        super().__init__()
        self.period1 = period1
        self.period2 = period2

    def on_start(self):
        self.sma1 = SMA(self.period1)
        self.sma2 = SMA(self.period2)
        self.position_open = False

    def on_event(self, event: Event, pit_data: PITData):
        if isinstance(event, MarketEvent):
            # Update indicators
            val1 = self.sma1.update(event.close)
            val2 = self.sma2.update(event.close)

            # Check crossover
            if not self.position_open and val1 > val2:
                self.broker.submit_order(
                    OrderEvent(
                        asset_id=event.asset_id,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        quantity=100
                    )
                )
                self.position_open = True
            elif self.position_open and val1 < val2:
                self.broker.close_position(event.asset_id)
                self.position_open = False
```

### Key Differences

1. **No metaclass magic** - Simple Python classes
2. **Explicit configuration** - Parameters as constructor args
3. **Clear event flow** - You control when indicators update
4. **Modern Python** - Type hints, dataclasses, enums

## Common Patterns

### Pattern 1: Data Access

```python
# Zipline
price = data.current(asset, 'price')
history = data.history(asset, 'close', 100, '1d')

# Backtrader
price = self.data.close[0]
history = self.data.close.get(size=100)

# QEngine
price = pit_data.get_latest_price(asset)
history = pit_data.get_history(asset, 'close', 100, '1d')
```

### Pattern 2: Order Management

```python
# Zipline
order_target_percent(asset, 0.5)
order_target_value(asset, 10000)

# Backtrader
self.order_target_percent(target=0.5)
self.order_target_value(target=10000)

# QEngine
self.broker.target_position(asset, 0.5)
self.broker.target_value(asset, 10000)
```

### Pattern 3: Portfolio Information

```python
# Zipline
positions = context.portfolio.positions
cash = context.portfolio.cash

# Backtrader
positions = self.broker.positions
cash = self.broker.cash

# QEngine
positions = self.broker.get_positions()
cash = self.broker.get_cash()
```

## Migration Helpers

QEngine provides utilities to ease migration:

```python
from qengine.migration import MigrationHelper

# Convert common patterns
helper = MigrationHelper()

# Map Zipline-style data access
def zipline_style_history(pit_data, asset, field, periods):
    return helper.get_history_like_zipline(
        pit_data, asset, field, periods
    )

# Map Backtrader-style indicators
sma = helper.wrap_backtrader_indicator(bt.indicators.SMA, period=50)
```

## Step-by-Step Migration Process

### 1. Analyze Your Strategy
- List all data sources used
- Identify order types needed
- Document indicator dependencies
- Note any custom logic

### 2. Set Up QEngine Structure
```python
class YourStrategy(Strategy):
    def __init__(self, **params):
        super().__init__()
        # Store parameters

    def on_start(self):
        # Initialize indicators
        # Subscribe to data

    def on_event(self, event, pit_data):
        # Core logic here
```

### 3. Port Indicators
- Use built-in QEngine indicators where available
- Wrap existing indicators with helpers
- Implement custom indicators as needed

### 4. Convert Order Logic
- Map order types to QEngine equivalents
- Update position sizing logic
- Adjust for QEngine's broker interface

### 5. Test Thoroughly
- Start with simple test cases
- Compare results with original
- Validate performance metrics
- Check for data leakage

## Why Migrate?

### Performance
- 10-100x faster than pure Python frameworks
- Efficient memory usage with Polars
- JIT compilation for hot paths

### Correctness
- Guaranteed PIT data access
- No data leakage by design
- Comprehensive testing framework

### Modern Development
- Type hints throughout
- Async support for live trading
- Clean, maintainable code
- Excellent debugging tools

### ML Integration
- First-class support for ML signals
- Built-in feature engineering
- Integrated with qfeatures/qeval

## Getting Help

### Resources
- [API Documentation](../api/)
- [Example Strategies](../../examples/)
- [GitHub Discussions](https://github.com/quantlab/qengine/discussions)
- [Discord Community](#)

### Common Issues

**Issue**: "My indicators don't match"
- Check data alignment
- Verify warmup periods
- Compare calculation methods

**Issue**: "Orders not executing as expected"
- Verify market hours
- Check order types
- Review fill models

**Issue**: "Performance worse than expected"
- Profile with py-spy
- Check for unnecessary data copies
- Optimize indicator calculations

## Conclusion

Migration to QEngine requires some effort, but the benefits are substantial:
- Better performance
- Stronger correctness guarantees
- Modern development experience
- Active community and support

Start with a simple strategy, get comfortable with the patterns, then migrate your complex strategies. The QEngine team and community are here to help!
