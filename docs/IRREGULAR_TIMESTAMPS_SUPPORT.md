# Irregular Timestamps Support in QEngine

## Overview

QEngine's event-driven architecture provides **native support for irregular timestamps**, allowing backtesting on any type of bar aggregation including volume bars, dollar bars, information bars, and other non-time-based sampling methods. This capability is a key differentiator that sets QEngine apart from many other backtesting frameworks.

## Why Irregular Timestamps Matter

Traditional backtesting frameworks typically assume regular time intervals (e.g., 1-minute, 5-minute, daily bars). However, financial markets exhibit:

- **Heteroskedastic trading activity**: Volume and volatility vary dramatically throughout the day
- **Microstructure effects**: Information arrival is irregular and event-driven
- **Regime changes**: Market conditions shift based on news, volatility, and flow patterns

Using time-based bars can introduce significant biases:
- Over-sampling during quiet periods
- Under-sampling during high-activity periods
- Structural breaks at market open/close
- Information leakage across regular intervals

## QEngine's Event-Driven Solution

QEngine's core architecture is built around **events**, not time intervals:

```python
from qengine.core.event import Event, MarketEvent
from qengine.core.types import MarketDataType
from datetime import datetime

# Volume bar event - triggers when 10,000 shares traded
volume_bar_event = MarketEvent(
    timestamp=datetime(2024, 3, 15, 9, 47, 23, 145000),  # Irregular timestamp
    asset_id="AAPL",
    data_type=MarketDataType.BAR,
    open=150.25,
    high=150.89,
    low=150.12,
    close=150.67,
    volume=10000,  # Exactly 10,000 shares
    metadata={"bar_type": "volume", "threshold": 10000}
)

# Dollar bar event - triggers when $1M traded
dollar_bar_event = MarketEvent(
    timestamp=datetime(2024, 3, 15, 9, 52, 7, 892000),   # Different irregular timestamp
    asset_id="AAPL",
    data_type=MarketDataType.BAR,
    open=150.67,
    high=150.95,
    low=150.55,
    close=150.88,
    volume=6632,  # Variable volume to reach $1M
    metadata={"bar_type": "dollar", "threshold": 1000000, "dollar_volume": 1000000}
)
```

## Supported Bar Types

QEngine can handle any bar aggregation method supported by qfeatures:

### 1. Volume Bars
```python
# From qfeatures volume bar aggregation
from qfeatures.bars.volume import VolumeBars

volume_bars = VolumeBars(threshold=10000)
irregular_data = volume_bars.transform(tick_data)

# QEngine processes these irregular timestamps naturally
for row in irregular_data.iter_rows(named=True):
    event = MarketEvent(
        timestamp=row["timestamp"],
        asset_id=row["asset_id"],
        open=row["open"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
        volume=row["volume"],
        data_type=MarketDataType.BAR
    )
    engine.process_event(event)
```

### 2. Dollar Bars
```python
# Dollar volume bars from qfeatures
dollar_data = dollar_bars.transform(tick_data)
# QEngine processes seamlessly - timestamps are completely irregular
```

### 3. Information Bars
```python
# Information-driven bars (e.g., based on VPIN, order flow imbalance)
info_data = info_bars.transform(tick_data)
# QEngine handles complex conditional logic for bar formation
```

### 4. Tick Bars
```python
# Fixed number of ticks per bar
tick_data = tick_bars.transform(raw_ticks)
# Completely irregular timing based on market activity
```

## Architecture Benefits

### Event-Driven Clock
```python
from qengine.core.clock import SimulationClock

# QEngine's clock is event-driven, not time-driven
clock = SimulationClock()

# Advances based on event timestamps, regardless of irregularity
events = [
    # Could be volume bars with highly irregular timestamps
    MarketEvent(timestamp=datetime(2024, 3, 15, 9, 32, 1, 123), ...),
    MarketEvent(timestamp=datetime(2024, 3, 15, 9, 47, 53, 892), ...),
    MarketEvent(timestamp=datetime(2024, 3, 15, 10, 3, 12, 445), ...),
]

for event in events:
    clock.advance_to(event.timestamp)  # Handles any timestamp sequence
    # Strategy receives event at exact irregular timing
```

### Point-in-Time Correctness
```python
# Strategy receives data at exact event timestamps
class VolumeBarStrategy(Strategy):
    def on_market_event(self, event: MarketEvent, pit_data):
        # pit_data contains only information available at this irregular timestamp
        # No look-ahead bias regardless of bar timing

        if event.volume > self.volume_threshold:
            # React to high-volume events immediately
            self.submit_order(...)
```

### Realistic Execution Timing
```python
# Orders execute at realistic irregular timestamps
def on_market_event(self, event: MarketEvent, pit_data):
    # Submit order at 9:47:23.145 (irregular volume bar completion)
    order = self.create_order(OrderType.MARKET, ...)
    self.submit_order(order)

    # Execution happens at next available event timestamp
    # Could be 9:52:07.892 (next dollar bar) - completely realistic timing
```

## Integration with QFeatures

QEngine is designed to seamlessly consume irregular data from qfeatures:

```python
import polars as pl
from qfeatures.bars.volume import VolumeBars
from qengine import QEngine
from qengine.data.feed import PolarsDataFeed

# 1. Create irregular bars with qfeatures
volume_bars = VolumeBars(threshold=50000)  # 50k share bars
irregular_df = volume_bars.fit_transform(tick_data)

# 2. Feed directly into QEngine
feed = PolarsDataFeed(irregular_df)
engine = QEngine()
engine.add_data_feed(feed)

# 3. Strategy processes irregular timestamps naturally
class IrregularTimingStrategy(Strategy):
    def on_market_event(self, event, pit_data):
        # Receives events at natural market timing
        # No artificial time grid imposed
        pass

# 4. Backtest with realistic timing
results = engine.run()
```

## Practical Example: Volume Bar Backtesting

```python
"""
Example: Backtesting a momentum strategy on volume bars
This captures market microstructure effects that time bars miss
"""

import polars as pl
from qfeatures.bars.volume import VolumeBars
from qengine import QEngine
from qengine.strategy import Strategy

class VolumeMomentumStrategy(Strategy):
    def __init__(self):
        self.lookback_periods = 20
        self.momentum_threshold = 0.02

    def on_market_event(self, event, pit_data):
        """Strategy triggered by volume bar completion, not time."""

        # Get recent volume bars (irregular timestamps)
        recent_data = pit_data.get_bars(self.lookback_periods)

        if len(recent_data) < self.lookback_periods:
            return

        # Calculate momentum over irregular intervals
        returns = recent_data["close"].pct_change()
        momentum = returns.tail(5).mean()

        # Volume bars naturally capture high-activity periods
        # Strategy reacts to market structure, not clock
        if abs(momentum) > self.momentum_threshold:
            if momentum > 0:
                self.submit_order(OrderSide.BUY, 1000)
            else:
                self.submit_order(OrderSide.SELL, 1000)

# Create volume bars from tick data
tick_data = pl.read_parquet("spy_ticks.parquet")
volume_bars = VolumeBars(threshold=100000)  # 100k share bars
volume_df = volume_bars.fit_transform(tick_data)

print(f"Original ticks: {len(tick_data)} rows")
print(f"Volume bars: {len(volume_df)} rows")
print(f"Compression ratio: {len(tick_data)/len(volume_df):.1f}x")

# Timestamps are highly irregular - more bars during active periods
print("\nSample irregular timestamps:")
for row in volume_df.head(10).iter_rows(named=True):
    print(f"{row['timestamp']} - Volume: {row['volume']:,}")

# Backtest on irregular timestamps
engine = QEngine()
engine.add_strategy(VolumeMomentumStrategy())
engine.add_data_feed(PolarsDataFeed(volume_df))

results = engine.run()
print(f"\nBacktest completed with {len(volume_df)} irregular events")
```

## Performance Benefits

Using irregular timestamps with QEngine provides several advantages:

### 1. Information Content
- **Higher signal-to-noise ratio**: Bars form based on market activity, not time
- **Reduced market microstructure noise**: Volume/dollar bars filter out low-activity periods
- **Better regime detection**: Natural adaptation to changing market conditions

### 2. Statistical Properties
- **More homoskedastic returns**: Volume bars tend to have more stable return distributions
- **Reduced autocorrelation**: Better independence assumptions for statistical models
- **Improved risk models**: Volatility estimates benefit from activity-based sampling

### 3. Execution Realism
- **Natural timing**: Strategy reactions align with actual market events
- **Reduced overfitting**: No artificial time grid creates more robust strategies
- **Better slippage modeling**: Execution costs vary naturally with market activity

## Comparison with Other Frameworks

| Framework | Time Bars | Volume Bars | Dollar Bars | Info Bars | Irregular Support |
|-----------|-----------|-------------|-------------|-----------|-------------------|
| **QEngine** | ✅ | ✅ | ✅ | ✅ | **Native** |
| Backtrader | ✅ | ❌ | ❌ | ❌ | Workarounds only |
| Zipline | ✅ | ❌ | ❌ | ❌ | Custom pipeline |
| VectorBT | ✅ | Limited | Limited | ❌ | Manual preprocessing |
| QuantConnect | ✅ | ✅ | ❌ | ❌ | Cloud only |

## Technical Implementation

### Event Processing
```python
# QEngine's core event loop handles any timestamp sequence
class QEngine:
    def process_events(self, events):
        # Sort events by timestamp (handles irregular sequences)
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        for event in sorted_events:
            # Advance clock to event time (irregular)
            self.clock.advance_to(event.timestamp)

            # Create point-in-time data view at this exact moment
            pit_data = self.data_manager.get_pit_data(event.timestamp)

            # Execute strategy logic at irregular timing
            for strategy in self.strategies:
                strategy.on_event(event, pit_data)

            # Process orders at realistic timing
            self.broker.process_event(event)
```

### Data Feed Flexibility
```python
# Any data with timestamps works
class PolarsDataFeed(DataFeed):
    def get_next_event(self):
        """Return next event regardless of timestamp regularity."""
        if self.current_index < len(self.data):
            row = self.data.row(self.current_index, named=True)
            return MarketEvent(
                timestamp=row["timestamp"],  # Could be any irregular timing
                asset_id=row["asset_id"],
                **row
            )
        return None
```

## Best Practices

### 1. Bar Selection
```python
# Choose bar type based on strategy logic
volume_bars = VolumeBars(threshold=50000)    # For volume-based strategies
dollar_bars = DollarBars(threshold=1000000)  # For institutional flow
tick_bars = TickBars(threshold=1000)         # For high-frequency strategies
```

### 2. Strategy Design
```python
class IrregularAwareStrategy(Strategy):
    def on_market_event(self, event, pit_data):
        # Don't assume regular intervals
        time_since_last = event.timestamp - self.last_timestamp

        # Adapt to irregular timing
        if time_since_last > timedelta(minutes=30):
            # Handle large gaps (market close/open)
            self.reset_signals()

        # Use bar metadata for context
        if event.metadata.get("bar_type") == "volume":
            threshold = event.metadata["threshold"]
            # Strategy logic aware of volume threshold
```

### 3. Risk Management
```python
def on_market_event(self, event, pit_data):
    # Risk checks account for irregular timing
    time_weighted_exposure = self.calculate_exposure(
        current_time=event.timestamp,
        irregular_intervals=True
    )

    # Position sizing adapts to activity levels
    if event.volume > self.high_activity_threshold:
        position_size *= self.activity_adjustment_factor
```

## Future Enhancements

QEngine's irregular timestamp support enables future innovations:

### 1. Multi-Asset Synchronization
```python
# Different assets can have different irregular bars
aapl_volume_bars = volume_bars.transform(aapl_ticks)    # Irregular timing A
tsla_dollar_bars = dollar_bars.transform(tsla_ticks)    # Irregular timing B

# QEngine synchronizes naturally without forcing artificial alignment
```

### 2. Alternative Data Integration
```python
# News-driven bars, social sentiment bars, etc.
news_events = create_news_bars(news_feed)  # Highly irregular
social_bars = create_sentiment_bars(twitter_feed)  # Event-driven timing

# All integrate seamlessly with existing market data
```

### 3. Real-Time Compatibility
```python
# Live trading with same irregular logic
live_volume_bars = VolumeBarStream(threshold=50000)
# Produces events at same irregular timing as backtest
```

## Conclusion

QEngine's **native support for irregular timestamps** is a fundamental architectural advantage that enables:

- **More realistic backtesting** through natural market timing
- **Better strategy development** using information-driven bars
- **Seamless integration** with qfeatures' advanced bar aggregation
- **Future-proof architecture** for alternative data and real-time trading

This capability, combined with QEngine's event-driven design, point-in-time correctness, and high-performance implementation, creates a uniquely powerful backtesting framework for modern quantitative strategies.

The ability to handle volume bars, dollar bars, information bars, and any other irregular aggregation method out-of-the-box is indeed a significant differentiator that should be prominently featured in QEngine's marketing and documentation.
