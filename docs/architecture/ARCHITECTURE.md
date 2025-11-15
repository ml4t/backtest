# ml4t.backtest Architecture Documentation

## Overview

ml4t.backtest implements a modern event-driven architecture optimized for ML-driven trading strategies while maintaining the flexibility for traditional algorithmic trading. The system is built on three core principles:

1. **Point-in-Time Correctness**: Architectural guarantees against data leakage
2. **Performance First**: Columnar data with Polars, JIT compilation with Numba
3. **Extensibility**: Everything is pluggable through well-defined interfaces

## Core Components

### 1. Event System (`core/event.py`)

The heart of ml4t.backtest is an event-driven architecture where all information flows through typed events:

```python
Event (base)
├── MarketEvent      # Price data (trades, quotes, bars)
├── SignalEvent      # ML model predictions
├── OrderEvent       # Order submissions
├── FillEvent        # Execution confirmations
└── CorporateActionEvent  # Splits, dividends, etc.
```

**Key Features:**
- Priority queue ensures chronological processing
- Type-safe event definitions using dataclasses
- Subscription-based event routing
- Thread-safe event bus for concurrent operations

### 2. Time Management (`core/clock.py`)

The Clock component serves as the master timekeeper, ensuring all events are processed in strict chronological order:

```python
Clock
├── Coordinates multiple data sources
├── Enforces PIT correctness
├── Manages trading calendars
└── Supports backtest/paper/live modes
```

**Key Responsibilities:**
- Merges events from multiple sources chronologically
- Prevents look-ahead bias by controlling time advancement
- Integrates with pandas-market-calendars for accurate session handling
- Provides market open/close awareness

### 3. Data Layer (`data/`)

Modern data management using Polars for exceptional performance:

```python
DataFeed (ABC)
├── ParquetDataFeed  # High-performance with lazy loading
├── CSVDataFeed      # Legacy format support
└── LiveDataFeed     # Real-time market data (future)

SignalSource (ABC)
├── ParquetSignalSource  # Batch ML signals
└── StreamingSignalSource  # Real-time predictions (future)
```

**Performance Optimizations:**
- Lazy loading with Polars scan operations
- Predicate pushdown for efficient filtering
- Zero-copy data access where possible
- Columnar storage for cache efficiency

### 4. Strategy Framework (`strategy/`)

Flexible strategy interface supporting multiple paradigms:

```python
Strategy (ABC)
├── SignalStrategy      # ML signal-based trading
├── IndicatorStrategy   # Technical indicator-based
└── CustomStrategy      # User-defined logic
```

**Lifecycle Methods:**
- `on_start()`: One-time initialization
- `on_event()`: Process each event
- `before_trading_start()`: Daily preparation
- `after_trading_end()`: End-of-day tasks
- `on_end()`: Cleanup and final analysis

### 5. Execution Layer (Planned)

Realistic order simulation with pluggable models:

```python
Broker (ABC)
├── SimulationBroker   # Backtesting
├── PaperBroker        # Paper trading
└── LiveBroker         # Live execution

Models (Pluggable)
├── FillModel          # How orders match
├── SlippageModel      # Price impact
├── CommissionModel    # Transaction costs
└── MarketImpactModel  # Almgren-Chriss, etc.
```

### 6. Portfolio Management (Planned)

Comprehensive portfolio tracking and analysis:

```python
PortfolioAccounting
├── Position tracking
├── P&L calculation (realized/unrealized)
├── Multi-currency support
└── Performance metrics
```

## Data Flow Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Data Feeds  │────▶│    Clock     │────▶│  EventBus   │
└─────────────┘     └──────────────┘     └─────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌─────────────┐
                    │ PIT Enforcer │     │  Strategy   │
                    └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Broker    │◀────│ Risk Manager │◀────│   Orders    │
└─────────────┘     └──────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ Portfolio   │────▶│   Reporter   │
└─────────────┘     └──────────────┘
```

## Key Design Decisions

### 1. Polars-First Data Architecture

**Rationale:** Polars provides 10-100x performance improvement over Pandas while maintaining a familiar API.

**Implementation:**
- All data operations use Polars DataFrames
- Lazy evaluation for memory efficiency
- Native Arrow integration for zero-copy operations

### 2. Event-Driven vs Vectorized

**Rationale:** Event-driven provides realism for complex strategies while vectorized operations enable performance.

**Hybrid Approach:**
- Event-driven core for order matching and state management
- Vectorized analytics for metrics and reporting
- Micro-batching for performance optimization

### 3. Abstract Base Classes

**Rationale:** Clear contracts enable extensibility without coupling.

**Benefits:**
- Users can implement custom components
- Testing via mock implementations
- Future extensions without breaking changes

### 4. Point-in-Time Architecture

**Rationale:** ML strategies require strict temporal isolation to prevent leakage.

**Guarantees:**
- Clock controls all time advancement
- Events processed in strict chronological order
- No future data accessible to strategies

## Performance Considerations

### Hot Paths Identified

1. **Event Processing Loop**: Target for Rust implementation
2. **Order Matching**: Numba JIT compilation
3. **Data Access**: Polars lazy evaluation
4. **Metric Calculation**: Vectorized NumPy operations

### Memory Management

- Streaming data processing (no full history in memory)
- Lazy loading of large datasets
- Configurable history buffers
- Efficient Arrow columnar format

### Concurrency Strategy

- Thread-safe event bus
- Parallel data feed processing
- Async I/O for live trading
- Multi-process optimization runs

## Extensibility Points

### Pluggable Components

All major components implement interfaces allowing custom implementations:

1. **Data Feeds**: Custom data sources
2. **Strategies**: User-defined trading logic
3. **Brokers**: New execution venues
4. **Models**: Custom slippage, impact, etc.
5. **Reporters**: Custom output formats

### Event Extensions

Users can define custom event types:

```python
@dataclass
class CustomEvent(Event):
    custom_field: Any

    def __post_init__(self):
        self.event_type = EventType.CUSTOM
```

## Testing Strategy

### Levels of Testing

1. **Unit Tests**: Individual component validation
2. **Integration Tests**: Component interaction
3. **Scenario Tests**: Complete backtest validation
4. **Performance Tests**: Benchmark regression detection
5. **Property Tests**: Mathematical invariants

### Test Infrastructure

```python
# Deterministic testing with fixed seeds
engine = ml4t.backtest(seed=42)

# Golden scenario validation
result = engine.run(scenario="split_dividend_test")
assert result.matches_golden_output()

# Performance benchmarking
@benchmark
def test_million_events():
    engine.run(events=1_000_000)
```

## Migration Path

### From Zipline

```python
# Compatibility layer
from ml4t.backtest.compat import ZiplineAlgorithm

class MyAlgo(ZiplineAlgorithm):
    def initialize(self, context):
        # Existing Zipline code works
        pass
```

### From Backtrader

```python
# Compatibility layer
from ml4t.backtest.compat import BacktraderStrategy

class MyStrategy(BacktraderStrategy):
    def next(self):
        # Existing Backtrader code works
        pass
```

## Future Enhancements

### Phase 1 (Current)
- Core event system ✓
- Basic data feeds ✓
- Strategy framework ✓
- Simple execution

### Phase 2 (Next)
- Advanced order types
- Market microstructure models
- Corporate actions
- ML signal integration

### Phase 3 (Future)
- Rust performance optimization
- Market impact models
- Portfolio optimization
- Live trading adapters

### Phase 4 (Long-term)
- Distributed backtesting
- Cloud deployment
- Strategy marketplace
- AutoML integration

## Conclusion

ml4t.backtest's architecture balances performance, correctness, and extensibility. By learning from existing solutions while leveraging modern Python tooling, ml4t.backtest provides a foundation for the next generation of algorithmic trading systems.
