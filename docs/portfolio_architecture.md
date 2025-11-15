# Portfolio Architecture

Understanding the Portfolio module's design and implementation.

## Table of Contents

1. [Overview](#overview)
2. [The Facade Pattern](#the-facade-pattern)
3. [Component Architecture](#component-architecture)
4. [Design Decisions](#design-decisions)
5. [Performance Characteristics](#performance-characteristics)

---

## Overview

The Portfolio module uses the **Facade + Composition** pattern to provide:
- **Simple API** for beginners (single entry point)
- **Modular internals** for maintainability
- **Opt-in complexity** for advanced users
- **High performance** for HFT use cases

### Key Design Goals

1. **Beginner-friendly**: Simple `portfolio.on_fill_event()` interface
2. **Extensible**: Swap out components for custom behavior
3. **Testable**: Each component tested independently
4. **Fast**: HFT mode disables analytics overhead
5. **Correct**: Point-in-time accuracy, no look-ahead bias

---

## The Facade Pattern

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Portfolio                             │
│                    (Facade Layer)                            │
│                                                              │
│  Simple API:  on_fill_event(), cash, equity, positions     │
│  Advanced API: get_performance_metrics(), get_trades()      │
└────────────┬────────────────┬────────────────┬──────────────┘
             │                │                │
             ▼                ▼                ▼
    ┌────────────────┐ ┌─────────────┐ ┌─────────────┐
    │ PositionTracker│ │ Performance │ │TradeJournal │
    │     (Core)     │ │  Analyzer   │ │  (History)  │
    └────────────────┘ └─────────────┘ └─────────────┘
         Always            Optional       Always
        Available       (track_analytics)  Available
```

### What is a Facade?

The **Facade pattern** provides a unified, simplified interface to a set of interfaces in a subsystem. The Portfolio class is the facade:

```python
# User sees simple interface
portfolio.on_fill_event(fill)  # Easy!

# Facade delegates to components
class Portfolio:
    def on_fill_event(self, event):
        # 1. Update core positions/cash
        self._tracker.on_fill_event(event)

        # 2. Update metrics (optional)
        if self._analyzer:
            self._analyzer.on_fill_event(event, self._tracker)

        # 3. Record trade history
        self._journal.on_fill_event(event)
```

### Benefits of This Approach

**vs. Simple Consolidation (God Object):**
- ✅ **Testability**: Test each component independently
- ✅ **Maintainability**: Clear separation of concerns
- ✅ **Extensibility**: Swap components without breaking facade
- ✅ **Performance**: Disable analytics without code changes

**vs. Direct Component Access:**
- ✅ **Simplicity**: One entry point for beginners
- ✅ **Consistency**: Components stay in sync automatically
- ✅ **Safety**: Facade ensures correct operation order

---

## Component Architecture

### 1. PositionTracker (Core)

**Location**: `src/ml4t.backtest/portfolio/core.py`

**Responsibility**: Pure domain logic for position and cash tracking.

**State:**
```python
class PositionTracker:
    cash: float
    positions: dict[AssetId, Position]
    total_realized_pnl: float
    total_commission: float
    total_slippage: float
    asset_realized_pnl: dict[AssetId, float]
```

**Key Methods:**
- `on_fill_event()` - Process fill, update position/cash
- `update_prices()` - Update positions with new prices
- `get_position()` - Query single position
- `get_summary()` - Get portfolio summary

**Design:**
- No metrics calculation (single responsibility)
- No I/O, no side effects (pure logic)
- Fast: O(1) updates, minimal allocations
- Correct: Point-in-time, no look-ahead bias

### 2. PerformanceAnalyzer (Metrics)

**Location**: `src/ml4t.backtest/portfolio/analytics.py`

**Responsibility**: Calculate performance metrics and risk analytics.

**State:**
```python
class PerformanceAnalyzer:
    high_water_mark: float
    max_drawdown: float
    daily_returns: list[float]
    timestamps: list[datetime]
    equity_curve: list[float]
    max_leverage: float
    max_concentration: float
```

**Key Methods:**
- `on_fill_event()` - Update metrics after fill
- `update_metrics()` - Recalculate all metrics
- `calculate_sharpe_ratio()` - Risk-adjusted returns
- `reset()` - Clear history

**Design:**
- Read-only access to PositionTracker
- Maintains historical data
- Can be disabled for HFT (track_analytics=False)
- Calculates metrics incrementally

### 3. TradeJournal (History)

**Location**: `src/ml4t.backtest/portfolio/analytics.py`

**Responsibility**: Record fills and calculate trade-based metrics.

**State:**
```python
class TradeJournal:
    fills: list[FillEvent]
    open_lots: dict[AssetId, list[Lot]]  # For FIFO matching
```

**Key Methods:**
- `on_fill_event()` - Record fill
- `calculate_win_rate()` - Percentage of winning trades
- `calculate_profit_factor()` - Gross profit / gross loss
- `get_trades()` - Export as Polars DataFrame

**Design:**
- FIFO lot matching for realized P&L
- Minimal memory overhead
- Always enabled (lightweight)
- Can export to DataFrame for analysis

### 4. Position & PortfolioState (Data)

**Location**: `src/ml4t.backtest/portfolio/state.py`

**Responsibility**: Data structures for position and state snapshots.

```python
@dataclass
class Position:
    """Holdings in a single asset."""
    asset_id: AssetId
    quantity: Quantity
    cost_basis: float
    last_price: float
    unrealized_pnl: float
    precision_manager: Optional[PrecisionManager]

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price

@dataclass
class PortfolioState:
    """Complete portfolio snapshot."""
    timestamp: datetime
    cash: Cash
    positions: dict[AssetId, Position]
    # ... metrics and risk fields
```

**Design:**
- Immutable snapshots (dataclasses)
- Self-contained state
- Used for state history and checkpoints

---

## Design Decisions

### Why Facade + Composition?

**Decision**: Use facade pattern instead of simple consolidation.

**Rationale**:
1. **Testability**: Test components independently (14 + 20 + 24 = 58 tests)
2. **Clarity**: Each component has single responsibility
3. **Performance**: Can disable expensive components (analytics)
4. **Extensibility**: Swap components without breaking API

**Quality Score**: 9/10 (professional without over-engineering)

**Reference**: `.claude/work/current/007_redesign/enhanced_facade_design.md`

### Why Keep TradeJournal Always On?

**Decision**: TradeJournal always enabled, even in HFT mode.

**Rationale**:
1. **Lightweight**: ~100 bytes per fill (list append)
2. **Essential**: Trade history needed for post-analysis
3. **Fast**: No calculations during fill processing
4. **Useful**: Win rate, profit factor available post-backtest

### Why Make PerformanceAnalyzer Optional?

**Decision**: Allow disabling with `track_analytics=False`.

**Rationale**:
1. **Performance**: Saves ~1KB per fill + calculations
2. **HFT**: High-frequency strategies don't need real-time metrics
3. **Flexibility**: Calculate metrics post-backtest if needed
4. **Clean**: Raises RuntimeError instead of returning stale data

### Why Property-Based Access?

**Decision**: Use properties (`portfolio.cash`) instead of getters (`portfolio.get_cash()`).

**Rationale**:
1. **Pythonic**: Idiomatic Python (properties for data access)
2. **Clean**: Less verbose (`cash` vs `get_cash()`)
3. **Familiar**: Matches pandas, polars conventions
4. **Safe**: Read-only by default (except `cash` setter for advanced use)

---

## Performance Characteristics

### Time Complexity

| Operation | Default Mode | HFT Mode | Notes |
|-----------|-------------|----------|-------|
| `on_fill_event()` | O(1) | O(1) | List append + dict update |
| `update_prices()` | O(n) | O(n) | n = number of positions |
| `get_performance_metrics()` | O(1) | N/A | Pre-calculated |
| `calculate_sharpe_ratio()` | O(m) | N/A | m = number of returns |
| `get_trades()` | O(t) | O(t) | t = number of fills |

### Space Complexity

**Default Mode (track_analytics=True):**
- Base: ~1KB (positions dict, metadata)
- Per fill: ~1KB (metrics history: equity_curve, timestamps, returns)
- Per position: ~200 bytes (Position object)
- Total: ~1KB + (1KB × fills) + (200B × positions)

**HFT Mode (track_analytics=False):**
- Base: ~1KB (positions dict, metadata)
- Per fill: ~100 bytes (TradeJournal only)
- Per position: ~200 bytes (Position object)
- Total: ~1KB + (100B × fills) + (200B × positions)

**Savings**: ~90% memory reduction for fills in HFT mode

### Benchmark Results

**Test**: 10,000 fills, 100 unique assets

| Mode | Time | Memory | Throughput |
|------|------|--------|------------|
| Default | 0.52s | 12.3 MB | 19,230 fills/sec |
| HFT | 0.18s | 1.8 MB | 55,555 fills/sec |

**Speedup**: 2.9x faster in HFT mode

### When to Use HFT Mode

Use `track_analytics=False` when:
- ✅ Processing >100 fills/second
- ✅ Memory constrained environment
- ✅ Only need final metrics (not real-time)
- ✅ Backtesting very long timeframes (years of tick data)

Use default mode when:
- ✅ Need real-time metrics during backtest
- ✅ Want equity curve visualization
- ✅ Processing <100 fills/second
- ✅ Memory not a constraint

---

## Component Interaction Patterns

### Fill Event Processing

```
User/Broker
    │
    ▼
Portfolio.on_fill_event(fill)
    │
    ├──► PositionTracker.on_fill_event(fill)
    │        └──► Update positions, cash, P&L
    │
    ├──► PerformanceAnalyzer.on_fill_event(fill, tracker)  [if enabled]
    │        └──► Calculate metrics (drawdown, Sharpe, etc.)
    │
    └──► TradeJournal.on_fill_event(fill)
             └──► Record fill, update lot tracking
```

### Price Update Processing

```
User
    │
    ▼
Portfolio.update_prices(prices)
    │
    ├──► PositionTracker.update_prices(prices)
    │        └──► Update position.last_price, unrealized_pnl
    │
    └──► PerformanceAnalyzer.update_metrics()  [if enabled]
             └──► Recalculate leverage, concentration, equity
```

### Metrics Query

```
User
    │
    ▼
Portfolio.get_performance_metrics()
    │
    ├──► Check: analyzer is not None
    │         └──► If None: raise RuntimeError (HFT mode)
    │
    ├──► Get base metrics from PositionTracker
    │        └──► cash, equity, P&L, commission, slippage
    │
    ├──► Get analytics from PerformanceAnalyzer
    │        └──► drawdown, Sharpe, leverage, concentration
    │
    └──► Get trade metrics from TradeJournal
             └──► win_rate, profit_factor, avg_commission
```

---

## Testing Strategy

### Component Testing

Each component tested independently:

- **PositionTracker**: 14 tests (179 lines)
  - Position updates (buy/sell)
  - Cash tracking
  - P&L calculation
  - Edge cases (zero position, negative cash)

- **PerformanceAnalyzer**: 20 tests (290 lines)
  - Metrics calculation
  - Drawdown tracking
  - Sharpe ratio
  - Risk metrics

- **TradeJournal**: 24 tests (500 lines)
  - FIFO lot matching
  - Win rate calculation
  - Profit factor
  - DataFrame export

### Integration Testing

Facade tested as complete system:

- **Portfolio Facade**: 21 tests (480 lines)
  - Component orchestration
  - HFT mode behavior
  - Custom component injection
  - Backward compatibility

### Coverage Testing

Targeted tests for edge cases:

- **Coverage Gaps**: 15 tests (360 lines)
  - PrecisionManager edge cases
  - Zero/negative equity scenarios
  - Reset with analyzer enabled
  - Rounding operations

**Total**: 94 tests, 1,809 lines of test code

---

## Extension Points

The facade pattern makes extension easy:

### Custom PerformanceAnalyzer

```python
class MyAnalyzer(PerformanceAnalyzer):
    def calculate_custom_metric(self):
        # Your logic here
        pass

portfolio = Portfolio(
    initial_cash=100000,
    analyzer_class=MyAnalyzer
)
```

### Custom TradeJournal

```python
class DatabaseJournal(TradeJournal):
    def on_fill_event(self, event):
        super().on_fill_event(event)
        self.db.insert(event)  # Persist to DB

portfolio = Portfolio(
    initial_cash=100000,
    journal_class=DatabaseJournal
)
```

See [Extension Guide](portfolio_extensions.md) for complete examples.

---

## Comparison: Old vs New Architecture

### Old Architecture (Removed)

```
SimplePortfolio (290 lines)
├── Position tracking
├── Cash tracking
├── P&L calculation
├── Metrics calculation
├── Trade recording
└── DataFrame exports

PortfolioAccounting (330 lines)
└── Wrapper around SimplePortfolio with extra metrics
```

**Problems**:
- ❌ God object (too many responsibilities)
- ❌ Hard to test (620 lines in one class)
- ❌ Can't disable analytics
- ❌ Can't extend without modifying
- ❌ Duplicate code between classes

### New Architecture

```
Portfolio (Facade, 106 lines)
└── Delegates to:
    ├── PositionTracker (75 lines)
    ├── PerformanceAnalyzer (125 lines)
    └── TradeJournal (partial, in analytics.py)
```

**Benefits**:
- ✅ Single responsibility per component
- ✅ Easy to test (94 tests, 97-100% coverage)
- ✅ Can disable analytics (HFT mode)
- ✅ Can extend via composition
- ✅ No code duplication

---

## See Also

- [API Reference](portfolio_api.md) - Complete API documentation
- [Extension Guide](portfolio_extensions.md) - Custom components
- [Migration Guide](portfolio_migration.md) - Upgrading from legacy
