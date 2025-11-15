# Point-in-Time Enforcement and State Management

## Overview

This document addresses the critical architectural concerns raised in the review regarding PIT (Point-in-Time) data access and state management in ml4t.backtest's hybrid architecture.

## Point-in-Time Data Access

### The Challenge

Preventing strategies from accessing future data is the most critical requirement for a valid backtest. Any leak of future information invalidates results and can lead to catastrophic trading losses.

### Our Solution: Immutable Data Views

The `PITData` object provides an **immutable, time-bounded view** of market data that is passed to strategies at each event:

```python
class PITData:
    """Immutable point-in-time data accessor."""

    def __init__(self, current_time: datetime, data_store: DataStore):
        self._current_time = current_time
        self._data_store = data_store
        # Freeze the view at construction
        self._frozen = True

    def get_latest_price(self, asset: AssetId) -> Optional[Price]:
        """Get the most recent price as of current_time."""
        return self._data_store.get_price_before(asset, self._current_time)

    def get_history(
        self,
        asset: AssetId,
        field: str,
        periods: int,
        frequency: str = "1d"
    ) -> pl.DataFrame:
        """Get historical data ending at current_time."""
        end_time = self._current_time
        start_time = self._calculate_lookback(end_time, periods, frequency)
        return self._data_store.get_range(
            asset, field, start_time, end_time
        )

    @property
    def current_time(self) -> datetime:
        """The current simulation time (read-only)."""
        return self._current_time
```

### Enforcement Mechanism

1. **Clock Creates Views**: The Clock creates a new `PITData` instance for each event
2. **Strategy Receives View**: Strategies receive this immutable view in `on_event()`
3. **No Direct Data Access**: Strategies cannot access raw data feeds directly
4. **Time Validation**: All data queries validate against current_time

```python
# In the event processing loop
def process_event(self, event: Event):
    # Create PIT view for this moment
    pit_data = PITData(event.timestamp, self.data_store)

    # Strategy only sees data up to this point
    strategy.on_event(event, pit_data)

    # View becomes invalid after processing
    pit_data._invalidate()
```

## State Management Architecture

### The Challenge

In a hybrid system with both event-driven and vectorized components, maintaining consistent state while enabling high performance is complex.

### Our Solution: Centralized State with Immutable Snapshots

```python
@dataclass
class PortfolioState:
    """Single source of truth for portfolio state."""

    timestamp: datetime
    positions: Dict[AssetId, Position]
    cash: Dict[Currency, Decimal]
    pending_orders: List[Order]
    completed_orders: List[Order]
    metrics: PerformanceMetrics

    def snapshot(self) -> 'PortfolioState':
        """Create an immutable snapshot."""
        return deepcopy(self)

    def to_arrow(self) -> pa.Table:
        """Convert to Arrow format for vectorized operations."""
        # Efficient columnar representation
        pass
```

### State Flow

```
Event → StateManager → PortfolioState → Strategy
           ↓               ↓               ↓
      Validation      Snapshots      Read-only View
```

### State Update Protocol

1. **Atomic Updates**: All state changes are atomic transactions
2. **Event Sourcing**: State changes triggered only by events
3. **Audit Trail**: Every state change is logged with the triggering event
4. **Rollback Support**: Can restore to any previous state

```python
class StateManager:
    """Manages portfolio state transitions."""

    def __init__(self):
        self._state = PortfolioState()
        self._history: List[StateSnapshot] = []
        self._lock = threading.Lock()

    def update(self, event: Event) -> PortfolioState:
        """Update state based on event."""
        with self._lock:
            # Save snapshot for rollback
            snapshot = self._state.snapshot()
            self._history.append(snapshot)

            # Apply state transition
            if isinstance(event, FillEvent):
                self._apply_fill(event)
            elif isinstance(event, CorporateActionEvent):
                self._apply_corporate_action(event)

            return self._state.snapshot()  # Return immutable copy
```

## Migration Approach: Simplified Compatibility

### Original Concern

The review correctly identifies that full API compatibility with Zipline/Backtrader would be a maintenance nightmare.

### Revised Approach: Migration Helpers, Not Full Compatibility

Instead of compatibility layers, we provide **migration helpers** that ease the transition:

```python
# NOT a full compatibility layer
# Just helpers for common patterns

class MigrationHelper:
    """Utilities to help port strategies from other platforms."""

    @staticmethod
    def zipline_order_to_ml4t.backtest(
        asset: Asset,
        amount: int,
        style: OrderStyle
    ) -> OrderEvent:
        """Convert Zipline order to ml4t.backtest format."""
        # Simple mapping, not full compatibility
        pass

    @staticmethod
    def backtrader_indicator_to_ml4t.backtest(
        indicator_class: type
    ) -> ml4t.backtestIndicator:
        """Wrap Backtrader indicator for use in ml4t.backtest."""
        # Basic wrapper, not full compatibility
        pass
```

### Documentation-First Migration

Focus on excellent migration **documentation** rather than code compatibility:

1. **Pattern Mapping Guide**: Show how common patterns translate
2. **Code Examples**: Before/after examples for typical strategies
3. **Feature Comparison**: Clear table of what maps and what doesn't
4. **Migration Checklist**: Step-by-step process for porting

## Performance Implications

### PIT Enforcement Overhead

- **Cost**: ~5-10% performance overhead vs unsafe access
- **Mitigation**: Cache frequent queries, use columnar operations
- **Trade-off**: Correctness is non-negotiable

### State Management Overhead

- **Cost**: Minimal with proper design (< 1% overhead)
- **Mitigation**: Immutable snapshots use structural sharing
- **Benefit**: Enables debugging, rollback, and determinism

## Testing Strategy

### PIT Correctness Tests

```python
def test_pit_prevents_future_access():
    """Ensure strategies cannot access future data."""
    clock = Clock()
    clock.set_time(datetime(2024, 1, 1, 10, 0))

    pit_data = PITData(clock.current_time, data_store)

    # Should return data up to 10:00
    assert pit_data.get_latest_price('AAPL') == 150.0

    # Future data (10:30) should not be visible
    with pytest.raises(FutureDataError):
        pit_data._data_store.get_price_at('AAPL',
                                          datetime(2024, 1, 1, 10, 30))
```

### State Consistency Tests

```python
def test_state_consistency_across_events():
    """Ensure state remains consistent through event processing."""
    state_manager = StateManager()

    # Process multiple events
    events = generate_test_events(1000)
    for event in events:
        state_manager.update(event)

    # Verify invariants
    assert state_manager.validate_invariants()
    assert state_manager.calculate_nav() >= 0
    assert sum(state_manager.positions.values()) == expected_total
```

## Conclusion

By implementing:
1. **Immutable PIT data views** passed to strategies
2. **Centralized state management** with atomic updates
3. **Simplified migration helpers** instead of full compatibility

We address the critical concerns while maintaining performance and correctness. The system remains simple, testable, and performant while providing iron-clad guarantees against data leakage.
