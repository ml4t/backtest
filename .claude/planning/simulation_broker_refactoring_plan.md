# SimulationBroker Refactoring Plan

## Executive Summary

**Goal**: Refactor the 862-line SimulationBroker class into focused, testable components following the Single Responsibility Principle.

**Status**: Planning phase - 3 of 4 component classes created
**Estimated Effort**: 8-12 hours total
**Progress**: ~2 hours invested (component design), ~6-10 hours remaining

---

## Problem Statement

### Current Issues
- **SimulationBroker**: 862 lines, 23 methods, violates SRP
- **on_market_event()**: 137 lines - too complex
- **_try_fill_order()**: 195 lines - handles 9 distinct responsibilities
- Difficult to test individual concerns in isolation
- Hard to extend without breaking existing functionality

### Code Smell Indicators
```python
# Single class manages ALL of these:
class SimulationBroker:
    # Order routing
    _open_orders, _stop_orders, _trailing_stops, _bracket_orders, _pending_orders

    # Position tracking
    _positions, cash

    # Order execution
    _try_fill_order() # 195 lines!

    # Commission/slippage/impact models
    commission_model, slippage_model, market_impact_model

    # Margin management
    margin_account

    # Statistics
    _total_commission, _total_slippage, _fill_count
```

---

## Proposed Architecture

### Component Hierarchy

```
SimulationBroker (Facade/Orchestrator)
├── PositionTracker (✅ Created)
│   ├── Tracks positions & cash
│   ├── Updates on fills
│   └── Cash constraint validation
├── OrderRouter (✅ Created)
│   ├── Routes orders to queues
│   ├── Manages order lifecycle
│   └── Queue queries & cleanup
├── BracketOrderManager (✅ Created)
│   ├── Creates bracket legs
│   ├── Manages OCO relationships
│   └── Cancels linked orders
└── FillSimulator (❌ Not yet created - MOST COMPLEX)
    ├── Order execution logic
    ├── Fill price calculation
    ├── Liquidity constraints
    ├── Margin requirements
    ├── Commission/slippage calculation
    └── FillEvent generation
```

---

## Component Specifications

### 1. PositionTracker ✅ COMPLETED

**File**: `src/qengine/execution/position_tracker.py` (145 lines)

**Responsibilities**:
- Track position quantities per asset
- Maintain cash balance
- Update positions on fills
- Validate cash constraints

**Public API**:
```python
class PositionTracker:
    def __init__(self, initial_cash: float)
    def update_position(asset_id, quantity, side, fill_price, commission, slippage)
    def get_position(asset_id) -> Quantity
    def get_all_positions() -> dict[AssetId, Quantity]
    def get_cash() -> float
    def has_sufficient_cash(required_cash: float) -> bool
    def reset()
    def get_statistics() -> dict
```

**Benefits**:
- Isolated position logic for easy testing
- Clear cash management interface
- No entanglement with order execution

---

### 2. OrderRouter ✅ COMPLETED

**File**: `src/qengine/execution/order_router.py` (243 lines)

**Responsibilities**:
- Route orders to appropriate queues (open, stop, trailing, bracket, pending)
- Store order history
- Query orders by criteria
- Remove orders from queues

**Public API**:
```python
class OrderRouter:
    def __init__(self, execution_delay: bool = True)
    def route_order(order: Order, timestamp: datetime)
    def activate_pending_orders(asset_id: AssetId) -> list[Order]
    def get_order(order_id: OrderId) -> Order | None
    def get_open_orders(asset_id: AssetId | None = None) -> list[Order]
    def get_stop_orders(asset_id: AssetId) -> list[Order]
    def get_trailing_stops(asset_id: AssetId) -> list[Order]
    def remove_order(order: Order) -> bool
    def register_bracket_order(parent_id, bracket_info)
    def get_bracket_info(parent_id) -> dict | None
    def remove_bracket(parent_id)
    def reset()
    def get_statistics() -> dict
```

**Benefits**:
- Centralized order queue management
- Easy to add new order types
- Clean separation from execution logic

---

### 3. BracketOrderManager ✅ COMPLETED

**File**: `src/qengine/execution/bracket_manager.py` (124 lines)

**Responsibilities**:
- Create stop-loss and take-profit legs after parent fill
- Link OCO orders
- Cancel linked orders when one fills

**Public API**:
```python
class BracketOrderManager:
    def __init__(self, submit_order_callback: Callable)
    def handle_bracket_fill(parent_order, fill_event) -> list[Order]
    def handle_oco_fill(filled_order, cancel_callback) -> list[str]
    def get_bracket_children(parent_id) -> list[str]
    def reset()
```

**Benefits**:
- Isolated bracket logic
- Reusable OCO handling
- Clear parent-child relationships

---

### 4. FillSimulator ❌ NOT YET CREATED - REQUIRES DETAILED DESIGN

**File**: `src/qengine/execution/fill_simulator.py` (estimated 300-400 lines)

**Responsibilities**:
- Determine if order can fill at market price
- Apply market impact
- Calculate fill price with slippage
- Apply liquidity constraints
- Check margin requirements (derivatives)
- Check cash constraints (equities)
- Calculate commission and slippage costs
- Generate FillEvent

**Proposed API**:
```python
class FillSimulator:
    def __init__(
        self,
        asset_registry: AssetRegistry,
        commission_model: CommissionModel | None = None,
        slippage_model: SlippageModel | None = None,
        market_impact_model: MarketImpactModel | None = None,
        liquidity_model: LiquidityModel | None = None,
        enable_margin: bool = False,
    ):
        ...

    def try_fill_order(
        self,
        order: Order,
        market_price: Price,
        current_cash: float,
        current_position: Quantity,
        timestamp: datetime | None = None,
    ) -> FillResult | None:
        """Attempt to fill an order.

        Returns:
            FillResult with fill_event, commission, slippage, or None if can't fill
        """
        ...

    # Private methods (extracted from current _try_fill_order):
    def _can_fill(self, order, market_price) -> bool
    def _apply_market_impact(self, order, market_price, timestamp) -> Price
    def _calculate_fill_price(self, order, impacted_price) -> Price
    def _apply_liquidity_constraints(self, order, fill_quantity, market_price) -> Quantity
    def _check_margin_requirements(self, order, fill_quantity, fill_price, position) -> Quantity
    def _check_cash_constraints(self, order, fill_quantity, fill_price, cash) -> Quantity
    def _calculate_costs(self, order, fill_quantity, fill_price, market_price) -> tuple[float, float]
    def _create_fill_event(self, order, fill_quantity, fill_price, commission, slippage, timestamp) -> FillEvent
```

**Complexity Breakdown**:

The current `_try_fill_order()` method (195 lines) handles:

1. **Order validation** (2 lines)
   ```python
   if not order.can_fill(market_price):
       return None
   ```

2. **Market impact** (5 lines)
   ```python
   impacted_market_price = self._get_market_price_with_impact(
       order, market_price, timestamp
   )
   ```

3. **Fill price calculation** (1 line)
   ```python
   fill_price = self._calculate_fill_price(order, impacted_market_price)
   ```

4. **Liquidity constraints** (9 lines)
   ```python
   fill_quantity = order.remaining_quantity
   if self.liquidity_model is not None:
       max_liquidity = self.liquidity_model.get_max_fill_quantity(...)
       fill_quantity = min(fill_quantity, max_liquidity)
       if fill_quantity < 0.01:
           return None
   ```

5. **Margin requirements** (31 lines) - Complex derivative logic
6. **Cash constraints** (51 lines) - Binary search for max affordable quantity
7. **Commission/slippage calculation** (15 lines)
8. **Position updates** (10 lines)
9. **FillEvent creation** (45 lines)

**Challenges**:
- Cash constraint logic uses binary search (lines 562-592) - needs careful extraction
- Margin logic interacts with MarginAccount - needs interface design
- Position updates should delegate to PositionTracker

---

## Refactored SimulationBroker

**New Size**: Estimated 250-300 lines (down from 862)

**Structure**:
```python
class SimulationBroker(Broker):
    """Orchestrates order execution via specialized components."""

    def __init__(self, initial_cash, ...):
        # Initialize components
        self.position_tracker = PositionTracker(initial_cash)
        self.order_router = OrderRouter(execution_delay)
        self.bracket_manager = BracketOrderManager(self.submit_order)
        self.fill_simulator = FillSimulator(
            asset_registry, commission_model, slippage_model,
            market_impact_model, liquidity_model, enable_margin
        )

        # Statistics
        self._total_commission = 0.0
        self._total_slippage = 0.0
        self._fill_count = 0

        # Event bus reference (set by engine)
        self.event_bus = None

    def submit_order(self, order, timestamp=None) -> OrderId:
        """Submit an order - delegates to OrderRouter."""
        # Validation (from recent fix)
        self._validate_order(order)

        # Update order state
        order.state = OrderState.SUBMITTED
        order.status = OrderStatus.SUBMITTED
        order.submitted_time = timestamp or datetime.now()

        # Route to appropriate queue
        self.order_router.route_order(order, timestamp)

        return order.order_id

    def cancel_order(self, order_id) -> bool:
        """Cancel an order - delegates to OrderRouter."""
        order = self.order_router.get_order(order_id)
        if not order:
            return False

        order.state = OrderState.CANCELLED
        order.status = OrderStatus.CANCELLED
        self.order_router.remove_order(order)

        return True

    def on_market_event(self, event: MarketEvent, timestamp: datetime):
        """Process market event - orchestrates components."""
        asset_id = event.asset_id
        market_price = event.close

        # Store last price
        self._last_prices[asset_id] = market_price

        # Activate pending orders (if execution delay)
        self.order_router.activate_pending_orders(asset_id)

        # Process stop orders
        self._process_stop_orders(asset_id, market_price, timestamp)

        # Process trailing stops
        self._process_trailing_stops(asset_id, market_price, timestamp)

        # Process open orders
        self._process_open_orders(asset_id, market_price, timestamp)

    def _process_open_orders(self, asset_id, market_price, timestamp):
        """Process open orders for an asset."""
        orders = list(self.order_router.get_open_orders(asset_id))

        for order in orders:
            # Try to fill via FillSimulator
            result = self.fill_simulator.try_fill_order(
                order,
                market_price,
                self.position_tracker.get_cash(),
                self.position_tracker.get_position(asset_id),
                timestamp,
            )

            if result:
                # Update position
                self.position_tracker.update_position(
                    asset_id, result.fill_quantity, order.side,
                    result.fill_price, result.commission, result.slippage
                )

                # Update statistics
                self._total_commission += result.commission
                self._total_slippage += result.slippage
                self._fill_count += 1

                # Remove from queue
                self.order_router.remove_order(order)

                # Handle bracket if applicable
                if order.order_type == OrderType.BRACKET:
                    self.bracket_manager.handle_bracket_fill(order, result.fill_event)

                # Handle OCO cancellation
                if order.child_order_ids:
                    self.bracket_manager.handle_oco_fill(order, self.cancel_order)

                # Publish fill event
                if self.event_bus:
                    self.event_bus.publish(result.fill_event)

    # Similar methods for _process_stop_orders, _process_trailing_stops
    # Delegation methods for get_position, get_cash, etc.
```

**Benefits of Refactored Structure**:
1. **SimulationBroker**: 250-300 lines (down from 862) - 65% reduction
2. **Each component**: <300 lines, single responsibility
3. **Independently testable**: Can test FillSimulator without OrderRouter
4. **Easy to extend**: Add new order types by extending OrderRouter
5. **Clear interfaces**: Component boundaries are explicit

---

## Migration Strategy

### Phase 1: Component Creation ✅ MOSTLY DONE
- [x] PositionTracker (145 lines)
- [x] OrderRouter (243 lines)
- [x] BracketOrderManager (124 lines)
- [ ] FillSimulator (300-400 lines) - **BLOCKED: Needs detailed design**

### Phase 2: SimulationBroker Refactoring
1. Update SimulationBroker to instantiate components
2. Delegate `submit_order()` to OrderRouter
3. Delegate `cancel_order()` to OrderRouter
4. Refactor `on_market_event()` to orchestrate components
5. Delegate position queries to PositionTracker
6. Update `reset()` to reset all components

### Phase 3: Testing
1. Unit tests for each component in isolation
2. Integration tests for SimulationBroker orchestration
3. Regression tests against existing test suite
4. Ensure no behavior changes

### Phase 4: Cleanup
1. Remove old code from broker.py
2. Update imports in dependent modules
3. Update documentation

---

## Risks & Mitigation

### Risk 1: Behavior Changes
**Mitigation**: Comprehensive test suite, side-by-side comparison

### Risk 2: Performance Degradation
**Mitigation**: Benchmark before/after, component overhead is minimal

### Risk 3: Breaking Tests
**Mitigation**: Tests should use public Broker API, internal changes shouldn't break

### Risk 4: Incomplete Extraction
**Mitigation**: Careful review of _try_fill_order logic, line-by-line extraction

---

## Effort Breakdown

| Task | Estimated Hours | Status |
|------|-----------------|--------|
| PositionTracker design & implementation | 1.5 | ✅ Done |
| OrderRouter design & implementation | 2.0 | ✅ Done |
| BracketOrderManager design & implementation | 1.0 | ✅ Done |
| FillSimulator design | 2.0 | ⏳ Pending |
| FillSimulator implementation | 3.0 | ⏳ Pending |
| SimulationBroker refactoring | 2.0 | ⏳ Pending |
| Test updates | 1.5 | ⏳ Pending |
| Documentation | 0.5 | ⏳ Pending |
| **Total** | **13.5 hours** | **31% complete** |

**Completed**: 4.5 hours (PositionTracker, OrderRouter, BracketOrderManager)
**Remaining**: 9 hours (FillSimulator, integration, testing)

---

## Decision Point

**Option A: Continue with Full Refactoring** (9 hours remaining)
- Complete FillSimulator design and implementation
- Refactor SimulationBroker to orchestrate components
- Update all tests
- Full separation of concerns achieved

**Option B: Defer FillSimulator, Use Hybrid Approach** (2 hours)
- Keep `_try_fill_order()` in SimulationBroker for now
- Use the 3 completed components (PositionTracker, OrderRouter, BracketOrderManager)
- Partial improvement, easier to test positions and routing
- FillSimulator extraction can be future work

**Option C: Stop Here, Document Learning** (0 hours)
- Keep the 3 component classes as reference implementations
- Document the refactoring approach for future work
- Apply only the 6 quick fixes already completed
- Revisit full refactoring when time permits

---

## Recommendation

Given the complexity of FillSimulator (195-line method with 9 responsibilities, binary search logic, margin/liquidity/cash constraints), I recommend:

**Option B: Hybrid Approach**

**Rationale**:
1. The 3 completed components (PositionTracker, OrderRouter, BracketOrderManager) provide immediate value
2. These are the easiest to integrate and test independently
3. FillSimulator requires more careful design due to:
   - Binary search for max affordable quantity (complex to extract)
   - Margin account interactions
   - Multiple constraint checks with early returns
4. Can be integrated incrementally without breaking existing tests
5. FillSimulator extraction can be tackled separately when more time is available

**Next Steps for Option B**:
1. Integrate PositionTracker into SimulationBroker (1 hour)
2. Integrate OrderRouter into SimulationBroker (1 hour)
3. Integrate BracketOrderManager into SimulationBroker (30 min)
4. Run tests to verify no regressions (30 min)
5. Document hybrid architecture (30 min)

**Total for Option B**: 3.5 hours to get immediate benefits

**Future Work**: Extract FillSimulator when time permits (6 hours)

---

## Files Created

1. `src/qengine/execution/position_tracker.py` (145 lines)
2. `src/qengine/execution/order_router.py` (243 lines)
3. `src/qengine/execution/bracket_manager.py` (124 lines)
4. `.claude/planning/simulation_broker_refactoring_plan.md` (this document)

**Total New Code**: 512 lines across 3 focused components

---

*Document created: 2025-09-30*
*Last updated: 2025-09-30*
*Status: Awaiting decision on Option A, B, or C*
