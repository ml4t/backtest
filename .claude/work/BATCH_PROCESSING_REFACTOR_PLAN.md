# BacktestEngine Batch Processing Refactor - Implementation Plan

**Created**: 2025-11-19
**Expert Reviews**: `.claude/code_review/20251119/response_01.md`, `response_02.md`
**Target**: Transform atomic event processing (126k iterations) → batch time-slice processing (252 iterations)

## Executive Summary

The BacktestEngine has three critical architectural flaws preventing correct multi-asset portfolio backtesting:

1. **Issue #1**: `Strategy.buy_percent()` accumulates positions instead of calculating target differences
   - **Impact**: 25-position strategy creates 75+ positions, loses -95% instead of +770%
   - **Root Cause**: Ignores current position when calculating order size

2. **Issue #2**: Broker processes events per-asset, preventing simultaneous portfolio rebalancing
   - **Impact**: Cannot make portfolio-level decisions (cross-asset ranking, optimization)
   - **Root Cause**: Atomic event loop (one asset at a time)

3. **Issue #3**: Execution delay timing causes duplicate orders (fills at T+2 instead of T+1)
   - **Impact**: Orders don't fill until 2 bars later, breaking realistic execution
   - **Root Cause**: Pending orders moved to open at wrong phase of event processing

**Solution**: Refactor from atomic event-driven to batch time-slice processing.

## Target Outcomes

### Phase 1: Correctness
- ✅ **Return**: +770% (matching manual loop reference)
- ✅ **Positions**: ~25 final positions (not 75+)
- ✅ **Execution Delay**: Orders fill at T+1 (not T+2)
- ✅ **Iterations**: 252 (not 126,000)
- ✅ **Performance**: 3,000+ events/sec (baseline)

### Phase 2: Performance
- ✅ **Throughput**: >50,000 events/sec (16.7× improvement)
- ✅ **Architecture**: Hybrid vectorized batch processing
- ✅ **Memory**: Stable, no leaks from batching

---

## Phase 1: Correctness (Priority Tasks)

### TASK 1: Add Strategy.order_target_percent() (Issue #1 Fix)

**File**: `src/ml4t/backtest/strategy/base.py`

**Problem**: Current `buy_percent()` accumulates positions
```python
# Current (WRONG)
def buy_percent(self, asset_id, percent, price):
    dollars_to_spend = portfolio_value * percent  # Ignores current position!
    quantity = dollars_to_spend / price
    # Each call ADDS to position
```

**Solution**: Add `order_target_percent()` that calculates deltas
```python
def order_target_percent(self, asset_id: str, target_pct: float):
    """Place order to adjust position to target percent of portfolio equity.
    
    This fixes Issue #1 by calculating the DIFFERENCE between target and current.
    """
    # Get current state
    current_equity = self.broker._internal_portfolio.equity
    current_qty = self.get_position(asset_id)
    price = self._current_market_data[asset_id].close
    
    if price is None or price <= 0:
        return
    
    # Calculate target quantity
    target_value = current_equity * target_pct
    target_qty = target_value / price
    
    # Calculate DELTA (the key fix)
    delta_qty = target_qty - current_qty
    
    # Round and submit
    trade_qty = int(delta_qty)
    if abs(trade_qty) > 0:
        side = OrderSide.BUY if trade_qty > 0 else OrderSide.SELL
        order = Order(
            asset_id=asset_id,
            side=side,
            quantity=abs(trade_qty),
            order_type=OrderType.MARKET
        )
        self.broker.submit_order(order)
```

**Additional Changes**:
- Keep `buy_percent()` for backward compatibility but add deprecation warning
- Add `_current_market_data` cache (populated by engine before strategy runs)
- Add docstring warning about accumulation behavior

**Testing**:
```python
def test_order_target_percent_no_accumulation():
    # Call twice for same asset at 4%
    strategy.order_target_percent("AAPL", 0.04)
    strategy.order_target_percent("AAPL", 0.04)
    
    # Should result in 4% position, not 8%
    expected = portfolio.equity * 0.04
    actual = position.quantity * price
    assert abs(actual - expected) < expected * 0.01
```

**Dependencies**: None (can start immediately)

**Estimated Effort**: 2 hours (implementation + tests)

---

### TASK 2: Add Broker.process_batch_fills() (Issue #2 & #3 Fix)

**File**: `src/ml4t/backtest/execution/broker.py`

**Problem**: Current `on_market_event()` processes one asset at a time
- Can't see all assets simultaneously
- Wrong timing for moving pending→open orders

**Solution**: Add batch fill processing method
```python
def process_batch_fills(
    self, 
    timestamp: datetime, 
    market_data_map: dict[AssetId, MarketEvent]
) -> list[FillEvent]:
    """Attempt to fill ALL open orders against current market data batch.
    
    This fixes Issues #2 and #3:
    - #2: Processes all assets simultaneously (batch)
    - #3: Orders placed at T fill at T+1 (correct timing)
    
    Args:
        timestamp: Current simulation time
        market_data_map: Dict of asset_id → MarketEvent for this timestamp
    
    Returns:
        List of FillEvent objects generated
    """
    fills = []
    
    # First, move pending orders to open (execution delay)
    # This happens at START of new timestamp
    self._activate_pending_orders(timestamp)
    
    # Iterate over open orders (copy since we modify during iteration)
    for asset_id, orders in list(self._open_orders.items()):
        # Skip if no price data for this asset
        if asset_id not in market_data_map:
            continue
        
        market_event = market_data_map[asset_id]
        
        # Process each order for this asset
        for order in list(orders):
            if not order.is_active:
                continue
            
            # Check fill conditions (reuses existing logic)
            fill_result = self.fill_simulator.try_fill_order(
                order,
                market_event,
                current_cash=self._internal_portfolio.cash,
                current_position=self.get_position(asset_id),
            )
            
            if fill_result:
                # Apply fill immediately
                quantity_change = (
                    fill_result.fill_quantity 
                    if order.side == OrderSide.BUY 
                    else -fill_result.fill_quantity
                )
                self._internal_portfolio.update_position(
                    asset_id,
                    quantity_change,
                    fill_result.fill_price,
                    fill_result.commission,
                    fill_result.slippage,
                )
                
                fills.append(fill_result.fill_event)
                
                # Remove if fully filled
                if order.is_filled:
                    self.order_router.remove_order(order)
    
    return fills

def _activate_pending_orders(self, timestamp: datetime):
    """Move pending orders to open orders (execution delay handling)."""
    for asset_id, pending_list in list(self._pending_orders.items()):
        for order, submit_time in pending_list:
            # Orders submitted at T become active at T+1
            if order.order_type == OrderType.MARKET:
                self._open_orders[asset_id].append(order)
            # ... handle other order types ...
        
        # Clear pending for this asset
        self._pending_orders[asset_id].clear()
```

**Integration Points**:
- Must work with existing `OrderRouter`, `FillSimulator`, `BracketManager`
- Must update `_internal_portfolio` (Position tracker)
- Must handle risk manager hooks (if present)

**Testing**:
```python
def test_execution_delay_t_plus_1():
    # Submit order at day 0
    broker.submit_order(order, timestamp=day_0)
    assert order.status == OrderStatus.SUBMITTED
    
    # Process day 0 batch (order stays pending)
    fills = broker.process_batch_fills(day_0, events_day_0)
    assert len(fills) == 0
    assert order.status == OrderStatus.SUBMITTED
    
    # Process day 1 batch (order activates and fills)
    fills = broker.process_batch_fills(day_1, events_day_1)
    assert len(fills) == 1
    assert order.status == OrderStatus.FILLED
```

**Dependencies**: Understanding existing broker architecture (OrderRouter, FillSimulator)

**Estimated Effort**: 4 hours (implementation + tests + integration)

---

### TASK 3: Add DataFeed.stream_by_timestamp()

**File**: `src/ml4t/backtest/data/multi_symbol_feed.py`

**Problem**: Current `stream()` yields events one at a time

**Solution**: Add batch streaming method
```python
def stream_by_timestamp(self) -> Iterator[tuple[datetime, list[MarketEvent]]]:
    """Yield events grouped by timestamp for batch processing.
    
    Yields:
        (timestamp, events): Tuple of timestamp and all MarketEvents at that time
    
    Example:
        >>> for timestamp, events in feed.stream_by_timestamp():
        ...     print(f"{timestamp}: {len(events)} assets")
        2024-01-01: 500 assets
        2024-01-02: 500 assets
        ...
    """
    # Group by timestamp
    grouped = self._merged_df.group_by('timestamp', maintain_order=True)
    
    for timestamp_df in grouped:
        timestamp = timestamp_df['timestamp'][0]
        events = []
        
        for row in timestamp_df.iter_rows(named=True):
            # Create MarketEvent (reuse existing logic)
            event = self._create_market_event(row)
            events.append(event)
        
        yield (timestamp, events)

def _create_market_event(self, row: dict) -> MarketEvent:
    """Extract method for event creation (DRY)."""
    return MarketEvent(
        timestamp=row['timestamp'],
        asset_id=row['asset_id'],
        data_type=MarketDataType.BAR,
        price=row['close'],
        open=row['open'],
        high=row['high'],
        low=row['low'],
        close=row['close'],
        volume=row['volume'],
        signals=self._extract_signals(row),
        context=self._extract_context(row['timestamp']),
    )
```

**Testing**:
```python
def test_stream_by_timestamp_grouping():
    feed = MultiSymbolDataFeed(price_df, signals_df, context_df)
    
    batches = list(feed.stream_by_timestamp())
    
    # Should have 252 batches (252 trading days)
    assert len(batches) == 252
    
    # Each batch should have 500 events (500 stocks)
    for timestamp, events in batches:
        assert len(events) == 500
        assert all(e.timestamp == timestamp for e in events)
```

**Dependencies**: None (can run parallel with Tasks 1-2)

**Estimated Effort**: 2 hours (implementation + tests)

---

### TASK 4: Refactor BacktestEngine.run() for Batch Processing

**File**: `src/ml4t/backtest/engine.py`

**Problem**: Current loop iterates event-by-event (126,000 iterations)

**Solution**: Refactor to iterate by timestamp (252 iterations)
```python
def run(
    self,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    max_events: int | None = None,
) -> dict[str, Any]:
    """Run backtest using Batch Time-Slice Processing.
    
    This replaces atomic event-driven with batch processing:
    - Iterates by TIMESTAMP (not by event)
    - Processes all assets simultaneously at each timestamp
    - Ensures correct execution order: Fills → Portfolio Update → Strategy
    """
    logger.info("Starting backtest engine (batch mode)")
    self.start_time = datetime.now()
    
    # Initialize components
    self.strategy.on_start(self.portfolio, self.clock)
    self.broker.initialize(self.portfolio, self.clock)
    self.reporter.on_start()
    
    # Main event loop: BATCH MODE
    self.events_processed = 0
    
    # Check if data feed supports batch streaming
    if hasattr(self.data_feed, 'stream_by_timestamp'):
        # Batch mode (new behavior)
        for timestamp, event_batch in self.data_feed.stream_by_timestamp():
            self._process_time_slice(timestamp, event_batch)
            self.events_processed += len(event_batch)
            
            if max_events and self.events_processed >= max_events:
                break
    else:
        # Fallback: atomic mode (backward compatibility)
        # ... existing event-by-event loop ...
        pass
    
    # Finalize
    self.strategy.on_end()
    self.broker.finalize()
    self.reporter.on_end()
    
    self.end_time = datetime.now()
    duration = (self.end_time - self.start_time).total_seconds()
    
    logger.info(
        f"Backtest complete: {self.events_processed:,} events in {duration:.2f}s "
        f"({self.events_processed / duration:.0f} events/sec)"
    )
    
    return self._compile_results()

def _process_time_slice(self, timestamp: datetime, events: list[MarketEvent]) -> None:
    """Process all events at a single timestamp.
    
    Critical ordering (from expert review):
    1. EXECUTION: Fill old orders using new prices
    2. STATE UPDATE: Mark-to-market portfolio
    3. STRATEGY: Make decisions with updated state
    4. RISK: Validate and apply constraints
    """
    # Map events for easy lookup
    market_map = {e.asset_id: e for e in events if e.event_type == EventType.MARKET}
    
    # --- PHASE 1: EXECUTION (Broker) ---
    # Fill orders submitted at T-1 using prices at T
    # This correctly implements execution delay
    fills = self.broker.process_batch_fills(timestamp, market_map)
    
    # --- PHASE 2: UPDATE STATE (Portfolio) ---
    # Mark-to-market: Update position values with new prices
    for asset_id, event in market_map.items():
        position = self.portfolio.get_position(asset_id)
        if position:
            position.update_price(event.close)
    
    # Recalculate portfolio equity
    self.portfolio._update_equity()
    
    # --- PHASE 3: STRATEGY (Decision) ---
    # Strategy sees UPDATED portfolio (with fills from Phase 1)
    
    # Store market data in strategy cache for order_target_percent()
    self.strategy._current_market_data = market_map
    
    # Call strategy with batch
    if hasattr(self.strategy, 'on_timestamp_batch'):
        # Batch mode strategy
        context_dict = events[0].context if events else {}
        self.strategy.on_timestamp_batch(timestamp, events, context_dict)
    else:
        # Fallback: call on_market_event for each (backward compat)
        for event in events:
            context_dict = event.context or {}
            self.strategy.on_market_event(event, context_dict)
    
    # --- PHASE 4: RISK (Validation) ---
    # Risk manager hooks are already integrated:
    # - Hook B: validate_order() called via wrapped broker.submit_order()
    # - Hook C: check_position_exits() could be called here
    # - Hook D: record_fill() already called via fill event subscription
    
    if self.risk_manager:
        # Hook C: Check exit conditions for all positions
        for event in events:
            exit_orders = self.risk_manager.check_position_exits(
                event, self.broker, self.portfolio
            )
            for order in exit_orders:
                self.broker.submit_order(order, timestamp)
    
    # Dispatch fills to subscribers (reporter, portfolio)
    for fill_event in fills:
        self.clock.dispatch_event(fill_event)
```

**Backward Compatibility**:
- Auto-detect batch vs atomic mode based on data feed type
- Keep atomic mode for existing strategies
- Gradual migration path

**Testing**:
```python
def test_batch_mode_iteration_count():
    feed = MultiSymbolDataFeed(...)  # 500 stocks × 252 days
    engine = BacktestEngine(data_feed=feed, ...)
    
    with patch.object(engine, '_process_time_slice') as mock_process:
        engine.run()
        
        # Should call _process_time_slice 252 times (not 126,000)
        assert mock_process.call_count == 252
```

**Dependencies**: Tasks 1, 2, 3 complete

**Estimated Effort**: 6 hours (implementation + tests + integration)

---

### TASK 5: Update Example Strategy

**File**: `examples/integrated/top25_using_engine.py`

**Problem**: Example uses `buy_percent()` which accumulates positions

**Solution**: Replace with `order_target_percent()`
```python
class Top25MLStrategy(Strategy):
    def _process_daily_batch(self):
        # ... ranking logic ...
        
        # Calculate target weights (4% per position)
        target_pct = 1.0 / self.n_positions
        target_weights = {asset_id: target_pct for asset_id in top_assets}
        
        # Use order_target_percent() instead of buy_percent()
        for asset_id, target_pct in target_weights.items():
            self.order_target_percent(asset_id, target_pct)
        
        # Or use rebalance_to_weights() helper:
        # current_prices = {e.asset_id: e.close for e in self.daily_events}
        # self.rebalance_to_weights(target_weights, current_prices, tolerance=0.01)
```

**Testing**:
Run updated example and verify:
- Final positions: ~25 (not 75+)
- Total return: ~+770% (not -95%)
- Position sizes: ~4% each (not 12%+)

**Dependencies**: Task 1 complete

**Estimated Effort**: 1 hour (update + validation)

---

### TASK 6: Integration Validation

**File**: `tests/integration/test_batch_processing.py`

**Goal**: Prove Phase 1 correctness

**Test Suite**:
```python
class TestBatchProcessing:
    def test_order_target_percent_no_accumulation(self):
        """Issue #1 fix: order_target_percent doesn't accumulate."""
        # Detailed test from Task 1
        
    def test_execution_delay_t_plus_1(self):
        """Issue #3 fix: Orders fill at T+1, not T+2."""
        # Detailed test from Task 2
        
    def test_batch_iteration_count(self):
        """Issue #2 fix: 252 iterations, not 126,000."""
        # Detailed test from Task 4
        
    def test_top25_strategy_returns(self):
        """Integration: Engine matches manual loop returns."""
        # Run example with engine
        results = run_top25_strategy_with_engine()
        
        # Compare to manual loop baseline
        assert abs(results['total_return'] - 770.0) < 5.0  # Within 5%
        assert 20 <= len(results['final_positions']) <= 30  # ~25 positions
        
    def test_cross_framework_validation(self):
        """Validation: Compare to VectorBT/Backtrader."""
        # Run same strategy in all 3 frameworks
        ml4t_results = run_with_ml4t_backtest()
        vectorbt_results = run_with_vectorbt()
        backtrader_results = run_with_backtrader()
        
        # Should agree within 1%
        assert abs(ml4t_results['return'] - vectorbt_results['return']) < 1.0
        assert abs(ml4t_results['return'] - backtrader_results['return']) < 1.0
```

**Success Criteria**:
✅ All 5 tests pass
✅ Engine returns: +770% ± 5%
✅ Final positions: 20-30
✅ Iterations: 252
✅ Execution delay: T+1

**Dependencies**: Tasks 1-5 complete

**Estimated Effort**: 3 hours (test writing + validation + debugging)

---

## Phase 1 Summary

**Total Estimated Effort**: 18 hours (2-3 days)

**Task Order**:
1. Task 1 (Strategy API) - 2 hours
2. Task 2 (Broker batching) - 4 hours  
3. Task 3 (DataFeed batching) - 2 hours [parallel with 1-2]
4. Task 4 (Engine refactor) - 6 hours
5. Task 5 (Example update) - 1 hour
6. Task 6 (Validation) - 3 hours

**Checkpoint**: Phase 1 complete when all Task 6 tests pass.

---

## Phase 2: Performance Optimization (Follow-up Tasks)

### TASK 7: Polars Batch Operations in Broker

**Goal**: Replace Python loops with Polars vectorized operations

**Target**: 2-3× speedup in fill processing

**Implementation**:
```python
def process_batch_fills_vectorized(
    self, 
    timestamp: datetime, 
    market_data_df: pl.DataFrame
) -> list[FillEvent]:
    """Vectorized fill processing using Polars."""
    # Convert orders to DataFrame
    orders_df = self._orders_to_polars()
    
    # Join with market data
    fillable = orders_df.join(market_data_df, on='asset_id', how='inner')
    
    # Vectorized fill conditions
    fills_df = fillable.filter(
        (pl.col('order_type') == 'MARKET') |
        ((pl.col('order_type') == 'LIMIT') & (pl.col('price') <= pl.col('limit_price')))
    )
    
    # Apply fills in bulk
    return self._apply_fills_bulk(fills_df)
```

**Estimated Effort**: 4 hours

---

### TASK 8: Numba JIT for Hot Paths

**Goal**: Compile calculation-heavy loops to machine code

**Target**: 3-5× speedup for position delta calculations

**Implementation**:
```python
from numba import njit

@njit
def calculate_position_deltas(
    target_values: np.ndarray,
    current_quantities: np.ndarray,
    prices: np.ndarray,
) -> np.ndarray:
    """Numba-compiled rebalancing delta calculation."""
    target_quantities = target_values / prices
    deltas = target_quantities - current_quantities
    return deltas

def order_target_percent_batch(self, target_weights: dict):
    """Batch version using Numba."""
    # Extract (zero-copy)
    assets = list(target_weights.keys())
    target_vals = np.array([self.portfolio.equity * w for w in target_weights.values()])
    current_qtys = np.array([self.get_position(a) for a in assets])
    prices = np.array([self._current_market_data[a].close for a in assets])
    
    # Process (C-speed)
    deltas = calculate_position_deltas(target_vals, current_qtys, prices)
    
    # Wrap
    for asset, delta in zip(assets, deltas):
        if abs(delta) > 0.01:
            self._submit_delta_order(asset, delta)
```

**Estimated Effort**: 3 hours

---

### TASK 9: Optimize Portfolio Mark-to-Market

**Goal**: Vectorize position value updates

**Target**: 2× speedup for portfolio updates

**Implementation**:
```python
def update_market_values_batch(self, market_data_map: dict[AssetId, MarketEvent]):
    """Vectorized mark-to-market."""
    # Extract into arrays
    asset_ids = list(self.positions.keys())
    prices = np.array([
        market_data_map[aid].close 
        for aid in asset_ids 
        if aid in market_data_map
    ])
    
    # Vectorized unrealized P&L
    cost_basis = np.array([self.positions[aid].cost_basis for aid in asset_ids])
    quantities = np.array([self.positions[aid].quantity for aid in asset_ids])
    
    current_values = prices * quantities
    unrealized_pnl = current_values - cost_basis
    
    # Update positions
    for i, asset_id in enumerate(asset_ids):
        self.positions[asset_id].unrealized_pnl = unrealized_pnl[i]
        self.positions[asset_id].last_price = prices[i]
```

**Estimated Effort**: 2 hours

---

### TASK 10: Performance Benchmark

**Goal**: Validate >50,000 events/sec throughput

**Testing**:
```python
def test_batch_processing_throughput():
    """Verify Phase 2 performance targets."""
    feed = MultiSymbolDataFeed(...)  # 500 stocks × 252 days = 126k events
    engine = BacktestEngine(data_feed=feed, ...)
    
    start = time.time()
    engine.run()
    elapsed = time.time() - start
    
    throughput = 126_000 / elapsed
    
    # Phase 2 target
    assert throughput > 50_000, f"Throughput {throughput:.0f} < 50k events/sec"
    
    # Memory stable
    assert memory_usage_mb < initial_memory * 1.5  # Max 50% growth
```

**Success Criteria**:
- ✅ Throughput: >50,000 events/sec
- ✅ Speedup: 16.7× vs baseline (3k events/sec)
- ✅ Memory: Stable (no leaks)

**Estimated Effort**: 2 hours (benchmark + profiling)

---

## Phase 2 Summary

**Total Estimated Effort**: 11 hours (1-2 days)

**Task Order**:
7. Polars batch operations - 4 hours
8. Numba JIT - 3 hours
9. Portfolio optimization - 2 hours
10. Performance benchmark - 2 hours

**Checkpoint**: Phase 2 complete when Task 10 benchmark passes.

---

## Risk Assessment & Mitigation

### High Risk: Breaking Changes

**Risk**: Changing `run()` method breaks existing code

**Mitigation**:
- Auto-detect batch vs atomic mode based on feed type
- Keep backward compatibility for atomic mode
- Add deprecation warnings for old API
- Provide migration guide

### Medium Risk: Integration with Risk Manager

**Risk**: Batch processing breaks risk manager hooks

**Mitigation**:
- Risk manager operates on MarketEvent objects (transparent)
- Hooks are already designed for batch (Hook B, C, D)
- Test with and without risk manager enabled

### Medium Risk: Order Router Complexity

**Risk**: process_batch_fills() conflicts with OrderRouter/BracketManager

**Mitigation**:
- Reuse existing _check_fill() logic
- Preserve order routing semantics
- Test bracket orders in batch mode

### Low Risk: Test Coverage

**Risk**: Insufficient validation of batch vs atomic equivalence

**Mitigation**:
- Cross-framework validation (VectorBT, Backtrader)
- Compare batch vs atomic mode results
- Comprehensive integration tests

---

## Execution Plan for /next Workflow

**Phase 1 (Correctness)**:
```bash
# Day 1
/next  # → Execute Task 1 (Strategy API)
/next  # → Execute Task 2 (Broker batching)
/next  # → Execute Task 3 (DataFeed batching)

# Day 2
/next  # → Execute Task 4 (Engine refactor)
/next  # → Execute Task 5 (Example update)

# Day 3
/next  # → Execute Task 6 (Validation)
# CHECKPOINT: All validation tests pass?
```

**Phase 2 (Performance)**:
```bash
# Day 4-5
/next  # → Execute Task 7 (Polars)
/next  # → Execute Task 8 (Numba)
/next  # → Execute Task 9 (Portfolio)
/next  # → Execute Task 10 (Benchmark)
# CHECKPOINT: >50k events/sec achieved?
```

---

## Success Metrics

### Phase 1 Complete:
- ✅ Return: +770% ± 5% (matching manual loop)
- ✅ Positions: 20-30 final positions
- ✅ Timing: Orders fill at T+1
- ✅ Iterations: 252 (not 126,000)
- ✅ Tests: 5/5 passing

### Phase 2 Complete:
- ✅ Throughput: >50,000 events/sec
- ✅ Speedup: 16.7× improvement
- ✅ Memory: Stable, no leaks
- ✅ Quality: All existing tests pass

---

## References

- Expert Review #1 (Architecture): `.claude/code_review/20251119/response_01.md`
- Expert Review #2 (Performance): `.claude/code_review/20251119/response_02.md`
- Working Reference: `examples/integrated/top25_ml_strategy_complete.py`
- Broken Example: `examples/integrated/top25_using_engine.py`
- Current Engine: `src/ml4t/backtest/engine.py`
- Current Strategy: `src/ml4t/backtest/strategy/base.py`
- Current Broker: `src/ml4t/backtest/execution/broker.py`

---

**END OF PLAN**
