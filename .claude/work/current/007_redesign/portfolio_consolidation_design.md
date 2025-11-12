# Portfolio Consolidation Design - Unified API

**Date**: 2025-11-12
**Goal**: Merge 3 portfolio classes into 1 world-class API
**Competitors**: VectorBT Pro (benchmark for UX)

---

## Design Principles

1. **User-First**: One clear API, organized by task
2. **Best-of-Breed**: Choose the best implementation from each class
3. **No Compromise**: Real-time tracking + comprehensive analytics
4. **Professional Polish**: VectorBT-level UX with institutional rigor
5. **Backward Compatible**: Maintain existing integration points

---

## API Organization (6 Functional Groups)

```python
class Portfolio:
    """Unified portfolio management with institutional-grade tracking."""

    # ============================================================
    # GROUP 1: CORE POSITION & CASH MANAGEMENT (8 methods)
    # ============================================================
    def __init__(initial_cash, currency="USD", track_history=True, precision_manager=None)
    def get_position(asset_id) -> Position | None
    def update_position(asset_id, quantity_change, price, commission, slippage, asset_precision_manager)
    def update_prices(prices: dict[AssetId, float]) -> None

    @property cash -> float
    @property equity -> float
    @property returns -> float
    @property unrealized_pnl -> float

    # ============================================================
    # GROUP 2: EVENT HANDLING (4 methods)
    # ============================================================
    def initialize() -> None
    def on_fill_event(event: FillEvent) -> None
    def on_market_event(event: MarketEvent) -> None  # Renamed from update_market_value
    def finalize(timestamp: datetime | None) -> None

    # ============================================================
    # GROUP 3: PERFORMANCE METRICS (8 methods)
    # ============================================================
    def get_performance_metrics() -> dict[str, Any]
    def calculate_sharpe_ratio() -> float
    def calculate_win_rate() -> float
    def calculate_profit_factor() -> float
    def calculate_avg_commission() -> float
    def calculate_avg_slippage() -> float

    @property max_drawdown -> float
    @property high_water_mark -> float

    # ============================================================
    # GROUP 4: TRADE & POSITION DATA (6 methods)
    # ============================================================
    def get_positions() -> pl.DataFrame
    def get_trades() -> pl.DataFrame
    def get_equity_curve() -> pl.DataFrame
    def get_returns_series() -> pl.Series
    def get_position_summary() -> dict[str, Any]
    def get_summary() -> dict[str, Any]  # Comprehensive

    # ============================================================
    # GROUP 5: STATE MANAGEMENT (3 methods)
    # ============================================================
    def get_current_state(timestamp) -> PortfolioState
    def save_state(timestamp) -> None
    def reset() -> None

    # ============================================================
    # GROUP 6: RISK ANALYTICS (properties, tracked real-time)
    # ============================================================
    @property max_leverage -> float
    @property max_concentration -> float
    @property current_leverage -> float
    @property current_concentration -> float
```

**Total**: 29 public methods + 10 properties = 39 API points

---

## Method-by-Method Design Decisions

### GROUP 1: Core Position & Cash Management

#### `__init__(initial_cash, currency="USD", track_history=True, precision_manager=None)`

**Sources**: Portfolio, SimplePortfolio, PortfolioAccounting (all three!)

**Design**:
```python
def __init__(
    self,
    initial_cash: Cash = 100000.0,
    currency: str = "USD",  # From SimplePortfolio
    track_history: bool = True,  # From PortfolioAccounting
    precision_manager: Optional[PrecisionManager] = None,  # From Portfolio
):
```

**Rationale**: Merge all constructor parameters. Defaults match current behavior.

**New Attributes** (merge all three):
```python
# From Portfolio (core)
self.initial_cash = float(initial_cash)
self.cash = float(initial_cash)
self.positions: dict[AssetId, Position] = {}
self.precision_manager = precision_manager
self.total_commission = 0.0
self.total_slippage = 0.0
self.total_realized_pnl = 0.0
self.asset_realized_pnl: dict[AssetId, float] = {}
self.state_history: list[PortfolioState] = []

# From SimplePortfolio (event-driven)
self.currency = currency
self.current_prices: dict[AssetId, float] = {}

# From PortfolioAccounting (analytics) - STORE FILLEVENT OBJECTS
self.fills: list[FillEvent] = []  # Better than dict storage
self.track_history = track_history

# Real-time metric tracking (PortfolioAccounting approach)
self.high_water_mark = float(initial_cash)
self.max_drawdown = 0.0
self.daily_returns: list[float] = []
self.timestamps: list[datetime] = []
self.equity_curve: list[float] = []
self.max_leverage = 0.0
self.max_concentration = 0.0
```

#### Core Methods: Keep Portfolio implementations

**No changes needed**:
- `get_position(asset_id)` - Portfolio implementation is perfect
- `update_position(...)` - Portfolio implementation is perfect
- `update_prices(...)` - Portfolio implementation is perfect
- Properties (`cash`, `equity`, `returns`, `unrealized_pnl`) - Portfolio implementations perfect

---

### GROUP 2: Event Handling

#### `initialize() -> None`

**Source**: SimplePortfolio only

**Keep as-is**:
```python
def initialize(self) -> None:
    """Initialize portfolio for new backtest."""
    logger.debug(f"Initializing portfolio with ${self.initial_cash:,.2f} {self.currency}")
```

**Rationale**: Useful lifecycle hook, no conflicts.

#### `on_fill_event(event: FillEvent) -> None`

**Sources**: SimplePortfolio (dict storage) vs PortfolioAccounting (FillEvent storage)

**Choose**: PortfolioAccounting approach (store FillEvent objects)

**Implementation**:
```python
def on_fill_event(self, event: FillEvent) -> None:
    """Handle fill event from broker.

    Args:
        event: Fill event with execution details
    """
    # Store full FillEvent (PortfolioAccounting approach - more complete)
    self.fills.append(event)

    # Determine quantity change
    quantity_change = event.fill_quantity if event.side.value in ["buy", "BUY"] else -event.fill_quantity

    # Update position (Portfolio core logic)
    self.update_position(
        asset_id=event.asset_id,
        quantity_change=quantity_change,
        price=float(event.fill_price),
        commission=event.commission,
        slippage=event.slippage,
    )

    # Update real-time metrics (PortfolioAccounting approach)
    self._update_metrics(event.timestamp)

    # Log (SimplePortfolio style)
    logger.info(
        f"Fill: {event.side.value.upper()} {event.fill_quantity} {event.asset_id} "
        f"@ ${float(event.fill_price):.2f} (commission: ${event.commission:.2f})"
    )
```

**Why FillEvent storage over dict?**
- Contains ALL fill details (15 fields vs 8 fields)
- Enables accurate lot matching (profit factor, win rate)
- No data loss
- Can always convert to dict for backward compatibility

#### `on_market_event(event: MarketEvent) -> None`

**Source**: SimplePortfolio (`update_market_value`)

**Renamed** for consistency with `on_fill_event`:
```python
def on_market_event(self, event: MarketEvent) -> None:
    """Update portfolio with latest market prices.

    Args:
        event: Market event with price data
    """
    # Update current price for the asset (SimplePortfolio logic)
    if hasattr(event, "close") and event.close is not None:
        self.current_prices[event.asset_id] = float(event.close)
    elif hasattr(event, "price") and event.price is not None:
        self.current_prices[event.asset_id] = float(event.price)

    # Update all positions with latest prices (Portfolio core logic)
    self.update_prices(self.current_prices)

    # Update metrics with new prices (PortfolioAccounting approach)
    self._update_metrics(event.timestamp)
```

**Backward compatibility**: Keep `update_market_value` as alias.

#### `finalize(timestamp: Optional[datetime] = None) -> None`

**Source**: SimplePortfolio only

**Keep but simplify** (remove trade P&L calculation - handled by lot matching):
```python
def finalize(self, timestamp: Optional[datetime] = None) -> None:
    """Finalize portfolio at end of backtest.

    Args:
        timestamp: Current simulation time
    """
    # Save final state
    self.save_state(timestamp or datetime.now())

    logger.info(f"Portfolio finalized. Final equity: ${self.equity:,.2f}")
```

**Rationale**: Trade P&L is calculated in real-time via lot matching, no need for post-processing.

---

### GROUP 3: Performance Metrics

#### `get_performance_metrics() -> dict[str, Any]`

**Source**: PortfolioAccounting (comprehensive)

**Keep PortfolioAccounting implementation** - it's superior:
```python
def get_performance_metrics(self) -> dict[str, Any]:
    """Get comprehensive performance metrics."""
    metrics = {
        # Returns
        "total_return": self.returns,
        "total_pnl": self.total_realized_pnl + self.unrealized_pnl,
        "realized_pnl": self.total_realized_pnl,
        "unrealized_pnl": self.unrealized_pnl,

        # Risk
        "max_drawdown": self.max_drawdown,
        "max_leverage": self.max_leverage,
        "max_concentration": self.max_concentration,

        # Current state
        "current_equity": self.equity,
        "current_cash": self.cash,

        # Costs
        "total_commission": self.total_commission,
        "total_slippage": self.total_slippage,

        # Trading
        "num_trades": len(self.fills),
        "win_rate": self.calculate_win_rate(),
        "profit_factor": self.calculate_profit_factor(),
        "avg_commission_per_trade": self.calculate_avg_commission(),
        "avg_slippage_per_trade": self.calculate_avg_slippage(),
    }

    # Sharpe ratio (if enough data)
    if len(self.daily_returns) > 1:
        sharpe = self.calculate_sharpe_ratio()
        if sharpe is not None:
            metrics["sharpe_ratio"] = sharpe

    return metrics
```

#### Individual metric methods

**All from PortfolioAccounting** (lot-matching based, more accurate):
- `calculate_sharpe_ratio()` - PortfolioAccounting
- `calculate_win_rate()` - PortfolioAccounting (lot matching)
- `calculate_profit_factor()` - PortfolioAccounting (lot matching)
- `calculate_avg_commission()` - PortfolioAccounting
- `calculate_avg_slippage()` - PortfolioAccounting

**Keep as-is** from PortfolioAccounting. These are well-implemented.

#### Properties

```python
@property
def max_drawdown(self) -> float:
    """Maximum drawdown (real-time tracked)."""
    return self.max_drawdown

@property
def high_water_mark(self) -> float:
    """Highest equity achieved."""
    return self.high_water_mark
```

---

### GROUP 4: Trade & Position Data

#### `get_positions() -> pl.DataFrame`

**Sources**: SimplePortfolio and PortfolioAccounting (nearly identical)

**Choose**: PortfolioAccounting (includes total_pnl column):
```python
def get_positions(self) -> pl.DataFrame:
    """Get DataFrame of current positions."""
    if not self.positions:
        return pl.DataFrame()

    positions_data = []
    for position in self.positions.values():
        realized_pnl = self.asset_realized_pnl.get(position.asset_id, 0.0)
        total_pnl = position.unrealized_pnl + realized_pnl

        positions_data.append({
            "asset_id": position.asset_id,
            "quantity": position.quantity,
            "cost_basis": position.cost_basis,
            "last_price": position.last_price,
            "market_value": position.market_value,
            "unrealized_pnl": position.unrealized_pnl,
            "realized_pnl": realized_pnl,
            "total_pnl": total_pnl,  # PortfolioAccounting adds this
        })

    return pl.DataFrame(positions_data)
```

#### `get_trades() -> pl.DataFrame`

**Sources**: SimplePortfolio (dict-based) vs PortfolioAccounting (FillEvent-based)

**Choose**: PortfolioAccounting (FillEvent storage is richer):
```python
def get_trades(self) -> pl.DataFrame:
    """Get all trades as DataFrame."""
    if not self.fills:
        return pl.DataFrame()

    trades_data = []
    for fill in self.fills:
        trades_data.append({
            "timestamp": fill.timestamp,
            "order_id": fill.order_id,
            "trade_id": fill.trade_id,
            "asset_id": fill.asset_id,
            "side": fill.side.value,
            "quantity": fill.fill_quantity,
            "price": fill.fill_price,
            "commission": fill.commission,
            "slippage": fill.slippage,
            "total_cost": fill.total_cost,
        })

    return pl.DataFrame(trades_data)
```

**Backward compatibility note**: SimplePortfolio included "pnl" column. We can add this via lot matching if needed, but FillEvent storage is more flexible.

#### `get_equity_curve() -> pl.DataFrame`

**Source**: PortfolioAccounting (`get_equity_curve_df`)

**Rename** for consistency and keep:
```python
def get_equity_curve(self) -> pl.DataFrame:
    """Get equity curve as DataFrame."""
    if not self.timestamps:
        return pl.DataFrame()

    return pl.DataFrame({
        "timestamp": self.timestamps,
        "equity": self.equity_curve,
        "returns": [0.0, *self.daily_returns],  # Pad with 0 for first day
    })
```

#### `get_returns_series() -> pl.Series`

**Source**: SimplePortfolio (`get_returns`)

**Rename** for clarity and keep:
```python
def get_returns_series(self) -> pl.Series:
    """Get returns as Polars Series."""
    if not self.state_history:
        return pl.Series([])

    returns = []
    prev_value = self.initial_cash

    for state in self.state_history:
        current_value = state.equity
        ret = (current_value - prev_value) / prev_value if prev_value != 0 else 0
        returns.append(ret)
        prev_value = current_value

    return pl.Series(returns)
```

#### `get_position_summary() -> dict[str, Any]`

**Source**: Portfolio base

**Keep as-is** - it's good.

#### `get_summary() -> dict[str, Any]`

**Source**: PortfolioAccounting

**Keep as-is** - combines position summary + metrics:
```python
def get_summary(self) -> dict[str, Any]:
    """Get comprehensive portfolio summary."""
    summary = self.get_position_summary()
    summary.update(self.get_performance_metrics())
    return summary
```

---

### GROUP 5: State Management

#### All from Portfolio base - keep as-is

- `get_current_state(timestamp) -> PortfolioState` - Portfolio
- `save_state(timestamp) -> None` - Portfolio
- `reset() -> None` - **Merge all three**

**`reset()` - Merged implementation**:
```python
def reset(self) -> None:
    """Reset portfolio to initial state."""
    # Core state (Portfolio)
    self.cash = self.initial_cash
    self.positions.clear()
    self.total_commission = 0.0
    self.total_slippage = 0.0
    self.total_realized_pnl = 0.0
    self.asset_realized_pnl.clear()
    self.state_history.clear()

    # Event-driven state (SimplePortfolio)
    self.current_prices.clear()
    self.fills.clear()  # Changed from trades

    # Analytics state (PortfolioAccounting)
    self.high_water_mark = float(self.initial_cash)
    self.max_drawdown = 0.0
    self.daily_returns.clear()
    self.timestamps.clear()
    self.equity_curve.clear()
    self.max_leverage = 0.0
    self.max_concentration = 0.0

    # Re-initialize equity curve if tracking
    if self.track_history:
        self.equity_curve.append(self.initial_cash)
```

---

### GROUP 6: Risk Analytics (Properties)

**Source**: PortfolioAccounting real-time tracking

**Add properties**:
```python
@property
def max_leverage(self) -> float:
    """Maximum leverage achieved."""
    return self._max_leverage

@property
def max_concentration(self) -> float:
    """Maximum position concentration."""
    return self._max_concentration

@property
def current_leverage(self) -> float:
    """Current leverage ratio."""
    if not self.positions:
        return 0.0
    state = self.get_current_state(datetime.now())
    return state.leverage

@property
def current_concentration(self) -> float:
    """Current position concentration."""
    if not self.positions:
        return 0.0
    state = self.get_current_state(datetime.now())
    return state.concentration
```

---

## Private Helper Methods

### `_update_metrics(timestamp: datetime) -> None`

**Source**: PortfolioAccounting

**Critical method** - handles real-time metric tracking:
```python
def _update_metrics(self, timestamp: datetime) -> None:
    """Update performance and risk metrics (real-time tracking).

    Called after every fill and price update.
    """
    current_equity = self.equity

    # Update high water mark and drawdown
    if current_equity > self.high_water_mark:
        self.high_water_mark = current_equity

    if self.high_water_mark > 0:
        current_drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

    # Track equity curve and returns
    if self.track_history:
        self.timestamps.append(timestamp)
        self.equity_curve.append(current_equity)

        # Calculate return if we have previous data
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]
            if prev_equity > 0:
                daily_return = (current_equity - prev_equity) / prev_equity
                self.daily_returns.append(daily_return)

    # Update risk metrics from current state
    state = self.get_current_state(timestamp)
    self.max_leverage = max(self.max_leverage, state.leverage)
    self.max_concentration = max(self.max_concentration, state.concentration)

    # Save state if tracking history
    if self.track_history:
        self.save_state(timestamp)
```

---

## Backward Compatibility Layer

### For SimplePortfolio users

**Keep these aliases**:
```python
# Aliases for backward compatibility (will deprecate in v2.0)
def update_market_value(self, event: MarketEvent) -> None:
    """Deprecated: Use on_market_event() instead."""
    warnings.warn(
        "update_market_value() is deprecated, use on_market_event()",
        DeprecationWarning,
        stacklevel=2
    )
    return self.on_market_event(event)

def get_total_value(self) -> float:
    """Deprecated: Use equity property instead."""
    warnings.warn(
        "get_total_value() is deprecated, use .equity property",
        DeprecationWarning,
        stacklevel=2
    )
    return self.equity

def calculate_metrics(self) -> dict[str, Any]:
    """Deprecated: Use get_performance_metrics() instead."""
    warnings.warn(
        "calculate_metrics() is deprecated, use get_performance_metrics()",
        DeprecationWarning,
        stacklevel=2
    )
    return self.get_performance_metrics()

def get_returns(self) -> pl.Series:
    """Deprecated: Use get_returns_series() instead."""
    warnings.warn(
        "get_returns() is deprecated, use get_returns_series()",
        DeprecationWarning,
        stacklevel=2
    )
    return self.get_returns_series()
```

### For PortfolioAccounting users

**Keep these aliases**:
```python
def process_fill(self, fill_event: FillEvent) -> None:
    """Deprecated: Use on_fill_event() instead."""
    warnings.warn(
        "process_fill() is deprecated, use on_fill_event()",
        DeprecationWarning,
        stacklevel=2
    )
    return self.on_fill_event(fill_event)

def get_trades_df(self) -> Optional[pl.DataFrame]:
    """Deprecated: Use get_trades() instead."""
    warnings.warn(
        "get_trades_df() is deprecated, use get_trades()",
        DeprecationWarning,
        stacklevel=2
    )
    return self.get_trades()

def get_equity_curve_df(self) -> Optional[pl.DataFrame]:
    """Deprecated: Use get_equity_curve() instead."""
    warnings.warn(
        "get_equity_curve_df() is deprecated, use get_equity_curve()",
        DeprecationWarning,
        stacklevel=2
    )
    return self.get_equity_curve()

def get_positions_df(self) -> Optional[pl.DataFrame]:
    """Deprecated: Use get_positions() instead."""
    warnings.warn(
        "get_positions_df() is deprecated, use get_positions()",
        DeprecationWarning,
        stacklevel=2
    )
    return self.get_positions()
```

---

## Implementation Plan

### Phase 1: Core Implementation (8 hours)
1. Create new `portfolio.py` with merged `__init__`
2. Keep all Portfolio core methods (already perfect)
3. Add event handlers (`on_fill_event`, `on_market_event`)
4. Add `_update_metrics` helper
5. Test: Position tracking + event handling

### Phase 2: Analytics (6 hours)
6. Add all metric calculation methods
7. Add DataFrame output methods
8. Add risk analytics properties
9. Test: Metrics calculation accuracy

### Phase 3: Backward Compatibility (2 hours)
10. Add deprecated method aliases
11. Add warnings
12. Test: Old code still works

### Phase 4: Integration Updates (12 hours)
13. Update BacktestEngine (remove SimplePortfolio import)
14. Update Reporting (remove PortfolioAccounting import)
15. Update __init__.py (remove old classes)
16. Test: Engine + Reporting work with new Portfolio

### Phase 5: Test Migration (16 hours)
17. Merge test_portfolio.py tests
18. Migrate test_portfolio_get_position.py tests
19. Update all test fixtures
20. Run full test suite
21. Fix any failures

### Phase 6: Documentation (4 hours)
22. Update API reference
23. Create migration guide
24. Add usage examples
25. Update CLAUDE.md

**Total: 48 hours (~1.5 weeks)**

---

## Success Criteria

1. ✅ All 48 existing tests pass (after migration)
2. ✅ BacktestEngine works with new Portfolio
3. ✅ Reporting works with new Portfolio
4. ✅ Validation tests pass (VectorBT exact matching)
5. ✅ No performance regression
6. ✅ Deprecation warnings work
7. ✅ API is intuitive (user testing)

---

## Risks & Mitigation

### Risk 1: Breaking existing code
**Mitigation**: Comprehensive backward compatibility layer with deprecation warnings

### Risk 2: Test failures during migration
**Mitigation**: Migrate tests incrementally, one group at a time

### Risk 3: Performance regression
**Mitigation**: Benchmark before/after, _update_metrics is opt-in via track_history

### Risk 4: Metric calculation differences
**Mitigation**: Document lot-matching algorithm, add tests comparing to VectorBT

---

## Next Steps

1. **Review this design** with user - get approval
2. **Implement Phase 1** - Core functionality
3. **Test Phase 1** - Verify basics work
4. **Continue through phases** incrementally
5. **Full validation** - VectorBT exact matching

---

**Design Status**: ✅ Complete - Ready for implementation
**Estimated Time**: 48 hours total (1.5 weeks)
**Target Release**: v1.0 with unified Portfolio API
