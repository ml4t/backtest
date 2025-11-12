# Enhanced Facade + Composition Architecture - Detailed Design

**Date**: 2025-11-12
**Status**: Approved - Ready for Implementation
**Target Quality**: 9/10 (professional without over-engineering)

---

## Architecture Overview

```
Portfolio (Facade)
├── PositionTracker (core position/cash tracking)
├── PerformanceAnalyzer (metrics and analytics)
└── TradeJournal (trade history and persistence)
```

---

## Component 1: PositionTracker

**File**: `portfolio/portfolio.py`
**Lines**: ~250
**Responsibility**: Core position and cash tracking ONLY (no analytics)

### API

```python
class PositionTracker:
    """Core position and cash tracking - pure domain logic."""

    def __init__(
        self,
        initial_cash: Cash = 100000.0,
        precision_manager: Optional[PrecisionManager] = None,
    ):
        """Initialize position tracker.

        Args:
            initial_cash: Starting cash balance
            precision_manager: PrecisionManager for cash rounding (USD precision)
        """
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: dict[AssetId, Position] = {}
        self.precision_manager = precision_manager

        # Cumulative costs and P&L
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_realized_pnl = 0.0
        self.asset_realized_pnl: dict[AssetId, float] = {}

    def get_position(self, asset_id: AssetId) -> Position | None:
        """Get position for an asset."""
        return self.positions.get(asset_id)

    def update_position(
        self,
        asset_id: AssetId,
        quantity_change: Quantity,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        asset_precision_manager: Optional[PrecisionManager] = None,
    ) -> None:
        """Update a position with a trade.

        Args:
            asset_id: Asset identifier
            quantity_change: Change in quantity (positive for buy, negative for sell)
            price: Execution price
            commission: Commission paid
            slippage: Slippage cost
            asset_precision_manager: PrecisionManager for asset-specific quantity precision
        """
        # Get or create position
        if asset_id not in self.positions:
            self.positions[asset_id] = Position(
                asset_id=asset_id,
                precision_manager=asset_precision_manager or self.precision_manager,
            )

        position = self.positions[asset_id]

        # Update position
        if quantity_change > 0:
            # Buy
            position.add_shares(quantity_change, price)
            cash_change = quantity_change * price + commission
            self.cash = float(self.cash) - cash_change
            if self.precision_manager:
                self.cash = self.precision_manager.round_cash(self.cash)
        else:
            # Sell
            realized_pnl = position.remove_shares(-quantity_change, price)
            cash_change = (-quantity_change) * price - commission
            self.cash = float(self.cash) + cash_change
            if self.precision_manager:
                self.cash = self.precision_manager.round_cash(self.cash)

            # Track realized P&L
            self.total_realized_pnl += realized_pnl
            if self.precision_manager:
                self.total_realized_pnl = self.precision_manager.round_cash(self.total_realized_pnl)

            if asset_id not in self.asset_realized_pnl:
                self.asset_realized_pnl[asset_id] = 0.0
            self.asset_realized_pnl[asset_id] += realized_pnl
            if self.precision_manager:
                self.asset_realized_pnl[asset_id] = self.precision_manager.round_cash(
                    self.asset_realized_pnl[asset_id]
                )

        # Track costs
        self.total_commission += commission
        self.total_slippage += slippage
        if self.precision_manager:
            self.total_commission = self.precision_manager.round_cash(self.total_commission)
            self.total_slippage = self.precision_manager.round_cash(self.total_slippage)

        # Remove empty positions
        is_empty = position.quantity == 0
        if asset_precision_manager:
            is_empty = is_empty or asset_precision_manager.is_position_zero(position.quantity)
        if is_empty:
            del self.positions[asset_id]

    def update_prices(self, prices: dict[AssetId, float]) -> None:
        """Update all positions with new market prices."""
        for asset_id, price in prices.items():
            if asset_id in self.positions:
                self.positions[asset_id].update_price(price)

    @property
    def equity(self) -> float:
        """Total equity (cash + positions)."""
        position_value = sum(p.market_value for p in self.positions.values())
        return float(self.cash) + position_value

    @property
    def returns(self) -> float:
        """Simple returns from initial capital."""
        if self.initial_cash == 0:
            return 0.0
        return (self.equity - float(self.initial_cash)) / float(self.initial_cash)

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def get_summary(self) -> dict[str, Any]:
        """Get summary of current state."""
        return {
            "cash": self.cash,
            "equity": self.equity,
            "positions": len(self.positions),
            "realized_pnl": self.total_realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_realized_pnl + self.unrealized_pnl,
            "returns": self.returns,
            "commission": self.total_commission,
            "slippage": self.total_slippage,
        }

    def reset(self) -> None:
        """Reset to initial state."""
        self.cash = self.initial_cash
        self.positions.clear()
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_realized_pnl = 0.0
        self.asset_realized_pnl.clear()
```

### Key Design Decisions

1. **No analytics**: PositionTracker does NOT track equity_curve, daily_returns, high_water_mark, etc. That's PerformanceAnalyzer's job.
2. **No state history**: State tracking belongs to Portfolio facade (for backward compat).
3. **Pure domain logic**: Only positions, cash, costs. Nothing else.
4. **Focused testing**: Can unit test position logic without any analytics overhead.

---

## Component 2: PerformanceAnalyzer

**File**: `portfolio/analytics.py`
**Lines**: ~300
**Responsibility**: Performance metrics and risk analytics

### API

```python
class PerformanceAnalyzer:
    """Performance metrics and risk analytics."""

    def __init__(self, tracker: PositionTracker):
        """Initialize analyzer with position tracker.

        Args:
            tracker: PositionTracker to analyze
        """
        self.tracker = tracker

        # Real-time metric tracking
        self.high_water_mark = tracker.initial_cash
        self.max_drawdown = 0.0
        self.daily_returns: list[float] = []
        self.timestamps: list[datetime] = []
        self.equity_curve: list[float] = []

        # Risk metrics
        self.max_leverage = 0.0
        self.max_concentration = 0.0

    def update(self, timestamp: datetime) -> None:
        """Update metrics after position/price change.

        Called by Portfolio facade after every fill or market event.
        """
        current_equity = self.tracker.equity

        # Update high water mark and drawdown
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity

        if self.high_water_mark > 0:
            current_drawdown = (self.high_water_mark - current_equity) / self.high_water_mark
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Track equity curve and returns
        self.timestamps.append(timestamp)
        self.equity_curve.append(current_equity)

        # Calculate return if we have previous data
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]
            if prev_equity > 0:
                daily_return = (current_equity - prev_equity) / prev_equity
                self.daily_returns.append(daily_return)

        # Update risk metrics
        # (Simplified - full implementation would use PortfolioState)
        position_values = [p.market_value for p in self.tracker.positions.values()]
        if position_values and current_equity > 0:
            max_position_value = max(abs(v) for v in position_values)
            total_position_value = sum(abs(v) for v in position_values)
            concentration = max_position_value / current_equity
            leverage = total_position_value / current_equity
            self.max_concentration = max(self.max_concentration, concentration)
            self.max_leverage = max(self.max_leverage, leverage)

    def calculate_sharpe_ratio(self) -> float | None:
        """Calculate Sharpe ratio (annualized)."""
        if len(self.daily_returns) < 2:
            return None

        import numpy as np
        returns = np.array(self.daily_returns)
        if returns.std() > 0:
            return (returns.mean() / returns.std()) * np.sqrt(252)
        return 0.0

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = {
            "total_return": self.tracker.returns,
            "total_pnl": self.tracker.total_realized_pnl + self.tracker.unrealized_pnl,
            "realized_pnl": self.tracker.total_realized_pnl,
            "unrealized_pnl": self.tracker.unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "current_equity": self.tracker.equity,
            "current_cash": self.tracker.cash,
            "total_commission": self.tracker.total_commission,
            "total_slippage": self.tracker.total_slippage,
            "max_leverage": self.max_leverage,
            "max_concentration": self.max_concentration,
        }

        # Add Sharpe ratio if available
        sharpe = self.calculate_sharpe_ratio()
        if sharpe is not None:
            metrics["sharpe_ratio"] = sharpe

        return metrics

    def get_equity_curve(self) -> pl.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.timestamps:
            return pl.DataFrame()

        return pl.DataFrame({
            "timestamp": self.timestamps,
            "equity": self.equity_curve,
            "returns": [0.0, *self.daily_returns],
        })

    def reset(self) -> None:
        """Reset analyzer state."""
        self.high_water_mark = self.tracker.initial_cash
        self.max_drawdown = 0.0
        self.daily_returns.clear()
        self.timestamps.clear()
        self.equity_curve.clear()
        self.max_leverage = 0.0
        self.max_concentration = 0.0
```

### Extension Point

Users can subclass PerformanceAnalyzer to add custom metrics:

```python
class MyAnalyzer(PerformanceAnalyzer):
    def calculate_sortino_ratio(self) -> float:
        # Custom metric
        downside_returns = [r for r in self.daily_returns if r < 0]
        if not downside_returns:
            return 0.0
        import numpy as np
        downside_std = np.std(downside_returns)
        if downside_std > 0:
            return (np.mean(self.daily_returns) / downside_std) * np.sqrt(252)
        return 0.0
```

---

## Component 3: TradeJournal

**File**: `portfolio/analytics.py`
**Lines**: ~150
**Responsibility**: Trade history and persistence

### API

```python
class TradeJournal:
    """Trade tracking and history management."""

    def __init__(self):
        """Initialize trade journal."""
        self.fills: list[FillEvent] = []

    def record_fill(self, fill_event: FillEvent) -> None:
        """Record a fill event.

        Args:
            fill_event: Fill event from broker
        """
        self.fills.append(fill_event)

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

    def calculate_win_rate(self) -> float:
        """Calculate win rate via lot matching."""
        if not self.fills:
            return 0.0

        # Lot matching algorithm (same as PortfolioAccounting)
        winning_trades = 0
        total_trades = 0
        position_lots: dict[str, list[dict[str, float]]] = {}

        for fill in self.fills:
            asset_id = fill.asset_id

            if fill.side.value == "buy":
                if asset_id not in position_lots:
                    position_lots[asset_id] = []
                position_lots[asset_id].append({
                    "quantity": fill.fill_quantity,
                    "price": float(fill.fill_price),
                })
            elif fill.side.value == "sell":
                if position_lots.get(asset_id):
                    buy_lot = position_lots[asset_id].pop(0)
                    pnl = (float(fill.fill_price) - buy_lot["price"]) * min(
                        fill.fill_quantity,
                        buy_lot["quantity"],
                    )
                    total_trades += 1
                    if pnl > 0:
                        winning_trades += 1

        return winning_trades / total_trades if total_trades > 0 else 0.0

    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        # Same algorithm as PortfolioAccounting.calculate_profit_factor()
        pass

    def reset(self) -> None:
        """Reset journal."""
        self.fills.clear()
```

---

## Component 4: Portfolio Facade

**File**: `portfolio/portfolio.py`
**Lines**: ~300
**Responsibility**: Simple public API + component orchestration

### API

```python
class Portfolio:
    """Unified portfolio management - simple API with modular internals."""

    def __init__(
        self,
        initial_cash: Cash = 100000.0,
        currency: str = "USD",
        track_analytics: bool = True,
        precision_manager: Optional[PrecisionManager] = None,
        analyzer_class: Optional[Type] = None,
        journal_class: Optional[Type] = None,
    ):
        """Initialize portfolio.

        Args:
            initial_cash: Starting cash balance
            currency: Base currency
            track_analytics: Whether to enable analytics (disable for HFT)
            precision_manager: PrecisionManager for cash rounding
            analyzer_class: Custom analyzer class (default: PerformanceAnalyzer)
            journal_class: Custom journal class (default: TradeJournal)
        """
        # Core tracking (always present)
        self._tracker = PositionTracker(initial_cash, precision_manager)

        # Optional analytics (can disable for performance)
        if track_analytics:
            from qengine.portfolio.analytics import PerformanceAnalyzer
            AnalyzerClass = analyzer_class or PerformanceAnalyzer
            self._analyzer = AnalyzerClass(self._tracker)
        else:
            self._analyzer = None

        # Trade journal
        from qengine.portfolio.analytics import TradeJournal
        JournalClass = journal_class or TradeJournal
        self._journal = JournalClass()

        # Portfolio-level attributes
        self.currency = currency
        self.initial_cash = initial_cash
        self.current_prices: dict[AssetId, float] = {}

        # State history (for backward compatibility)
        self.state_history: list[PortfolioState] = []

    # ===== Event Handlers =====
    def on_fill_event(self, event: FillEvent) -> None:
        """Handle fill event from broker."""
        # Record in journal
        self._journal.record_fill(event)

        # Update position
        quantity_change = event.fill_quantity if event.side.value in ["buy", "BUY"] else -event.fill_quantity
        self._tracker.update_position(
            asset_id=event.asset_id,
            quantity_change=quantity_change,
            price=float(event.fill_price),
            commission=event.commission,
            slippage=event.slippage,
        )

        # Update analytics
        if self._analyzer:
            self._analyzer.update(event.timestamp)

        logger.info(
            f"Fill: {event.side.value.upper()} {event.fill_quantity} {event.asset_id} "
            f"@ ${float(event.fill_price):.2f}"
        )

    # ===== Delegate to PositionTracker =====
    @property
    def cash(self) -> float:
        return self._tracker.cash

    @property
    def equity(self) -> float:
        return self._tracker.equity

    def get_position(self, asset_id: AssetId) -> Position | None:
        return self._tracker.get_position(asset_id)

    # ===== Delegate to PerformanceAnalyzer =====
    def get_performance_metrics(self) -> dict[str, Any]:
        if not self._analyzer:
            raise ValueError("Analytics disabled. Set track_analytics=True")
        return self._analyzer.get_metrics()

    def calculate_sharpe_ratio(self) -> float | None:
        if not self._analyzer:
            return None
        return self._analyzer.calculate_sharpe_ratio()

    # ===== Delegate to TradeJournal =====
    def get_trades(self) -> pl.DataFrame:
        return self._journal.get_trades()

    # ===== For advanced users =====
    @property
    def tracker(self) -> PositionTracker:
        """Access position tracker (advanced users)."""
        return self._tracker

    @property
    def analyzer(self) -> Optional[PerformanceAnalyzer]:
        """Access performance analyzer (advanced users)."""
        return self._analyzer

    @property
    def journal(self) -> TradeJournal:
        """Access trade journal (advanced users)."""
        return self._journal
```

---

## Implementation Checklist

### Phase 1: PositionTracker ✅
- [ ] Create PositionTracker class
- [ ] Move position/cash logic from Portfolio
- [ ] Unit tests for PositionTracker
- [ ] Verify equity calculations

### Phase 2: PerformanceAnalyzer
- [ ] Create PerformanceAnalyzer class
- [ ] Move metrics from PortfolioAccounting
- [ ] Unit tests with mock tracker
- [ ] Verify Sharpe ratio calculation

### Phase 3: TradeJournal
- [ ] Create TradeJournal class
- [ ] Move fill storage logic
- [ ] Unit tests for win rate
- [ ] Verify DataFrame export

### Phase 4: Portfolio Facade
- [ ] Create Portfolio facade
- [ ] Add component delegation
- [ ] Add track_analytics flag
- [ ] Integration tests

### Phase 5: Integration
- [ ] Update BacktestEngine
- [ ] Update Reporting
- [ ] Backward compatibility aliases

### Phase 6: Testing
- [ ] Migrate 48 tests
- [ ] Validation suite
- [ ] Performance benchmark

---

**Design Status**: ✅ Complete - Ready for implementation
**Next Step**: Implement PositionTracker (Phase 1)
