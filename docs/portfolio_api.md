# Portfolio API Reference

Complete reference for the ml4t.backtest Portfolio module.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Portfolio Class](#portfolio-class)
3. [Progressive Disclosure](#progressive-disclosure)
4. [Component Access](#component-access)
5. [Complete API](#complete-api)

---

## Quick Start

### Basic Usage

```python
from ml4t.backtest.portfolio import Portfolio

# Create portfolio
portfolio = Portfolio(initial_cash=100000.0, currency="USD")

# Process trades (called by Broker)
portfolio.on_fill_event(fill_event)

# Query state
print(f"Cash: ${portfolio.cash:,.2f}")
print(f"Equity: ${portfolio.equity:,.2f}")
print(f"Positions: {len(portfolio.positions)}")

# Get metrics
metrics = portfolio.get_performance_metrics()
print(f"Total P&L: ${metrics['total_pnl']:,.2f}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### High-Frequency Trading (HFT) Mode

Disable analytics for maximum performance:

```python
# No performance tracking overhead
portfolio = Portfolio(
    initial_cash=100000.0,
    track_analytics=False  # Disables PerformanceAnalyzer
)

# Only core position/cash tracking available
print(f"Cash: ${portfolio.cash}")
print(f"Positions: {portfolio.positions}")

# Metrics methods raise RuntimeError
# portfolio.get_performance_metrics()  # Error!
```

---

## Portfolio Class

### Initialization

```python
Portfolio(
    initial_cash: float = 100000.0,
    currency: str = "USD",
    track_analytics: bool = True,
    precision_manager: Optional[PrecisionManager] = None,
    analyzer_class: Optional[Type] = None,
    journal_class: Optional[Type] = None
)
```

**Parameters:**
- `initial_cash` - Starting capital (default: 100,000)
- `currency` - Base currency (default: "USD")
- `track_analytics` - Enable performance tracking (default: True)
- `precision_manager` - Custom precision rules (optional)
- `analyzer_class` - Custom PerformanceAnalyzer (advanced)
- `journal_class` - Custom TradeJournal (advanced)

**Returns:** Portfolio instance

---

## Progressive Disclosure

The Portfolio API is designed for progressive disclosure - beginners use simple methods, advanced users access components directly.

### Level 1: Beginner (Simple API)

```python
# Event processing
portfolio.on_fill_event(fill_event)

# Basic queries
cash = portfolio.cash
equity = portfolio.equity
returns = portfolio.returns

# Simple metrics
metrics = portfolio.get_performance_metrics()
sharpe = portfolio.calculate_sharpe_ratio()
```

### Level 2: Intermediate (Position Management)

```python
# Position queries
position = portfolio.get_position("BTC")
all_positions = portfolio.positions  # dict[AssetId, Position]

# Position details
if position:
    print(f"Quantity: {position.quantity}")
    print(f"Cost basis: ${position.cost_basis:,.2f}")
    print(f"Market value: ${position.market_value:,.2f}")
    print(f"Unrealized P&L: ${position.unrealized_pnl:,.2f}")

# Price updates
portfolio.update_prices({"BTC": 51000.0, "ETH": 3200.0})
```

### Level 3: Advanced (Component Access)

```python
# Direct component access
tracker = portfolio.tracker         # PositionTracker (core)
analyzer = portfolio.analyzer       # PerformanceAnalyzer (metrics)
journal = portfolio.journal         # TradeJournal (history)

# Advanced queries
state = portfolio.get_current_state(datetime.now())
summary = portfolio.get_position_summary()

# Trades DataFrame
trades_df = portfolio.get_trades()  # Polars DataFrame
```

---

## Component Access

### PositionTracker (Core)

Always available. Handles position and cash tracking.

```python
# Access core tracker
tracker = portfolio.tracker

# Properties
print(f"Initial cash: {tracker.initial_cash}")
print(f"Current cash: {tracker.cash}")
print(f"Total realized P&L: {tracker.total_realized_pnl}")
print(f"Total commission: {tracker.total_commission}")
print(f"Total slippage: {tracker.total_slippage}")

# Methods
position = tracker.get_position("BTC")
summary = tracker.get_summary()
tracker.update_prices({"BTC": 51000.0})
```

### PerformanceAnalyzer (Metrics)

Available when `track_analytics=True` (default).

```python
# Access analyzer
analyzer = portfolio.analyzer  # None if track_analytics=False

if analyzer:
    # Metrics
    print(f"Max drawdown: {analyzer.max_drawdown:.2%}")
    print(f"High water mark: ${analyzer.high_water_mark:,.2f}")
    print(f"Max leverage: {analyzer.max_leverage:.2f}x")

    # Time series
    print(f"Equity curve: {len(analyzer.equity_curve)} data points")
    print(f"Daily returns: {len(analyzer.daily_returns)} values")
```

### TradeJournal (History)

Always available. Records all fills and trades.

```python
# Access journal
journal = portfolio.journal

# Metrics
print(f"Win rate: {journal.calculate_win_rate():.1%}")
print(f"Profit factor: {journal.calculate_profit_factor():.2f}")
print(f"Avg commission: ${journal.calculate_avg_commission():.2f}")
print(f"Avg slippage: ${journal.calculate_avg_slippage():.2f}")

# Trade data
trades_df = journal.get_trades()  # Polars DataFrame
if trades_df is not None:
    print(f"Total trades: {len(trades_df)}")
    print(trades_df.head())
```

---

## Complete API

### Properties

```python
# Read-only properties
portfolio.cash                    # float - Current cash balance
portfolio.equity                  # float - Total equity (cash + positions)
portfolio.returns                 # float - Total return percentage
portfolio.unrealized_pnl          # float - Unrealized P&L
portfolio.total_realized_pnl      # float - Realized P&L
portfolio.total_commission        # float - Total commission paid
portfolio.total_slippage          # float - Total slippage cost
portfolio.positions               # dict[AssetId, Position] - All positions

# Read-write properties
portfolio.cash = 50000.0          # Set cash directly (advanced)

# Component access
portfolio.tracker                 # PositionTracker (always available)
portfolio.analyzer                # PerformanceAnalyzer (None if HFT mode)
portfolio.journal                 # TradeJournal (always available)

# Metadata
portfolio.initial_cash            # float - Starting capital
portfolio.currency                # str - Base currency
portfolio.current_prices          # dict[AssetId, float] - Latest prices
portfolio.state_history           # list[PortfolioState] - Historical states
```

### Methods

#### Event Processing

```python
portfolio.on_fill_event(event: FillEvent) -> None
```
Process a fill event from the broker. Updates positions, cash, and all metrics.

#### Position Queries

```python
portfolio.get_position(asset_id: str) -> Position | None
```
Get position for specific asset. Returns None if no position.

```python
portfolio.get_all_positions() -> dict[str, float]
```
Get all positions as dict of quantities.

#### Price Updates

```python
portfolio.update_prices(prices: dict[str, float]) -> None
```
Update all positions with new market prices.

**Example:**
```python
portfolio.update_prices({
    "BTC": 51000.0,
    "ETH": 3200.0,
    "AAPL": 175.50
})
```

#### Position Management

```python
portfolio.update_position(
    asset_id: str,
    quantity_change: float,
    price: float,
    commission: float = 0.0,
    slippage: float = 0.0
) -> None
```
Manually update position (advanced usage, normally handled by `on_fill_event`).

#### Metrics

```python
portfolio.get_performance_metrics() -> dict[str, Any]
```
Get comprehensive performance metrics. Raises RuntimeError in HFT mode.

**Returns:**
```python
{
    "total_return": 0.05,              # 5% return
    "total_pnl": 5000.0,               # Total P&L
    "realized_pnl": 3000.0,            # Closed trades P&L
    "unrealized_pnl": 2000.0,          # Open positions P&L
    "max_drawdown": 0.03,              # 3% max drawdown
    "sharpe_ratio": 1.5,               # Risk-adjusted return
    "current_equity": 105000.0,
    "current_cash": 50000.0,
    "total_commission": 150.0,
    "total_slippage": 75.0,
    "num_trades": 10,
    "max_leverage": 1.2,
    "max_concentration": 0.4,
    "win_rate": 0.6,                   # 60% winning trades
    "profit_factor": 2.0,              # Wins/losses ratio
    "avg_commission_per_trade": 15.0,
    "avg_slippage_per_trade": 7.5
}
```

```python
portfolio.calculate_sharpe_ratio() -> float | None
```
Calculate Sharpe ratio (annualized). Returns None if insufficient data.

```python
portfolio.get_trades() -> pl.DataFrame | None
```
Get all trades as Polars DataFrame. Returns None if no trades.

**DataFrame columns:**
- timestamp, order_id, trade_id, asset_id
- side, quantity, price
- commission, slippage, total_cost

#### State Management

```python
portfolio.get_current_state(timestamp: datetime) -> PortfolioState
```
Get complete portfolio state snapshot at timestamp.

**Returns PortfolioState with:**
- timestamp, cash, positions
- total_value, total_realized_pnl, total_unrealized_pnl
- total_commission, total_slippage
- leverage, concentration, max_position_value

```python
portfolio.save_state(timestamp: datetime) -> None
```
Save current state to history for later analysis.

```python
portfolio.get_position_summary() -> dict[str, Any]
```
Get summary of positions and portfolio state.

**Returns:**
```python
{
    "cash": 50000.0,
    "equity": 105000.0,
    "positions": 3,                    # Number of positions
    "total_realized_pnl": 3000.0,
    "unrealized_pnl": 2000.0,
    "total_commission": 150.0,
    "total_slippage": 75.0,
    "returns": 0.05                    # 5%
}
```

```python
portfolio.reset() -> None
```
Reset portfolio to initial state. Clears all positions, metrics, and history.

---

## Position Class

The `Position` dataclass represents holdings in a single asset.

```python
@dataclass
class Position:
    asset_id: str                      # Asset identifier
    quantity: float                    # Current quantity held
    cost_basis: float                  # Total cost (avg_price * quantity)
    last_price: float                  # Latest market price
    unrealized_pnl: float              # Current unrealized P&L
    precision_manager: PrecisionManager | None  # Precision rules

    @property
    def market_value(self) -> float:  # quantity * last_price
        ...
```

**Usage:**
```python
position = portfolio.get_position("BTC")
if position:
    avg_price = position.cost_basis / position.quantity
    print(f"Avg entry: ${avg_price:,.2f}")
    print(f"Current: ${position.last_price:,.2f}")
    print(f"Gain: ${position.unrealized_pnl:,.2f}")
```

---

## PortfolioState Class

Complete snapshot of portfolio at a point in time.

```python
@dataclass
class PortfolioState:
    timestamp: datetime
    cash: float
    positions: dict[str, Position]
    pending_orders: list[Any]
    filled_orders: list[Any]

    # Performance metrics
    total_value: float
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_commission: float
    total_slippage: float

    # Risk metrics
    leverage: float
    max_position_value: float
    concentration: float

    @property
    def equity(self) -> float:
        ...

    @property
    def total_pnl(self) -> float:
        ...
```

---

## Error Handling

### Common Errors

```python
# HFT mode - analytics disabled
portfolio = Portfolio(track_analytics=False)
try:
    metrics = portfolio.get_performance_metrics()
except RuntimeError as e:
    print(f"Error: {e}")
    # "Performance analytics disabled (track_analytics=False)"

# Sharpe ratio with insufficient data
sharpe = portfolio.calculate_sharpe_ratio()
if sharpe is None:
    print("Not enough data for Sharpe ratio")

# No trades yet
trades_df = portfolio.get_trades()
if trades_df is None:
    print("No trades recorded")
```

---

## Performance Notes

### Memory Usage

- **Default mode**: ~1KB per fill event (stores metrics history)
- **HFT mode**: ~100 bytes per fill (no metrics tracking)
- **State history**: ~2KB per saved state

### Computational Cost

- `on_fill_event()`: O(1) - constant time
- `update_prices()`: O(n) - where n = number of positions
- `get_performance_metrics()`: O(1) - pre-calculated
- `calculate_sharpe_ratio()`: O(m) - where m = number of returns

### Optimization Tips

1. **Use HFT mode** for high-frequency strategies (>100 trades/sec)
2. **Batch price updates** instead of updating per-asset
3. **Limit state saves** to key checkpoints only
4. **Access components directly** for hot paths

---

## See Also

- [Architecture Guide](portfolio_architecture.md) - How Portfolio is structured
- [Extension Guide](portfolio_extensions.md) - Custom analyzers and journals
- [Migration Guide](portfolio_migration.md) - Upgrading from legacy code
