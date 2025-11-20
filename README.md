# ml4t.backtest - Event-Driven Backtesting Engine

**Status**: Beta - Accounting System Complete
**Current version**: 0.2.0
**Performance**: 30x faster than Backtrader
**Accounting**: Cash and margin accounts with proper constraints

---

## Overview

ml4t.backtest is a high-performance, event-driven backtesting engine for quantitative trading strategies with institutional-grade accounting.

### Key Features

- ✅ **Realistic Accounting**: Cash and margin account support with proper constraints
- ✅ **Performance**: 30x faster than Backtrader, 5x faster than VectorBT
- ✅ **Point-in-Time Correctness**: No look-ahead bias
- ✅ **Flexible Execution**: Same-bar and next-bar modes
- ✅ **Comprehensive Testing**: 160+ tests with 69% coverage
- ✅ **Type-Safe**: Full type hints with mypy validation

---

## Installation

```bash
pip install ml4t-backtest
```

Or for development:

```bash
git clone https://github.com/yourusername/ml4t-backtest
cd ml4t-backtest
uv sync  # or pip install -e .
```

---

## Quick Start

```python
import polars as pl
from datetime import datetime, timedelta
from ml4t.backtest import Engine, Strategy, DataFeed, PerShareCommission

# Create sample data
dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)]
data = pl.DataFrame({
    "timestamp": dates,
    "asset": ["AAPL"] * 100,
    "open": [100 + i * 0.5 for i in range(100)],
    "high": [101 + i * 0.5 for i in range(100)],
    "low": [99 + i * 0.5 for i in range(100)],
    "close": [100 + i * 0.5 for i in range(100)],
    "volume": [1_000_000] * 100,
})

# Define strategy
class SimpleStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if "AAPL" not in data:
            return

        price = data["AAPL"]["close"]
        position = broker.get_position("AAPL")

        # Buy on day 1, sell on day 50
        if timestamp.day == 1 and (position is None or position.quantity == 0):
            broker.submit_order("AAPL", 100)  # Buy 100 shares
        elif timestamp.day == 50 and position and position.quantity > 0:
            broker.submit_order("AAPL", -position.quantity)  # Sell all

# Run backtest
feed = DataFeed(prices_df=data)
strategy = SimpleStrategy()

engine = Engine(
    feed,
    strategy,
    initial_cash=50_000.0,
    account_type="cash",  # or "margin"
    commission_model=PerShareCommission(0.01),
)

results = engine.run()

print(f"Final Value: ${results['final_value']:,.2f}")
print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Total Trades: {results['num_trades']}")
```

---

## Account Types

ml4t.backtest supports two account types with realistic constraints:

### Cash Account

Cash accounts enforce strict constraints:
- ✅ **No leverage**: Can only trade with available cash
- ✅ **No short selling**: Cannot hold negative positions
- ✅ **Buying power = cash**: Simple calculation
- ✅ **Order rejection**: Orders rejected if insufficient cash

**Example: Cash Account**

```python
from ml4t.backtest import Engine, Strategy, DataFeed

class BuyAndHoldStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if "AAPL" not in data:
            return

        position = broker.get_position("AAPL")

        # Try to buy on first day
        if timestamp.day == 1 and (position is None or position.quantity == 0):
            price = data["AAPL"]["close"]
            # Try to buy $60,000 worth (will be rejected with $50k initial cash)
            quantity = int(60_000 / price)
            order = broker.submit_order("AAPL", quantity)

            if order is None:
                print(f"Order rejected: Insufficient cash")
                # Adjust to affordable quantity
                quantity = int(broker.account.cash / price)
                broker.submit_order("AAPL", quantity)

# Run with cash account
engine = Engine(
    feed,
    strategy,
    initial_cash=50_000.0,
    account_type="cash",  # Cash account (default)
)

results = engine.run()
```

**Constraints Enforced**:
- Orders exceeding available cash are rejected
- Short selling attempts are rejected
- Exit orders always execute (closing positions)

### Margin Account

Margin accounts allow leverage and short selling:
- ✅ **Leverage enabled**: Can trade up to 2x cash (50% initial margin)
- ✅ **Short selling allowed**: Can hold negative positions
- ✅ **Margin requirements**: Initial and maintenance margin enforced
- ✅ **Buying power formula**: `(NLV - MM) / IM`

**Example: Margin Account**

```python
from ml4t.backtest import Engine, Strategy, DataFeed

class FlippingStrategy(Strategy):
    """Strategy that flips between long and short positions."""

    def __init__(self):
        self.bar_count = 0

    def on_data(self, timestamp, data, context, broker):
        if "AAPL" not in data:
            return

        self.bar_count += 1
        position = broker.get_position("AAPL")
        current_qty = position.quantity if position else 0

        # Odd bars: long, Even bars: short
        target_qty = 100 if (self.bar_count % 2 == 1) else -100
        order_qty = target_qty - current_qty

        if order_qty != 0:
            broker.submit_order("AAPL", order_qty)

# Run with margin account
engine = Engine(
    feed,
    strategy,
    initial_cash=50_000.0,
    account_type="margin",  # Margin account
    initial_margin=0.5,     # 50% initial margin (2x leverage)
    maintenance_margin=0.25,  # 25% maintenance margin
)

results = engine.run()
```

**Margin Calculations**:
- **Net Liquidation Value (NLV)**: `cash + sum(position_values)`
- **Maintenance Margin (MM)**: `sum(abs(position_values) × maintenance_margin)`
- **Buying Power (BP)**: `(NLV - MM) / initial_margin`

**Constraints Enforced**:
- Orders exceeding buying power are rejected
- Position reversals (long→short) are split into close + open
- Short positions tracked with negative quantities
- Margin calls enforced when equity falls below maintenance margin

---

## Order Rejection Scenarios

Understanding when orders are rejected helps design robust strategies.

### Cash Account Rejections

```python
class CashAccountStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        price = data["AAPL"]["close"]
        cash = broker.account.cash

        # Scenario 1: Insufficient cash
        expensive_quantity = int(100_000 / price)  # Try to buy $100k worth
        order = broker.submit_order("AAPL", expensive_quantity)
        if order is None:
            print(f"Rejected: Insufficient cash (have ${cash:.0f}, need $100,000)")

        # Scenario 2: Attempted short sale
        order = broker.submit_order("AAPL", -100)  # Try to short 100 shares
        if order is None:
            print(f"Rejected: Cash accounts cannot short sell")

        # Scenario 3: Successful purchase
        affordable_quantity = int(cash / price)
        order = broker.submit_order("AAPL", affordable_quantity)
        if order is not None:
            print(f"Accepted: Bought {affordable_quantity} shares")
```

### Margin Account Rejections

```python
class MarginAccountStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        price = data["AAPL"]["close"]
        position = broker.get_position("AAPL")

        # Get buying power (includes leverage)
        buying_power = broker.get_buying_power()

        # Scenario 1: Within buying power (accepted)
        quantity = int(buying_power / price)
        order = broker.submit_order("AAPL", quantity)
        if order is not None:
            print(f"Accepted: Bought {quantity} shares using leverage")

        # Scenario 2: Exceeds buying power (rejected)
        excessive_quantity = int((buying_power * 2) / price)
        order = broker.submit_order("AAPL", excessive_quantity)
        if order is None:
            print(f"Rejected: Exceeds buying power")

        # Scenario 3: Position reversal (split into close + open)
        if position and position.quantity > 0:
            # Going from long 100 to short 100 (reversal)
            reversal_order = broker.submit_order("AAPL", -200)
            # Internally split into: close 100, then open -100
            # Both must satisfy margin requirements
```

---

## API Reference

### Engine

```python
Engine(
    feed: DataFeed,
    strategy: Strategy,
    initial_cash: float = 100_000.0,
    account_type: str = "cash",  # "cash" or "margin"
    initial_margin: float = 0.5,  # For margin accounts only
    maintenance_margin: float = 0.25,  # For margin accounts only
    commission_model: Optional[CommissionModel] = None,
    slippage_model: Optional[SlippageModel] = None,
    execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
)
```

**Parameters**:
- `feed`: DataFeed object containing price data
- `strategy`: Strategy object implementing trading logic
- `initial_cash`: Starting cash amount (default: $100,000)
- `account_type`: `"cash"` (no leverage, no shorts) or `"margin"` (leverage + shorts)
- `initial_margin`: Initial margin requirement (default: 0.5 = 50% = 2x leverage)
- `maintenance_margin`: Maintenance margin requirement (default: 0.25 = 25%)
- `commission_model`: Commission model (default: no commission)
- `slippage_model`: Slippage model (default: no slippage)
- `execution_mode`: When orders fill (default: SAME_BAR)

**Returns**: Dictionary with results:
```python
{
    "initial_cash": float,
    "final_value": float,
    "total_return": float,
    "total_return_pct": float,
    "max_drawdown": float,
    "max_drawdown_pct": float,
    "num_trades": int,
    "winning_trades": int,
    "losing_trades": int,
    "win_rate": float,
    "total_commission": float,
    "total_slippage": float,
    "trades": List[Trade],
    "equity_curve": List[Tuple[datetime, float]],
    "fills": List[Fill],
}
```

### Strategy

```python
class Strategy:
    def on_start(self, broker: Broker):
        """Called once before backtest starts."""
        pass

    def on_data(self, timestamp: datetime, data: Dict[str, Dict[str, float]],
                context: Any, broker: Broker):
        """Called on each bar/tick.

        Args:
            timestamp: Current event timestamp
            data: Dict[asset, Dict[field, value]] - e.g., {"AAPL": {"close": 150.0}}
            context: Context object (reserved for future use)
            broker: Broker interface for order submission
        """
        pass

    def on_end(self, broker: Broker):
        """Called once after backtest ends."""
        pass
```

### Broker API

```python
# Submit orders
order = broker.submit_order(asset: str, quantity: int) -> Optional[Order]
# Returns None if order rejected, Order object if accepted

# Get positions
position = broker.get_position(asset: str) -> Optional[Position]
# Returns None if no position, Position object otherwise

# Query account
cash = broker.account.cash  # Available cash
equity = broker.account.equity  # Total account value
positions = broker.account.positions  # Dict[asset, Position]
buying_power = broker.get_buying_power()  # Available buying power
```

---

## Commission Models

```python
from ml4t.backtest import NoCommission, PerShareCommission, PercentageCommission, TieredCommission

# No commission (default)
NoCommission()

# Per-share commission (e.g., $0.01/share)
PerShareCommission(0.01)

# Percentage commission (e.g., 0.1% = 10bps)
PercentageCommission(0.001)

# Tiered commission
TieredCommission(tiers=[
    (0, 10_000, 0.01),    # $0.01/share for 0-10k shares
    (10_000, 100_000, 0.005),  # $0.005/share for 10k-100k
    (100_000, float('inf'), 0.001),  # $0.001/share for 100k+
])
```

---

## Slippage Models

```python
from ml4t.backtest import NoSlippage, FixedSlippage, PercentageSlippage, VolumeShareSlippage

# No slippage (default)
NoSlippage()

# Fixed slippage (e.g., $0.05 per share)
FixedSlippage(0.05)

# Percentage slippage (e.g., 0.05% = 5bps)
PercentageSlippage(0.0005)

# Volume-based slippage
VolumeShareSlippage(
    volume_limit=0.025,  # 2.5% of bar volume
    price_impact=0.1,    # 10% price impact if limit hit
)
```

---

## Validation

ml4t.backtest has been validated against industry-standard frameworks:

- **vs Backtrader**: 0.39% P&L difference (same-bar mode)
- **vs VectorBT**: <1% difference (cash-constrained mode)
- **Test Coverage**: 69% with 160+ tests

See `tests/validation/` for validation studies.

---

## Performance

Benchmarks on 250 assets × 252 trading days:

| Framework | Runtime | vs ml4t.backtest |
|-----------|---------|------------------|
| **ml4t.backtest** | 0.6s | 1x |
| **VectorBT** | 3.4s | 5.7x slower |
| **Backtrader** | 18.7s | 31x slower |

---

## Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run validation tests
pytest tests/validation/

# Run with coverage
pytest --cov=src/ml4t/backtest --cov-report=html
```

---

## Examples

See `examples/` directory for:
- Simple moving average crossover
- ML-based strategies
- Multi-asset portfolios
- Crypto basis trading
- Commission and slippage demos

---

## Documentation

- **Architecture**: See `.claude/memory/` for design decisions
- **Margin Calculations**: See `docs/margin_calculations.md` (coming soon)
- **Trade Schema**: See `.claude/reference/TRADE_SCHEMA.md`

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

---

## License

MIT License - See LICENSE file

---

## Changelog

### v0.2.0 (2025-11-20)
- ✅ Complete accounting system overhaul
- ✅ Cash account support with proper constraints
- ✅ Margin account support (leverage + short selling)
- ✅ Position reversal handling
- ✅ 160+ tests with validation studies
- ✅ Exit-first order sequencing

### v0.1.0 (2025-01-20)
- Initial prototype
- Event-driven architecture
- Basic order execution
- Performance benchmarks

---

**Status**: Beta - Accounting system complete, ready for testing
**Last updated**: 2025-11-20
