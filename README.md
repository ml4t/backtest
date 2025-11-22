# ml4t.backtest - Event-Driven Backtesting Engine

**Status**: Beta
**Version**: 0.2.0
**Tests**: 154 passing

---

## Overview

ml4t.backtest is an event-driven backtesting engine for quantitative trading strategies. It supports single-asset and multi-asset strategies with realistic execution modeling.

### Key Features

- **Event-Driven Execution**: Bar-by-bar strategy execution with `on_data()` callbacks
- **Multi-Asset Support**: Trade multiple assets simultaneously with unified portfolio tracking
- **Polars-Based Data**: Efficient data handling with lazy evaluation support
- **Realistic Accounting**: Cash accounts (no leverage) and margin accounts (with short selling)
- **Framework Compatibility**: Configuration presets to match VectorBT, Backtrader, and Zipline behavior
- **Pluggable Models**: Swappable commission and slippage models
- **Order Types**: Market, limit, stop, stop-limit, trailing stop
- **Exit-First Processing**: Exit orders processed before entries for capital efficiency

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

# Create price data
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
class MomentumStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        if "AAPL" not in data:
            return

        price = data["AAPL"]["close"]
        position = broker.get_position("AAPL")

        # Simple momentum: buy if no position, sell after 20 bars
        if position is None or position.quantity == 0:
            broker.submit_order("AAPL", 100)  # Buy 100 shares
        elif position.bars_held >= 20:
            broker.close_position("AAPL")

# Run backtest
feed = DataFeed(prices_df=data)
strategy = MomentumStrategy()

engine = Engine(
    feed,
    strategy,
    initial_cash=50_000.0,
    account_type="cash",
    commission_model=PerShareCommission(0.01),
)

results = engine.run()

print(f"Final Value: ${results['final_value']:,.2f}")
print(f"Total Return: {results['total_return_pct']:.2f}%")
print(f"Total Trades: {results['num_trades']}")
```

---

## Account Types

### Cash Account

Standard cash account with no leverage or short selling:

```python
engine = Engine(
    feed, strategy,
    initial_cash=100_000.0,
    account_type="cash",  # No leverage, no shorts
)
```

**Constraints**:
- Orders rejected if insufficient cash
- Short selling not allowed
- Buying power equals available cash

### Margin Account

Margin account with leverage and short selling support:

```python
engine = Engine(
    feed, strategy,
    initial_cash=100_000.0,
    account_type="margin",
    initial_margin=0.5,      # 50% = 2x leverage (Reg T)
    maintenance_margin=0.25,  # 25% maintenance requirement
)
```

**Features**:
- Short selling enabled
- Buying power formula: `(NLV - MM) / initial_margin`
- Position reversals allowed (long → short in single order)
- Margin calls when equity falls below maintenance

---

## Order Types

```python
# Market order (default)
broker.submit_order("AAPL", 100)

# Limit order
broker.submit_order("AAPL", 100, order_type=OrderType.LIMIT, limit_price=150.0)

# Stop order
broker.submit_order("AAPL", -100, order_type=OrderType.STOP, stop_price=145.0)

# Trailing stop
broker.submit_order("AAPL", -100, order_type=OrderType.TRAILING_STOP, trail_amount=5.0)

# Bracket order (entry + take-profit + stop-loss)
entry, tp, sl = broker.submit_bracket("AAPL", 100, take_profit=160.0, stop_loss=145.0)
```

**Order Management**:
```python
# Update pending order (stop price, limit price, quantity)
broker.update_order(order.order_id, stop_price=new_stop)

# Cancel order
broker.cancel_order(order.order_id)

# Close entire position
broker.close_position("AAPL")
```

---

## Execution Modes

```python
from ml4t.backtest import ExecutionMode

# Same-bar execution (default) - orders fill at current bar's close
engine = Engine(feed, strategy, execution_mode=ExecutionMode.SAME_BAR)

# Next-bar execution - orders fill at next bar's open (like Backtrader)
engine = Engine(feed, strategy, execution_mode=ExecutionMode.NEXT_BAR)
```

---

## Commission Models

```python
from ml4t.backtest import (
    NoCommission,
    PerShareCommission,
    PercentageCommission,
    TieredCommission,
)

# No commission
NoCommission()

# Per-share ($0.01/share)
PerShareCommission(0.01)

# Percentage (0.1% = 10bps)
PercentageCommission(0.001)

# Tiered by volume
TieredCommission(tiers=[
    (0, 10_000, 0.01),       # $0.01/share for 0-10k shares
    (10_000, 100_000, 0.005), # $0.005/share for 10k-100k
    (100_000, float('inf'), 0.001),  # $0.001/share for 100k+
])
```

---

## Slippage Models

```python
from ml4t.backtest import (
    NoSlippage,
    FixedSlippage,
    PercentageSlippage,
    VolumeShareSlippage,
)

# No slippage
NoSlippage()

# Fixed ($0.05 per share)
FixedSlippage(0.05)

# Percentage (0.05% = 5bps)
PercentageSlippage(0.0005)

# Volume-based
VolumeShareSlippage(
    volume_limit=0.025,  # 2.5% of bar volume
    price_impact=0.1,    # 10% price impact if limit hit
)
```

---

## Framework Compatibility

ml4t.backtest can be configured to match the behavior of other backtesting frameworks:

```python
from ml4t.backtest import BacktestConfig

# Match VectorBT behavior
config = BacktestConfig.from_preset("vectorbt")
# - Same-bar fills at close
# - Fractional shares
# - Process all signals

# Match Backtrader behavior
config = BacktestConfig.from_preset("backtrader")
# - Next-bar fills at open
# - Integer shares only
# - Check position before acting

# Match Zipline behavior
config = BacktestConfig.from_preset("zipline")
# - Next-bar fills at open
# - Integer shares
# - Per-share commission
# - Volume-based slippage
```

---

## Multi-Asset Strategies

```python
class MultiAssetStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        # data contains all assets at this timestamp
        for asset, bar in data.items():
            price = bar["close"]
            signals = bar.get("signals", {})

            position = broker.get_position(asset)
            current_qty = position.quantity if position else 0

            # Act on signals
            if signals.get("buy") and current_qty == 0:
                broker.submit_order(asset, 100)
            elif signals.get("sell") and current_qty > 0:
                broker.close_position(asset)
```

---

## Data Feed

The DataFeed accepts Polars DataFrames with prices, signals, and context:

```python
from ml4t.backtest import DataFeed

# From DataFrames
feed = DataFeed(
    prices_df=prices,   # Required: timestamp, asset, open, high, low, close, volume
    signals_df=signals, # Optional: timestamp, asset, signal_columns...
    context_df=context, # Optional: timestamp, context_columns... (market-wide data)
)

# From Parquet files (uses lazy loading)
feed = DataFeed(
    prices_path="prices.parquet",
    signals_path="signals.parquet",
    context_path="context.parquet",
)
```

---

## Strategy Interface

```python
class Strategy:
    def on_start(self, broker):
        """Called once before backtest starts."""
        pass

    def on_data(self, timestamp, data, context, broker):
        """Called on each bar.

        Args:
            timestamp: Current datetime
            data: Dict[asset, Dict[field, value]]
                  e.g., {"AAPL": {"close": 150.0, "signals": {"momentum": 0.5}}}
            context: Dict of market-wide data (VIX, SPY, etc.)
            broker: Broker interface for orders and positions
        """
        pass

    def on_end(self, broker):
        """Called once after backtest ends."""
        pass
```

---

## Results

```python
results = engine.run()

# Portfolio metrics
results["initial_cash"]      # Starting cash
results["final_value"]       # Ending portfolio value
results["total_return"]      # Dollar return
results["total_return_pct"]  # Percent return
results["max_drawdown"]      # Maximum drawdown (dollars)
results["max_drawdown_pct"]  # Maximum drawdown (percent)

# Trade metrics
results["num_trades"]        # Total completed trades
results["winning_trades"]    # Number of winners
results["losing_trades"]     # Number of losers
results["win_rate"]          # Win rate (0-1)

# Costs
results["total_commission"]  # Total commission paid
results["total_slippage"]    # Total slippage cost

# Detailed data
results["trades"]            # List[Trade] - completed round-trips
results["fills"]             # List[Fill] - all order fills
results["equity_curve"]      # List[Tuple[datetime, float]] - equity over time
```

---

## Validation

ml4t.backtest is validated against established frameworks in isolated environments:

| Framework | Environment | Status |
|-----------|-------------|--------|
| VectorBT Pro | `.venv-vectorbt-pro` | Scenario-based |
| Backtrader | `.venv-backtrader` | Scenario-based |
| Zipline | `.venv-zipline` | Excluded (bundle issues) |

Validation uses per-framework isolated environments to avoid dependency conflicts.

---

## Testing

```bash
# Run all tests
source .venv/bin/activate
pytest tests/ -q

# Run with coverage
pytest tests/ --cov=src/ml4t/backtest --cov-report=html
```

---

## Documentation

- **Project Map**: `.claude/PROJECT_MAP.md`
- **Architecture Decisions**: `.claude/memory/decisions.md`
- **Validation Strategy**: `.claude/memory/validation_methodology.md`

---

## License

MIT License - See LICENSE file

---

## Changelog

### v0.2.0 (2025-11-22)
- Major repository cleanup (99.2% code reduction)
- Complete accounting system (cash + margin accounts)
- Exit-first order processing
- Framework compatibility presets
- Per-framework validation strategy
- 154 tests passing

### v0.1.0 (2025-01-20)
- Initial prototype
- Event-driven architecture
- Basic order execution
