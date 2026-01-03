# ml4t.backtest - Agent Guide

## What This Library Does

**ml4t.backtest** is an event-driven backtesting engine with institutional-grade execution fidelity.

- **100% exact match** with VectorBT Pro, Backtrader, and Zipline
- **16.6x faster** than Backtrader
- **Point-in-time correctness** - No look-ahead bias
- **Multi-asset support** - Natural iteration over asset universe
- **Composable risk management** - Protocol-based position/portfolio rules

---

## Quick Start

```python
from ml4t.backtest import Engine, Strategy, DataFeed, run_backtest

# Your strategy
class MyStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, d in data.items():
            if d.get("signals", {}).get("buy_signal"):
                broker.submit_order(asset, 100)

# Run backtest
result = run_backtest(feed, MyStrategy(), initial_cash=100_000)
print(f"Final value: ${result['final_value']:,.0f}")
```

---

## Core Concepts

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| **Engine** | Event loop orchestration | `run()` |
| **Broker** | Order execution, positions | `submit_order()`, `get_position()`, `close_position()` |
| **Strategy** | User logic | `on_data()`, `on_start()`, `on_end()` |
| **DataFeed** | Price + signal iteration | Constructor only |

---

## Strategy Interface

```python
class Strategy(ABC):
    def on_data(self, timestamp, data, context, broker) -> None:
        """Called for each bar with available data."""
        pass

    def on_start(self, broker) -> None:
        """Called before backtest starts."""
        pass

    def on_end(self, broker) -> None:
        """Called after backtest ends."""
        pass
```

### Data Access

```python
def on_data(self, timestamp, data, context, broker):
    for asset, d in data.items():
        # Price data
        price = d["close"]
        open_price = d["open"]
        high = d["high"]
        low = d["low"]
        volume = d["volume"]

        # Signals (user-provided)
        signals = d.get("signals", {})
        ml_score = signals.get("ml_score", 0.5)

    # Global context
    vix = context.get("vix", 15)
```

---

## Broker API

### Order Submission

```python
# Market order (positive = buy, negative = sell)
order = broker.submit_order(asset, 100)        # Buy 100 shares
order = broker.submit_order(asset, -100)       # Sell 100 shares

# Limit order
broker.submit_order(asset, 100, order_type=OrderType.LIMIT, limit_price=150.0)

# Stop order
broker.submit_order(asset, -100, order_type=OrderType.STOP, stop_price=95.0)

# Bracket order (entry + stop + take profit)
broker.submit_bracket(asset, 100, take_profit=105.0, stop_loss=95.0)

# Close all shares
broker.close_position(asset)
```

### Convenience Methods (Zipline-style)

```python
# Target percentage of portfolio (like Zipline's order_target_percent)
broker.order_target_percent("AAPL", 0.10)      # Target 10% weight
broker.order_target_percent("AAPL", 0.0)       # Close position

# Target dollar value
broker.order_target_value("AAPL", 10000)       # Target $10k position
broker.order_target_value("AAPL", -5000)       # Target short $5k

# Portfolio rebalancing (processes sells first, then buys)
broker.rebalance_to_weights({
    "AAPL": 0.30,
    "GOOGL": 0.30,
    "MSFT": 0.40,
})
```

### Position Queries

```python
pos = broker.get_position(asset)
if pos:
    qty = pos.quantity           # Current quantity
    entry = pos.entry_price      # Average entry price
    pnl = pos.unrealized_pnl(current_price)
    mfe = pos.max_favorable_excursion
    mae = pos.max_adverse_excursion
```

### Account Queries

```python
cash = broker.get_cash()               # Available cash
value = broker.get_account_value()     # Total portfolio value (cash + positions)
buying_power = broker.get_buying_power()  # Available for new positions
```

---

## Risk Management

### Position Rules (Auto-Evaluated Each Bar)

```python
# Risk rules are now directly importable from main package
from ml4t.backtest import StopLoss, TrailingStop, VolatilityStop, RuleChain, TimeExit

# Set global rules (all positions)
broker.set_position_rules(StopLoss(pct=0.05))

# Set per-asset rules
broker.set_position_rules(TrailingStop(pct=0.03), asset="AAPL")

# Combine rules
rules = RuleChain([
    StopLoss(pct=0.05),           # 5% stop loss
    TrailingStop(pct=0.02),       # 2% trailing
])
broker.set_position_rules(rules)
```

### Available Position Rules

| Rule | Description | Key Parameters |
|------|-------------|----------------|
| `StopLoss` | Fixed percentage stop | `pct` |
| `TrailingStop` | Trailing percentage stop | `pct`, `activation_pct` |
| `TighteningTrailingStop` | Tightens as profit grows | `initial_pct`, `final_pct` |
| `VolatilityStop` | ATR-based stop | `atr_multiplier` |
| `TimeExit` | Exit after N bars | `bars` |
| `SignalExit` | Exit on signal | `signal_key`, `threshold` |
| `RuleChain` | Combine multiple rules | `rules` list |

### Portfolio Risk Limits

```python
from ml4t.backtest.risk.portfolio.limits import MaxPositions, MaxExposure

# Configure via Engine
engine = Engine(
    feed, strategy,
    portfolio_limits=[
        MaxPositions(max_count=10),
        MaxExposure(max_long=1.0, max_short=0.5),
    ]
)
```

---

## Configuration

### Execution Modes

```python
from ml4t.backtest import ExecutionMode

# Same-bar: Signal at bar N, fill at bar N close (default)
engine = Engine(feed, strategy, execution_mode=ExecutionMode.SAME_BAR)

# Next-bar: Signal at bar N, fill at bar N+1 open
engine = Engine(feed, strategy, execution_mode=ExecutionMode.NEXT_BAR)
```

### Commission & Slippage

```python
from ml4t.backtest import PercentageCommission, PercentageSlippage

engine = Engine(
    feed, strategy,
    commission_model=PercentageCommission(rate=0.001),  # 0.1%
    slippage_model=PercentageSlippage(rate=0.001),      # 0.1%
)
```

### Presets (Framework Compatibility)

```python
from ml4t.backtest import BacktestConfig

# Load preset matching other framework behavior
config = BacktestConfig.from_preset("backtrader")
config = BacktestConfig.from_preset("vectorbt_pro")
config = BacktestConfig.from_preset("zipline")

engine = Engine.from_config(feed, strategy, config=config)
```

### Account Types

```python
# Cash account (no shorts, no margin)
engine = Engine(feed, strategy, account_type="cash")

# Margin account (shorts allowed, leverage)
engine = Engine(feed, strategy, account_type="margin", initial_margin=0.5)
```

---

## Results

### Access Results

```python
result = engine.run()

# Dictionary access (backward compatible)
print(result["final_value"])
print(result["num_trades"])
print(result["sharpe"])

# Metrics
metrics = result.metrics
print(metrics["sharpe"])
print(metrics["sortino"])
print(metrics["max_drawdown_pct"])
```

### Export to DataFrames

```python
# Trades DataFrame
trades_df = result.to_trades_dataframe()
# Columns: asset, entry_time, exit_time, entry_price, exit_price, quantity,
#          pnl, pnl_percent, bars_held, commission, slippage, exit_reason

# Equity DataFrame
equity_df = result.to_equity_dataframe()
# Columns: timestamp, equity
```

### Persist to Disk

```python
# Save to Parquet
result.to_parquet("./results/my_backtest")

# Load later
from ml4t.backtest import BacktestResult
loaded = BacktestResult.from_parquet("./results/my_backtest")
```

### Signal Enrichment

```python
from ml4t.backtest import enrich_trades_with_signals

# Add ML features at entry/exit times
enriched = enrich_trades_with_signals(
    trades_df,
    signals_df,  # DataFrame with timestamp + signal columns
    signal_cols=["ml_score", "momentum", "volatility"]
)
# Adds: entry_ml_score, exit_ml_score, entry_momentum, exit_momentum, ...
```

---

## File Map

```
src/ml4t/backtest/
├── engine.py          (394 lines)  - Event loop orchestration
├── broker.py          (1,015 lines) - Order execution, position tracking
├── strategy.py        (29 lines)   - Strategy base class
├── datafeed.py        (~130 lines) - DataFeed iterator
├── types.py           (375 lines)  - Order, Position, Fill, Trade dataclasses
├── config.py          (~400 lines) - Configuration enums and presets
├── models.py          (135 lines)  - Commission/slippage models
├── result.py          (~200 lines) - BacktestResult class
├── calendar.py        (773 lines)  - Trading calendar integration
│
├── accounting/        (1,189 lines) - Account policies
│   ├── account.py     - AccountState class
│   ├── policy.py      - CashAccountPolicy, MarginAccountPolicy
│   └── gatekeeper.py  - Order validation
│
├── analytics/         (~800 lines) - Performance metrics
│   ├── trades.py      - TradeAnalyzer
│   ├── metrics.py     - Sharpe, Sortino, Calmar, drawdown
│   └── equity.py      - EquityCurve analysis
│
├── execution/         (~820 lines) - Advanced execution
│   ├── impact.py      - Market impact models
│   ├── limits.py      - Volume participation limits
│   └── rebalancer.py  - Portfolio rebalancing
│
└── risk/              (1,944 lines) - Risk management
    ├── types.py       - RiskContext, PositionState, ActionType
    ├── position/      - Position-level rules (stop, trail, signal)
    └── portfolio/     - Portfolio-level limits
```

---

## Common Patterns

### Multi-Asset Strategy

```python
class MultiAssetStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, d in data.items():
            pos = broker.get_position(asset)
            score = d.get("signals", {}).get("score", 0.5)

            if pos is None and score > 0.7:
                # Size to 5% of portfolio
                value = broker.get_account_value() * 0.05
                qty = value / d["close"]
                broker.submit_order(asset, qty)

            elif pos is not None and score < 0.3:
                broker.close_position(asset)
```

### Stop Loss with Trailing

```python
from ml4t.backtest.risk.position.static import StopLoss
from ml4t.backtest.risk.position.dynamic import TrailingStop
from ml4t.backtest.risk.position.composite import RuleChain

class ProtectedStrategy(Strategy):
    def on_start(self, broker):
        # 5% stop, then 2% trailing once 3% profitable
        broker.set_position_rules(RuleChain([
            StopLoss(pct=0.05),
            TrailingStop(pct=0.02, activation_pct=0.03),
        ]))

    def on_data(self, timestamp, data, context, broker):
        # Entry logic only - exits handled by rules
        for asset, d in data.items():
            if broker.get_position(asset) is None:
                if d.get("signals", {}).get("buy"):
                    broker.submit_order(asset, 100)
```

### Rebalancing

```python
from ml4t.backtest import TargetWeightExecutor, RebalanceConfig

class RebalanceStrategy(Strategy):
    def __init__(self):
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(min_trade_value=100)
        )

    def on_data(self, timestamp, data, context, broker):
        # Monthly rebalance
        if timestamp.day == 1:
            target_weights = {"AAPL": 0.3, "GOOGL": 0.3, "MSFT": 0.4}
            orders = self.executor.execute(broker, target_weights, data)
```

---

## Validation Status

| Framework | Scenarios | Trades Matched | Status |
|-----------|-----------|----------------|--------|
| VectorBT Pro | 11/11 | 1,022 | 100% exact |
| Backtrader | All | 12,600 | 100% exact |
| Zipline | 10/10 | 119,577 | 100% exact |

---

## Key Design Decisions

1. **No historical data access**: Signals should be pre-computed. Use `signals` dict for lookback indicators.
2. **Exit-first processing**: Exit orders execute before entries to free capital.
3. **VBT Pro compatibility**: Same-bar re-entry prevention, HWM update timing matches VBT Pro.
4. **Point-in-time correctness**: Stop evaluation uses previous bar's water marks.

---

## See Also

- `README.md` - Full documentation
- `.claude/PROJECT_MAP.md` - Detailed project structure
- `docs/cross_library_api.md` - Cross-implementation API specification
- `validation/` - Framework validation scripts
