# ml4t-backtest

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Event-driven backtesting engine for quantitative trading strategies. Designed for correctness over speed, with comprehensive validation against VectorBT Pro, Backtrader, and Zipline.

## Features

- **Event-Driven Architecture**: Point-in-time correctness with no look-ahead bias
- **Exit-First Processing**: Matches real broker order execution behavior
- **Risk Management**: Stop-loss, take-profit, trailing stops, bracket orders
- **Multi-Asset Support**: Efficient handling of portfolios with 500+ assets
- **Cash and Margin Accounts**: Realistic position sizing with buying power constraints
- **Configurable Execution**: Same-bar or next-bar fills, multiple slippage models
- **Futures Support**: Contract multipliers, overnight sessions

## Installation

```bash
pip install ml4t-backtest
```

## Quick Start

```python
from ml4t.backtest import Engine, Strategy, BacktestConfig, DataFeed
from ml4t.backtest.risk import StopLoss, TakeProfit, RuleChain

class TrendFollowing(Strategy):
    def __init__(self, fast=10, slow=30):
        self.fast = fast
        self.slow = slow

    def on_data(self, timestamp, data, context, broker):
        close = data["close"]
        fast_ma = close.rolling(self.fast).mean().iloc[-1]
        slow_ma = close.rolling(self.slow).mean().iloc[-1]

        position = broker.get_position("SPY")

        if fast_ma > slow_ma and position is None:
            broker.submit_order("SPY", quantity=100, side="BUY")
        elif fast_ma < slow_ma and position is not None:
            broker.close_position("SPY")

# Configure backtest
config = BacktestConfig(
    initial_cash=100_000,
    commission_rate=0.001,
)

# Run backtest
feed = DataFeed(price_data)
engine = Engine(feed, TrendFollowing(), config)
result = engine.run()

# Analyze results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
```

## Architecture

```
Engine (event loop)
├── DataFeed (price iteration)
├── Broker (order execution)
│   ├── AccountPolicy (cash/margin)
│   ├── Gatekeeper (buying power)
│   └── FillExecutor (price simulation)
├── RiskManager (position/portfolio rules)
└── Strategy (user logic)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `engine.py` | Event loop orchestration |
| `broker.py` | Order execution, position tracking |
| `strategy.py` | Strategy base class |
| `datafeed.py` | Price and signal iteration |
| `risk/` | Position and portfolio risk rules |
| `accounting/` | Cash and margin account policies |
| `analytics/` | Performance metrics, trade analysis |
| `execution/` | Fill simulation, market impact |

## Risk Management

Position-level exit rules:

```python
from ml4t.backtest.risk import (
    StopLoss,           # Fixed percentage stop
    TakeProfit,         # Fixed percentage target
    TrailingStop,       # High-water mark trailing
    VolatilityTrailingStop,  # ATR-based trailing
    RuleChain,          # Combine multiple rules
)

class MyStrategy(Strategy):
    def on_start(self, broker):
        # Set default rules for all positions
        broker.set_position_rules(RuleChain([
            StopLoss(pct=0.05),      # 5% stop-loss
            TakeProfit(pct=0.15),    # 15% take-profit
            TrailingStop(pct=0.03),  # 3% trailing stop
        ]))
```

Portfolio-level risk controls:

```python
from ml4t.backtest.risk import (
    MaxPositions,       # Limit number of positions
    MaxDrawdown,        # Close all on drawdown limit
    DailyLossLimit,     # Daily P&L limit
)
```

## Execution Modes

Match behavior of different backtesting frameworks:

```python
from ml4t.backtest import ExecutionMode, StopFillMode

# VectorBT Pro / VectorBT OSS style (same-bar fills)
config = BacktestConfig(
    execution_mode=ExecutionMode.SAME_BAR,
    stop_fill_mode=StopFillMode.STOP_PRICE,
)

# Backtrader style (next-bar fills)
config = BacktestConfig(
    execution_mode=ExecutionMode.NEXT_BAR,
    stop_fill_mode=StopFillMode.STOP_PRICE,
)

# Zipline style (next-bar open fills)
config = BacktestConfig(
    execution_mode=ExecutionMode.NEXT_BAR,
    stop_fill_mode=StopFillMode.NEXT_BAR_OPEN,
)
```

## Commission and Slippage

```python
from ml4t.backtest import (
    PercentCommission,    # 0.1% of trade value
    PerShareCommission,   # $0.005 per share
    FixedSlippage,        # Fixed amount per share
    PercentSlippage,      # Percentage of price
    NoCommission,
    NoSlippage,
)

config = BacktestConfig(
    commission_model=PercentCommission(rate=0.001),
    slippage_model=PercentSlippage(rate=0.0005),
)
```

## Multi-Asset Portfolios

```python
class RankingStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        # Rank assets by momentum
        returns = data["close"].pct_change(20)
        ranked = returns.iloc[-1].sort_values(ascending=False)

        # Long top 10, short bottom 10
        longs = ranked.head(10).index.tolist()
        shorts = ranked.tail(10).index.tolist()

        # Rebalance
        for asset in longs:
            if broker.get_position(asset) is None:
                broker.submit_order(asset, quantity=100, side="BUY")

        for asset in shorts:
            if broker.get_position(asset) is None:
                broker.submit_order(asset, quantity=100, side="SELL")
```

## Result Analysis

```python
result = engine.run()

# Summary metrics
print(result.metrics)
# {'total_return': 0.15, 'sharpe_ratio': 1.2, 'max_drawdown': 0.08, ...}

# Trade-level analysis
trades = result.trades
print(f"Win rate: {(trades['pnl'] > 0).mean():.1%}")
print(f"Avg trade: ${trades['pnl'].mean():,.2f}")

# Equity curve
equity = result.equity_curve
equity.plot()

# Export to Parquet
result.to_parquet("backtest_result.parquet")
```

## Validation

The library is validated against three established backtesting frameworks:

### Test Coverage Matrix

| Feature | VectorBT Pro | VectorBT OSS | Backtrader | Zipline |
|---------|--------------|--------------|------------|---------|
| Long only | Exact match | Exact match | Exact match | Exact match |
| Long/Short | Exact match | Exact match | Exact match | Exact match |
| Multi-asset (500 assets) | 100% match | 100% match | 100% match | 100% match |
| Stop-loss | Exact match | Exact match | Exact match | Exact match |
| Take-profit | Exact match | Exact match | Exact match | Exact match |
| Trailing stop | Exact match | Pending | Pending | Pending |
| Commission (%) | Exact match | Pending | Pending | Pending |
| Slippage (%) | Exact match | Pending | Pending | Pending |

### Large-Scale Validation

- **119,000+ trades** verified trade-by-trade against VectorBT Pro, Backtrader, and Zipline
- 500 assets x 10 years (2,520 daily bars)
- **100% PnL match** on all common trades

### Validation Methodology

Each framework runs in an isolated virtual environment to avoid dependency conflicts. Validation scripts use identical pre-computed signals to eliminate strategy variance.

```bash
# Run validation (requires separate venvs)
cd validation/
source .venv-vectorbt-pro/bin/activate
python vectorbt_pro/scenario_01_long_only.py
```

See [validation/README.md](validation/README.md) for complete validation documentation.

## Performance

| Configuration | Backtrader | ml4t-backtest | Notes |
|---------------|------------|---------------|-------|
| 1,000 bars x 1 asset | 0.28s | 0.38s | Comparable |
| 1,000 bars x 10 assets | 2.17s | 0.54s | 4x faster |
| 1,000 bars x 50 assets | 10.49s | 1.24s | 8x faster |

Multi-asset portfolios benefit from efficient data handling. Single-asset backtests are comparable to Backtrader.

## Testing

```bash
# Run all tests (919 tests)
uv run pytest tests/ -q

# Run with coverage
uv run pytest tests/ --cov=src/ml4t/backtest

# Type checking
uv run ty check
```

## API Reference

### Core Classes

```python
from ml4t.backtest import (
    Engine,           # Backtest orchestrator
    Strategy,         # Strategy base class
    DataFeed,         # Price data iterator
    BacktestConfig,   # Configuration
    BacktestResult,   # Results container
)
```

### Risk Rules

```python
from ml4t.backtest.risk import (
    # Position-level
    StopLoss, TakeProfit, TrailingStop, VolatilityTrailingStop,
    SignalExit, RuleChain,
    # Portfolio-level
    MaxPositions, MaxDrawdown, DailyLossLimit,
)
```

### Execution

```python
from ml4t.backtest import (
    ExecutionMode,        # SAME_BAR, NEXT_BAR
    StopFillMode,         # STOP_PRICE, NEXT_BAR_OPEN
    StopLevelBasis,       # FILL_PRICE, SIGNAL_PRICE
)
```

### Commission and Slippage

```python
from ml4t.backtest import (
    PercentCommission, PerShareCommission, NoCommission,
    PercentSlippage, FixedSlippage, NoSlippage,
)
```

## Integration with ML4T Libraries

ml4t-backtest is part of the ML4T library ecosystem:

```python
from ml4t.data import DataManager
from ml4t.engineer import compute_features
from ml4t.engineer.labeling import triple_barrier_labels
from ml4t.backtest import Engine, Strategy
from ml4t.diagnostic.evaluation import TradeAnalysis

# Complete workflow
data = DataManager().fetch("SPY", "2020-01-01", "2023-12-31")
features = compute_features(data, ["rsi", "macd", "atr"])
# ... train model ...
result = engine.run()
analysis = TradeAnalysis(result.trades)
```

## Ecosystem

- **ml4t-data**: Market data acquisition and storage
- **ml4t-engineer**: Feature engineering and indicators
- **ml4t-diagnostic**: Statistical validation and evaluation
- **ml4t-backtest**: Event-driven backtesting (this library)
- **ml4t-live**: Live trading platform

## Development

```bash
git clone https://github.com/applied-ai/ml4t-backtest.git
cd ml4t-backtest

# Install with dev dependencies
uv sync

# Run tests
uv run pytest tests/ -q

# Type checking
uv run ty check

# Linting
uv run ruff check src/
```

## Known Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for documented assumptions and edge cases:

- Partial fills not supported (all-or-nothing execution)
- No intrabar stop simulation (uses bar OHLC for trigger detection)
- Fractional shares supported but not all brokers allow them
- Calendar overnight sessions require explicit configuration

## License

MIT License - see [LICENSE](LICENSE) for details.
