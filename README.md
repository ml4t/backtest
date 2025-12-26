# ml4t.backtest - Event-Driven Backtesting Engine

**Status**: Beta
**Version**: 0.2.0
**Tests**: 474 passing (73% coverage)
**Lines of Code**: ~7,700 (source)

---

## Overview

ml4t.backtest is a **minimal, high-fidelity event-driven backtesting engine** for quantitative trading strategies. It supports single-asset and multi-asset strategies with institutional-grade execution modeling.

### Design Philosophy

- **Minimal Core**: ~7,700 lines of production code
- **Point-in-Time Correctness**: No look-ahead bias, realistic execution
- **Event-Driven**: Bar-by-bar strategy execution matching live trading
- **Polars-First**: Efficient data handling with lazy evaluation
- **Pluggable Components**: Commission, slippage, account policies, execution models
- **Framework Compatible**: Presets to match VectorBT, Backtrader, Zipline

### Target Use Cases

- Academic research (replication studies, strategy validation)
- Algorithmic trading development
- Portfolio optimization and rebalancing
- Multi-asset trading systems
- Transition from backtesting to live trading

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Features](#core-features)
4. [Account Types](#account-types)
5. [Order Types](#order-types)
6. [Execution Modes](#execution-modes)
7. [Portfolio Rebalancing](#portfolio-rebalancing)
8. [Commission & Slippage Models](#commission--slippage-models)
9. [Execution Realism](#execution-realism)
10. [Analytics & Metrics](#analytics--metrics)
11. [Analysis & Diagnostic Integration](#analysis--diagnostic-integration)
12. [Market Calendar Integration](#market-calendar-integration)
13. [Multi-Asset Strategies](#multi-asset-strategies)
14. [Configuration System](#configuration-system)
15. [Framework Compatibility](#framework-compatibility)
16. [Validation](#validation)
17. [API Reference](#api-reference)

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

**Dependencies**:
- Python 3.11+
- polars >= 1.15.0
- numpy >= 1.26.4
- pandas >= 2.2.3
- pandas_market_calendars >= 4.4.0 (optional, for calendar features)

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
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Total Trades: {results['num_trades']}")
```

---

## Core Features

### 1. Event-Driven Architecture

Iterates through market events bar-by-bar:
```python
for timestamp, prices, signals, context in datafeed:
    broker.update_time(timestamp, prices)
    broker.process_orders()  # Fill pending orders
    strategy.on_data(timestamp, prices, context, broker)
    broker.process_orders()  # Process new orders
```

### 2. Point-in-Time Correctness

- Strategy decisions use only data available at decision time
- No future peeking, no look-ahead bias
- Realistic order fills (not always at ideal prices)

### 3. Multi-Asset Support

- Trade multiple assets simultaneously
- Unified position tracking across portfolio
- Per-asset commission/slippage
- Cross-asset context data (VIX, SPY, sector indices)

### 4. Polars-Based Data

- Efficient memory usage with lazy evaluation
- Fast DataFeed iteration (~100k events/sec)
- Parquet file support for large datasets

### 5. Exit-First Order Processing

- Exit orders processed before entry orders
- Frees capital immediately for reallocation
- Prevents "locked capital" in rebalancing scenarios

### 6. Realistic Execution

- Volume participation limits (max % of bar volume)
- Market impact modeling (price moves against you)
- Partial fills (fill what's available, queue remainder)
- OHLC-bounded fills (price must be within bar range)

### 7. Flexible Position Tracking

Every position tracks:
```python
position.asset              # Asset symbol
position.quantity           # Current quantity (+ long, - short)
position.entry_price        # Average entry price
position.entry_time         # First entry timestamp
position.current_price      # Latest market price
position.market_value       # quantity * current_price
position.unrealized_pnl     # Market value - cost basis
position.bars_held          # Number of bars since entry
```

### 8. Comprehensive Order Types

- Market orders
- Limit orders
- Stop orders (stop-loss, stop-limit)
- Trailing stops (follow price movement)
- Bracket orders (entry + take-profit + stop-loss in one call)

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
- Short selling prohibited
- Buying power = available cash
- No margin calls

**Use Case**: Conservative strategies, retirement accounts, educational use

### Margin Account

Margin account with leverage and short selling:

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
- Leverage via buying power calculation
- Position reversals (long → short in single order)
- Margin call simulation

**Buying Power Formula**:
```
Buying Power = (Net Liquidation Value - Maintenance Margin) / Initial Margin
```

**Maintenance Margin Calculation**:
```
MM = sum(abs(position.market_value) * maintenance_margin)
```

**Use Case**: Hedge funds, long-short equity, market-neutral strategies

### Futures Account (Planned for v0.3)

```python
engine = Engine(
    feed, strategy,
    account_type="futures",
    initial_margin_per_contract=5_000,  # CME ES requirement
    maintenance_margin_per_contract=4_500,
)
```

---

## Order Types

### Market Orders

Fill at current market price:
```python
# Buy 100 shares at market
broker.submit_order("AAPL", 100)

# Sell 50 shares at market (signed quantity)
broker.submit_order("AAPL", -50)
```

### Limit Orders

Only fill at specified price or better:
```python
broker.submit_order(
    "AAPL", 100,
    order_type=OrderType.LIMIT,
    limit_price=150.0,  # Buy at $150 or lower
)
```

### Stop Orders

Trigger when price crosses stop level:
```python
# Stop-loss: sell when price drops to $145
broker.submit_order(
    "AAPL", -100,
    order_type=OrderType.STOP,
    stop_price=145.0,
)

# Stop-limit: convert to limit order when stop triggers
broker.submit_order(
    "AAPL", -100,
    order_type=OrderType.STOP_LIMIT,
    stop_price=145.0,
    limit_price=144.0,
)
```

### Trailing Stops

Follow price movement, lock in gains:
```python
broker.submit_order(
    "AAPL", -100,
    order_type=OrderType.TRAILING_STOP,
    trail_amount=5.0,  # Trail $5 below high
)
```

### Bracket Orders

Entry + take-profit + stop-loss in one call:
```python
entry, take_profit, stop_loss = broker.submit_bracket(
    asset="AAPL",
    quantity=100,
    take_profit=160.0,  # Sell at profit
    stop_loss=145.0,    # Sell at loss
)
```

### Order Management

```python
# Update pending order
broker.update_order(order.order_id, stop_price=new_stop)

# Cancel order
broker.cancel_order(order.order_id)

# Close entire position
broker.close_position("AAPL")
```

---

## Execution Modes

### Same-Bar Execution (Default)

Orders fill at current bar (realistic for intraday bars):
```python
engine = Engine(feed, strategy, execution_mode=ExecutionMode.SAME_BAR)
```

**Fill Logic**:
- Market orders: Fill at close
- Limit/Stop orders: Fill if price available in OHLC

**Use Case**: Minute/hourly bars, realistic for liquid markets

### Next-Bar Execution

Orders fill at next bar's open (Backtrader-style):
```python
engine = Engine(feed, strategy, execution_mode=ExecutionMode.NEXT_BAR)
```

**Fill Logic**:
- All orders queued until next bar
- Fill at open price of next bar

**Use Case**: Daily bars, conservative modeling, replicating Backtrader

---

## Portfolio Rebalancing

### TargetWeightExecutor

Rebalance portfolio to target weights:

```python
from ml4t.backtest import TargetWeightExecutor, RebalanceConfig

class WeightedStrategy(Strategy):
    def __init__(self):
        self.executor = TargetWeightExecutor(
            config=RebalanceConfig(
                min_trade_value=500,      # Skip trades < $500
                allow_partial=True,       # Allow partial fills
                max_rebalance_pct=0.5,    # Limit single rebalance to 50% turnover
            )
        )

    def on_data(self, timestamp, data, context, broker):
        # Define target weights
        targets = {
            "AAPL": 0.30,  # 30% of portfolio
            "GOOG": 0.25,  # 25%
            "MSFT": 0.20,  # 20%
            "AMZN": 0.15,  # 15%
            # Cash: 10% (implicit, = 1.0 - sum(weights))
        }

        # Execute rebalance (creates orders)
        result = self.executor.execute(targets, data, broker)

        # Check results
        if not result.success:
            print(f"Rebalancing failed: {result.message}")
```

**Features**:
- Calculates required trades to reach target weights
- Respects account constraints (cash available, buying power)
- Handles partial fills gracefully
- Volume participation limits (optional)
- Market impact modeling (optional)

**ExecutionResult**:
```python
result.success           # True if all orders submitted
result.orders_placed     # List[Order] submitted
result.orders_skipped    # List[(asset, reason)] not submitted
result.message           # Human-readable summary
result.projected_weights # Expected weights after fills
```

---

## Commission & Slippage Models

### Commission Models

```python
from ml4t.backtest import (
    NoCommission,
    PerShareCommission,
    PercentageCommission,
    TieredCommission,
    CombinedCommission,
)

# No commission
NoCommission()

# Per-share ($0.01/share, $1 min)
PerShareCommission(cost_per_share=0.01, min_cost=1.0)

# Percentage (0.1% = 10bps)
PercentageCommission(percentage=0.001)

# Tiered by volume
TieredCommission(tiers=[
    (0, 10_000, 0.01),           # $0.01/share for 0-10k shares
    (10_000, 100_000, 0.005),     # $0.005/share for 10k-100k
    (100_000, float('inf'), 0.001), # $0.001/share for 100k+
])

# Combined (percentage + per-share + min)
CombinedCommission(
    percentage=0.0001,  # 1bp
    per_share=0.005,    # $0.005/share
    min_cost=1.0,       # $1 minimum
)
```

### Slippage Models

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
FixedSlippage(amount=0.05)

# Percentage (0.05% = 5bps)
PercentageSlippage(percentage=0.0005)

# Volume-based (realistic for large orders)
VolumeShareSlippage(
    volume_limit=0.025,  # Max 2.5% of bar volume
    price_impact=0.1,    # 10% price impact if limit hit
)
```

---

## Execution Realism

### Volume Participation Limits

Prevent filling more than X% of bar volume:

```python
from ml4t.backtest import VolumeParticipationLimit

limits = VolumeParticipationLimit(max_participation=0.10)  # Max 10% of volume

engine = Engine(
    feed, strategy,
    execution_limits=limits,
)
```

**Behavior**:
- Order fills up to `max_participation * bar_volume`
- Remainder queued for future bars
- Realistic for large orders in low-volume stocks

### Market Impact Modeling

Price moves against you when trading large sizes:

```python
from ml4t.backtest import LinearImpact, SquareRootImpact

# Linear impact: price moves linearly with size
impact = LinearImpact(impact_coefficient=0.1)

# Square-root impact: price_impact = k * sqrt(size / volume)
impact = SquareRootImpact(impact_coefficient=0.1)

engine = Engine(
    feed, strategy,
    market_impact_model=impact,
)
```

**Execution Price Calculation**:
```
Effective Price = Base Price * (1 + impact_pct * direction)
  where:
    impact_pct = impact_coefficient * (order_size / bar_volume)^0.5
    direction = +1 for buys (price up), -1 for sells (price down)
```

---

## Analytics & Metrics

### EquityCurve

Track portfolio value over time:

```python
from ml4t.backtest import EquityCurve

curve = EquityCurve(initial_cash=100_000)

# Update after each bar
curve.update(timestamp=datetime.now(), equity=portfolio_value)

# Get equity curve data
curve.to_dataframe()  # Returns pl.DataFrame
curve.to_list()       # Returns List[Tuple[datetime, float]]
```

### TradeAnalyzer

Analyze completed round-trip trades:

```python
from ml4t.backtest import TradeAnalyzer

analyzer = TradeAnalyzer()

# After backtest
results = engine.run()
trades = results["trades"]  # List[Trade]

# Analyze
stats = analyzer.analyze(trades)

# Trade statistics
stats["num_trades"]         # Total trades
stats["winning_trades"]     # Number of winners
stats["losing_trades"]      # Number of losers
stats["win_rate"]           # Win rate (0-1)
stats["profit_factor"]      # Gross profit / gross loss
stats["avg_win"]            # Average winning trade
stats["avg_loss"]           # Average losing trade
stats["largest_win"]        # Largest winning trade
stats["largest_loss"]       # Largest losing trade
stats["avg_bars_held"]      # Average holding period
```

### Performance Metrics

```python
from ml4t.backtest import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    cagr,
    volatility,
)

# Sharpe ratio (risk-adjusted return)
sharpe = sharpe_ratio(returns, risk_free_rate=0.02)

# Sortino ratio (downside risk-adjusted)
sortino = sortino_ratio(returns, target_return=0.0)

# Calmar ratio (return / max drawdown)
calmar = calmar_ratio(returns)

# Maximum drawdown
max_dd, max_dd_pct = max_drawdown(equity_curve)

# Compound annual growth rate
cagr_value = cagr(initial_value, final_value, num_years)

# Volatility (annualized)
vol = volatility(returns, periods_per_year=252)
```

### Results Dictionary

```python
results = engine.run()

# Portfolio metrics
results["initial_cash"]      # Starting cash
results["final_value"]       # Ending portfolio value
results["total_return"]      # Dollar return
results["total_return_pct"]  # Percent return
results["max_drawdown"]      # Maximum drawdown (dollars)
results["max_drawdown_pct"]  # Maximum drawdown (percent)
results["sharpe_ratio"]      # Sharpe ratio
results["sortino_ratio"]     # Sortino ratio
results["calmar_ratio"]      # Calmar ratio
results["cagr"]              # Compound annual growth rate

# Trade metrics
results["num_trades"]        # Total completed trades
results["winning_trades"]    # Number of winners
results["losing_trades"]     # Number of losers
results["win_rate"]          # Win rate (0-1)
results["profit_factor"]     # Gross profit / gross loss
results["avg_win"]           # Average winning trade
results["avg_loss"]          # Average losing trade

# Costs
results["total_commission"]  # Total commission paid
results["total_slippage"]    # Total slippage cost

# Detailed data
results["trades"]            # List[Trade] - completed round-trips
results["fills"]             # List[Fill] - all order fills
results["equity_curve"]      # List[Tuple[datetime, float]]
```

---

## Analysis & Diagnostic Integration

### BacktestAnalyzer

Bridge backtest results to `ml4t.diagnostic` for comprehensive analysis:

```python
from ml4t.backtest import Engine, BacktestAnalyzer, TradeStatistics

# Run backtest
engine = Engine(feed, strategy, initial_cash=100_000)
results = engine.run()

# Create analyzer
analyzer = BacktestAnalyzer(engine)

# Get trade statistics (no diagnostic dependency required)
stats = analyzer.trade_statistics()
print(f"Win Rate: {stats.win_rate:.2%}")
print(f"Profit Factor: {stats.profit_factor:.2f}")
print(f"Expectancy: ${stats.expectancy:.2f}")
print(f"Avg Holding: {stats.avg_bars_held:.1f} bars")
```

### TradeStatistics

Quick metrics without external dependencies:

```python
stats.total_trades       # Total round-trip trades
stats.winning_trades     # Number of winners
stats.losing_trades      # Number of losers
stats.win_rate           # Win rate (0.0-1.0)
stats.profit_factor      # Gross profit / gross loss
stats.payoff_ratio       # Avg win / avg loss
stats.expectancy         # Expected value per trade
stats.largest_win        # Best single trade
stats.largest_loss       # Worst single trade
stats.avg_bars_held      # Average holding period
stats.total_commission   # Total fees paid
```

### Diagnostic Library Integration

For advanced analysis, convert trades to diagnostic format:

```python
from ml4t.backtest import to_trade_records, to_returns_series

# Convert trades for diagnostic library
records = to_trade_records(engine.broker.trades)

# Use with ml4t.diagnostic
from ml4t.diagnostic.evaluation import TradeAnalysis
analysis = TradeAnalysis(records)
worst = analysis.worst_trades(n=20)

# SHAP-based error analysis (for ML strategies)
from ml4t.diagnostic.evaluation import TradeShapAnalyzer
shap_analyzer = TradeShapAnalyzer(model, features, shap_values)
patterns = shap_analyzer.explain_worst_trades(worst)
```

### Example Workflows

See `examples/analysis/` for complete examples:

| Example | Description |
|---------|-------------|
| `01_single_asset_trade_stats.py` | Basic trade statistics |
| `02_multi_asset_portfolio.py` | Per-asset breakdown |
| `03_ml_linear_regression.py` | ML strategy analysis |
| `04_ml_gradient_boosting.py` | SHAP integration |
| `05_benchmark_comparison.py` | Statistical significance |

---

## Market Calendar Integration

Integration with `pandas_market_calendars` for realistic trading schedules:

```python
from ml4t.backtest import (
    get_calendar,
    get_trading_days,
    is_trading_day,
    filter_to_trading_days,
    get_holidays,
)

# Get NYSE calendar
nyse = get_calendar("NYSE")

# Filter DataFrame to trading days only
trading_data = filter_to_trading_days(df, calendar="NYSE")

# Check if date is a trading day
if is_trading_day(datetime(2024, 12, 25), "NYSE"):
    # Market is open
    pass

# Get holidays
holidays = get_holidays("NYSE", start_date="2024-01-01", end_date="2024-12-31")

# Generate trading minutes (for intraday strategies)
from ml4t.backtest import generate_trading_minutes
minutes = generate_trading_minutes(
    start_date="2024-01-01",
    end_date="2024-01-31",
    calendar="NYSE",
)
```

**Supported Calendars**:
- NYSE, NASDAQ, ARCA, BATS
- LSE, TSX, JPX, HKEX, ASX
- CME, ICE, EUREX
- 50+ global exchanges

---

## Multi-Asset Strategies

### Basic Multi-Asset

```python
class MultiAssetStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        # data: Dict[asset, Dict[field, value]]
        for asset, bar in data.items():
            price = bar["close"]
            signals = bar.get("signals", {})

            position = broker.get_position(asset)
            current_qty = position.quantity if position else 0

            # Per-asset logic
            if signals.get("buy") and current_qty == 0:
                broker.submit_order(asset, 100)
            elif signals.get("sell") and current_qty > 0:
                broker.close_position(asset)
```

### Portfolio Rebalancing

```python
class PortfolioStrategy(Strategy):
    def __init__(self):
        self.executor = TargetWeightExecutor()

    def on_data(self, timestamp, data, context, broker):
        # Rebalance monthly
        if timestamp.day == 1:
            # Equal weight across available assets
            assets = list(data.keys())
            weight = 0.95 / len(assets)
            targets = {a: weight for a in assets}

            self.executor.execute(targets, data, broker)
```

### Market-Wide Context

```python
# DataFeed with context
feed = DataFeed(
    prices_df=prices,
    context_df=market_context,  # timestamp, VIX, SPY_return, sector_rotation
)

class ContextAwareStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        vix = context.get("VIX", 0)
        spy_return = context.get("SPY_return", 0)

        # Adjust strategy based on market conditions
        if vix > 30:  # High volatility
            # Reduce exposure
            pass
        elif spy_return < -0.02:  # Market down >2%
            # Go defensive
            pass
```

---

## Configuration System

### BacktestConfig

Centralized configuration for all engine parameters:

```python
from ml4t.backtest import BacktestConfig, ExecutionMode, FillTiming

config = BacktestConfig(
    # Execution
    execution_mode=ExecutionMode.SAME_BAR,
    fill_timing=FillTiming.CLOSE_ONLY,
    fractional_shares=True,

    # Account
    initial_cash=100_000.0,
    account_type="cash",
    initial_margin=0.5,
    maintenance_margin=0.25,

    # Commission
    commission_model=PerShareCommission(0.01),

    # Slippage
    slippage_model=FixedSlippage(0.05),

    # Limits
    max_positions=20,
    max_position_size=50_000.0,
    max_order_size=10_000.0,

    # Rebalancing
    rebalance_frequency="monthly",
    target_weight_tolerance=0.05,
)

engine = Engine.from_config(feed, strategy, config)
```

### Preset Configurations

Match behavior of other frameworks:

```python
# VectorBT-compatible
config = BacktestConfig.from_preset("vectorbt")
# - Same-bar fills at close
# - Fractional shares
# - Process all signals

# Backtrader-compatible
config = BacktestConfig.from_preset("backtrader")
# - Next-bar fills at open
# - Integer shares only
# - Check position before acting

# Zipline-compatible
config = BacktestConfig.from_preset("zipline")
# - Next-bar fills at open
# - Integer shares
# - Per-share commission
# - Volume-based slippage
```

---

## Framework Compatibility

ml4t.backtest provides presets and adapters to match other frameworks:

| Framework | Execution | Shares | Commission | Slippage | Preset |
|-----------|-----------|--------|------------|----------|--------|
| VectorBT Pro | Same-bar, close | Fractional | None | None | `vectorbt` |
| Backtrader | Next-bar, open | Integer | Per-share | Fixed | `backtrader` |
| Zipline | Next-bar, open | Integer | Per-share | Volume-based | `zipline` |

### Validation Status

| Framework | Environment | Match Status |
|-----------|-------------|--------------|
| VectorBT Pro | `.venv-vectorbt-pro` | ✅ EXACT (4/4 scenarios) |
| VectorBT OSS | `.venv-validation` | ✅ EXACT (4/4 scenarios) |
| Backtrader | `.venv-validation` | ✅ EXACT (4/4 scenarios) |
| Zipline | `.venv-validation` | ✅ Within tolerance (strategy-level stops) |

**Validation Scenarios**:
1. Long-only basic trades
2. Long-short with position flipping
3. Stop-loss execution
4. Take-profit execution

---

## Validation

### Test Suite

```bash
# Run all tests
source .venv/bin/activate
pytest tests/ -q

# Run with coverage
pytest tests/ --cov=src/ml4t/backtest --cov-report=html

# Test specific module
pytest tests/accounting/ -v
```

**Test Organization**:
```
tests/
├── __init__.py
├── test_core.py              # Engine, Broker, DataFeed
├── test_broker.py            # Broker-specific tests
├── test_calendar.py          # Calendar function tests
├── accounting/               # Account system tests
│   ├── test_account_state.py
│   ├── test_cash_account_policy.py
│   ├── test_margin_account_policy.py
│   ├── test_gatekeeper.py
│   └── test_position.py
├── execution/                # Execution model tests
│   ├── test_impact.py
│   └── test_rebalancer.py
└── risk/                     # Risk management tests
    ├── test_static_rules.py
    ├── test_dynamic_rules.py
    ├── test_signal_rules.py
    ├── test_composite_rules.py
    ├── test_portfolio_limits.py
    └── test_portfolio_manager.py
```

### Framework Validation

**Strategy**:
1. Isolated virtual environments per framework (avoid dependency conflicts)
2. Scenario-based validation (long-only, long-short, stops, take-profit)
3. Exact numeric match verification (trades, fills, final equity)

**Run Validation**:
```bash
# VectorBT Pro (commercial license required)
source .venv-vectorbt-pro/bin/activate
python validation/vectorbt_pro/benchmark_performance.py

# Open-source frameworks
source .venv-validation/bin/activate
python validation/backtrader/benchmark_performance.py
```

---

## API Reference

### Engine

```python
engine = Engine(
    datafeed: DataFeed,
    strategy: Strategy,
    initial_cash: float = 100_000.0,
    account_type: str = "cash",  # "cash" or "margin"
    initial_margin: float = 0.5,
    maintenance_margin: float = 0.25,
    commission_model: CommissionModel = NoCommission(),
    slippage_model: SlippageModel = NoSlippage(),
    execution_mode: ExecutionMode = ExecutionMode.SAME_BAR,
    execution_limits: ExecutionLimits | None = None,
    market_impact_model: MarketImpactModel | None = None,
)

results = engine.run()
```

### Strategy

```python
class Strategy:
    def on_start(self, broker: Broker) -> None:
        """Called once before backtest."""

    def on_data(
        self,
        timestamp: datetime,
        data: dict[str, dict],  # {asset: {field: value}}
        context: dict,           # Market-wide data
        broker: Broker,
    ) -> None:
        """Called on each bar."""

    def on_end(self, broker: Broker) -> None:
        """Called once after backtest."""
```

### Broker

```python
# Orders
broker.submit_order(asset, quantity, side, order_type, limit_price, stop_price)
broker.submit_bracket(asset, quantity, take_profit, stop_loss)
broker.update_order(order_id, **updates)
broker.cancel_order(order_id)
broker.close_position(asset)

# State queries
broker.get_position(asset) -> Position | None
broker.get_account_value() -> float
broker.get_cash() -> float
broker.positions -> dict[str, Position]
broker.pending_orders -> list[Order]
```

### DataFeed

```python
feed = DataFeed(
    prices_df: pl.DataFrame | None = None,
    signals_df: pl.DataFrame | None = None,
    context_df: pl.DataFrame | None = None,
    prices_path: str | None = None,
    signals_path: str | None = None,
    context_path: str | None = None,
)

# Iterate
for timestamp, data, context in feed:
    # data: Dict[asset, Dict[field, value]]
    # context: Dict[field, value]
    pass
```

### Position

```python
position.asset           # str
position.quantity        # float (+ long, - short)
position.entry_price     # float (average cost)
position.entry_time      # datetime
position.current_price   # float | None
position.market_value    # float (quantity * current_price)
position.cost_basis      # float (quantity * entry_price)
position.unrealized_pnl  # float (market_value - cost_basis)
position.bars_held       # int
```

### Order

```python
order.order_id         # str
order.asset            # str
order.side             # OrderSide.BUY or OrderSide.SELL
order.quantity         # float
order.order_type       # OrderType (MARKET, LIMIT, STOP, etc.)
order.status           # OrderStatus (PENDING, FILLED, CANCELLED, etc.)
order.limit_price      # float | None
order.stop_price       # float | None
order.trail_amount     # float | None
order.filled_price     # float | None
order.filled_quantity  # float | None
order.filled_at        # datetime | None
order.created_at       # datetime
```

### Trade

```python
trade.asset            # str
trade.entry_time       # datetime
trade.exit_time        # datetime
trade.entry_price      # float
trade.exit_price       # float
trade.quantity         # float
trade.pnl              # float (realized P&L)
trade.pnl_pct          # float (return %)
trade.bars_held        # int
trade.commission       # float
trade.slippage         # float
```

---

## Documentation

- **Project Map**: `.claude/PROJECT_MAP.md` - Architecture overview
- **Architecture Decisions**: `docs/*.md` - Design rationale
- **Validation Methodology**: `.claude/memory/validation_methodology.md`
- **Code Examples**: `examples/` - Strategy templates

---

## Roadmap

### v0.3 (Q1 2025)

- [ ] Futures support (contract specifications, margin calculations)
- [ ] Options support (Black-Scholes pricing, Greeks)
- [ ] Advanced order types (OCO, OSO, iceberg)
- [ ] Risk management framework (position sizing rules, portfolio limits)
- [ ] Performance attribution (factor analysis, Brinson model)

### v0.4 (Q2 2025)

- [ ] Live trading integration (`ml4t-live` companion package)
- [ ] Real-time data feeds (IB, Alpaca)
- [ ] Paper trading mode
- [ ] Web-based dashboard (Dash/Streamlit)

---

## License

MIT License - See LICENSE file

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

---

## Citation

If you use ml4t.backtest in academic research:

```bibtex
@software{ml4t_backtest,
  author = {Your Name},
  title = {ml4t.backtest: Event-Driven Backtesting Engine},
  year = {2025},
  url = {https://github.com/yourusername/ml4t-backtest},
  version = {0.2.0}
}
```

---

## Changelog

### v0.2.0 (2025-11-22)

- ✅ Major repository cleanup (99.2% code reduction: 739K → 5.5K lines)
- ✅ Complete accounting system (cash + margin accounts)
- ✅ Exit-first order processing
- ✅ Framework compatibility presets (VectorBT, Backtrader, Zipline)
- ✅ Per-framework validation strategy (EXACT matches achieved)
- ✅ Portfolio rebalancing (TargetWeightExecutor)
- ✅ Execution realism (volume limits, market impact, partial fills)
- ✅ Analytics suite (EquityCurve, TradeAnalyzer, metrics)
- ✅ Market calendar integration (pandas_market_calendars)
- ✅ 474 tests passing (73% coverage)

### v0.1.0 (2025-01-20)

- Initial prototype
- Event-driven architecture
- Basic order execution

---

*ml4t.backtest - Minimal, correct, fast.*
