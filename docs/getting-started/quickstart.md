# Quickstart

Build and run your first backtest in 5 minutes.

## Minimal Example

```python
import polars as pl
from ml4t.backtest import Engine, DataFeed, Strategy

class BuyAndHold(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            if broker.get_position(asset) is None:
                broker.submit_order(asset, 100)

# Create sample price data
prices = pl.DataFrame({
    "timestamp": pl.datetime_range(
        start=pl.datetime(2023, 1, 2),
        end=pl.datetime(2023, 6, 30),
        interval="1d",
        eager=True,
    ),
    "asset": "AAPL",
    "open": [150.0 + i * 0.1 for i in range(180)],
    "high": [151.0 + i * 0.1 for i in range(180)],
    "low": [149.0 + i * 0.1 for i in range(180)],
    "close": [150.5 + i * 0.1 for i in range(180)],
    "volume": [1_000_000] * 180,
})

feed = DataFeed(prices_df=prices)
engine = Engine(feed=feed, strategy=BuyAndHold())
result = engine.run()

print(f"Final Value:  ${result.metrics['final_value']:,.2f}")
print(f"Total Return: {result.metrics['total_return_pct']:.1f}%")
print(f"Sharpe Ratio: {result.metrics['sharpe']:.2f}")
print(f"Trades:       {result.metrics['num_trades']}")
```

## Data Format

DataFeed expects a Polars DataFrame with these columns:

| Column | Type | Required |
|--------|------|----------|
| `timestamp` | Datetime | Yes |
| `asset` | String | Yes |
| `open` | Float | Yes |
| `high` | Float | Yes |
| `low` | Float | Yes |
| `close` | Float | Yes |
| `volume` | Float | Yes |

For multi-asset backtests, stack all assets in a single DataFrame -- the engine
handles partitioning by timestamp automatically.

## Strategy Callbacks

Every strategy subclasses `Strategy` and implements `on_data`:

```python
class MyStrategy(Strategy):
    def on_start(self, broker):
        """Called once before the backtest starts. Set up risk rules here."""
        pass

    def on_data(self, timestamp, data, context, broker):
        """Called on every bar. Generate orders here.

        Args:
            timestamp: Current bar's datetime
            data: Dict of {asset: {open, high, low, close, volume, signals}}
            context: Dict of context data (if provided)
            broker: Broker for submitting orders and querying positions
        """
        pass

    def on_end(self, broker):
        """Called once after the backtest ends."""
        pass
```

## Signal-Based Strategy

Pass pre-computed signals alongside prices:

```python
from ml4t.backtest import run_backtest

class SignalStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            signal = bar.get("signals", {}).get("prediction", 0)
            position = broker.get_position(asset)

            if signal > 0.7 and position is None:
                # Buy 10% of portfolio value
                equity = broker.get_account_value()
                shares = int(equity * 0.10 / bar["close"])
                if shares > 0:
                    broker.submit_order(asset, shares)

            elif signal < 0.3 and position is not None:
                broker.close_position(asset)

# Signals DataFrame has same timestamp/asset columns plus your signal columns
result = run_backtest(prices, SignalStrategy(), signals=signals_df)
```

## Adding Transaction Costs

```python
from ml4t.backtest import BacktestConfig
from ml4t.backtest.config import CommissionType, SlippageType

config = BacktestConfig(
    initial_cash=100_000,
    commission_type=CommissionType.PERCENTAGE,
    commission_rate=0.001,       # 0.1% per trade
    slippage_type=SlippageType.PERCENTAGE,
    slippage_rate=0.0005,        # 0.05% slippage
)

result = run_backtest(prices, strategy, config=config)
```

## Using Framework Profiles

Match the exact behavior of another backtesting framework:

```python
# Backtrader-compatible: next-bar execution, integer shares, margin account
result = run_backtest(prices, strategy, config="backtrader")

# VectorBT-compatible: same-bar execution, fractional shares, no costs
result = run_backtest(prices, strategy, config="vectorbt")

# Zipline-compatible: next-bar execution, per-share commission
result = run_backtest(prices, strategy, config="zipline")

# Conservative production settings
result = run_backtest(prices, strategy, config="realistic")
```

See [Profiles](../user-guide/profiles.md) for all available presets and their settings.

## Adding Risk Management

Set position rules in `on_start` to automatically manage exits:

```python
from ml4t.backtest import StopLoss, TakeProfit, TrailingStop, RuleChain

class ProtectedStrategy(Strategy):
    def on_start(self, broker):
        # Rules evaluate in order; first trigger wins
        broker.set_position_rules(RuleChain([
            StopLoss(pct=0.05),        # Exit at -5% loss
            TakeProfit(pct=0.15),      # Exit at +15% profit
            TrailingStop(pct=0.03),    # Trail 3% from high water mark
        ]))

    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            if broker.get_position(asset) is None:
                equity = broker.get_account_value()
                shares = int(equity * 0.10 / bar["close"])
                if shares > 0:
                    broker.submit_order(asset, shares)
```

## Analyzing Results

`engine.run()` returns a `BacktestResult` with metrics, trades, and export methods:

```python
result = engine.run()

# Key metrics (dict)
print(result.metrics["sharpe"])
print(result.metrics["max_drawdown_pct"])
print(result.metrics["win_rate"])
print(result.metrics["profit_factor"])

# Trades as Polars DataFrame
trades_df = result.to_trades_dataframe()
print(trades_df.head())

# Equity curve as Polars DataFrame
equity_df = result.to_equity_dataframe()
print(equity_df.head())

# Export to Parquet for analysis with ml4t-diagnostic
result.to_parquet("./results/my_backtest")
```

## Convenience Function

For quick experiments, `run_backtest` combines DataFeed + Engine in one call:

```python
from ml4t.backtest import run_backtest

# Accepts DataFrames or file paths
result = run_backtest(
    prices="data/prices.parquet",
    strategy=MyStrategy(),
    signals="data/signals.parquet",
    config="backtrader",
)
```

## Next Steps

- [How It Works](../concepts/how-it-works.md) -- understand the execution model
- [Execution Semantics](../user-guide/execution-semantics.md) -- fill ordering, stops, timing
- [Configuration](../user-guide/configuration.md) -- all 40+ behavioral knobs
- [Risk Management](../user-guide/risk-management.md) -- stops, trails, portfolio limits
