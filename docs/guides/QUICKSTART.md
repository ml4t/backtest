# ml4t.backtest Quick Start

Get up and running with ml4t.backtest in 5 minutes.

## Installation

```bash
pip install ml4t-backtest
```

## Core Concepts

ml4t.backtest has four main components:

1. **DataFeed** - Delivers OHLCV data + pre-computed signals
2. **Strategy** - Your trading logic (just implement `on_data()`)
3. **Broker** - Executes orders, tracks positions
4. **Engine** - Orchestrates the event loop

```
DataFeed ──► Engine ──► Strategy
                │            │
                └── Broker ◄─┘
```

## Minimal Example

```python
import polars as pl
from ml4t.backtest import Strategy, DataFeed, Engine

# 1. Prepare data (OHLCV with timestamp and asset columns)
prices_df = pl.DataFrame({
    'timestamp': pd.date_range('2020-01-01', periods=252, freq='D'),
    'asset': 'AAPL',
    'open': np.random.uniform(100, 150, 252),
    'high': np.random.uniform(100, 150, 252),
    'low': np.random.uniform(100, 150, 252),
    'close': np.random.uniform(100, 150, 252),
    'volume': np.random.uniform(1e6, 1e7, 252),
})

# 2. Pre-compute signals
signals_df = prices_df.select(['timestamp', 'asset']).with_columns([
    pl.Series('sma_cross', np.random.choice([0, 1], 252)),  # Your actual signal
])

# 3. Define strategy
class SimpleStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            signal = bar['signals'].get('sma_cross', 0)
            pos = broker.get_position(asset)

            if signal == 1 and pos is None:
                qty = int(broker.get_cash() * 0.95 / bar['close'])
                broker.submit_order(asset, qty)
            elif signal == 0 and pos is not None:
                broker.close_position(asset)

# 4. Run backtest
feed = DataFeed(prices_df=prices_df, signals_df=signals_df)
engine = Engine(feed, SimpleStrategy(), initial_cash=100000)
result = engine.run()

# 5. Analyze results
print(f"Final value: ${broker.get_account_value():,.2f}")
print(f"Trades: {len(broker.trades)}")
```

## Data Format

### prices_df (required)

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Bar timestamp |
| asset | str | Asset identifier |
| open | float | Open price |
| high | float | High price |
| low | float | Low price |
| close | float | Close price |
| volume | float | Volume |

### signals_df (optional)

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Must match prices_df |
| asset | str | Must match prices_df |
| *your_signals* | any | Any signal columns you need |

### context_df (optional)

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Bar timestamp |
| *your_context* | any | Market-wide context (VIX, regime, etc.) |

## Strategy Interface

```python
class Strategy(ABC):
    @abstractmethod
    def on_data(self, timestamp, data, context, broker) -> None:
        """Called for each bar with all available data.

        Args:
            timestamp: Current bar timestamp
            data: Dict of {asset: {open, high, low, close, volume, signals: {...}}}
            context: Dict of market-wide context data
            broker: Broker instance for order submission
        """
        pass

    def on_start(self, broker) -> None:
        """Called before backtest starts. Set up rules here."""
        pass

    def on_end(self, broker) -> None:
        """Called after backtest ends. Final cleanup."""
        pass
```

## Broker API

### Submitting Orders

```python
# Market order (most common)
broker.submit_order('AAPL', 100)      # Buy 100 shares
broker.submit_order('AAPL', -100)     # Sell 100 shares (or short)

# Limit order
from ml4t.backtest import OrderType
broker.submit_order('AAPL', 100, order_type=OrderType.LIMIT, limit_price=150.0)

# Stop order
broker.submit_order('AAPL', -100, order_type=OrderType.STOP, stop_price=145.0)

# Bracket order (entry + take-profit + stop-loss)
entry, tp, sl = broker.submit_bracket('AAPL', 100, take_profit=160.0, stop_loss=145.0)
```

### Closing Positions

```python
broker.close_position('AAPL')  # Close entire position
```

### Querying State

```python
# Position info
pos = broker.get_position('AAPL')
if pos:
    print(pos.quantity)        # Shares held (negative = short)
    print(pos.entry_price)     # Average entry price
    print(pos.bars_held)       # Bars since entry
    print(pos.unrealized_pnl(current_price))

# All positions
for asset, pos in broker.positions.items():
    print(f"{asset}: {pos.quantity}")

# Cash and value
cash = broker.get_cash()
total_value = broker.get_account_value()
```

### Order Management

```python
# Update pending order
broker.update_order(order_id, stop_price=new_stop)

# Cancel order
broker.cancel_order(order_id)

# Get pending orders
pending = broker.get_pending_orders()
pending_for_asset = broker.get_pending_orders('AAPL')
```

## Configuration Options

### Engine Parameters

```python
engine = Engine(
    feed,
    strategy,
    initial_cash=100000,

    # Account type
    account_type='cash',        # or 'margin'
    initial_margin=0.5,         # For margin accounts
    maintenance_margin=0.25,    # For margin accounts

    # Execution mode
    execution_mode=ExecutionMode.SAME_BAR,   # Fill at close (default)
    # execution_mode=ExecutionMode.NEXT_BAR,  # Fill at next bar's open

    # Commission model
    commission_model=PercentageCommission(0.001),  # 0.1% commission
    # commission_model=PerShareCommission(0.01),   # $0.01 per share

    # Slippage model
    slippage_model=PercentageSlippage(0.0005),    # 0.05% slippage
    # slippage_model=FixedSlippage(0.01),          # $0.01 per share

    # Futures support
    contract_specs={'ES': es_spec},
)
```

### Execution Modes

- **SAME_BAR**: Orders fill at the same bar's close price (default)
- **NEXT_BAR**: Orders fill at the next bar's open price (more realistic)

```python
from ml4t.backtest import ExecutionMode

engine = Engine(
    feed, strategy,
    execution_mode=ExecutionMode.NEXT_BAR,
)
```

## Risk Management

```python
from ml4t.backtest.risk import StopLoss, TakeProfit, TimeExit, RuleChain

class MyStrategy(Strategy):
    def on_start(self, broker):
        rules = RuleChain([
            StopLoss(pct=0.02),           # 2% stop-loss
            TakeProfit(pct=0.05),         # 5% take-profit
            TimeExit(max_bars=20),        # Exit after 20 bars
        ])
        broker.set_position_rules(rules)
```

## Analyzing Results

```python
from ml4t.backtest import sharpe_ratio, max_drawdown, cagr

# After running backtest
trades = broker.trades
fills = broker.fills

# Calculate metrics on equity curve
equity = [broker.initial_cash]  # You'd track this in on_data
sr = sharpe_ratio(returns)
mdd = max_drawdown(equity)
annual_return = cagr(equity, periods_per_year=252)
```

## Multi-Asset Example

```python
class RotationStrategy(Strategy):
    def __init__(self, max_positions: int = 5):
        self.max_positions = max_positions

    def on_data(self, timestamp, data, context, broker):
        # Rank assets by signal strength
        ranked = sorted(
            data.items(),
            key=lambda x: x[1]['signals'].get('momentum', 0),
            reverse=True
        )

        # Get top N
        top_assets = [asset for asset, _ in ranked[:self.max_positions]
                      if ranked[0][1]['signals'].get('momentum', 0) > 0]

        # Exit positions not in top
        for asset in list(broker.positions.keys()):
            if asset not in top_assets:
                broker.close_position(asset)

        # Enter top positions
        account_value = broker.get_account_value()
        per_position = account_value / self.max_positions

        for asset in top_assets:
            if broker.get_position(asset) is None:
                price = data[asset]['close']
                qty = int(per_position / price)
                if qty > 0:
                    broker.submit_order(asset, qty)
```

## Next Steps

- [Migration Guide](./MIGRATION_GUIDE.md) - Coming from Zipline or Backtrader
- [Differentiators](./DIFFERENTIATORS.md) - What ml4t.backtest does better
- [API Reference](../api/) - Full API documentation
