# Account Policies

Configure account behavior for your backtests using simple boolean flags.

## Account Types

ml4t-backtest uses a **unified configuration** approach with two key flags:

| Flag | Description |
|------|-------------|
| `allow_short_selling` | Whether short positions are allowed |
| `allow_leverage` | Whether margin leverage is allowed |

These flags map to traditional account types:

| Account Type | `allow_short_selling` | `allow_leverage` |
|--------------|----------------------|------------------|
| Cash | `False` | `False` |
| Crypto | `True` | `False` |
| Margin | `True` | `True` |

## Cash Account (Default)

Long-only with no leverage - simplest and safest:

```python
from ml4t.backtest import BacktestConfig, Engine

config = BacktestConfig(
    initial_cash=100_000,
    allow_short_selling=False,  # Default
    allow_leverage=False,       # Default
)

# Or equivalently, just use defaults:
config = BacktestConfig(initial_cash=100_000)
```

## Crypto Account

Short selling allowed, but no leverage:

```python
config = BacktestConfig(
    initial_cash=100_000,
    allow_short_selling=True,
    allow_leverage=False,
)
```

## Margin Account

Full margin trading with short selling and leverage:

```python
config = BacktestConfig(
    initial_cash=100_000,
    allow_short_selling=True,
    allow_leverage=True,
    initial_margin=0.5,              # 50% initial margin (2x leverage)
    long_maintenance_margin=0.25,    # 25% maintenance for longs
    short_maintenance_margin=0.30,   # 30% maintenance for shorts
)
```

Common margin configurations:

| Use Case | `initial_margin` | Max Leverage |
|----------|------------------|--------------|
| Standard margin | 0.50 | 2x |
| Day trading | 0.25 | 4x |
| Futures-style | 0.10 | 10x |

## Using Engine Directly

You can also pass account settings directly to Engine:

```python
from ml4t.backtest import Engine, DataFeed

engine = Engine(
    feed=feed,
    strategy=strategy,
    initial_cash=100_000,
    allow_short_selling=True,
    allow_leverage=True,
    initial_margin=0.5,
)
```

## Using Broker.from_config()

For advanced use cases, create a broker from config:

```python
from ml4t.backtest import Broker, BacktestConfig

config = BacktestConfig(
    initial_cash=100_000,
    allow_short_selling=True,
    allow_leverage=True,
)

broker = Broker.from_config(config)
```

## Transaction Costs

Configure commission and slippage:

```python
from ml4t.backtest import BacktestConfig
from ml4t.backtest.config import CommissionModel, SlippageModel

config = BacktestConfig(
    initial_cash=100_000,
    commission_model=CommissionModel.PERCENTAGE,
    commission_rate=0.001,     # 0.1% per trade
    slippage_model=SlippageModel.PERCENTAGE,
    slippage_rate=0.0005,      # 0.05% slippage
)
```

## Presets

Use presets for common scenarios:

```python
from ml4t.backtest import BacktestConfig

# Backtrader-compatible settings
config = BacktestConfig.from_preset("backtrader")

# VectorBT-compatible settings
config = BacktestConfig.from_preset("vectorbt")

# Conservative production settings
config = BacktestConfig.from_preset("realistic")
```

## Insufficient Funds

Orders that exceed available buying power will be rejected. Use the Gatekeeper to check orders before submission:

```python
from ml4t.backtest.accounting import Gatekeeper

gatekeeper = Gatekeeper(account_state, policy)
is_valid, reason = gatekeeper.validate_order(order)
if not is_valid:
    print(f"Order rejected: {reason}")
```

## Migration from `account_type` (Pre-1.0)

If you were using the old `account_type` string parameter, update as follows:

```python
# Old API (deprecated)
broker = Broker(account_type="margin")

# New API
broker = Broker(allow_short_selling=True, allow_leverage=True)

# Or with config
config = BacktestConfig(allow_short_selling=True, allow_leverage=True)
broker = Broker.from_config(config)
```
