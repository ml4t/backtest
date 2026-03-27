# Account Policies

Account policy determines what the broker is allowed to do with cash, leverage, and
short sale proceeds. Use this page when you need to decide whether your strategy
should behave like a long-only cash account, a short-enabled crypto-style account,
or a Reg T margin account.

The configuration is intentionally simple: instead of switching between account
"types", you set the policy flags directly and let the broker enforce the resulting
buying-power rules.

## Quick Example

```python
from ml4t.backtest import BacktestConfig

config = BacktestConfig(
    initial_cash=100_000,
    allow_short_selling=True,
    allow_leverage=True,
    initial_margin=0.5,
    long_maintenance_margin=0.25,
    short_maintenance_margin=0.30,
)
```

Use this pattern when you want realistic shorting and leverage constraints instead
of long-only cash-account behavior.

## When to Use Which Policy

- use a cash account for long-only equity strategies with no borrowing
- use a crypto-style account when you want shorting but no leverage
- use a margin account when leverage, short maintenance, and buying-power checks matter

## Account Types

`ml4t-backtest` uses a unified configuration model with two main flags:

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

Use the default cash-account policy for long-only strategies where proceeds from sales
must settle back into cash before they can be reused:

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

Use this combination when shorting is allowed but leverage is not:

```python
config = BacktestConfig(
    initial_cash=100_000,
    allow_short_selling=True,
    allow_leverage=False,
)
```

## Margin Account

Use a margin account when you need borrowing capacity, leverage, and maintenance
constraints:

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

You can also pass account policy directly to `Engine`:

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

For advanced workflows, create the broker from a resolved config:

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

Account policy often interacts with trading costs, especially when leverage or high
turnover magnifies drag:

```python
from ml4t.backtest import BacktestConfig
from ml4t.backtest.config import CommissionType, SlippageType

config = BacktestConfig(
    initial_cash=100_000,
    commission_type=CommissionType.PERCENTAGE,
    commission_rate=0.001,     # 0.1% per trade
    slippage_type=SlippageType.PERCENTAGE,
    slippage_rate=0.0005,      # 0.05% slippage
)
```

## Presets

Presets are useful when you want framework-style account and execution behavior
without configuring each knob by hand:

```python
from ml4t.backtest import BacktestConfig

# Sensible defaults for general use
config = BacktestConfig.from_preset("default")

# Fast iteration (no costs, simplified execution)
config = BacktestConfig.from_preset("fast")

# Backtrader-compatible settings
config = BacktestConfig.from_preset("backtrader")

# VectorBT-compatible settings
config = BacktestConfig.from_preset("vectorbt")

# Zipline-compatible settings
config = BacktestConfig.from_preset("zipline")

# QuantConnect LEAN-compatible settings
config = BacktestConfig.from_preset("lean")

# Conservative production settings
config = BacktestConfig.from_preset("realistic")
```

Each preset sets all 40+ behavioral knobs to match the target framework's behavior.
Strict variants (`backtrader_strict`, `vectorbt_strict`, `zipline_strict`, `lean_strict`)
are also available for exact parity testing.

## Insufficient Funds

Orders that exceed available buying power are rejected by the gatekeeper. Use this
directly when you want to inspect why an order would fail:

```python
from ml4t.backtest.accounting import Gatekeeper

gatekeeper = Gatekeeper(account_state, policy)
is_valid, reason = gatekeeper.validate_order(order)
if not is_valid:
    print(f"Order rejected: {reason}")
```

## Migration from `account_type`

If you still have older code using an `account_type` string, migrate to explicit
policy flags:

```python
# Old API (deprecated)
broker = Broker(account_type="margin")

# New API
broker = Broker(allow_short_selling=True, allow_leverage=True)

# Or with config
config = BacktestConfig(allow_short_selling=True, allow_leverage=True)
broker = Broker.from_config(config)
```

## See It in Action

The [Machine Learning for Trading](https://github.com/stefan-jansen/machine-learning-for-trading)
materials use account policy most clearly in these workflows:

- **Ch16 case studies** — reusable `BacktestConfig` objects control cash use, leverage, and
  portfolio behavior across equities, futures, crypto, and options examples
- **Ch17** (`portfolio_construction`) — allocator comparisons depend on explicit account and
  turnover assumptions instead of hidden notebook defaults
- **Ch19** (`risk_management`) — leverage, shorting, and maintenance rules interact directly
  with position sizing and portfolio limits

Use the [Book Guide](../book-guide/index.md) when you want to jump from a notebook or
case-study path to the production account-policy workflow.

## Next Steps

- [Book Guide](../book-guide/index.md) -- chapter and case-study map for account and portfolio workflows
- [Configuration](configuration.md) -- full account, margin, and cash-management parameter reference
- [Risk Management](risk-management.md) -- position rules and portfolio limits that interact with buying power
- [Rebalancing](rebalancing.md) -- portfolio-weight execution under explicit account constraints
- [Results & Analysis](results.md) -- inspect turnover, fills, and portfolio-state effects of account policy
