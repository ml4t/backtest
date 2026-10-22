# Account Policies

The [accounts and constraints tutorial](../tutorials/accounts-and-constraints.md) compares
accepted orders, structured rejections, and resulting portfolio state under each setting.

Account policy determines what the broker is allowed to do with cash, leverage, and
short sale proceeds. Use this page when you need to decide whether your strategy
should behave like a long-only cash account, a short-enabled crypto-style account,
or a Reg T margin account.

The configuration is intentionally simple: instead of switching between account
"types", you set the policy flags directly and let the broker enforce the resulting
buying-power rules.

Run the linked accounts tutorial for complete order and portfolio-state comparisons under each policy.

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

## Futures cash and margin

A contract multiplier changes notional exposure and price-move P&L. A fixed or
percentage margin schedule constrains buying power; it does not select a
variation-cash settlement method. The default signed-notional ledger moves full
notional through cash at fills and includes signed position market value in equity.
A leveraged account's cash balance can therefore be negative.

For futures variation, select `ContractSpec(accounting_method="futures_variation")`
and a declared settlement policy. The [futures-accounting guide](futures-accounting.md)
runs both scheduled cash settlement and a mark-driven callback migration, with
separate exposure, liquidation, collateral and available-funds observations.

## Using Engine Directly

Pass the account policy to `Engine` through `BacktestConfig`. This complete example
submits a short sale on the first bar and checks its next-bar fill:

<!-- ml4t-doc-test: account-engine-direct -->
```python
from datetime import datetime

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, OrderSide, Strategy


class SellOnce(Strategy):
    def __init__(self):
        self.submitted = False

    def on_data(self, timestamp, data, context, broker):
        if not self.submitted:
            broker.submit_order("AAPL", 10, OrderSide.SELL)
            self.submitted = True


prices = pl.DataFrame({
    "timestamp": [datetime(2024, 1, 2), datetime(2024, 1, 3)],
    "asset": ["AAPL", "AAPL"],
    "open": [100.0, 100.0],
    "high": [100.0, 100.0],
    "low": [100.0, 100.0],
    "close": [100.0, 100.0],
    "volume": [10_000.0, 10_000.0],
})

config = BacktestConfig(
    initial_cash=100_000,
    allow_short_selling=True,
    allow_leverage=True,
    initial_margin=0.5,
)
engine = Engine(feed=DataFeed(prices_df=prices), strategy=SellOnce(), config=config)
result = engine.run()

assert [(fill.side.value, fill.quantity) for fill in result.fills] == [("sell", 10.0)]
print("short sale filled: 10 AAPL")
```

<!-- ml4t-doc-output: account-engine-direct -->
```text
short sale filled: 10 AAPL
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

## Read a marked account snapshot

`broker.get_account_snapshot()` returns an immutable observation of the current
native ledger: `timestamp`, `equity`, `cash`, `positions_value`, `gross_exposure`,
`net_exposure` and `leverage`. `timestamp` is the broker's query/callback clock,
including pre-open clocks; it does not assert that every position has a new
quote. Read holdings exports for individual mark timestamps and sources.

Position value is supplied directly by the marked ledger. It is not recovered
by subtracting cash from rounded NAV. The current signed-notional account method
has the same position value and net exposure. Under explicit futures variation,
position value includes unpaid P&L rather than principal notional. Calling this
getter does not add a bar or result observation.
Request a new snapshot after an event or fill when updated values are needed.

<!-- ml4t-doc-test: native-account-snapshot -->
```python
from datetime import UTC, datetime
import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine, Strategy
from ml4t.backtest.config import CommissionType, ExecutionMode

seen = []
class ReadAccount(Strategy):
    def on_data(self, timestamp, data, context, broker):
        broker.submit_order("A", 2)
        state = broker.get_account_snapshot()
        assert abs(state.cash - 99798.8) < 1e-8
        assert abs(state.equity - 99998.8) < 1e-8
        assert state.positions_value == 200.0
        assert state.gross_exposure == state.net_exposure == 200.0
        seen.append(state.timestamp)

clock = datetime(2025, 1, 6, 21, tzinfo=UTC)
result = Engine(
    DataFeed(prices_df=pl.DataFrame({"timestamp": [clock], "asset": ["A"], "close": [100.0]})),
    ReadAccount(),
    BacktestConfig(calendar=None, initial_cash=100000, execution_mode=ExecutionMode.SAME_BAR,
                    immediate_fill=True, commission_type=CommissionType.PER_ORDER,
                    commission_per_share=0.1, commission_per_trade=1.0),
).run()
assert seen == [clock]
assert result.to_fills_dataframe()["commission"].to_list() == [1.2]
print("The snapshot includes the actual fill fee and marked position value.")
```

Contract multipliers must be finite positive real numbers; Boolean and unknown
values are invalid. `Engine` captures copies of supplied `ContractSpec` objects,
and `broker.get_contract_spec(asset)` returns a detached copy. Editing preparation
objects or a returned descriptor does not change a bound run's execution units.
Prepare explicit dated instruments or conversion events for unit changes.

## Transaction Costs

In development source, fill-time affordability uses the actual price after market
impact and slippage, including the commission at that price. With leverage
disabled, a ten-share buy at $100 with 1% slippage requires $1,010. A $1,000
account rejects it when `partial_fills_allowed=False`, or fills nine integer
shares when partial fills are enabled. Quantity-dependent impact is recomputed
for the affordable quantity. Cash sizing retains the originally requested
quantity in the order's `requested_quantity` field.

Custom price models must support independent `deepcopy` for non-consuming quotes
and produce deterministic costs. Only actual executions advance their model
state. Explicit `skip_cash_validation=True` retains unconstrained behavior;
leveraged accounts continue to use their configured margin policy.

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

Each preset configures the measured behavior used by its comparison protocol. The native defaults,
explicit overrides, and adapter-emulated behavior are listed in the
[validation methodology](https://github.com/ml4t/backtest/blob/main/validation/METHODOLOGY.md).
Strict aliases are also available for
the retained comparison commands.

## Insufficient Funds

Use `Gatekeeper` directly when you need to validate an order before execution.
It requires an account state, a commission model, and the expected fill price:

<!-- ml4t-doc-test: account-gatekeeper -->
```python
from ml4t.backtest import Order, OrderSide
from ml4t.backtest.accounting import AccountState, Gatekeeper, UnifiedAccountPolicy
from ml4t.backtest.models import NoCommission

account = AccountState(initial_cash=100_000, policy=UnifiedAccountPolicy())
gatekeeper = Gatekeeper(account, NoCommission())
order = Order(asset="AAPL", side=OrderSide.BUY, quantity=1_500)
is_valid, reason = gatekeeper.validate_order(order, price=100.0)

assert not is_valid
print(reason)
```

<!-- ml4t-doc-output: account-gatekeeper -->
```text
Insufficient cash: need $150000.00, have $100000.00
```

## Migration from the beta `account_type` keyword

The beta-only `Broker(account_type=...)` keyword was removed before 0.1 and is not part of the
stable compatibility boundary. A single string selected several independent shorting and leverage
policies, so preserving it could silently change buying-power and liquidation behavior. Migrate to
the explicit policy flags:

```python
broker = Broker(allow_short_selling=True, allow_leverage=True)

# Or with config
config = BacktestConfig(allow_short_selling=True, allow_leverage=True)
broker = Broker.from_config(config)
```

The reviewed 0.1 compatibility snapshot is `tests/compatibility/snapshots/v0.1.json`. Run
`uv run python validation/generate_compatibility_snapshot.py` to check it. An intentional API or
schema change requires `--write` and review of the resulting snapshot diff.

## In the book

Chapter 17, Section 17.4, [Conformal position sizing](https://github.com/stefan-jansen/machine-learning-for-trading/blob/2d6e8f95eeccaee66906245606471f570b5807e5/17_portfolio_construction/07_conformal_position_sizing.ipynb) turns prediction uncertainty into position sizes. The account tutorial here shows how buying power and share precision affect those sizes.

## Next Steps

- [Book Guide](../book-guide/index.md) -- chapter and case-study map for account and portfolio workflows
- [Configuration](configuration.md) -- full account, margin, and cash-management parameter reference
- [Risk Management](risk-management.md) -- position rules and portfolio limits that interact with buying power
- [Rebalancing](rebalancing.md) -- portfolio-weight execution under explicit account constraints
- [Results & Analysis](results.md) -- inspect turnover, fills, and portfolio-state effects of account policy

Engine captures its input configuration when constructed, including nested mappings.
Prepare settings before constructing the Engine. Editing the original configuration
does not reconfigure a bound run, and the returned result keeps a detached copy of
the settings used for that run.

### Gross entry and commission-adjusted reporting basis

`broker.get_position(asset).entry_price` is the gross weighted entry price used
for native P&L. `commission_adjusted_cost_basis` is a separate reporting price:
it includes each transaction fee divided by the remaining signed quantity and
contract multiplier. A partial reduction retains the prior basis before adding
its fee; a reversal resets the basis to the fill price before adding the whole
reversal fee. This follows the position-reporting convention tested against
Zipline Reloaded 3.2.0. Cash and trade P&L continue to account for actual fees
once, using their existing entry/exit allocation.

For example, buying 10 shares at 100 with a fee of 1 reports a basis of 100.1.
Buying another 10 at 110 with a fee of 1 reports 105.1. Selling 5 at 120 with a
fee of 0.5 reports 105.13333333333334 on the remaining 15 shares, while their
gross entry price stays 105. The property is a live position view; it does not
add a field to saved holdings artifacts.
