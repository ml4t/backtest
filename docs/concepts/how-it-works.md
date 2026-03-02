# How It Works

This page explains the architecture, key abstractions, and execution flow of the backtesting engine.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                      Engine                          │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌─────────────────┐ │
│  │ DataFeed │──>│  Broker   │<──│    Strategy      │ │
│  └──────────┘   └────┬─────┘   └─────────────────┘ │
│                      │                               │
│         ┌────────────┼──────────────┐               │
│         │            │              │               │
│  ┌──────┴───┐  ┌─────┴────┐  ┌─────┴──────┐       │
│  │OrderBook │  │Gatekeeper│  │FillExecutor│       │
│  └──────────┘  └──────────┘  └────────────┘       │
│         │            │              │               │
│  ┌──────┴───┐  ┌─────┴────┐  ┌─────┴──────┐       │
│  │ RiskEng  │  │ Account  │  │  ImpactMdl │       │
│  └──────────┘  └──────────┘  └────────────┘       │
└─────────────────────────────────────────────────────┘
```

**Engine** orchestrates the main loop -- iterating bars, calling the strategy, and recording equity.

**DataFeed** partitions a Polars DataFrame by timestamp and iterates bar-by-bar across all assets. It pre-extracts OHLCV data for O(1) per-bar access.

**Broker** is the strategy's interface to the market. It accepts orders, manages positions, evaluates risk rules, and delegates fills to the execution pipeline.

**Strategy** is the user's code. It receives `(timestamp, data, context, broker)` on each bar and submits orders through the broker.

## Key Abstractions

### BacktestConfig

A single dataclass with 40+ fields controlling every behavioral choice -- fill ordering, stop modes, commission models, cash policies, settlement delays, and more. Instead of subclassing or monkey-patching, you change behavior by setting config values.

```python
from ml4t.backtest import BacktestConfig

# Default settings
config = BacktestConfig()

# Match another framework
config = BacktestConfig.from_preset("backtrader")

# Override specific knobs
config = BacktestConfig.from_preset("backtrader")
config.commission_rate = 0.002
```

### Profiles

Pre-configured settings that replicate the exact behavior of VectorBT, Backtrader, Zipline, and LEAN. A "strict" variant of each profile tunes additional knobs (like cash validation and settlement) for maximum parity.

See [Profiles](../user-guide/profiles.md) for the full comparison.

### Position Rules

Declarative exit rules (stop-loss, take-profit, trailing stop) that the broker evaluates automatically on every bar. Rules compose via `RuleChain` (first trigger wins), `AllOf` (all must trigger), or `AnyOf`.

```python
from ml4t.backtest import StopLoss, TrailingStop, RuleChain

rules = RuleChain([
    StopLoss(pct=0.05),
    TrailingStop(pct=0.03),
])
broker.set_position_rules(rules)
```

Rules are set in `on_start()` and apply globally, or per-asset via `broker.set_position_rules(rules, asset="AAPL")`.

## Execution Flow

The engine processes each bar in this order:

```
for each bar:
    1. Update broker with current OHLCV prices
    2. Process pending exits from previous bar (NEXT_BAR mode)
    3. Evaluate position rules (stops, trails) → generate exit orders
    4. Process pending orders (fills at open or close)
    5. Call strategy.on_data()
    6. Process new orders (SAME_BAR mode only)
    7. Update water marks for trailing stops
    8. Record equity
```

### NEXT_BAR vs SAME_BAR

In **NEXT_BAR** mode (default, realistic), orders submitted in `on_data()` are filled at the *next* bar's open price. The strategy decides based on today's close, and the order executes at tomorrow's open. This prevents look-ahead bias.

In **SAME_BAR** mode, orders fill at the current bar's close price. This is useful for vectorized comparisons with VectorBT but carries look-ahead risk for production strategies.

### Exit-First Processing

Within a single bar, all exits are processed before entries. This frees capital from closed positions before new positions need it. This matches real broker behavior where sell proceeds are available for buying in the same session.

```
Exits first:    [SL exit AAPL] → [TP exit MSFT] → cash freed
Then entries:   [Buy GOOG] → [Buy TSLA] → cash used
```

The ordering is controlled by `fill_ordering` (EXIT_FIRST, FIFO, SEQUENTIAL) and `entry_order_priority` (SUBMISSION, NOTIONAL_DESC, NOTIONAL_ASC).

### Order Validation (Gatekeeper)

Before any order fills, the Gatekeeper checks:

- Sufficient cash or buying power for the order
- Account policy compliance (short selling allowed? leverage allowed?)
- Position limits and exposure constraints

Rejected orders are recorded with a reason and accessible via `broker.get_rejected_orders()`.

## Data Flow

```
Polars DataFrame
    │
    ▼
DataFeed (partition by timestamp, pre-extract columns)
    │
    ▼
Engine loop (iterate bars)
    │
    ▼
Strategy.on_data(timestamp, data, context, broker)
    │
    ▼
broker.submit_order(asset, quantity)
    │
    ▼
OrderBook → Gatekeeper → FillExecutor → Position updates
    │
    ▼
BacktestResult (trades, equity, metrics)
```

### What `data` Contains

On each bar, `data` is a dict mapping asset names to bar dicts:

```python
data = {
    "AAPL": {
        "open": 150.0,
        "high": 152.0,
        "low": 149.5,
        "close": 151.0,
        "volume": 1000000,
        "signals": {"prediction": 0.85, "momentum": 0.12},
    },
    "MSFT": {
        "open": 280.0,
        ...
    },
}
```

Signals are nested under a `"signals"` key and come from the optional signals DataFrame passed to DataFeed.

## Result Structure

`Engine.run()` returns a `BacktestResult` with:

- **`metrics`** -- dict of performance metrics (sharpe, max drawdown, win rate, etc.)
- **`trades`** -- list of Trade objects (entry/exit times, prices, PnL)
- **`fills`** -- list of Fill objects (every order execution)
- **`equity_curve`** -- list of (timestamp, portfolio_value) tuples
- **`to_trades_dataframe()`** -- Polars DataFrame of all trades
- **`to_equity_dataframe()`** -- Polars DataFrame of equity curve
- **`to_parquet(path)`** -- export for ml4t-diagnostic integration

See [Results & Analysis](../user-guide/results.md) for details.

## Next Steps

- [Execution Semantics](../user-guide/execution-semantics.md) -- deep dive into fill ordering, stop modes, and timing
- [Configuration](../user-guide/configuration.md) -- all 40+ knobs explained
- [Quickstart](../getting-started/quickstart.md) -- write your first strategy
