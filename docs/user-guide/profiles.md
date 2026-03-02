# Profiles

Profiles are pre-configured `BacktestConfig` settings that replicate the exact behavior of other backtesting frameworks. Instead of reverse-engineering each framework's quirks, load a profile and get identical results.

## Available Profiles

### Core Profiles

| Profile | Description |
|---------|-------------|
| `default` | Sensible defaults for general use |
| `fast` | Zero-cost, no validation -- fastest possible execution |
| `backtrader` | Match Backtrader's default behavior |
| `vectorbt` | Match VectorBT's default behavior |
| `zipline` | Match Zipline Reloaded's default behavior |
| `lean` | Match QuantConnect LEAN's default behavior |
| `realistic` | Conservative settings for production |

### Strict Profiles

Strict variants tune additional knobs (cash validation, settlement, short policies) for maximum parity on large-scale comparisons:

| Profile | Base | Additional Tuning |
|---------|------|-------------------|
| `backtrader_strict` | backtrader | Submission precheck, simple cash check |
| `vectorbt_strict` | vectorbt | Lock notional for shorts, FIFO ordering |
| `zipline_strict` | zipline | Skip cash validation, allow shorts |
| `lean_strict` | lean | Buying power reservation, T+2 settlement |

### Aliases

| Alias | Resolves To |
|-------|-------------|
| `vectorbt_pro` | vectorbt |
| `vectorbt_oss` | vectorbt |
| `quantconnect` | lean |

## Usage

```python
from ml4t.backtest import BacktestConfig

# Load a profile
config = BacktestConfig.from_preset("backtrader")

# Use with run_backtest
from ml4t.backtest import run_backtest
result = run_backtest(prices, strategy, config="zipline")

# Override specific settings
config = BacktestConfig.from_preset("backtrader")
config.commission_rate = 0.002
config.initial_cash = 500_000
```

## Profile Comparison

### Execution

| Setting | default | backtrader | vectorbt | zipline | lean | realistic |
|---------|---------|-----------|----------|---------|------|-----------|
| Execution mode | next_bar | next_bar | same_bar | next_bar | same_bar | next_bar |
| Execution price | open | open | close | open | close | open |

### Stops

| Setting | default | backtrader | vectorbt | zipline | lean | realistic |
|---------|---------|-----------|----------|---------|------|-----------|
| Fill mode | stop_price | stop_price | stop_price | stop_price | stop_price | next_bar_open |
| Level basis | fill_price | signal_price | fill_price | fill_price | fill_price | fill_price |
| Trail HWM | close | close | bar_extreme | close | close | close |
| Trail timing | lagged | lagged | intrabar | lagged | lagged | lagged |

### Account

| Setting | default | backtrader | vectorbt | zipline | lean | realistic |
|---------|---------|-----------|----------|---------|------|-----------|
| Short selling | No | Yes (margin) | Yes | No | Yes | No |
| Leverage | No | Yes (50%) | No | No | No | No |
| Share type | fractional | integer | fractional | integer | integer | integer |

### Costs

| Setting | default | backtrader | vectorbt | zipline | lean | realistic |
|---------|---------|-----------|----------|---------|------|-----------|
| Commission | 0.1% | 0.1% | none | $0.005/share | $0.005/share | 0.2% |
| Slippage | 0.1% | 0.1% | none | 10% volume | 0.1% | 0.2% |
| Stop slippage | 0 | 0 | 0 | 0 | 0 | 0.1% |
| Cash buffer | 0% | 0% | 0% | 0% | 0% | 2% |

### Order Processing

| Setting | default | backtrader | vectorbt | zipline | lean | realistic |
|---------|---------|-----------|----------|---------|------|-----------|
| Fill ordering | exit_first | fifo | exit_first | exit_first | exit_first | exit_first |
| Reject insuff. | yes | yes | no | yes | yes | yes |
| Partial fills | no | no | yes | yes | no | no |
| Rebalance mode | incremental | snapshot | hybrid | snapshot | snapshot | incremental |

## Parity Validation

Each profile has been validated at two levels:

1. **Scenario-level** (16 scenarios per framework): Exact trade-by-trade matching on synthetic data covering entries, exits, stops, trailing stops, brackets, and multi-asset strategies.

2. **Large-scale** (250 assets x 20 years): Trade-by-trade comparison on real market data using strict profiles.

| Profile | Trades Compared | Trade Gap | Value Gap |
|---------|----------------|-----------|-----------|
| `zipline_strict` | 225,583 | 0 (0.00%) | $19 (0.0001%) |
| `backtrader_strict` | 216,980 | 1 (0.0005%) | $503 (0.004%) |
| `vectorbt_strict` | 210,352 | 91 (0.04%) | $0 (0.00%) |
| `lean_strict` | 226,172 | 589 (0.26%) | $7.2K (0.66%) |

## Performance

Benchmarked on the same workload (250 assets x 20 years, real data) using each framework:

| Framework | Speed Comparison | Notes |
|-----------|-----------------|-------|
| Backtrader | **19x faster** | Same behavioral settings |
| Zipline | **8x faster** | Same behavioral settings |
| LEAN | **5x faster** | Same behavioral settings |
| VectorBT OSS | 0.04x | VectorBT is vectorized (different paradigm) |

Performance comparisons are only valid when behavioral semantics are identical (same profile).

## Listing Profiles

```python
from ml4t.backtest.profiles import list_profiles

print(list_profiles())
# ['backtrader', 'default', 'lean', 'realistic', 'vectorbt', 'zipline']
```

## Next Steps

- [Configuration](configuration.md) -- understand each parameter
- [Execution Semantics](execution-semantics.md) -- why these settings produce different results
