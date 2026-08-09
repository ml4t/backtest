# Profiles

Profiles are pre-configured `BacktestConfig` settings for framework-specific execution semantics.
The retained evidence in this page states which pinned framework scenarios currently match
exactly and which remain release-blocking.

## Available Profiles

### Core Profiles

| Profile | Description |
|---------|-------------|
| `default` | Sensible defaults for general use with integer-share execution |
| `fast` | Zero-cost, integer-share execution -- fastest possible execution |
| `backtrader` | Match Backtrader's default behavior |
| `vectorbt` | Match VectorBT's default behavior (including fractional shares) |
| `zipline` | Match Zipline Reloaded's default behavior |
| `lean` | Match QuantConnect LEAN's default behavior |
| `realistic` | Conservative settings for production |

### Broker Presets

| Profile | Description |
|---------|-------------|
| `ibkr_us_stocks_fixed` | Interactive Brokers US stocks fixed commission schedule |

The broker preset also supports a modular alias:

| Alias | Resolves To |
|-------|-------------|
| `ibkr:us:stocks:fixed` | `ibkr_us_stocks_fixed` |

### Strict Profiles

Strict variants tune additional knobs (cash validation, settlement, short policies) for maximum parity on large-scale comparisons:

| Profile | Base | Additional Tuning |
|---------|------|-------------------|
| `backtrader_strict` | backtrader | Submission precheck, simple cash check |
| `vectorbt_strict` | vectorbt | Lock notional for shorts, FIFO ordering |
| `zipline_strict` | zipline | Skip cash validation, allow shorts |

### Aliases

| Alias | Resolves To |
|-------|-------------|
| `vectorbt_pro` | vectorbt |
| `vectorbt_oss` | vectorbt |
| `quantconnect` | lean |
| `ibkr:us:stocks:fixed` | ibkr_us_stocks_fixed |

## Usage

```python
from ml4t.backtest import BacktestConfig

# Load a profile
config = BacktestConfig.from_preset("backtrader")

# Or build from structured broker assumptions
config = BacktestConfig.from_assumptions(
    broker="ibkr",
    region="us",
    asset_class="stocks",
    plan="fixed",
)

# Use with run_backtest
from ml4t.backtest import run_backtest
result = run_backtest(prices, strategy, config="zipline")

# Override specific settings
config = BacktestConfig.from_preset("backtrader")
config.commission_rate = 0.002
config.initial_cash = 500_000
```

Profiles define behavioral defaults. Quote-aware feeds layer on top of them: you can start from a preset, then override `execution_price`, `mark_price`, and the feed's `price_col` / quote columns without changing the rest of the profile.

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
| Share type | integer | integer | fractional | integer | integer | integer |

### Costs

| Setting | default | backtrader | vectorbt | zipline | lean | realistic |
|---------|---------|-----------|----------|---------|------|-----------|
| Commission | none | 0.1% | none | $0.005/share | $0.005/share | 0.2% |
| Slippage | none | 0.1% | none | 10% volume | 0.1% | 0.2% |
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

Framework profiles can be validated at two levels:

1. **Scenario-level** (16 scenarios per framework): Exact trade-by-trade matching on synthetic data covering entries, exits, stops, trailing stops, brackets, and multi-asset strategies.

2. **Large-scale**: Trade-by-trade comparison on a retained real-data workload. A claim is omitted
   when no passing artifact is retained for that framework.

<!-- parity-claims:start -->
<!-- Generated by validation/generate_parity_claims.py. Do not edit by hand. -->

Scenario claims use the retained release-candidate matrix. "Exact" appears only when every required scenario has zero canonical gap.

| Profile | Pinned framework | Required scenarios | Evidence |
|---|---|---:|---|
| `vectorbt_strict` | [VectorBT Pro 2025.12.31](https://github.com/polakowo/vectorbt.pro) | 16/16 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `vectorbt` | [VectorBT OSS 0.28.2](https://pypi.org/project/vectorbt/0.28.2/) | 15/15 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `backtrader_strict` | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 16/16 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `zipline_strict` | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | 15/15 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |

Large-scale claims are published only when a retained workload has zero canonical gap.

| Profile | Pinned framework | Compared | Trade gap | Terminal value | Evidence |
|---|---|---:|---:|---:|---|
| `vectorbt_strict` | VectorBT Pro 2025.12.31 (`1305a1e19743`) | 225,844 trades | 0 | 685179.007330 | [large-scale evidence](https://github.com/ml4t/backtest/blob/main/validation/vectorbt_pro/large_scale_parity.json) |

No large-scale claim is published for Backtrader, Zipline, VectorBT OSS, or LEAN without a passing retained artifact.
<!-- parity-claims:end -->

## Performance

Performance comparisons are valid only when each framework runs the same workload with equivalent
execution semantics. The project does not publish cross-framework speed ratios without a retained
benchmark artifact meeting those conditions.

## Listing Profiles

```python
from ml4t.backtest.profiles import list_profiles

print(list_profiles())
# ['backtrader', 'default', 'lean', 'realistic', 'vectorbt', 'zipline']
```

## Next Steps

- [Configuration](configuration.md) -- understand each parameter
- [Execution Semantics](execution-semantics.md) -- why these settings produce different results
