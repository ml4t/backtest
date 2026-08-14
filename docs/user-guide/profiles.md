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
| `zipline` | Match the documented Zipline comparison protocol |
| `lean` | Match the frozen LEAN daily US-equity comparison protocol |
| `realistic` | Conservative settings for production |

### Broker Presets

| Profile | Description |
|---------|-------------|
| `ibkr_us_stocks_fixed` | Interactive Brokers US stocks fixed commission schedule |

The broker preset also supports a modular alias:

| Alias | Resolves To |
|-------|-------------|
| `ibkr:us:stocks:fixed` | `ibkr_us_stocks_fixed` |

The `lean` profile is scoped to daily US equities submitted from `OnData` with
`DefaultBrokerageModel`, a margin account, and 2x security leverage. LEAN delegates fees,
slippage, margin, and fills to selected brokerage and security models, so this profile is not a
claim about every LEAN asset class, resolution, or brokerage model.

### Strict Profiles

Strict variants are the names used by retained comparison commands:

| Profile | Base | Additional Tuning |
|---------|------|-------------------|
| `backtrader_strict` | backtrader | Submission precheck, simple cash check |
| `vectorbt_strict` | vectorbt | Same settings as `vectorbt` |
| `zipline_strict` | zipline | Same settings as `zipline` |

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

These tables report configured values. A profile value is not automatically an equivalence claim.
The [behavior coverage map](https://github.com/ml4t/backtest/blob/main/validation/behavior_coverage.toml)
identifies which field groups have both a native oracle and a cross-engine comparison. Target
sizing, insufficient-cash boundaries, competing same-session orders, partial fills, missing bars,
and late assets remain excluded from scenario-level equivalence where the map marks them
unpublished.

### Execution

| Setting | default | backtrader | vectorbt | zipline | lean | realistic |
|---------|---------|-----------|----------|---------|------|-----------|
| Execution mode | next_bar | next_bar | same_bar | next_bar | next_bar | next_bar |
| Execution price | open | open | close | open | open | open |

### Stops

| Setting | default | backtrader | vectorbt | zipline | lean | realistic |
|---------|---------|-----------|----------|---------|------|-----------|
| Fill mode | stop_price | stop_price | stop_price | next_bar_open | stop_price* | next_bar_open |
| Level basis | fill_price | signal_price | fill_price | fill_price | fill_price* | fill_price |
| Trail HWM | close | close | bar_extreme | bar_extreme | close* | close |
| Trail timing | lagged | lagged | intrabar | intrabar | lagged* | lagged |

`*` The LEAN stop settings are ML4T profile fallbacks. The current native LEAN oracle does not
claim stop-order parity.

### Account

| Setting | default | backtrader | vectorbt | zipline | lean | realistic |
|---------|---------|-----------|----------|---------|------|-----------|
| Short selling | No | Yes | Yes | Yes | Yes | No |
| Leverage | No | No | No | Cash validation disabled | Yes, 2x | No |
| Share type | integer | integer | fractional | integer | integer | integer |

### Costs

| Setting | default | backtrader | vectorbt | zipline | lean | realistic |
|---------|---------|-----------|----------|---------|------|-----------|
| Commission | none | none | none | none | $0.005/share | 0.2% |
| Slippage | none | none | none | none | none | 0.2% |
| Stop slippage | 0 | 0 | 0 | 0 | 0 | 0.1% |
| Cash buffer | 0% | 0% | 0% | 0% | 0% | 2% |
| Target-weight multiplier | 100% | 100% | 100% | 100% | 99.75% | 100% |

### Order Processing

| Setting | default | backtrader | vectorbt | zipline | lean | realistic |
|---------|---------|-----------|----------|---------|------|-----------|
| Fill ordering | exit_first | fifo | exit_first | fifo | sequential | exit_first |
| Reject insuff. | yes | yes | no | no | yes | yes |
| Partial fills | no | no | yes | no | no | no |
| Rebalance mode | incremental | snapshot | hybrid | snapshot | snapshot | incremental |

## Parity Validation

Framework profiles are validated on the workloads each retained artifact declares:

1. **Scenario-level**: Exact ordered trade and fill matching on the required synthetic matrix for
   VectorBT, Backtrader, and Zipline. Capability declarations identify reconstructed and
   unavailable result surfaces.

2. **Large-scale**: Exact target-intent, native-fill, fill-derived closed-trade, and terminal-state
   comparison on a reconstructable 250-asset, 5,040-session workload. Each framework row states
   the surface it exposes; the claim does not include unavailable order-lifecycle fields.

LEAN has separate native-behavior and Chapter 16 case-study evidence. The frozen engine produced
47,652 fills across three case studies; ML4T matched every canonical fill and each terminal value
at the declared $0.0001 quantum. See `validation/native/evidence/lean-18001.json` and
`validation/lean/case_study_evidence.json`.

<!-- parity-claims:start -->
<!-- Generated by validation/generate_parity_claims.py. Do not edit by hand. -->

Scenario claims use the retained release-candidate matrix. "Exact" means terminal values, ordered closed trades, and ordered fills match after 1e-8 quantization for every retained surface the framework exposes.

| Profile | Pinned framework | Required scenarios | Evidence |
|---|---|---:|---|
| `vectorbt_strict` | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | 17/17 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `vectorbt` | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 16/16 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `backtrader_strict` | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 17/17 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `zipline_strict` | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | 16/16 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |

Large-scale claims are published only when a retained workload has zero canonical gap.

| Profile | Pinned framework | Compared | Trade gap | Terminal value | Evidence |
|---|---|---:|---:|---:|---|
| `vectorbt_strict` | VectorBT Pro 2025.12.31 (`1305a1e19743`) | 225,844 trades | 0 | 685179.007330 | [large-scale evidence](https://github.com/ml4t/backtest/blob/main/validation/vectorbt_pro/large_scale_parity.json) |

No large-scale claim is published for Backtrader, Zipline, VectorBT OSS, or LEAN without a passing retained artifact.
<!-- parity-claims:end -->

## Performance

The stable release does not publish cross-framework speed ratios. ML4T-only regression evidence is
defined in `validation/performance_baselines.json`. The separate retained cross-framework artifact
uses one warm-up and ten process-isolated measurements per runner, with raw samples, median and 95%
intervals, process-tree peak RSS, exact output checksums, and an idiomatic view that makes no
equivalence claim.

## Listing Profiles

```python
from ml4t.backtest.profiles import list_profiles

print(list_profiles())
# ['backtrader', 'default', 'lean', 'realistic', 'vectorbt', 'zipline']
```

## Next Steps

- [Configuration](configuration.md) -- understand each parameter
- [Execution Semantics](execution-semantics.md) -- why these settings produce different results
