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

1. **Real strategies**: Complete canonical fill streams, every shared equity timestamp, and
   terminal values for supported pairs drawn from ETF allocation, CME futures, and crypto
   perpetual-funding case studies. Frozen model-derived targets are shared by both engines.

2. **Synthetic scenarios**: Exact ordered trade and fill matching on the required matrix for
   VectorBT, Backtrader, and Zipline. Capability declarations identify reconstructed and
   unavailable result surfaces.

3. **Synthetic stress**: Exact target-intent, native-fill, fill-derived closed-trade, and
   terminal-state comparison on a reconstructable 250-asset, 5,040-session workload. Each
   framework row states the surface it exposes; the claim does not include unavailable
   order-lifecycle fields.

LEAN has separate native-behavior and Chapter 16 case-study evidence. The frozen engine produced
47,652 fills across three case studies; ML4T matched every canonical fill and each terminal value
at the declared $0.0001 quantum. See `validation/native/evidence/lean-18001.json` and
`validation/lean/case_study_evidence.json`.

<!-- parity-claims:start -->
<!-- Generated by validation/generate_parity_claims.py. Do not edit by hand. -->

### Real-strategy audit

17/17 required pairs pass; 8 pairs are declared unsupported. The audit uses five real-data strategy workloads with frozen historical market data and model-derived targets. A pass requires identical valuation timestamp coverage, complete fill streams with quantities equal at 1e-5 and prices equal at 1e-8, and account monetary values that round to the same cent. The FX workload uses the USD-quoted pairs in its frozen target stream so every required engine uses native USD valuation.

The parity protocol disables transaction costs and position rules on both sides. It tests target sizing, order sequencing, fills, cash and margin behavior, funding where applicable, and valuation. It does not claim to reproduce each selected case-study production result with its original costs and risk overlays.

| Real strategy | Pinned framework | Current result | Evidence |
|---|---|---|---|
| ETF allocation | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | fills equal at declared field precision; 1,995 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| ETF allocation | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | fills equal at declared field precision; 1,995 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| ETF allocation | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | fills equal at declared field precision; 1,995 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| ETF allocation | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | fills equal at declared field precision; 1,995 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| ETF allocation | [LEAN 18001](https://github.com/QuantConnect/Lean) | fills equal at declared field precision; 1,995 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| CME futures | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | fills equal at declared field precision; 1,595 valuations within $0.01 (max raw gap $0.00000010); terminal within $0.01 (raw gap $0.00000007) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| CME futures | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | fills equal at declared field precision; 1,595 valuations within $0.01 (max raw gap $0.00000015); terminal within $0.01 (raw gap $0.00000015) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| Crypto perpetual funding | [LEAN 18001](https://github.com/QuantConnect/Lean) | fills equal at declared field precision; 2,426 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| FX allocation (USD-quoted pairs) | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | fills equal at declared field precision; 2,108 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| FX allocation (USD-quoted pairs) | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | fills equal at declared field precision; 2,108 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| FX allocation (USD-quoted pairs) | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | fills equal at declared field precision; 2,108 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| FX allocation (USD-quoted pairs) | [LEAN 18001](https://github.com/QuantConnect/Lean) | fills equal at declared field precision; 2,108 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| US equity panel | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | fills equal at declared field precision; 4,146 valuations within $0.01 (max raw gap $0.00001950); terminal within $0.01 (raw gap $0.00001880) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| US equity panel | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | fills equal at declared field precision; 4,146 valuations within $0.01 (max raw gap $0.00001910); terminal within $0.01 (raw gap $0.00001870) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| US equity panel | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | fills equal at declared field precision; 4,146 valuations within $0.01 (max raw gap $0.00000170); terminal within $0.01 (raw gap $0.00000160) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| US equity panel | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | fills equal at declared field precision; 4,027 valuations within $0.01 (max raw gap $0.00000190); terminal within $0.01 (raw gap $0.00000030) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| US equity panel | [LEAN 18001](https://github.com/QuantConnect/Lean) | fills equal at declared field precision; 4,027 valuations within $0.01 (max raw gap $0.00000460); terminal within $0.01 (raw gap $0.00000420) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |

Engine-only timing samples for all 17 passing pairs are retained in [real-strategy performance evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_PERFORMANCE.json). These measurements support only the named strategy, framework version, input bundle, and machine.

### Synthetic diagnostic scenarios

The scenario matrix contains synthetic conformance tests. "Exact" means terminal values, ordered closed trades, and ordered fills match after 1e-8 quantization. Each record declares whether a surface is native, reconstructed, aggregate-only, input-only, or unavailable. These results test isolated conventions, not realistic strategy equivalence.

| Profile | Pinned framework | Required scenarios | Evidence |
|---|---|---:|---|
| `vectorbt_strict` | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | 17/17 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `vectorbt_oss_strict` | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 16/16 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `backtrader_strict` | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 17/17 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `zipline_strict` | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | 16/16 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |

The synthetic stress workload contains 250 assets and 5,040 daily sessions (1,260,000 bars). Every row has zero canonical gap for target intents, native fills, closed trades reconstructed from those fills, and terminal state reconstructed from the fill ledger and final marks. Fill records use 1e-8 precision; monetary totals use cent precision.

| Profile | Current framework | Target intents | Native fills | Fill-derived closed trades | Terminal value | Evidence |
|---|---|---:|---:|---:|---:|---|
| `vectorbt_strict` | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | 427,790 | 423,313 | 222,751 | 1,285,886.320000 | [scale evidence](https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json) |
| `vectorbt_oss_strict` | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 427,790 | 417,941 | 211,322 | 1,345,348.850000 | [scale evidence](https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json) |
| `backtrader_strict` | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 427,790 | 343,813 | 182,019 | -9,166,273.560000 | [scale evidence](https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json) |
| `zipline_strict` | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | 427,790 | 427,696 | 226,434 | 10,504,095.900000 | [scale evidence](https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json) |
| `lean` | [LEAN 18001](https://github.com/QuantConnect/Lean) | 427,790 | 361,297 | 191,297 | 184,538.130000 | [scale evidence](https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json) |
<!-- parity-claims:end -->

## Performance

The stable release does not publish cross-framework speed ratios. ML4T-only regression evidence is
defined in `validation/performance_baselines.json`. The separate retained cross-framework artifact
uses one warm-up and ten process-isolated measurements per runner, with raw samples, median and 95%
intervals, process-tree peak RSS, exact output checksums, and an idiomatic view that makes no
equivalence claim.

`validation/REAL_STRATEGY_PERFORMANCE.json` separately retains engine-only timings for the six
real-strategy pairs that passed correctness. It excludes data loading, inference, target
construction, adapter preparation, extraction, and reporting. The results apply only to the named
versions, bundles, and machine.

## Listing Profiles

```python
from ml4t.backtest.profiles import list_profiles

print(list_profiles())
# ['backtrader', 'default', 'lean', 'realistic', 'vectorbt', 'zipline']
```

## Next Steps

- [Configuration](configuration.md) -- understand each parameter
- [Execution Semantics](execution-semantics.md) -- why these settings produce different results
