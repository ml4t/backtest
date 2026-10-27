# ML4T Backtest

Event-driven backtesting engine with configurable execution semantics, validated against four independent frameworks.

Use `ml4t-backtest` when notebook research is no longer enough and you need explicit,
reproducible answers to practical execution questions: when orders fill, how stops trigger,
how cash is reserved, and how results change when you match another framework's behavior.

<div class="grid cards" markdown>

-   :material-play-circle:{ .lg .middle } __Run Your First Backtest__
    ---
    Run a complete strategy on bundled synthetic bars and inspect its fills.
    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-tune:{ .lg .middle } __User Guide__
    ---
    Find the workflow for data, strategies, orders, account rules, costs,
    risk, and result analysis.
    [:octicons-arrow-right-24: User Guide](user-guide/index.md)

-   :material-check-all:{ .lg .middle } __Validated Against 4 Frameworks__
    ---
    Compare retained scenario evidence for VectorBT, Backtrader, and Zipline,
    plus native and Chapter 16 evidence for LEAN.
    [:octicons-arrow-right-24: Profiles](user-guide/profiles.md)

-   :material-book-open-variant:{ .lg .middle } __Chapters 16-19__
    ---
    The book develops the ideas in notebooks. This library turns them into
    reusable execution and reporting workflows.
    [:octicons-arrow-right-24: Book Guide](book-guide/index.md)

</div>

## Overview

`ml4t-backtest` is the simulation layer in the ML4T stack. It sits between research and
deployment:

- `ml4t-data` prepares canonical market datasets
- `ml4t-engineer` produces labels and features
- `ml4t-diagnostic` validates signals, models, and portfolio behavior
- `ml4t-backtest` simulates execution with explicit, configurable semantics
- `ml4t-live` reuses the same strategy surface for paper and live rollout

## Quick Example

This run uses a bundled synthetic AAPL panel, so it needs no API key or data
download. The [first backtest](getting-started/quickstart.md) walks through the
orders and result records.

<!-- ml4t-doc-test: home-example -->
```python
from importlib.metadata import version

import polars as pl
from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices

prices = load_example_prices("equity").filter(pl.col("asset") == "AAPL")
result = Engine(
    DataFeed(prices_df=prices),
    ExampleRoundTrip("AAPL", 100),
    BacktestConfig(initial_cash=100_000),
).run()
print("ml4t-backtest " + version("ml4t-backtest"))
print(f"fills={len(result.fills)} final=${result.metrics['final_value']:.2f}")
```

<!-- ml4t-doc-output: home-example -->
```text
ml4t-backtest {package_version}
fills=2 final=$100300.00
```

Each `Engine` instance is single-use. Create a new instance for every independent run.

Moving a Zipline strategy? Follow the [task-level migration map](user-guide/migrate-from-zipline.md)
and run its checked target-weight example before comparing framework results.

The convenience function accepts the same price panel and strategy directly:

<!-- ml4t-doc-test: home-convenience -->
```python
import polars as pl
from ml4t.backtest import run_backtest
from ml4t.backtest.example_data import ExampleRoundTrip, load_example_prices

prices = load_example_prices("equity").filter(pl.col("asset") == "AAPL")
result = run_backtest(prices, ExampleRoundTrip("AAPL", 100), config="backtrader")
print(f"fills={len(result.fills)} final=${result.metrics['final_value']:.2f}")
```

<!-- ml4t-doc-output: home-convenience -->
```text
fills=2 final=$100300.00
```

## Why ML4T Backtest?

**Configurable execution semantics.** Framework differences such as fill ordering, stop modes,
cash policies, and settlement are named configuration parameters. Retained evidence below states
which pinned framework scenarios currently match exactly.

**Quote-aware when you need it.** The feed can cache bid, ask, midpoint, and quote sizes additively. Market execution and position marking can use `price`, `bid`, `ask`, `quote_mid`, or `quote_side`.

**Retained validation evidence.** The primary audit uses five real-data strategy workloads with
frozen inputs. Separate synthetic scenario and 250-asset stress suites isolate conventions and
exercise high event counts.

| Feature | Description |
|---------|-------------|
| Event-driven | Explicit decision and fill timing; same-bar settings require a causal-data check |
| Configurable behavior | Set fill timing, cash, costs, and order processing explicitly |
| Quote-aware execution | Side-aware fills and separate mark pricing |
| 10 framework profiles | Configure VectorBT, Backtrader, Zipline, and LEAN semantics |
| Risk management | Stop-loss, take-profit, trailing stops, portfolio limits |
| Multi-asset | Rebalancing, weight targets, exit-first ordering |
| Rich persistence | Export trades, fills, equity, portfolio state, and daily P&L to Parquet |

## Parity Validation

Real-strategy comparisons run frozen market data and model-derived targets through each supported
engine pair. The synthetic scenario and stress suites provide narrower conformance evidence.

<!-- parity-claims:start -->
<!-- Generated by validation/generate_parity_claims.py. Do not edit by hand. -->

### Real-strategy audit

17/17 required pairs pass; 8 pairs are declared unsupported. The audit uses five real-data strategy workloads with frozen historical market data and model-derived targets. A pass requires identical valuation timestamp coverage, complete fill streams with quantities equal at 1e-5 and prices equal at 1e-8, and account monetary values that round to the same cent. The FX workload uses the USD-quoted pairs in its frozen target stream so every required engine uses native USD valuation.

The parity protocol disables transaction costs and position rules on both sides. It tests target sizing, order sequencing, fills, cash and margin behavior, funding where applicable, and valuation. It does not claim to reproduce each selected case-study production result with its original costs and risk overlays.

| Real strategy | Pinned framework | Current result | Evidence |
|---|---|---|---|
| ETF allocation | [VectorBT Pro 2026.6.27](https://vectorbt.pro/) | fills equal at declared field precision; 1,995 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| ETF allocation | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | fills equal at declared field precision; 1,995 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| ETF allocation | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | fills equal at declared field precision; 1,995 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| ETF allocation | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | fills equal at declared field precision; 1,995 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| ETF allocation | [LEAN 18001](https://github.com/QuantConnect/Lean) | fills equal at declared field precision; 1,995 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| CME futures | [VectorBT Pro 2026.6.27](https://vectorbt.pro/) | fills equal at declared field precision; 1,595 valuations within $0.01 (max raw gap $0.00000010); terminal within $0.01 (raw gap $0.00000007) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| CME futures | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | fills equal at declared field precision; 1,595 valuations within $0.01 (max raw gap $0.00000015); terminal within $0.01 (raw gap $0.00000015) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| Crypto perpetual funding | [LEAN 18001](https://github.com/QuantConnect/Lean) | fills equal at declared field precision; 2,426 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| FX allocation (USD-quoted pairs) | [VectorBT Pro 2026.6.27](https://vectorbt.pro/) | fills equal at declared field precision; 2,108 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| FX allocation (USD-quoted pairs) | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | fills equal at declared field precision; 2,108 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| FX allocation (USD-quoted pairs) | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | fills equal at declared field precision; 2,108 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| FX allocation (USD-quoted pairs) | [LEAN 18001](https://github.com/QuantConnect/Lean) | fills equal at declared field precision; 2,108 valuations and terminal exact at 1e-8 | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| US equity panel | [VectorBT Pro 2026.6.27](https://vectorbt.pro/) | fills equal at declared field precision; 4,146 valuations within $0.01 (max raw gap $0.00001950); terminal within $0.01 (raw gap $0.00001880) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| US equity panel | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | fills equal at declared field precision; 4,146 valuations within $0.01 (max raw gap $0.00001910); terminal within $0.01 (raw gap $0.00001870) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| US equity panel | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | fills equal at declared field precision; 4,146 valuations within $0.01 (max raw gap $0.00000170); terminal within $0.01 (raw gap $0.00000160) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| US equity panel | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | fills equal at declared field precision; 4,027 valuations within $0.01 (max raw gap $0.00000190); terminal within $0.01 (raw gap $0.00000030) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |
| US equity panel | [LEAN 18001](https://github.com/QuantConnect/Lean) | fills equal at declared field precision; 4,027 valuations within $0.01 (max raw gap $0.00000460); terminal within $0.01 (raw gap $0.00000420) | [real-strategy evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_RESULTS.json) |

### Real-strategy engine performance

The table reports engine-call wall time for all 17 correctness-passing pairs. The ratio is framework median / ML4T median; values above 1 mean ML4T completed the engine call faster.

| Real strategy | Pinned framework | Framework median (95% CI), s | ML4T median (95% CI), s | Framework / ML4T median |
|---|---|---:|---:|---:|
| ETF allocation | [VectorBT Pro 2026.6.27](https://vectorbt.pro/) | 0.317 (0.297-0.330) | 0.456 (0.452-0.460) | 0.695x |
| ETF allocation | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 0.173 (0.170-0.175) | 0.451 (0.448-0.455) | 0.383x |
| ETF allocation | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 9.408 (9.378-9.537) | 0.475 (0.473-0.478) | 19.802x |
| ETF allocation | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | 3.846 (3.841-3.867) | 0.741 (0.737-0.747) | 5.191x |
| ETF allocation | [LEAN 18001](https://github.com/QuantConnect/Lean) | 2.571 (2.515-2.635) | 0.874 (0.868-0.879) | 2.942x |
| CME futures | [VectorBT Pro 2026.6.27](https://vectorbt.pro/) | 0.286 (0.283-0.291) | 0.527 (0.524-0.537) | 0.542x |
| CME futures | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 2.475 (2.465-2.535) | 0.542 (0.537-0.548) | 4.570x |
| Crypto perpetual funding | [LEAN 18001](https://github.com/QuantConnect/Lean) | 2.793 (2.751-2.871) | 0.744 (0.736-0.753) | 3.753x |
| FX allocation (USD-quoted pairs) | [VectorBT Pro 2026.6.27](https://vectorbt.pro/) | 0.291 (0.283-0.294) | 0.159 (0.159-0.161) | 1.828x |
| FX allocation (USD-quoted pairs) | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 0.146 (0.138-0.152) | 0.159 (0.158-0.162) | 0.919x |
| FX allocation (USD-quoted pairs) | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 0.439 (0.434-0.447) | 0.163 (0.160-0.164) | 2.701x |
| FX allocation (USD-quoted pairs) | [LEAN 18001](https://github.com/QuantConnect/Lean) | 0.961 (0.935-0.979) | 0.170 (0.168-0.172) | 5.656x |
| US equity panel | [VectorBT Pro 2026.6.27](https://vectorbt.pro/) | 0.865 (0.851-1.004) | 24.782 (24.511-24.923) | 0.035x |
| US equity panel | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 17.246 (17.166-17.345) | 25.313 (25.119-25.615) | 0.681x |
| US equity panel | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 505.641 (499.073-515.494) | 29.242 (28.887-29.792) | 17.292x |
| US equity panel | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | 110.874 (105.635-113.330) | 25.622 (25.400-26.416) | 4.327x |
| US equity panel | [LEAN 18001](https://github.com/QuantConnect/Lean) | 49.967 (48.757-50.759) | 31.049 (30.472-31.514) | 1.609x |

Backtest measured 2026-10-21; reference measurements dated 2026-10-13 were retained unchanged. Host: `Linux-7.0.0-38-generic-x86_64-with-glibc2.43` with 24 logical CPUs. Each side used one isolated warm-up process and ten isolated measured processes. The timer includes only the engine call; it excludes input loading, model inference, target construction, adapter preparation, result extraction, serialization, reporting. These measurements apply only to the named strategy, framework version, frozen input bundle, and machine. Raw samples and bootstrap intervals are retained in [real-strategy performance evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_PERFORMANCE.json).

A performance-impact review retains these measurements for engine source `28c47d6eb270`. The measured source, locked dependencies, review rationale and current correctness evidence are recorded with the raw samples. Measurements are repeated when changes materially affect timed execution; a new source revision alone does not require a benchmark rerun. Dependency validation checks selected packages and artifacts; allowing prereleases without changing those selections does not invalidate the measurements.

### Synthetic diagnostic scenarios

The scenario matrix contains synthetic conformance tests. "Exact" means terminal values, ordered closed trades, and ordered fills match after 1e-8 quantization. Each record declares whether a surface is native, reconstructed, aggregate-only, input-only, or unavailable. These results test isolated conventions, not realistic strategy equivalence.

| Profile | Pinned framework | Required scenarios | Evidence |
|---|---|---:|---|
| `vectorbt_strict` | [VectorBT Pro 2026.6.27](https://vectorbt.pro/) | 17/17 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `vectorbt_oss_strict` | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 16/16 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `backtrader_strict` | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 17/17 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |
| `zipline_strict` | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | 16/16 exact | [scenario evidence](https://github.com/ml4t/backtest/blob/main/validation/CORRECTNESS_RESULTS.json) |

The synthetic stress workload contains 250 assets and 5,040 daily sessions (1,260,000 bars). Every row has zero canonical gap for target intents, native fills, closed trades reconstructed from those fills, and terminal state reconstructed from the fill ledger and final marks. Fill records use 1e-8 precision; monetary totals use cent precision.

| Profile | Current framework | Target intents | Native fills | Fill-derived closed trades | Terminal value | Evidence |
|---|---|---:|---:|---:|---:|---|
| `vectorbt_strict` | [VectorBT Pro 2026.6.27](https://vectorbt.pro/) | 427,790 | 423,313 | 222,751 | 1,285,886.320000 | [scale evidence](https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json) |
| `vectorbt_oss_strict` | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 427,790 | 417,941 | 211,322 | 1,345,348.850000 | [scale evidence](https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json) |
| `backtrader_strict` | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 427,790 | 343,813 | 182,019 | -9,166,273.560000 | [scale evidence](https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json) |
| `zipline_strict` | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | 427,790 | 427,696 | 226,434 | 10,504,095.900000 | [scale evidence](https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json) |
| `lean` | [LEAN 18001](https://github.com/QuantConnect/Lean) | 427,790 | 361,297 | 191,297 | 184,538.130000 | [scale evidence](https://github.com/ml4t/backtest/blob/main/validation/LARGE_SCALE_RESULTS.json) |
<!-- parity-claims:end -->

Release performance evidence runs deterministic single-asset, 250-asset daily, quote-aware,
rebalance, and partial-fill workloads in isolated processes. It reports setup and engine runtime
separately, measures whole-process peak RSS, and verifies retained financial-output checksums and
counts. The 250-asset workload periodically enters and exits 50 positions. Sample spread is
reported for diagnosis, while the
instrument-free hotpath benchmark enforces the runtime regression limit. These ML4T-only
measurements are regression baselines, not cross-machine performance claims.

The separate cross-framework performance artifact retains ten isolated measurements per runner
after one warm-up. It reports complete-process wall time, process-tree and LEAN-container peak RSS,
raw samples, 95% bootstrap intervals, exact output checksums, and semantic disclosures. It is
retained as supporting audit evidence. The published table above instead reports engine-only
timings for the realistic, correctness-passing workloads and states the workload, versions, host,
date, and uncertainty needed to interpret each ratio.

## Installation

```bash
pip install ml4t-backtest
```

## Next Steps

- **New here?** Start with the [Quickstart](getting-started/quickstart.md)
- **Coming from the book?** Use the [Book Guide](book-guide/index.md)
- **Debugging execution behavior?** Read [Execution Semantics](user-guide/execution-semantics.md)
- **Looking up functions and parameters?** See the [API Reference](api/index.md)

## From Book to Library

If you are reading *Machine Learning for Trading, Third Edition*, use the docs in this order:

1. learn the execution or reporting concept in the notebook
2. use the [Book Guide](book-guide/index.md) to find the matching production workflow
3. move to the relevant user-guide page for the reusable API
4. look up functions and parameters in the [API Reference](api/index.md)

This is especially important for quote-aware execution, realistic reporting, rebalancing,
and strategy portability into `ml4t-live`.

## Part of the ML4T Ecosystem

```
ml4t-data --> ml4t-engineer --> ml4t-diagnostic --> ml4t-backtest --> ml4t-live
```

The same `Strategy` class works in both backtest and live trading via `ml4t-live`.
