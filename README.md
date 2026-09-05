# ml4t-backtest

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/ml4t-backtest)](https://pypi.org/project/ml4t-backtest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Event-driven backtesting engine for quantitative trading strategies with realistic execution modeling.

## Part of the ML4T Library Ecosystem

This library is one of six interconnected libraries supporting the machine learning for trading workflow described in [Machine Learning for Trading](https://ml4trading.io):

![ML4T Library Ecosystem](docs/images/ml4t_ecosystem_workflow_color.png)

Together they cover data infrastructure, feature engineering, modeling, signal evaluation, strategy backtesting, and live deployment.

## What This Library Does

Backtesting requires accurate simulation of order execution, position tracking, and risk management. ml4t-backtest provides:

- Event-driven architecture with point-in-time correctness (no look-ahead bias)
- Exit-first order processing matching real broker behavior
- Configurable execution modes (same-bar or next-bar fills)
- Quote-aware execution and marking with `price`, bid, ask, midpoint, and side-aware sources
- Position-level risk rules (stop-loss, take-profit, trailing stops)
- Portfolio-level constraints (max positions, drawdown limits)
- Cash, margin, and crypto account policies
- First-class trade, fill, and portfolio-state export for audit and downstream analysis
- 40+ behavioral knobs for framework-specific parity

The same Strategy class used in backtesting works unchanged in ml4t-live for production deployment.

![ml4t-backtest Architecture](docs/images/ml4t_backtest_architecture_print.jpeg)

## Installation

```bash
pip install ml4t-backtest
```

## Quick Start

<!-- ml4t-doc-test: readme-quickstart -->
```python
from datetime import datetime

import polars as pl
from ml4t.backtest import Engine, Strategy, BacktestConfig, DataFeed

class SignalStrategy(Strategy):
    def on_data(self, timestamp, data, context, broker):
        for asset, bar in data.items():
            signal = bar.get("signals", {}).get("prediction", 0)
            price = bar.get("price", bar.get("close", 0))
            position = broker.get_position(asset)

            if position is None and signal > 0.5:
                shares = (broker.get_account_value() * 0.10) / price
                if shares > 0:
                    broker.submit_order(asset, shares)
            elif position is not None and signal < -0.5:
                broker.close_position(asset)

timestamps = [datetime(2024, 1, day) for day in (2, 3, 4, 5)]
prices = pl.DataFrame(
    {
        "timestamp": timestamps,
        "asset": ["AAPL"] * 4,
        "close": [100.0, 101.0, 103.0, 102.0],
    }
)
signals = pl.DataFrame(
    {
        "timestamp": timestamps,
        "asset": ["AAPL"] * 4,
        "prediction": [1.0, 1.0, -1.0, -1.0],
    }
)

config = BacktestConfig(initial_cash=100_000)
feed = DataFeed(prices_df=prices, signals_df=signals)
engine = Engine(feed, SignalStrategy(), config)
result = engine.run()

print(f"Total Return: {result.metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {result.metrics['sharpe']:.2f}")
print(result.to_fills_dataframe().head())
```

Each `Engine` instance is single-use. Create a new instance for every independent run.

`bar["price"]` follows `FeedSpec.price_col` when you provide one, so the same strategy works for close-based bars and quote-aware feeds.

## Risk Management

Position-level exit rules:

```python
from ml4t.backtest import Strategy, StopLoss, TakeProfit, TrailingStop, RuleChain

class MyStrategy(Strategy):
    def on_start(self, broker):
        broker.set_position_rules(RuleChain([
            StopLoss(pct=0.05),
            TakeProfit(pct=0.15),
            TrailingStop(pct=0.03),
        ]))
```

Portfolio-level controls:

```python
from ml4t.backtest.risk.portfolio.limits import MaxDrawdownLimit, DailyLossLimit
```

## Framework Profiles

Built-in profiles configure the behavioral semantics used by major backtesting frameworks:

```python
from ml4t.backtest import BacktestConfig

# Match VectorBT behavior (same-bar close fills, fractional shares)
config = BacktestConfig.from_preset("vectorbt")

# Match Backtrader behavior (next-bar open fills, integer shares)
config = BacktestConfig.from_preset("backtrader")

# Match the documented Zipline comparison protocol (next-bar open, no default costs)
config = BacktestConfig.from_preset("zipline")

# Match the frozen LEAN daily US-equity protocol (next-session open, integer shares)
config = BacktestConfig.from_preset("lean")

# Conservative production settings (higher costs, cash buffer)
config = BacktestConfig.from_preset("realistic")
```

Each profile sets 40+ behavioral knobs, including fill timing, execution price, share type,
commission model, and order processing. Current exact-match evidence appears below.

## Execution Modes

```python
from ml4t.backtest import ExecutionMode, StopFillMode

# Same-bar fills (VectorBT style)
config = BacktestConfig(
    execution_mode=ExecutionMode.SAME_BAR,
    stop_fill_mode=StopFillMode.STOP_PRICE,
)

# Next-bar fills (Backtrader style)
config = BacktestConfig(
    execution_mode=ExecutionMode.NEXT_BAR,
    stop_fill_mode=StopFillMode.STOP_PRICE,
)
```

## Quote-Aware Execution

```python
from ml4t.backtest import BacktestConfig, DataFeed
from ml4t.backtest.config import ExecutionPrice

feed = DataFeed(
    prices_df=quotes,
    price_col="mid_price",
    bid_col="bid",
    ask_col="ask",
    bid_size_col="bid_size",
    ask_size_col="ask_size",
)

config = BacktestConfig(
    execution_price=ExecutionPrice.QUOTE_SIDE,
    mark_price=ExecutionPrice.QUOTE_SIDE,
)
```

With `QUOTE_SIDE`, buys fill at the ask and sells fill at the bid when quotes are present. `mark_price` is configured separately, so you can trade on one source and mark the book on another.

Quote-aware runs also preserve the microstructure context in the result surface:

- `result.to_fills_dataframe()` includes bid/ask/midpoint/spread/size context
- `result.to_trades_dataframe()` includes nullable entry/exit quote summaries
- `result.to_portfolio_state_dataframe()` reflects the configured mark source over time
- `result.to_predictions_dataframe()` preserves the raw model/input surface for downstream
  diagnostics

## Reproducible Config Snapshots

`BacktestConfig` is also the serializable backtest preset surface. You can keep
input configs sparse, then persist the fully resolved config that actually ran.

```python
config = BacktestConfig.from_yaml("config/my_backtest.yaml")
result = Engine(feed, strategy, config).run()

resolved_config = result.config.to_dict()
runtime_spec = result.to_spec_dict()
written = result.to_parquet("results/run_001")
```

The exported result directory includes:

- `config.yaml` for the replayable resolved config payload
- `spec.yaml` for the richer runtime snapshot with library version and realized run window

Use top-level `feed` in `BacktestConfig` for generic feed semantics and top-level
`metadata` for user-defined provenance like input paths or strategy ids.

## Commission and Slippage

```python
from ml4t.backtest import BacktestConfig, CommissionType
from ml4t.backtest.config import SlippageType, SpreadConvention

config = BacktestConfig(
    commission_rate=0.001,         # 10 bps percentage
    slippage_rate=0.0005,          # 5 bps slippage
    stop_slippage_rate=0.001,      # Additional slippage for stop exits
)

# Or per-share (Interactive Brokers style)
config = BacktestConfig(
    commission_type=CommissionType.PER_SHARE,
    commission_per_share=0.005,
    commission_minimum=1.0,
)

# Or bar-only spread approximation in currency units
config = BacktestConfig(
    slippage_type=SlippageType.SPREAD,
    slippage_spread=0.02,
    slippage_spread_convention=SpreadConvention.FULL_SPREAD,
)
```

## Multi-Asset Rebalancing

```python
from ml4t.backtest import Strategy, TargetWeightExecutor, RebalanceConfig

class WeightStrategy(Strategy):
    def __init__(self):
        self.executor = TargetWeightExecutor(RebalanceConfig(
            min_trade_value=100,    # Optional: skip tiny dollar trades
            min_weight_change=0.01, # Optional: skip tiny weight changes
        ))
        self.bar_count = 0

    def on_data(self, timestamp, data, context, broker):
        self.bar_count += 1
        if self.bar_count % 21 != 1:  # Monthly rebalance
            return

        # ML predictions → portfolio weights
        weights = {}
        for asset, bar in data.items():
            signal = bar.get("signals", {}).get("prediction", 0)
            if signal and signal > 0:
                weights[asset] = signal
        if weights:
            total = sum(weights.values())
            weights = {a: w / total for a, w in weights.items()}
            self.executor.execute(weights, data, broker)
```

`RebalanceConfig` defaults both `min_trade_value` and `min_weight_change` to
`0.0`, so these filters are opt-in.

`BacktestConfig()` defaults to neutral costs: `commission_type=NONE` and
`slippage_type=NONE`. Broker-specific fee models and synthetic slippage are
opt-in.

## Cross-Framework Validation

The primary audit uses frozen data and targets from ETF allocation, CME futures, crypto
perpetual-funding, FX, and US equity-panel case studies. Profiles configure framework-specific
execution behavior, and only genuinely supported framework and asset combinations are required.
The real-strategy gate covers 17 required pairs and records eight unsupported pairs separately.

The scenario matrix and 250-asset workload remain useful synthetic diagnostic and stress tests.
They do not establish realistic strategy equivalence.

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

### Real-strategy engine performance

The table reports engine-call wall time for all 17 correctness-passing pairs. The ratio is framework median / ML4T median; values above 1 mean ML4T completed the engine call faster.

| Real strategy | Pinned framework | Framework median (95% CI), s | ML4T median (95% CI), s | Framework / ML4T median |
|---|---|---:|---:|---:|
| ETF allocation | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | 0.287 (0.286-0.289) | 0.415 (0.414-0.419) | 0.692x |
| ETF allocation | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 0.171 (0.171-0.174) | 0.415 (0.413-0.417) | 0.412x |
| ETF allocation | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 9.355 (9.303-9.424) | 0.430 (0.426-0.432) | 21.762x |
| ETF allocation | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | 3.872 (3.867-3.910) | 0.632 (0.628-0.638) | 6.127x |
| ETF allocation | [LEAN 18001](https://github.com/QuantConnect/Lean) | 2.564 (2.487-2.599) | 0.750 (0.749-0.757) | 3.418x |
| CME futures | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | 0.286 (0.284-0.289) | 0.432 (0.429-0.435) | 0.663x |
| CME futures | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 2.495 (2.488-2.526) | 0.429 (0.427-0.433) | 5.812x |
| Crypto perpetual funding | [LEAN 18001](https://github.com/QuantConnect/Lean) | 2.801 (2.709-2.891) | 0.657 (0.655-0.661) | 4.266x |
| FX allocation (USD-quoted pairs) | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | 0.283 (0.281-0.286) | 0.146 (0.145-0.147) | 1.939x |
| FX allocation (USD-quoted pairs) | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 0.139 (0.138-0.141) | 0.146 (0.145-0.147) | 0.952x |
| FX allocation (USD-quoted pairs) | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 0.425 (0.420-0.432) | 0.146 (0.145-0.147) | 2.920x |
| FX allocation (USD-quoted pairs) | [LEAN 18001](https://github.com/QuantConnect/Lean) | 0.913 (0.877-0.953) | 0.153 (0.152-0.157) | 5.952x |
| US equity panel | [VectorBT Pro 2026.6.27](https://github.com/polakowo/vectorbt.pro) | 0.795 (0.787-0.838) | 20.555 (20.486-20.600) | 0.039x |
| US equity panel | [VectorBT OSS 1.1.0](https://pypi.org/project/vectorbt/1.1.0/) | 16.995 (16.960-17.024) | 20.495 (20.453-20.534) | 0.829x |
| US equity panel | [Backtrader 1.9.78.123](https://pypi.org/project/backtrader/1.9.78.123/) | 495.553 (494.227-510.273) | 21.470 (21.386-21.543) | 23.082x |
| US equity panel | [Zipline Reloaded 3.1.1](https://pypi.org/project/zipline-reloaded/3.1.1/) | 107.609 (106.834-109.150) | 22.132 (22.082-22.234) | 4.862x |
| US equity panel | [LEAN 18001](https://github.com/QuantConnect/Lean) | 47.683 (47.333-48.015) | 26.368 (26.222-26.473) | 1.808x |

Measured 2026-09-03 on `Linux-6.8.0-138-generic-x86_64-with-glibc2.39` with 24 logical CPUs. Each side used one isolated warm-up process and ten isolated measured processes. The timer includes only the engine call; it excludes input loading, model inference, target construction, adapter preparation, result extraction, serialization, reporting. These measurements apply only to the named strategy, framework version, frozen input bundle, and machine. Raw samples and bootstrap intervals are retained in [real-strategy performance evidence](https://github.com/ml4t/backtest/blob/main/validation/REAL_STRATEGY_PERFORMANCE.json).

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

See [validation/README.md](validation/README.md) for methodology and detailed results.

Release-gate commands:

```bash
# Fast parity contract gate (scenario 01 across vectorbt/backtrader/zipline)
ML4T_COMPARISON_INPROC=1 uv run pytest tests/contracts/test_cross_engine_contracts.py -q

# Full correctness runner (selected scenarios)
python validation/run_all_correctness.py --framework vectorbt_oss --scenarios 01,03,05,09
python validation/run_all_correctness.py --framework backtrader --scenarios 01,03,05,09
python validation/run_all_correctness.py --framework zipline --scenarios 01,03,05,09
```

## Performance

Release performance evidence covers deterministic single-asset, 250-asset daily, quote-aware,
rebalance, and partial-fill workloads. Each workload runs three times in a fresh child process.
The 250-asset workload periodically enters and exits a 50-position portfolio.
The evidence separates setup from `Engine.run()`, measures peak RSS over the whole child process,
reports runtime and memory sample spread, and verifies retained financial-output checksums and
counts. The dedicated instrument-free hotpath benchmark enforces the runtime regression limit.

Run the release baselines and the instrument-free feed regression check locally:

```bash
uv run python validation/performance_baseline.py --output release-performance-evidence.json
uv run pytest tests/benchmark/test_hotpath_benchmarks.py::test_optimized_feed_runtime_vs_legacy_baseline --no-cov
```

Workload definitions and expected checksums are retained in
`validation/performance_baselines.json`. These ML4T-only measurements are regression baselines,
not cross-machine performance claims.

Cross-framework performance evidence uses a common 50-asset, 252-session controlled workload.
Each runner receives one isolated warm-up followed by ten isolated measurements. The retained
artifact contains raw samples, whole-process wall time, process-tree peak RSS, deterministic 95%
bootstrap intervals, output checksums, framework identities, and semantic disclosures for the
idiomatic view. This controlled synthetic workload remains supporting audit evidence rather than
the basis for the published real-strategy comparison above.

The real-strategy performance artifact times only engine execution for correctness-passing pairs.
Inputs, model inference, target construction, adapter preparation, extraction, and reporting are
excluded. The table above publishes the measured ratios with their date, versions, workload, host,
and uncertainty. They describe those executions and do not establish a hardware-independent speed
ranking.

## Documentation

- [Getting Started](docs/getting-started/quickstart.md) — your first backtest
- [Data Feed](docs/user-guide/data-feed.md) — `price_col`, quote columns, and feed wiring
- [Strategies](docs/user-guide/strategies.md) — strategy interface and templates
- [Stateful Strategies](docs/user-guide/stateful-strategies.md) — advanced event-driven patterns (Kelly sizing, pairs trading, circuit breakers)
- [Execution Semantics](docs/user-guide/execution-semantics.md) — fill timing, ordering, stops
- [Configuration](docs/user-guide/configuration.md) — 40+ behavioral knobs
- [Risk Management](docs/user-guide/risk-management.md) — stops, trails, portfolio limits
- [Rebalancing](docs/user-guide/rebalancing.md) — weight-based portfolio management
- [Results & Analysis](docs/user-guide/results.md) — trades, fills, equity, and Parquet export
- [Market Impact](docs/user-guide/market-impact.md) — commission, slippage, and impact models
- [Profiles](docs/user-guide/profiles.md) — framework parity presets

## Technical Characteristics

- **Event-driven**: Each bar processes sequentially with configurable order sequencing
- **Causal lifecycle**: Per-bar callbacks receive the current bar; `on_prepare` receives configuration but no future feed timestamps
- **Configurable fills**: Match behavior of different backtesting frameworks
- **Quote-aware**: Optional bid/ask/mid/size caches with side-aware market fills
- **Parquet export**: Trades, fills, equity, daily P&L, and config are serializable
- **Type-safe**: 0 type diagnostics (ty/Astral), full type annotations

## Related Libraries

- **ml4t-data**: Market data acquisition and storage
- **ml4t-engineer**: Feature engineering and technical indicators
- **ml4t-diagnostic**: Signal evaluation and statistical validation
- **ml4t-live**: Live trading with broker integration

## Development

```bash
git clone https://github.com/ml4t/backtest.git
cd backtest
uv sync
uv run pytest tests/ -q
uv run ty check
```

## Known Limitations

See [LIMITATIONS.md](LIMITATIONS.md) for documented assumptions:

- Bar data cannot identify the path or queue order of intrabar events
- Corporate actions, borrow costs, taxes, and currency conversion are not modeled
- The pre-stable strategy lifecycle still depends on the shared `ml4t-live` contract

## License

MIT License - see [LICENSE](LICENSE) for details.
