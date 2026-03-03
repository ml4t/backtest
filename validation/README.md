# ml4t.backtest Validation Suite

## Related Documents

- [Validation Methodology](METHODOLOGY.md) -- philosophy, behavioral matrix, profile system
- [Known Limitations](../LIMITATIONS.md) -- edge cases, what's not modeled

## Overview

ml4t-backtest validates correctness by being **configurable enough to perfectly replicate** every
major external backtester through profiles. See [METHODOLOGY.md](METHODOLOGY.md) for the full
approach.

Validation is performed per-framework in **isolated virtual environments** due to dependency
conflicts between frameworks (VBT Pro, Backtrader, Zipline).

## Test Coverage Matrix

16 scenarios x 4 frameworks (all pass):

| Feature | VBT Pro | VBT OSS | Backtrader | Zipline |
|---------|---------|---------|------------|---------|
| 01: Long only | PASS | PASS | PASS | PASS |
| 02: Long/Short | PASS | PASS | PASS | PASS |
| 03: Stop-loss | PASS | PASS | PASS | PASS |
| 04: Take-profit | PASS | PASS | PASS | PASS |
| 05: % Commission | PASS | PASS | PASS | PASS |
| 06: Per-share commission | PASS | PASS | PASS | PASS |
| 07: Fixed slippage | PASS | PASS | PASS | PASS |
| 08: % Slippage | PASS | PASS | PASS | PASS |
| 09: Trailing stop | PASS | PASS | PASS | PASS |
| 10: Bracket order | PASS | PASS | PASS | N/A |
| 11: Short only | PASS | PASS | PASS | PASS |
| 12: Short trailing stop | PASS | PASS | PASS | PASS |
| 13: TSL + TP combo | PASS | PASS | PASS | PASS |
| 14: TSL + SL combo | PASS | PASS | PASS | PASS |
| 15: Triple rule | PASS | PASS | PASS | PASS |
| 16: Stress (1500 bars) | PASS | PASS | PASS | PASS |

## Large-Scale Parity (250 assets x 20 years, real data)

Data: `us_equities.parquet` (250 US equities, 1998-2018). Strategy: long top 25, short bottom 25.

| Profile | ml4t Trades | Ref Trades | Trade Gap | Value Gap | Speed |
|---------|-------------|------------|-----------|-----------|-------|
| **zipline_strict** | 226,723 | 226,723 | **0 (0.00%)** | $19 (0.0001%) | **8.4x faster** |
| **backtrader_strict** | 216,980 | 216,981 | 1 (0.0005%) | $503 (0.004%) | **19.3x faster** |
| **vectorbt_strict** | 210,352 | 210,261 | 91 (0.04%) | $0 (0.00%) | 0.04x |
| **lean_strict** | 226,246 | 225,583 | 663 (0.29%) | $13,150 (1.2%) | **5.3x faster** |

## How to Run

### Parameterized runner (new)

```bash
# Single scenario
python validation/run_scenario.py --scenario 01 --framework backtrader

# All scenarios for one framework
python validation/run_scenario.py --framework vectorbt_oss

# Full matrix
python validation/run_scenario.py --all

# Dry run (list combinations)
python validation/run_scenario.py --dry-run
```

### Individual scripts (legacy, in archived/scenarios_v1/)

Each framework has 16 standalone scripts that can be run individually:

```bash
# VectorBT Pro
source .venv-vectorbt-pro/bin/activate
python validation/archived/scenarios_v1/vectorbt_pro/scenario_01_long_only.py

# Backtrader
.venv-backtrader/bin/python3 validation/archived/scenarios_v1/backtrader/scenario_01_long_only.py

# VectorBT OSS
.venv-vectorbt/bin/python3 validation/archived/scenarios_v1/vectorbt_oss/scenario_01_long_only.py

# Zipline
.venv-zipline/bin/python3 validation/archived/scenarios_v1/zipline/scenario_01_long_only.py
```

### Large-scale benchmarks

```bash
# Framework benchmark
python validation/benchmark_suite.py --profile backtrader_strict --framework backtrader

# All correctness scenarios
python validation/run_all_correctness.py

# Full validation (correctness + benchmarks)
python validation/run_full_validation.py
```

## Virtual Environment Setup

```bash
# VectorBT OSS
python3 -m venv .venv-vectorbt
.venv-vectorbt/bin/pip install vectorbt pandas numpy polars pyyaml pydantic numba

# Backtrader
python3 -m venv .venv-backtrader
.venv-backtrader/bin/pip install backtrader pandas numpy polars pyyaml pydantic numba exchange_calendars

# Zipline
python3 -m venv .venv-zipline
.venv-zipline/bin/pip install zipline-reloaded pandas numpy polars pyyaml pydantic numba exchange_calendars

# Never mix VBT OSS and Pro in the same environment
```

## File Organization

```
validation/
├── README.md                  # This file
├── METHODOLOGY.md             # Validation philosophy and behavioral matrix
├── common/                    # Shared infrastructure (types, data generators, comparator)
├── scenarios/                 # Declarative scenario definitions (16 configs)
├── frameworks/                # Parameterized framework drivers (4 modules)
├── run_scenario.py            # Unified CLI runner
├── run_all_correctness.py     # Legacy scenario correctness runner
├── run_all_benchmarks.py      # Framework benchmark loop
├── benchmark_suite.py         # Large-scale benchmark runner
├── run_full_validation.py     # Complete validation pipeline
├── archived/
│   ├── scenarios_v1/          # Original 64 standalone scripts (by framework)
│   └── *.py                   # Debug/comparison one-offs
├── lean/                      # LEAN integration
├── trade_logs/                # Golden file CSVs (gitignored)
└── nautilus/                  # Nautilus Trader evaluation
```

## Remaining Gaps

| Profile | Gap | Root Cause | Next Step |
|---------|-----|------------|-----------|
| zipline_strict | 0 trades, $19 | Floating point | **DONE** |
| backtrader_strict | 1 trade, $503 | Unknown | Per-bar investigation |
| vectorbt_strict | 91 trades, $0 | Unknown | Signal processing audit |
| lean_strict | 663 trades, $13K | Buying-power reservation | Add `buying_power_reservation` knob |
