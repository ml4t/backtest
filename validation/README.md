# ml4t.backtest Validation Suite

## Related Documents

- [Validation Methodology](METHODOLOGY.md) -- philosophy, behavioral matrix, profile system
- [Known Limitations](../LIMITATIONS.md) -- edge cases, what's not modeled
- [LEAN Validation Workflow](lean/README.md) -- exact LEAN setup used in parity/benchmark runs

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
| **zipline_strict** | 226,723 | 226,723 | **0 (0.00%)** | $10.30 (0.0014%) | **8.0x faster** |
| **backtrader_strict** | 226,535 | 226,535 | **0 (0.00%)** | ~$0 (float noise) | **7.7x faster** |
| **vectorbt_strict** | 210,352 | 210,261 | 91 (0.04%) | $0 (0.00%) | 0.04x |
| **lean** | 428,459 fills | 428,459 fills | **0 (0.00%)** | $1.55 (0.0002%) | **3.4x faster** |

## How to Run

### Current-environment runner

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

The command returns nonzero when a required framework or scenario does not execute. Use
`--result-json PATH` to retain machine-readable terminal records. An explicitly unsupported pair
is reported as `unsupported`; it is visible in the matrix and does not count as a pass.

### Isolated release-gate runner

```bash
# Complete required matrix using each framework's dedicated environment
python validation/run_all_correctness.py

# Selected framework/scenario matrix
python validation/run_all_correctness.py --framework backtrader --scenarios 01,03,05,09
```

The isolated runner writes `validation/CORRECTNESS_RESULTS.json` and distinguishes comparison
failures, unsupported pairs, unavailable environments, adapter failures, subprocess failures,
timeouts, skips, malformed records, and missing scenarios. Every status except `pass` and an
explicitly optional `unsupported` record fails the release gate.

### Large-scale benchmarks

```bash
# Framework benchmark
python validation/benchmark_suite.py --profile backtrader_strict --framework backtrader

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
├── run_all_correctness.py     # Isolated release-gate correctness runner
├── run_all_benchmarks.py      # Framework benchmark loop
├── benchmark_suite.py         # Large-scale benchmark runner
├── run_full_validation.py     # Complete validation pipeline
├── lean/                      # LEAN integration
├── trade_logs/                # Golden file CSVs (gitignored)
└── nautilus/                  # Nautilus Trader evaluation
```

## Remaining Gaps

| Profile | Gap | Root Cause | Next Step |
|---------|-----|------------|-----------|
| zipline_strict | 0 trades, $10.30 | Small terminal-value residual | **DONE** |
| backtrader_strict | 0 trades, float noise | Floating point | **DONE** |
| vectorbt_strict | 91 trades, $0 | Unknown | Signal processing audit |
| lean | 0 fills, $1.55 | Price precision / mark-to-market rounding | **DONE** |
