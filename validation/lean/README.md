# LEAN Validation Workflow

## Status

LEAN is part of the validation/parity workflow, but not through the old static
`scenario_01_long_only/` project in this directory.

What actually runs today is the LEAN adapter in
[`validation/benchmark_suite.py`](../benchmark_suite.py), which:

- supports the benchmark/parity workflow, not the old scenario matrix
- supports daily-data scenarios only
- generates a temporary LEAN project and LEAN-format data on the fly
- runs LEAN through the CLI in Docker

The old `scenario_01_long_only/` folder is legacy scaffolding, not the current
source of truth.

## What Made It Work

These are the pieces that mattered in practice:

1. We used the Dockerized LEAN CLI path, not a direct `.NET` launcher workflow.
2. We did not require a permanently installed `lean` binary.
   The adapter falls back to:

   ```bash
   uvx --python 3.12 --with "setuptools<81" lean
   ```

3. We required a machine-local LEAN workspace config at:

   ```text
   validation/lean/workspace/lean.json
   ```

4. We let `benchmark_suite.py` generate the LEAN project, algorithm, target
   file, and LEAN-format equity data for each run.
5. Docker had to be running before invoking the benchmark.

## One-Time Bootstrap

From the repository root:

```bash
mkdir -p validation/lean/workspace
cd validation/lean/workspace
uvx --python 3.12 --with "setuptools<81" lean init
```

If `lean` is already installed on `PATH`, this also works:

```bash
lean init
```

The important outcome is that `validation/lean/workspace/lean.json` exists and
is valid on the local machine. That file is not committed in the repo.

## Runtime Prerequisites

- Docker daemon available
- either `lean` on `PATH` or `uvx` on `PATH`
- Python 3.12 available for the `uvx` fallback path
- initialized LEAN workspace config at
  `validation/lean/workspace/lean.json`

The adapter itself also sets:

```text
UV_CACHE_DIR=/tmp/uv-cache
UV_TOOL_DIR=/tmp/uv-tools
```

to keep the transient LEAN CLI tool environment out of the repo.

## Exact Command We Run

Use the benchmark suite, not the legacy `run_all_correctness.py` LEAN path.

Typical invocation:

```bash
python validation/benchmark_suite.py \
  --framework lean \
  --scenario daily_baseline \
  --data-source real \
  --real-data-path /path/to/us_equities.parquet
```

To compare the ml4t side using the LEAN-style profile:

```bash
python validation/benchmark_suite.py \
  --framework ml4t-lean-strict \
  --scenario daily_baseline \
  --data-source real \
  --real-data-path /path/to/us_equities.parquet
```

Notes:

- `--framework lean` invokes the LEAN adapter.
- `--framework ml4t-lean-strict` runs `ml4t-backtest` with the LEAN-strict
  parity profile.
- LEAN currently returns an error for non-daily scenarios.

## What The Adapter Generates

For each run, `benchmark_suite.py` writes a generated LEAN project under:

```text
validation/lean/workspace/ml4t_benchmark/
```

Key generated files:

- `main.py`: generated QCAlgorithm using canonical target shares
- `config.json`: minimal LEAN project config
- `symbols.csv`: generated synthetic tickers for the selected assets
- `targets.csv`: daily target share instructions

It also exports LEAN-format daily equity data under:

```text
validation/lean/workspace/data/equity/usa/
```

including:

- `daily/*.zip`
- `map_files/*.csv`
- `factor_files/*.csv`

To avoid re-exporting identical daily data every run, it maintains:

```text
validation/lean/workspace/data/equity/usa/ml4t_manifest.json
```

and uses a signature-based cache check.

## The Exact LEAN Invocation

Once the generated project and data are in place, the adapter runs:

```bash
lean backtest validation/lean/workspace/ml4t_benchmark \
  --lean-config validation/lean/workspace/lean.json \
  --no-update \
  --output <generated-output-dir>
```

If `lean` is not installed, the code uses the `uvx` form instead.

Results are read back from the generated LEAN summary JSON in the output
directory.

## Source Of Truth In Code

The working implementation lives here:

- [`validation/benchmark_suite.py`](../benchmark_suite.py): LEAN adapter,
  project generation, data export, CLI invocation, result parsing

These files are legacy and should not be treated as the primary workflow:

- [`validation/lean/scenario_01_long_only/main.py`](scenario_01_long_only/main.py)
- [`validation/run_all_correctness.py`](../run_all_correctness.py)

## Practical Caveats

- LEAN is currently benchmark/parity coverage, not full scenario-matrix coverage.
- The adapter currently supports daily data only.
- The local `lean.json` is a manual prerequisite and is intentionally not
  committed.
- Docker startup and LEAN container initialization make LEAN slower and more
  operationally fragile than the Python-native backtest validation paths.
