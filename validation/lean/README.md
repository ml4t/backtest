# LEAN Validation Workflow

The retained LEAN comparison uses a frozen engine, CLI, data protocol, and account model:

- LEAN engine 18001 at source commit `278fcb3d1b815b63ccadba68d7ae54422e34b792`
- image `quantconnect/lean@sha256:ecd62b0e418d40d1d7c0cd95e90a94e397642a21d2c8810614830c8a4e9a8f70`
- LEAN CLI 1.0.228 from the locked environment
- daily adjusted US equities, `DefaultBrokerageModel`, margin account, and 2x security leverage

This is a scoped comparison protocol. It is not a claim about every LEAN asset class,
resolution, brokerage model, or order type.

## Build the Locked Environment

Docker must be running. From the repository root:

```bash
uv run python validation/build_framework_env.py --framework lean
```

This creates `.venv-lean`, verifies the exact CLI distribution, and verifies the immutable engine
image. The runners do not install a CLI dynamically and do not accept a mutable image tag.

## Native Behavior Evidence

Run the native oracle before changing the `lean` profile:

```bash
.venv-lean/bin/python validation/native/lean_behavior.py \
  --lean-command .venv-lean/bin/lean \
  --output /tmp/lean-native.json
```

The oracle checks the event and model behavior used by the profile, including next-session market
fills, final-bar orders, model selection, margin, target sizing, submission order, buying power,
explicit costs, fill-forward data, full fills, and terminal holdings. The retained result is
`validation/native/evidence/lean-18001.json`.

The `lean` profile's stop settings are ML4T fallbacks. Stop-order equivalence is not part of this
native evidence.

## Chapter 16 Case Studies

The current large comparison reruns three retained Chapter 16 projects against the frozen image:

```bash
.venv-lean/bin/python validation/run_lean_case_studies.py \
  --lean-command .venv-lean/bin/lean \
  --output /tmp/lean-case-studies.json
```

Add `--promote` only when intentionally replacing the tracked LEAN outputs and retained evidence.
Promotion is atomic and occurs only after all projects pass.

The three projects compare 47,652 canonical fills. They currently have zero fill-surface gap and
zero terminal-value gap at a `$0.0001` quantum. The machine-readable result is
`validation/lean/case_study_evidence.json`.

These projects specify their own percentage fee and zero-slippage settings. They do not test the
fee and slippage models selected by `DefaultBrokerageModel`; the native oracle tests that model
selection separately.

## Benchmark Adapter

The daily-data adapter remains available through the benchmark suite:

```bash
uv run python validation/benchmark_suite.py \
  --framework lean \
  --scenario daily_baseline \
  --data-source real \
  --real-data-path /path/to/us_equities.parquet
```

To run the ML4T side with the corresponding profile, use `--framework ml4t-lean-strict`.

The adapter requires an exact CLI version and passes the frozen image digest to every LEAN run.
It supports daily scenarios only.

## Reproducibility Boundary

The native and Chapter 16 runners construct temporary LEAN roots from tracked inputs:

- `validation/lean/support/lean.json`
- `validation/lean/support/data/market-hours/market-hours-database.json`
- `validation/lean/support/data/symbol-properties/symbol-properties-database.csv`
- the tracked project files and daily equity archives under `validation/lean/workspace/`

Map and factor files are generated from the first date in each tracked daily archive. No ignored
machine-local `lean.json`, map file, factor file, or data directory is required.

The old `scenario_01_long_only/` project and `validation/run_all_correctness.py` LEAN path are
legacy scaffolding. They are not evidence for the current claim.
