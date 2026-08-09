# Shared project context: ml4t-backtest

## Contract authority

- Runtime-neutral lifecycle, intent, execution-policy, and position-rule contracts belong in
  `ml4t-specs`.
- Backtest behavior must preserve causal information boundaries and cross-framework validation.

## Workflow

```bash
uv sync
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check
uv run pytest
pre-commit run --all-files
```
