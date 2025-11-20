# Private/Commercial Dependency Tests

This directory contains tests that require **commercial or private dependencies** that cannot be included in the open-source distribution.

## VectorBT Pro Tests

The following tests require [VectorBT Pro](https://vectorbt.pro/), a commercial library:

- `test_vectorbtpro.py` - Basic VectorBT Pro functionality tests
- `test_vectorbtpro_adapter.py` - Adapter tests for cross-framework validation
- `vectorbtpro_5000_trades.py` - Large-scale performance validation
- `benchmark_vectorbtpro_performance.py` - Performance benchmarking

### Installation

If you have a VectorBT Pro license, install it with:

```bash
uv pip install -U "vectorbtpro[base] @ git+ssh://git@github.com/polakowo/vectorbt.pro.git"
```

### Running Private Tests

These tests are **excluded from the default test suite**. To run them:

```bash
# Run all private tests
pytest tests/private/ -v

# Run specific test file
pytest tests/private/test_vectorbtpro_adapter.py -v

# Run with coverage
pytest tests/private/ --cov=ml4t.backtest --cov-report=html
```

## Important Notes

⚠️ **DO NOT commit VectorBT Pro to dependencies** before public release

- These tests are for **internal development validation only**
- VectorBT Pro is a commercial product and cannot be distributed with open-source code
- Before release, ensure `pyproject.toml` does NOT include `vectorbtpro` in dependencies
- The `comparison` extra may include `vectorbt` (open-source version) but not `vectorbtpro`

## CI/CD Considerations

- Default CI runs should NOT include these tests (no VectorBT Pro license)
- Optional validation workflows can run these tests if secrets are configured
- Consider using a separate private validation job that runs on-demand

## Alternative: Open-Source Validation

For public validation, use the open-source comparison frameworks:

```bash
# Install open-source comparison frameworks
uv pip install -e ".[comparison]"

# Run validation tests (will use vectorbt, backtrader, zipline)
pytest tests/validation/ -v
```

These provide 90%+ of the validation coverage without requiring commercial licenses.
