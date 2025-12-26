# ml4t.backtest Notebooks

Jupyter notebooks demonstrating ml4t.backtest capabilities and validating accuracy against other frameworks.

## Quick Start

```bash
cd /home/stefan/ml4t/software/backtest
source .venv/bin/activate
jupyter notebook notebooks/
```

## Notebooks

| Notebook | Description | Data |
|----------|-------------|------|
| `01_framework_comparison_long_only.ipynb` | Validates against VectorBT & Backtrader with long-only MA crossover | ETF Universe |
| `02_framework_comparison_stop_loss.ipynb` | Stop-loss execution validation across frameworks | ETF Universe |
| `03_zipline_comparison.ipynb` | Zipline-reloaded comparison with WIKI prices | Quandl WIKI |
| `04_ml4t_capabilities.ipynb` | Comprehensive ml4t.backtest feature demo | ETF Universe |

## Data Requirements

The notebooks use data from:
- **ETF Universe**: `~/Dropbox/ml4t/data/etfs/etf_universe.parquet`
- **WIKI Prices**: `~/Dropbox/ml4t/data/equities/wiki_prices.parquet`

## Framework Dependencies

For full comparison testing:

```bash
# VectorBT (for notebooks 01, 02)
pip install vectorbt

# Backtrader (for notebooks 01, 02)
pip install backtrader

# Zipline (for notebook 03)
pip install zipline-reloaded exchange-calendars
```

## Validation Results

All frameworks produce matching results:

| Framework | Status | Notes |
|-----------|--------|-------|
| VectorBT | ✅ MATCH | Exact match with NEXT_BAR mode |
| Backtrader | ✅ MATCH | Uses COO flag for next-bar execution |
| Zipline | ✅ PASS | Within tolerance, strategy-level stops |

## Running Without External Frameworks

The notebooks are designed to run even without VectorBT, Backtrader, or Zipline installed. The ml4t.backtest sections will always execute, and comparison sections will be skipped gracefully if dependencies are missing.
