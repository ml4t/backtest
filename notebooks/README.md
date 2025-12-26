# ml4t.backtest Notebooks

Jupyter notebooks demonstrating ml4t.backtest capabilities and validating accuracy against other frameworks.

## Quick Start

```bash
cd /home/stefan/ml4t/software/backtest
source .venv/bin/activate
jupyter notebook notebooks/
```

## Notebooks

| Notebook | Description | Data | Trades |
|----------|-------------|------|--------|
| `01_framework_comparison_long_only.ipynb` | Single-asset long-only MA crossover | ETF Universe | ~10 |
| `02_framework_comparison_stop_loss.ipynb` | Stop-loss execution validation | ETF Universe | ~10 |
| `03_zipline_comparison.ipynb` | Zipline-reloaded comparison | Quandl WIKI | ~10 |
| `04_ml4t_capabilities.ipynb` | Feature demonstration | ETF Universe | Various |
| **`05_comprehensive_validation.ipynb`** | **Multi-asset validation with benchmarks** | **Synthetic (5 assets, 3 years)** | **50+** |

## Recommended: Comprehensive Validation

**`05_comprehensive_validation.ipynb`** is the definitive validation notebook:

- **5-asset synthetic universe** with different market dynamics:
  - TECH: High volatility (σ=35%, GBM model)
  - UTIL: Low volatility utility (σ=12%, GBM model)
  - BANK: Financial with stochastic volatility (Heston model)
  - RETAIL: Cyclical with volatility clustering (GARCH model)
  - PHARMA: Healthcare with jump diffusion (GBM+jumps)

- **3 years of daily data** (756 bars per asset, 3,780 total bars)

- **Multi-asset momentum strategy** generating **50+ round-trip trades**

- **Full validation suite**:
  - Side-by-side equity curves
  - Trade-by-trade comparison
  - Final value matching (to $0.01)
  - Execution speed benchmarks

- **Uses ml4t.data.SyntheticProvider** for reproducible data generation

## Data Requirements

### Real Data (Notebooks 01-04)
- **ETF Universe**: `~/Dropbox/ml4t/data/etfs/etf_universe.parquet`
- **WIKI Prices**: `~/Dropbox/ml4t/data/equities/wiki_prices.parquet`

### Synthetic Data (Notebook 05)
No external data required - uses `ml4t.data.providers.SyntheticProvider` or built-in fallback generator.

## Framework Dependencies

For full comparison testing:

```bash
# VectorBT (for notebooks 01, 02, 05)
pip install vectorbt

# Backtrader (for notebooks 01, 02, 05)
pip install backtrader

# Zipline (for notebook 03)
pip install zipline-reloaded exchange-calendars
```

## Validation Results

All frameworks produce matching results:

| Framework | Scenarios | Status | Notes |
|-----------|-----------|--------|-------|
| VectorBT Pro | All 4 | ✅ EXACT MATCH | Same-bar signals + OHLC |
| VectorBT OSS | All 4 | ✅ EXACT MATCH | Next-bar execution |
| Backtrader | All 4 | ✅ EXACT MATCH | COO flag for next-bar |
| Zipline | All 4 | ✅ PASS | Within 0.002% tolerance |

### Validation Scenarios
1. **Long-only** - Basic MA crossover entries/exits
2. **Long-short** - Position flipping
3. **Stop-loss** - Exit on 5% loss
4. **Take-profit** - Exit at 10% gain

## Running Without External Frameworks

The notebooks are designed to run even without VectorBT, Backtrader, or Zipline installed. The ml4t.backtest sections will always execute, and comparison sections will be skipped gracefully if dependencies are missing.

## Execution Speed (Typical Results)

Benchmarks from `05_comprehensive_validation.ipynb` (5 assets, 3 years):

| Framework | Time (ms) | Notes |
|-----------|-----------|-------|
| VectorBT | ~15ms | Vectorized, fastest for signals |
| ml4t.backtest | ~50ms | Event-driven, flexible |
| Backtrader | ~300ms | Feature-rich, slower |
