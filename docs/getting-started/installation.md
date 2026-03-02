# Installation

## Requirements

- Python 3.11 or higher
- Polars
- PyYAML
- NumPy
- Pydantic

## Install from PyPI

```bash
pip install ml4t-backtest
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add ml4t-backtest
```

## Install from Source

```bash
git clone https://github.com/ml4t/ml4t-backtest.git
cd ml4t-backtest
pip install -e .
```

## Verify Installation

```python
from ml4t.backtest import Engine, Strategy, DataFeed, BacktestConfig, run_backtest
from ml4t.backtest import StopLoss, TakeProfit, TrailingStop, RuleChain

print("ml4t-backtest installed successfully!")
```

## Optional Dependencies

**Trading calendars** for session enforcement (skip weekends/holidays):

```bash
pip install pandas-market-calendars
```

**ml4t-diagnostic** for post-backtest analysis (tearsheets, trade analytics):

```bash
pip install ml4t-diagnostic
```

## Next Steps

- [Quickstart](quickstart.md) -- build and run your first backtest
