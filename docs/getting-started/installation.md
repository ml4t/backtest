# Installation

## Requirements

- Python 3.9 or higher
- Polars 0.20+
- NumPy

## Install from PyPI

```bash
pip install ml4t-backtest
```

## Install from Source

```bash
git clone https://github.com/stefan-jansen/ml4t-backtest.git
cd ml4t-backtest
pip install -e .
```

## Verify Installation

```python
from ml4t.backtest import Engine, Strategy, BacktestConfig

print("ml4t-backtest installed successfully!")
```

## Optional: VectorBT Pro Validation

To validate results against VectorBT Pro:

```bash
pip install vectorbt-pro
```

Then use the validation utilities:

```python
from ml4t.backtest.validation import validate_against_vectorbt

is_valid = validate_against_vectorbt(ml4t_result, vbt_result)
```
