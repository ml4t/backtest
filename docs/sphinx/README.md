# ml4t.backtest Sphinx Documentation

This directory contains the Sphinx-generated API documentation for ml4t.backtest.

## Building the Documentation

From the project root:

```bash
# Install documentation dependencies (if not already installed)
uv pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

# Build HTML documentation
.venv/bin/python -m sphinx -b html docs/sphinx/source docs/sphinx/build/html

# Or use make (if available)
cd docs/sphinx
make html
```

## Viewing the Documentation

After building, open `docs/sphinx/build/html/index.html` in your browser:

```bash
# On Linux with xdg-open
xdg-open docs/sphinx/build/html/index.html

# On macOS
open docs/sphinx/build/html/index.html

# Or use a simple HTTP server
cd docs/sphinx/build/html
python -m http.server 8000
# Then visit http://localhost:8000
```

## Documentation Structure

```
docs/sphinx/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation page
│   └── modules/             # Module-specific documentation
│       ├── core.rst         # Core types, events, assets
│       ├── execution.rst    # Execution engine, orders, fills
│       ├── portfolio.rst    # Portfolio management
│       ├── strategy.rst     # Strategy base classes
│       ├── data.rst         # Data feeds and schemas
│       └── reporting.rst    # Results and reporting
├── build/html/              # Generated HTML (git-ignored)
├── Makefile                 # Build convenience script
└── README.md                # This file
```

## Features

- **Auto-generated API documentation** from docstrings using Sphinx autodoc
- **Type hints support** via sphinx-autodoc-typehints
- **Google/NumPy docstring styles** supported via Napoleon
- **Read the Docs theme** for professional appearance
- **Cross-references** to Python, NumPy, Pandas, and Polars documentation
- **Source code links** via viewcode extension

## Maintenance

When adding new modules or reorganizing code:

1. Update the relevant `.rst` file in `source/modules/`
2. Rebuild the documentation
3. Verify the output in your browser

The documentation is automatically generated from docstrings in the source code, so improving docstrings directly improves the documentation.

## Docstring Style

ml4t.backtest uses Google-style docstrings. Example:

```python
def calculate_quantity(
    self,
    price: float,
    available_cash: float,
    commission_model: CommissionModel,
) -> float:
    """Calculate position size for an order.

    Args:
        price: Current market price (base price before slippage)
        available_cash: Available cash for the trade
        commission_model: Commission model to estimate fees

    Returns:
        Quantity to trade (always positive, sign handled by order side)

    Raises:
        ValueError: If insufficient cash to cover fees or minimum quantity

    Example:
        >>> sizer = VectorBTInfiniteSizer(granularity=0.001)
        >>> qty = sizer.calculate_quantity(50000.0, 10000.0, NoCommission(), NoSlippage(), order)
        >>> print(f"Size: {qty:.3f} BTC")
        Size: 0.200 BTC
    """
```

## Coverage

The documentation includes:
- 5 core modules (types, events, assets, clock, constants)
- 9 execution modules (broker, orders, sizing, commission, slippage, fills, tracking)
- 4 portfolio modules (portfolio, simple, accounting, margin)
- 3 strategy modules (base, adapters, crypto basis)
- 3 data modules (feed, schemas, asset registry)
- 4 reporting modules (base, HTML, Parquet, reporter)

**Total: 28 modules documented with full API reference**
