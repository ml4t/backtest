# VectorBT Pro Initial Findings

## Overview
VectorBT Pro is a closed-source, commercial backtesting framework (version 2025.7.27) with 320,775 lines of Python code across 271 files. It's a significant enhancement over the open-source VectorBT.

## Key Statistics
- **Code Size**: 320,775 lines of Python
- **Files**: 271 Python files
- **Performance**: 188,869 trades/second (after JIT warm-up)
- **Speed**: 206x faster than standard VectorBT, 22x faster than QEngine
- **License**: Commercial, proprietary (no redistribution allowed)

## Package Structure

```
vectorbtpro/
├── base/           # Core framework components
├── data/           # Data loading and management
│   └── custom/     # Integrations (Alpaca, Binance, CCXT, etc.)
├── generic/        # Generic operations and accessors
├── indicators/     # Technical indicators
├── labels/         # Labeling for ML
├── ohlcv/          # OHLC data handling
├── portfolio/      # Portfolio simulation engine
├── px/             # Plotly Express integrations
├── records/        # Trade records and analysis
├── registries/     # Plugin registries
├── returns/        # Return analytics
├── signals/        # Signal generation
└── utils/          # Utilities and helpers

```

## Core Components Identified

### 1. Data Layer (`data/`)
- **Multiple data sources**: Alpaca, Binance, CCXT, TV (TradingView), NDL, FinPy
- **Storage backends**: HDF5, DuckDB, CSV, Feather, SQL
- **Remote data fetching**: Built-in updater and saver
- **Synthetic data**: GBM (Geometric Brownian Motion) generator

### 2. Portfolio Engine (`portfolio/`)
- Core backtesting logic
- Order execution simulation
- Position tracking
- Performance metrics

### 3. Indicators (`indicators/`)
- Technical analysis indicators
- Custom indicator framework

### 4. Machine Learning Support (`labels/`)
- Labeling framework for ML
- Integration with feature engineering

### 5. Visualization (`px/`)
- Plotly Express integration
- Interactive charts and dashboards

### 6. Records System (`records/`)
- Trade record management
- Performance analytics
- Detailed execution tracking

## Performance Architecture

### JIT Compilation
- First run: 0.369 seconds for 2,475 trades
- After JIT: 0.013 seconds (28x speedup)
- Heavy use of Numba for core loops

### Vectorization
- Fully vectorized operations
- NumPy/Pandas at core
- Batch processing of signals

### Memory Efficiency
- Columnar data storage
- Lazy evaluation where possible
- Efficient record structures

## Key Features Observed

1. **Multi-Asset Support**
   - `group_by=True` for portfolio grouping
   - `cash_sharing=True` for shared capital
   - Proper position sizing in shares

2. **Advanced Metrics**
   - Sharpe ratio
   - Sortino ratio
   - Maximum drawdown
   - Win rate
   - Expectancy
   - MAE (Maximum Adverse Excursion)

3. **Data Integrations**
   - 10+ data source integrations
   - Real-time and historical data
   - Multiple storage formats

4. **Optimization Framework**
   - Parameter optimization
   - Walk-forward analysis
   - Monte Carlo simulations

## Comparison with QEngine

### VectorBT Pro Advantages:
1. **Speed**: 22x faster for vectorized strategies
2. **Maturity**: Production-ready with extensive testing
3. **Integrations**: 10+ data sources out-of-box
4. **Visualization**: Rich Plotly-based visualizations
5. **Documentation**: Comprehensive (though proprietary)

### QEngine Advantages:
1. **Event-Driven**: More realistic order execution
2. **ML Integration**: Native support for ML strategies
3. **Open Source**: Full transparency and customization
4. **Architecture**: Clean, modern design
5. **Extensibility**: Everything pluggable via ABCs

### Different Use Cases:
- **VectorBT Pro**: Best for vectorized strategies, parameter optimization, rapid prototyping
- **QEngine**: Best for complex ML strategies, realistic simulation, custom execution logic

## Technical Insights

### File Analysis Highlights:
- `_settings.py`: 85,599 lines - massive configuration system
- `_typing.py`: 20,270 lines - comprehensive type system
- `mcp_server.py`: 20,570 lines - Model Context Protocol server
- `accessors.py`: 11,530 lines - Pandas accessor extensions

### Architecture Patterns:
1. **Accessor Pattern**: Extends pandas DataFrames with `.vbt` accessor
2. **Registry Pattern**: Plugin system for indicators and data sources
3. **Builder Pattern**: Portfolio construction with method chaining
4. **Strategy Pattern**: Pluggable components throughout

## Next Steps for Deep Dive

1. **Portfolio Engine Analysis**: Understand exact execution logic
2. **Performance Secrets**: How does it achieve 189K trades/sec?
3. **Numba Usage**: Identify JIT-compiled hot paths
4. **Data Pipeline**: How data flows through the system
5. **Signal Processing**: Entry/exit signal optimization
6. **Memory Management**: How it handles large datasets
7. **Parallel Processing**: Multi-core utilization strategies

## Initial Conclusions

VectorBT Pro is a highly optimized, commercial-grade backtesting framework that achieves exceptional performance through:
- Aggressive JIT compilation with Numba
- Pure vectorized operations
- Efficient memory layout
- Extensive caching

While it excels at speed for vectorized strategies, it operates in a different paradigm than event-driven simulators like QEngine. The two serve complementary purposes in a quant's toolkit.

## Files to Examine in Detail

Priority files for deep analysis:
1. `/portfolio/base.py` - Core portfolio logic
2. `/portfolio/nb/` - Numba implementations
3. `/data/base.py` - Data pipeline architecture
4. `/records/` - Trade execution records
5. `/utils/jitting.py` - JIT compilation strategies
6. `/_settings.py` - Configuration system
