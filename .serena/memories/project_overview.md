# ml4t.backtest Project Overview

## Purpose
ml4t.backtest is a state-of-the-art event-driven backtesting engine designed for machine learning-driven trading strategies. It's part of the QuantLab monorepo ecosystem.

## Tech Stack
- **Language**: Python 3.12+
- **Core Libraries**: 
  - Polars (DataFrames, 10-100x faster than Pandas)
  - NumPy (numerical computations)
  - Numba (JIT compilation for hot paths)
  - Arrow (zero-copy data transfer)
- **Testing**: pytest, hypothesis (property-based testing)
- **Code Quality**: ruff (formatting and linting), mypy (type checking)
- **Optional**: Rust extensions for core event loop

## Architecture
- **Event-Driven Core**: Realistic simulation with ML support
- **Hybrid Approach**: Event-driven execution, vectorized analytics
- **Point-in-Time Correctness**: Architectural guarantees against data leakage
- **Components**:
  - Core: Event system, clock, types
  - Data: Data feeds and schemas
  - Strategy: Strategy framework
  - Execution: Order execution (broker simulation)
  - Portfolio: Portfolio management and tracking
  - Reporting: Output generation

## Integration Context
Part of QuantLab monorepo with:
- **qfeatures**: Feature engineering (upstream)
- **qeval**: Statistical validation (upstream)
- **qdata**: Market data management
- **ml4t.backtest**: This project - backtesting engine

## Key Design Principles
1. Point-in-Time correctness (no data leakage)
2. Performance first (Polars, Numba, optional Rust)
3. Extensibility (pluggable components via ABCs)
4. Repository tidiness (clear organization)
5. Code quality (modern tooling)
6. Test-driven development (80% minimum coverage)