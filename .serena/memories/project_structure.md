# ml4t.backtest Project Structure

## Directory Layout
```
ml4t.backtest/
├── src/ml4t.backtest/           # Source code
│   ├── core/              # Event system, clock, types
│   │   ├── event.py       # Event definitions
│   │   ├── event_bus.py   # Event bus implementation
│   │   ├── clock.py       # Time management
│   │   └── types.py       # Type definitions
│   ├── data/              # Data feeds and schemas
│   │   ├── feed.py        # DataFeed ABC
│   │   ├── parquet.py     # Parquet data feed
│   │   └── csv_feed.py    # CSV data feed
│   ├── strategy/          # Strategy framework
│   │   ├── base.py        # Strategy ABC
│   │   └── examples/      # Example strategies
│   ├── execution/         # Order execution
│   │   └── broker.py      # SimulationBroker
│   ├── portfolio/         # Portfolio management
│   │   └── simple.py      # SimplePortfolio
│   ├── reporting/         # Output generation
│   │   └── reporter.py    # InMemoryReporter
│   └── engine.py          # Main BacktestEngine
├── tests/                 # Test suite
│   ├── unit/              # Fast, isolated tests
│   ├── integration/       # Component interaction tests
│   ├── scenarios/         # Golden scenario tests
│   ├── comparison/        # Backtester comparison tests
│   ├── validation/        # Validation tests
│   └── conftest.py        # Pytest configuration
├── docs/                  # Documentation
│   ├── architecture/      # System design docs
│   └── guides/           # How-to guides
├── examples/              # Example strategies
├── resources/             # Reference implementations (read-only)
│   ├── backtrader-master/
│   ├── vectorbt.pro-main/
│   └── zipline-reloaded/
├── .claude/               # Claude workspace
│   ├── planning/          # Plans and roadmaps
│   ├── reference/         # Design documents
│   └── sprints/          # Sprint tracking
├── CLAUDE.md             # Project guidelines
├── README.md             # User documentation
└── pyproject.toml        # Python configuration
```

## Key Files
- `src/ml4t.backtest/engine.py` - Main BacktestEngine class
- `src/ml4t.backtest/core/event_bus.py` - Event-driven architecture core
- `src/ml4t.backtest/core/clock.py` - Time management for PIT correctness
- `tests/conftest.py` - Shared test fixtures
- `CLAUDE.md` - Development guidelines
- `.claude/reference/ARCHITECTURE.md` - Architecture documentation
- `.claude/reference/SIMULATION.md` - Simulation design

## Important Patterns
1. All components inherit from ABCs (Abstract Base Classes)
2. Event-driven communication via EventBus
3. Clock controls all time advancement
4. Strategies receive PIT-safe data views
5. Everything is pluggable and extensible

## File Naming
- Python modules: snake_case.py
- Test files: test_<module>.py
- Documentation: UPPER_CASE.md for important docs
- Examples: descriptive_name_example.py
