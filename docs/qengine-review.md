This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.
The content has been processed where line numbers have been added.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: README.md, CLAUDE.md, src/ml4t.backtest/__init__.py, src/ml4t.backtest/engine.py, src/ml4t.backtest/core/*.py, src/ml4t.backtest/data/*.py, src/ml4t.backtest/execution/*.py, src/ml4t.backtest/portfolio/*.py, src/ml4t.backtest/strategy/*.py, src/ml4t.backtest/reporting/*.py
- Files matching these patterns are excluded: .venv/**, docs/**, examples/**, tests/**, resources/**, htmlcov/**, __pycache__/**, *.pyc, benchmarks/**, config/**, src/**/*.py
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Line numbers have been added to the beginning of each line
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
CLAUDE.md
README.md
```

# Files

## File: CLAUDE.md
````markdown
  1: # CLAUDE.md - ml4t.backtest Development Guidelines
  2:
  3: ## Project Overview
  4:
  5: ml4t.backtest is a state-of-the-art event-driven backtesting engine designed for machine learning-driven trading strategies. Built on modern Python tooling (Polars, Arrow, Numba), it provides institutional-grade simulation capabilities while preventing data leakage through architectural guarantees.
  6:
  7: ## Key Documentation
  8:
  9: ### Architecture & Design
 10: - **Engine Architecture**: `.claude/reference/ARCHITECTURE.md`
 11: - **Simulation Design**: `.claude/reference/SIMULATION.md`
 12: - **Functionality Inventory**: `FUNCTIONALITY_INVENTORY.md` - Complete list of implemented features (UPDATE EACH SPRINT)
 13: - **Integration with Monorepo**: See `/home/stefan/quantlab/CLAUDE.md`
 14:
 15: ### Session Management
 16: - **Current Session**: Managed at monorepo level in `/home/stefan/quantlab/.claude/memory/CURRENT_SESSION_CONTEXT.md`
 17: - **Integration Tracking**: `/home/stefan/quantlab/.claude/tracking/INTEGRATION_TRACKER.md`
 18:
 19: ## Core Principles
 20:
 21: 1. **Point-in-Time Correctness**: Architectural guarantees against data leakage
 22: 2. **Performance First**: Polars columnar data, Numba JIT, optional Rust extensions
 23: 3. **Extensibility**: Everything pluggable through well-defined ABCs
 24: 4. **Repository Tidiness**: Clear separation of planning, documentation, and code
 25: 5. **Code Quality**: State-of-the-art tooling (Ruff, MyPy, pre-commit)
 26: 6. **Test-Driven Development**: Write tests first, minimum 80% coverage
 27:
 28: ## Monorepo Context
 29:
 30: ml4t.backtest is part of the **QuantLab monorepo**, a comprehensive quantitative finance ecosystem:
 31:
 32: - **qfeatures**: Feature engineering and labeling (upstream)
 33: - **qeval**: Statistical validation (upstream)
 34: - **ml4t.backtest** (this project): Event-driven backtesting engine
 35:
 36: ### Development Setup
 37:
 38: Follows monorepo conventions:
 39:
 40: ```bash
 41: # From monorepo root
 42: make setup       # One-time setup
 43: make test-qng    # Test ml4t.backtest specifically
 44: make quality     # Format, lint, type-check all projects
 45: ```
 46:
 47: ### Calendar Integration
 48:
 49: - **Exchange Calendars**: Follow `/home/stefan/quantlab/.claude/reference/CALENDAR_USAGE_GUIDE.md`
 50: - **Sensible Defaults**: Auto-detect calendar based on asset type
 51: - **Optional Alignment**: Support both sparse and dense time series
 52:
 53: ### Integration Points
 54:
 55: ml4t.backtest consumes:
 56: 1. **Features from qfeatures**: Standardized DataFrames with features and labels
 57: 2. **Models from qeval**: Validated ML models ready for backtesting
 58:
 59: ```python
 60: # Expected input from qfeatures/qeval pipeline
 61: {
 62:     "event_time": pl.Datetime("ns"),    # Point-in-time anchor
 63:     "asset_id": pl.Utf8,                # Instrument identifier
 64:     "features...": pl.Float64,          # Feature columns
 65:     "signal": pl.Float64,               # Model predictions
 66: }
 67: ```
 68:
 69: ## Project Structure
 70:
 71: ```
 72: ml4t.backtest/
 73: ├── CLAUDE.md               # This file - project guidelines
 74: ├── README.md               # User-facing project overview
 75: ├── LICENSE                 # Apache 2.0 license
 76: ├── pyproject.toml          # Modern Python packaging
 77: ├── Makefile                # Developer workflow commands
 78: ├── .pre-commit-config.yaml # Code quality hooks
 79: ├── .claude/                # Claude AI workspace (NEVER in root!)
 80: │   ├── planning/           # Implementation plans, roadmaps
 81: │   │   ├── IMPLEMENTATION_PLAN.md
 82: │   │   ├── FRAMEWORK_ANALYSIS.md
 83: │   │   └── ROADMAP.md
 84: │   ├── reference/          # Design reviews, architecture decisions
 85: │   │   ├── DESIGN.md       # Original design specification
 86: │   │   └── ARCHITECTURE_REVIEW.md
 87: │   ├── sprints/            # Sprint-based development
 88: │   └── PROJECT_GUIDELINES.md # Repo organization rules
 89: ├── docs/                   # User documentation
 90: │   ├── architecture/       # System design
 91: │   │   ├── ARCHITECTURE.md
 92: │   │   └── PIT_AND_STATE.md
 93: │   └── guides/            # How-to guides
 94: │       └── MIGRATION_GUIDE.md
 95: ├── src/ml4t.backtest/           # Source code
 96: │   ├── core/              # Event system, clock, types
 97: │   ├── data/              # Data feeds and schemas
 98: │   ├── strategy/          # Strategy framework
 99: │   ├── execution/         # Order execution (TODO)
100: │   ├── portfolio/         # Portfolio management (TODO)
101: │   └── reporting/         # Output generation (TODO)
102: ├── tests/                 # Test suite
103: │   ├── unit/              # Fast, isolated tests
104: │   ├── integration/       # Component interaction tests
105: │   ├── scenarios/         # Golden scenario tests
106: │   ├── comparison/        # Backtester comparison tests
107: │   └── conftest.py        # Pytest configuration
108: ├── examples/              # Example strategies
109: ├── benchmarks/            # Performance benchmarks
110: └── resources/             # Reference implementations (read-only)
111:     ├── backtrader-master/ # Backtrader source
112:     ├── vectorbt.pro-main/ # VectorBT Pro source
113:     └── zipline-reloaded/  # Zipline source
114: ```
115:
116: ## Critical Design Decisions (from DESIGN.md)
117:
118: ### 1. Architecture Choice
119: - **Event-Driven Core**: Chosen over pure vectorized for realism and ML support
120: - **Hybrid Approach**: Event-driven for execution, vectorized for analytics
121: - **Rationale**: Balances performance with correctness for complex strategies
122:
123: ### 2. Technology Stack
124: - **Polars over Pandas**: 10-100x performance improvement
125: - **Arrow Format**: Zero-copy data transfer, language interoperability
126: - **Numba JIT**: Hot path optimization without leaving Python
127: - **Optional Rust**: Core event loop for C-level performance
128: - **Comparison Backtesters**: VectorBT Pro, Zipline-Reloaded, Backtrader (for validation)
129:
130: ### 3. Data Leakage Prevention
131: - **Clock as Master**: Controls all time advancement
132: - **Immutable PIT Views**: Strategies receive time-bounded data snapshots
133: - **No Direct Access**: Strategies cannot access raw data feeds
134:
135: ### 4. Migration Strategy (Simplified per Review)
136: - **No Full Compatibility**: Migration helpers, not compatibility layers
137: - **Documentation Focus**: Clear guides over complex code
138: - **Better UX**: Intuitive API will drive adoption
139:
140: ## Development Workflow
141:
142: ### 1. Code Quality Standards
143:
144: **ALWAYS run before ANY code changes:**
145: ```bash
146: make format      # Ruff formatting
147: make lint        # Ruff linting
148: make type-check  # MyPy validation
149: ```
150:
151: **Or all at once:**
152: ```bash
153: make check       # Format + lint + type-check
154: ```
155:
156: **Pre-commit hooks (MANDATORY):**
157: ```bash
158: make dev-install  # Installs pre-commit hooks
159: pre-commit run --all-files  # Manual run
160: ```
161:
162: ### 2. Repository Organization Rules
163:
164: **ABSOLUTE RULES:**
165: 1. **Root Directory**: ONLY README.md, LICENSE, pyproject.toml, Makefile, configs
166: 2. **Planning Docs**: ALWAYS in `.claude/planning/`
167: 3. **Reference Docs**: ALWAYS in `.claude/reference/`
168: 4. **User Docs**: ALWAYS in `docs/`
169: 5. **NEVER**: Leave work-in-progress files in root
170: 6. **NEVER**: Create compatibility layers without explicit request
171:
172: ### 3. Test-Driven Development
173:
174: **Workflow:**
175: 1. Write test FIRST in `tests/unit/` or `tests/integration/`
176: 2. Run test to see it fail: `pytest tests/unit/test_feature.py::test_specific -v`
177: 3. Implement minimal code to pass
178: 4. Refactor while keeping tests green
179: 5. Run full suite: `make test`
180:
181: **Testing Standards:**
182: - Minimum 80% coverage (enforced)
183: - Use pytest fixtures from `conftest.py`
184: - Mock external dependencies
185: - Property-based tests for algorithms
186:
187: ### 4. Performance Considerations
188:
189: **From DESIGN.md - Performance Targets:**
190: - >1M events/second for simple strategies
191: - <30 seconds for 1 year tick data
192: - <10 seconds for 10 years daily data (1000 assets)
193: - <1GB memory for 10-year backtest
194:
195: **Optimization Priority:**
196: 1. Algorithmic complexity first
197: 2. Polars query optimization
198: 3. Numba JIT for tight loops
199: 4. Rust only for proven bottlenecks
200:
201: ### 5. Git Workflow
202:
203: **Branch Strategy:**
204: ```bash
205: git checkout -b feature/add-broker-simulation
206: git checkout -b fix/event-ordering-bug
207: git checkout -b docs/improve-pit-explanation
208: ```
209:
210: **Commit Messages:**
211: ```bash
212: git commit -m "feat: Add SimulationBroker with basic order matching
213:
214: - Implement Market and Limit order types
215: - Add order state machine
216: - Include unit tests with 95% coverage"
217: ```
218:
219: **NEVER commit unless:**
220: - All tests pass: `make test`
221: - Code quality checks pass: `make qa`
222: - No files in wrong directories
223:
224: ## Implementation Phases (from IMPLEMENTATION_PLAN.md)
225:
226: ### Phase 1: MVP Core Engine (Current)
227: - [x] Project structure and infrastructure
228: - [x] Core event system and Clock
229: - [x] Basic data feeds (Parquet, CSV)
230: - [x] Strategy framework
231: - [ ] Simple broker simulation
232: - [ ] Portfolio accounting
233: - [ ] Basic reporting
234:
235: ### Phase 2: Realistic Simulation
236: - [ ] Advanced order types (Stop, Bracket, OCO)
237: - [ ] Pluggable models (slippage, commission, impact)
238: - [ ] Corporate actions and calendars
239: - [ ] Futures support
240: - [ ] ML signal integration
241:
242: ### Phase 3: Performance & Differentiation
243: - [ ] Rust core optimization
244: - [ ] Almgren-Chriss market impact
245: - [ ] Portfolio optimization adapters
246: - [ ] TOML/YAML configuration
247:
248: ### Phase 4: Live Trading
249: - [ ] Broker adapters (Alpaca, IB)
250: - [ ] Real-time data feeds
251: - [ ] State persistence
252:
253: ## Common Patterns & Solutions
254:
255: ### Adding a New Component
256: 1. Define interface in appropriate module's `__init__.py`
257: 2. Create ABC in module directory
258: 3. Write comprehensive unit tests
259: 4. Implement concrete class
260: 5. Add integration tests
261: 6. Update documentation
262:
263: ### Data Feed Implementation
264: ```python
265: # Always follow this pattern from data/feed.py
266: class CustomDataFeed(DataFeed):
267:     def get_next_event(self) -> Optional[Event]:
268:         # Implementation
269:
270:     def peek_next_timestamp(self) -> Optional[datetime]:
271:         # Implementation
272:
273:     @property
274:     def is_exhausted(self) -> bool:
275:         # Implementation
276: ```
277:
278: ### Strategy Implementation
279: ```python
280: # Follow pattern from strategy/base.py
281: class CustomStrategy(Strategy):
282:     def on_start(self):
283:         # One-time setup
284:
285:     def on_event(self, event: Event, pit_data: PITData):
286:         # Core logic with PIT-safe data access
287: ```
288:
289: ## Key Technical Decisions
290:
291: ### From Architecture Review Feedback
292:
293: 1. **PIT Enforcement**: Immutable data views, not trust-based access
294: 2. **State Management**: Centralized PortfolioState with atomic updates
295: 3. **Migration Approach**: Helpers and docs, not full compatibility
296: 4. **Performance Trade-offs**: Correctness over speed, but optimize hot paths
297:
298: ### From Framework Analysis
299:
300: 1. **Avoid Backtrader's**: Metaclass magic, pure Python performance
301: 2. **Avoid Zipline's**: Heavy dependencies, Cython complexity
302: 3. **Avoid VectorBT's**: Sacrificing realism for speed
303: 4. **Adopt Best Practices**: Clean APIs, modern tooling, extensibility
304:
305: ## Testing Requirements
306:
307: ### Test Categories
308: - **Unit Tests**: `tests/unit/` - Fast, isolated, mocked
309: - **Integration Tests**: `tests/integration/` - Component interactions
310: - **Scenario Tests**: `tests/scenarios/` - Golden outputs
311: - **Benchmark Tests**: `tests/benchmarks/` - Performance tracking
312: - **Comparison Tests**: `tests/comparison/` - Validation against established backtesters
313:
314: ### Running Tests
315: ```bash
316: make test           # All tests
317: make test-unit      # Unit tests only
318: make test-cov       # With coverage report
319: pytest tests/unit/test_event_bus.py -v  # Specific test
320: ```
321:
322: ## Documentation Standards
323:
324: ### Docstrings (Google Style)
325: ```python
326: def calculate_returns(prices: np.ndarray, method: str = "simple") -> np.ndarray:
327:     """Calculate returns from price series.
328:
329:     Args:
330:         prices: Array of prices
331:         method: Calculation method ('simple' or 'log')
332:
333:     Returns:
334:         Array of returns
335:
336:     Raises:
337:         ValueError: If prices array is empty
338:     """
339: ```
340:
341: ### Type Hints (REQUIRED)
342: ```python
343: from typing import Optional, List, Dict
344: from datetime import datetime
345:
346: def process_events(
347:     events: List[Event],
348:     timestamp: datetime
349: ) -> Optional[Dict[str, Any]]:
350:     ...
351: ```
352:
353: ## Common Pitfalls to Avoid
354:
355: 1. **Don't Add to Root**: Keep it clean, use .claude/ and docs/
356: 2. **Don't Skip Tests**: TDD is mandatory
357: 3. **Don't Ignore Types**: MyPy must pass
358: 4. **Don't Create Deep Nesting**: Keep structure simple
359: 5. **Don't Use sys.path Hacks**: Use proper packaging
360: 6. **Don't Implement Full Compatibility**: Migration helpers only
361:
362: ## Quick Commands Reference
363:
364: ```bash
365: # Development
366: make dev-install    # Set up development environment
367: make format         # Auto-format code
368: make lint           # Check code style
369: make type-check     # Validate types
370: make test           # Run tests
371: make qa             # All quality checks
372:
373: # Git
374: git status          # Check changes
375: git add -p          # Stage selectively
376: git commit          # With descriptive message
377: git push            # After all checks pass
378:
379: # Testing
380: pytest -v           # Verbose test output
381: pytest -x           # Stop on first failure
382: pytest --pdb        # Debug on failure
383: pytest -k pattern   # Run matching tests
384:
385: # Profiling
386: python -m cProfile -o profile.stats examples/benchmark.py
387: py-spy record -o profile.svg -- python examples/benchmark.py
388: ```
389:
390: ## Session Continuity
391:
392: ### Starting a New Session
393: 1. Read `.claude/reference/DESIGN.md` for design context
394: 2. Check `docs/architecture/ARCHITECTURE.md` for implementation
395: 3. Review `.claude/planning/ROADMAP.md` for current status
396: 4. Continue from documented next steps
397:
398: ### Before Ending Session
399: 1. Update relevant tracking documents
400: 2. Commit completed work (if requested)
401: 3. Document next steps clearly
402: 4. Ensure no temporary files in root
403:
404: ## Notes
405:
406: - This document supersedes generic templates for ml4t.backtest
407: - Always prioritize correctness over performance
408: - Keep migration simple - good UX beats compatibility
409: - Reference DESIGN.md for any architectural questions
410: - Follow PROJECT_GUIDELINES.md for repo organization
````

## File: README.md
````markdown
  1: # ml4t.backtest
  2:
  3: ml4t.backtest is a state-of-the-art event-driven backtesting engine designed for high-performance backtesting of machine learning-driven trading strategies. It provides realistic market simulation with architectural guarantees against data leakage.
  4:
  5: ## Installation
  6:
  7: ```bash
  8: # From source (recommended)
  9: git clone <repository-url>
 10: cd ml4t.backtest
 11: pip install -e .
 12: ```
 13:
 14: ## Quick Start
 15:
 16: ```python
 17: import polars as pl
 18: import ml4t.backtest as qe
 19:
 20: # Load market data
 21: data_feed = qe.DataFeed.from_parquet("market_data.parquet")
 22:
 23: # Define a simple strategy
 24: class BuyAndHoldStrategy(qe.Strategy):
 25:     def on_market_data(self, event, pit_data):
 26:         if not self.portfolio.has_position("AAPL"):
 27:             order = qe.MarketOrder(
 28:                 asset_id="AAPL",
 29:                 quantity=100,
 30:                 side="BUY"
 31:             )
 32:             self.submit_order(order)
 33:
 34: # Create and run backtest
 35: engine = qe.BacktestEngine(
 36:     data_feed=data_feed,
 37:     strategy=BuyAndHoldStrategy(),
 38:     initial_capital=100000
 39: )
 40:
 41: results = engine.run()
 42: print(f"Final Value: ${results.final_value:,.2f}")
 43: print(f"Total Return: {results.total_return:.2%}")
 44: ```
 45:
 46: ## Key Features
 47:
 48: ### Event-Driven Architecture
 49: - **True Event Processing**: Bar-by-bar simulation with realistic timing
 50: - **Point-in-Time Data**: Architectural guarantees against look-ahead bias
 51: - **Event System**: Extensible event-driven framework for complex strategies
 52:
 53: ### Realistic Execution Simulation
 54: - **Order Types**: Market, Limit, Stop, Trailing Stop, Bracket orders
 55: - **Slippage Models**: 7 models including Almgren-Chriss market impact
 56: - **Commission Models**: 9 models including tiered and percentage-based
 57: - **Order Matching**: Realistic order book simulation
 58:
 59: ### Machine Learning Integration
 60: - **Strategy Adapters**: Bridge existing ML models to backtesting framework
 61: - **Model Lifecycle**: Handle model training, prediction, and updates
 62: - **Feature Pipeline**: Integration with QFeatures for online feature generation
 63:
 64: ### Performance & Accuracy
 65: - **Validated Results**: 100% agreement with VectorBT on identical strategies
 66: - **High Throughput**: 8,552 trades/second processing capability
 67: - **Memory Efficient**: Polars-based data handling for large datasets
 68:
 69: ## Architecture
 70:
 71: ml4t.backtest follows a clean event-driven architecture:
 72:
 73: ### Core Components
 74: ```python
 75: # Event system
 76: from ml4t.backtest.core import Event, EventBus, Clock
 77:
 78: # Data pipeline
 79: from ml4t.backtest.data import DataFeed
 80:
 81: # Strategy framework
 82: from ml4t.backtest.strategy import Strategy
 83:
 84: # Portfolio management
 85: from ml4t.backtest.portfolio import Portfolio
 86:
 87: # Order execution
 88: from ml4t.backtest.execution import Broker, Order
 89: ```
 90:
 91: ### Execution Pipeline
 92: 1. **Clock**: Controls simulation time advancement
 93: 2. **DataFeed**: Provides market data events
 94: 3. **Strategy**: Receives events and generates orders
 95: 4. **Broker**: Executes orders with realistic simulation
 96: 5. **Portfolio**: Tracks positions, cash, and performance
 97: 6. **Reporter**: Generates results and analytics
 98:
 99: ## Order Types
100:
101: ml4t.backtest supports sophisticated order types:
102:
103: ```python
104: # Market orders
105: market_order = qe.MarketOrder(asset_id="AAPL", quantity=100, side="BUY")
106:
107: # Limit orders
108: limit_order = qe.LimitOrder(
109:     asset_id="AAPL",
110:     quantity=100,
111:     side="BUY",
112:     limit_price=150.0
113: )
114:
115: # Stop orders
116: stop_order = qe.StopOrder(
117:     asset_id="AAPL",
118:     quantity=100,
119:     side="SELL",
120:     stop_price=140.0
121: )
122:
123: # Bracket orders (OCO)
124: bracket = qe.BracketOrder(
125:     parent=market_order,
126:     take_profit=155.0,
127:     stop_loss=145.0
128: )
129: ```
130:
131: ## Execution Models
132:
133: ### Slippage Models
134: ```python
135: from ml4t.backtest.execution.slippage import LinearImpactSlippage, VolumeShareSlippage
136:
137: # Linear market impact
138: slippage = LinearImpactSlippage(impact_coefficient=0.1)
139:
140: # Volume-based slippage
141: vol_slippage = VolumeShareSlippage(max_participation=0.1)
142: ```
143:
144: ### Commission Models
145: ```python
146: from ml4t.backtest.execution.commission import PercentageCommission, TieredCommission
147:
148: # Simple percentage
149: commission = PercentageCommission(rate=0.001)  # 10 basis points
150:
151: # Tiered pricing
152: tiered = TieredCommission({
153:     0: 0.005,      # First $100k
154:     100000: 0.003, # Next tier
155:     500000: 0.001  # Highest tier
156: })
157: ```
158:
159: ## Strategy Development
160:
161: ### Basic Strategy
162: ```python
163: class MomentumStrategy(qe.Strategy):
164:     def __init__(self, lookback=20):
165:         super().__init__()
166:         self.lookback = lookback
167:
168:     def on_market_data(self, event, pit_data):
169:         # Calculate momentum signal
170:         returns = pit_data.returns.tail(self.lookback)
171:         momentum = returns.mean()
172:
173:         if momentum > 0.01:  # Buy signal
174:             self.rebalance_to_target("AAPL", 0.5)  # 50% allocation
175:         elif momentum < -0.01:  # Sell signal
176:             self.close_position("AAPL")
177: ```
178:
179: ### ML Strategy Integration
180: ```python
181: from ml4t.backtest.strategy.adapters import MLStrategyAdapter
182:
183: class MLMomentumStrategy(MLStrategyAdapter):
184:     def predict(self, features):
185:         return self.model.predict(features.to_numpy())
186:
187:     def generate_orders(self, predictions, pit_data):
188:         for asset_id, signal in predictions.items():
189:             if signal > 0.6:
190:                 yield qe.MarketOrder(asset_id, 100, "BUY")
191:             elif signal < 0.4:
192:                 yield qe.MarketOrder(asset_id, 100, "SELL")
193: ```
194:
195: ## Performance Analysis
196:
197: ml4t.backtest provides comprehensive performance analytics:
198:
199: ```python
200: results = engine.run()
201:
202: # Portfolio metrics
203: print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
204: print(f"Max Drawdown: {results.max_drawdown:.2%}")
205: print(f"Win Rate: {results.win_rate:.1%}")
206:
207: # Trade analysis
208: print(f"Total Trades: {len(results.trades)}")
209: print(f"Avg Trade Return: {results.avg_trade_return:.2%}")
210:
211: # Generate reports
212: results.to_html("backtest_report.html")
213: results.to_parquet("detailed_trades.parquet")
214: ```
215:
216: ## Integration with QuantLab
217:
218: ml4t.backtest integrates seamlessly with other QuantLab libraries:
219:
220: ```python
221: import qfeatures as qf
222: import qeval as qe
223: import ml4t.backtest as qng
224:
225: # Feature engineering with QFeatures
226: features = qf.Pipeline([
227:     qf.features.microstructure.add_returns,
228:     qf.features.volatility.add_garch_features,
229:     qf.labeling.apply_triple_barrier
230: ]).transform(price_data)
231:
232: # Model validation with QEval
233: cv = qe.PurgedWalkForwardCV(n_splits=5)
234: model_results = qe.Evaluator(cv).evaluate(model, features)
235:
236: # Strategy backtesting with ml4t.backtest
237: strategy = qng.MLStrategyAdapter(model)
238: backtest_results = qng.BacktestEngine(strategy=strategy).run()
239: ```
240:
241: ## Validation & Testing
242:
243: ml4t.backtest has been extensively validated:
244:
245: - **154 unit tests** with comprehensive coverage
246: - **Cross-framework validation**: 100% agreement with VectorBT
247: - **Multi-asset testing**: 5,000 trades across 30 stocks
248: - **Performance benchmarking**: 8,552 trades/second processing
249:
250: ## Contributing
251:
252: 1. Install development dependencies: `pip install -e ".[dev]"`
253: 2. Run code quality checks: `ruff format . && ruff check . --fix && mypy src/ml4t.backtest`
254: 3. Run tests: `pytest tests/`
255: 4. Follow event-driven patterns and maintain type safety
256:
257: ## License
258:
259: Apache License 2.0
````
