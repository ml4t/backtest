# ml4t.backtest

**Production-ready** event-driven backtesting engine for ML-driven trading strategies with architectural guarantees against data leakage. All critical issues resolved (September 2025).

## Installation

```bash
# Development setup (from monorepo root)
make setup
make test-backtest

# Or standalone
pip install -e .
```

## Quick Start

```python
from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.data import ParquetDataFeed
from ml4t.backtest.execution import MarketOrder

# Create strategy
class MomentumStrategy(Strategy):
    def on_market_event(self, event, context):
        # Access point-in-time safe data
        if context.signals['momentum'] > 0.02:
            self.submit_order(MarketOrder("AAPL", 100, "BUY"))

# Run backtest
engine = BacktestEngine(
    data_feed=ParquetDataFeed("data.parquet"),
    strategy=MomentumStrategy(),
    initial_capital=100_000
)

results = engine.run()
print(f"Sharpe: {results.sharpe_ratio:.2f}, Return: {results.total_return:.1%}")
```

## Key Features

- **Event-Driven Core**: Point-in-time correctness with no data leakage
- **Temporal Accuracy**: Execution delay prevents lookahead bias
- **ML Signal Integration**: First-class support for ML predictions with helper methods (NEW)
- **Context-Aware Strategies**: Market-wide indicators (VIX, regime) for sophisticated logic (NEW)
- **Advanced Orders**: Market, Limit, Stop, Bracket with realistic execution
- **Execution Models**: Slippage (7 models), Commission (9 models), Market Impact
- **Multi-Asset Support**: Synchronized multi-feed data handling
- **Performance**: 8,000+ events/sec, Polars-based, memory-efficient ContextCache (2-5x savings)
- **Robust Execution**: No negative fills, proper cash constraints
- **Validation**: 100% agreement with VectorBT, 535 unit tests

## Architecture

```
DataFeed → EventBus → Strategy → Orders → Broker → Fills → Portfolio
              ↑                                              ↓
            Clock (Time Control)                       Performance
```

**Core Components:**
- `EventBus`: Priority-queue event routing with ~100k events/sec
- `Clock`: Centralized time control preventing data leakage  
- `Strategy`: Base class with lifecycle hooks (on_start, on_market_event, on_fill)
- `Broker`: Realistic order matching with slippage/impact models
- `Portfolio`: Position tracking, P&L, and metrics calculation

## ML Signal Integration

**New in November 2025:** ml4t.backtest now treats ML predictions as first-class citizens.

```python
from ml4t.backtest import BacktestEngine, Strategy
from ml4t.backtest.data import ParquetDataFeed

# 1. Data with ML predictions (pre-computed)
data = pl.DataFrame({
    "timestamp": [...],
    "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...],
    "prediction": [0.72, 0.85, 0.61, ...],  # ML signal: prob of up move
    "confidence": [0.88, 0.92, 0.75, ...],  # Model confidence
})

# 2. Extract signals via DataFeed
feed = ParquetDataFeed(
    path="data.parquet",
    asset_id="AAPL",
    signal_columns=["prediction", "confidence"],  # Extract as signals
)

# 3. Use in strategy with helper methods
class MLStrategy(Strategy):
    def on_market_event(self, event, context=None):
        # Access ML signals
        prediction = event.signals["prediction"]
        confidence = event.signals["confidence"]

        # Use helper methods for clean code
        if prediction > 0.7 and confidence > 0.8:
            self.size_by_confidence(  # Kelly-like sizing
                asset_id=event.asset_id,
                confidence=confidence,
                max_percent=0.20,
                price=event.close,
            )
        elif self.get_unrealized_pnl_pct(event.asset_id) and \
             self.get_unrealized_pnl_pct(event.asset_id) >= 0.15:
            self.close_position(event.asset_id)  # Take profit at 15%

# 4. Run backtest with context (market-wide indicators)
engine = BacktestEngine(
    data_feed=feed,
    strategy=MLStrategy(),
    context_data={"VIX": [...], "regime": [...]},  # Optional context
    initial_capital=100_000,
)

results = engine.run()
```

**See comprehensive guide:** [docs/ml_signals.md](docs/ml_signals.md)

**Features:**
- 9 helper methods (buy_percent, size_by_confidence, rebalance_to_weights, etc.)
- Context integration for market-wide indicators (VIX, SPY, regime)
- Memory-efficient ContextCache (2-5x savings for multi-asset strategies)
- Comprehensive test fixtures for rapid ML strategy development

## Usage Examples

### Order Types & Execution
```python
# Orders with realistic execution and temporal accuracy
order = qe.LimitOrder("AAPL", 100, "BUY", limit_price=150.0)
broker = SimulationBroker(
    slippage=LinearImpactSlippage(0.1),
    commission=PercentageCommission(0.001),  # 10bps
    execution_delay=True  # Prevents lookahead bias (default)
)
```

```python
class MLStrategy(qe.Strategy):
    def on_event(self, event):
        # Point-in-time safe ML predictions
        if event.event_type == EventType.MARKET:
            signal = self.get_signal(event.asset_id)
            if signal > 0.6:
                self.submit_order(qe.MarketOrder(event.asset_id, 100, "BUY"))
```

## QuantLab Integration

```python
# Future integration (in development)
from qfeatures import Pipeline
from qeval import PurgedWalkForwardCV

# Feature engineering → Model validation → Backtesting
features = Pipeline().fit_transform(data)
validated_model = qeval.validate(model, features)
results = ml4t.backtest.backtest(validated_model, features)
```

## Recent Updates

### November 2025: ML Signal Integration (Phase 1 & 1b)

**ML predictions as first-class citizens:**

- ✅ **Signal Support**: Unlimited ML predictions via `event.signals` dict
- ✅ **Context Integration**: Market-wide indicators (VIX, regime) with ContextCache
- ✅ **Helper Methods**: 9 methods for clean strategy code (buy_percent, size_by_confidence, etc.)
- ✅ **Test Fixtures**: 6 market scenarios with realistic ML predictions
- ✅ **Performance**: 2-5x memory savings validated (ContextCache benchmarks)
- ✅ **Examples**: Complete ML strategy example (examples/ml_strategy_example.py)
- ✅ **Test Coverage**: 535 tests (81% coverage, up from 77%)

See [docs/ml_signals.md](docs/ml_signals.md) for comprehensive ML integration guide.

### September 2025: Critical Issues Resolved

**All Critical Issues Resolved** - ml4t.backtest is now production-ready:

- ✅ **Event Flow Fixed**: Complete event routing from market data to portfolio
- ✅ **Temporal Accuracy**: Execution delay prevents lookahead bias
- ✅ **Multi-Feed Sync**: Stable ordering for multiple data feeds
- ✅ **P&L Calculations**: Clarified for all asset classes (options, FX, crypto)
- ✅ **Cash Constraints**: Robust handling prevents negative fill quantities
- ✅ **Corporate Actions**: Integrated stock splits, dividends processing
- ✅ **Test Coverage**: Comprehensive edge case and integration testing

See [docs/DELIVERY_SUMMARY.md](docs/DELIVERY_SUMMARY.md) for detailed fix documentation.

## Development

See [CLAUDE.md](CLAUDE.md) for development guidelines, code standards, and contributing instructions.

## License

Apache License 2.0
