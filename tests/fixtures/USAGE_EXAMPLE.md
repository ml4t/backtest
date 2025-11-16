# Quick Usage Example

## Simple Test with ML Signals

```python
# tests/unit/test_my_strategy.py

def test_ml_strategy_basic(ml_signal_data):
    """Test basic ML strategy with neutral market."""
    from ml4t.backtest import BacktestEngine, Strategy
    from ml4t.backtest.data.feed import ParquetDataFeed

    data_path, df = ml_signal_data

    class MyMLStrategy(Strategy):
        def on_event(self, event):
            pass

        def on_market_event(self, event, context=None):
            # Access ML signals
            prediction = event.signals.get("prediction", 0.0)
            confidence = event.signals.get("confidence", 0.0)

            # Simple entry logic
            if prediction > 0.7 and confidence > 0.8:
                self.buy_percent("TEST", 0.10, event.close)

    # Create data feed
    feed = ParquetDataFeed(
        path=data_path,
        asset_id="TEST",
        timestamp_column="timestamp",
        signal_columns=["prediction", "confidence"],
    )

    # Run backtest
    engine = BacktestEngine(
        data_feed=feed,
        strategy=MyMLStrategy(),
        initial_capital=100000.0,
    )

    results = engine.run()
    assert results["final_value"] > 0
```

## Test with Market Context (VIX, Regime)

```python
def test_ml_strategy_with_context(bull_market_data):
    """Test ML strategy with VIX filtering."""
    from ml4t.backtest import BacktestEngine, Strategy
    from ml4t.backtest.data.feed import ParquetDataFeed

    data_path, context = bull_market_data

    class VIXFilteredStrategy(Strategy):
        def on_event(self, event):
            pass

        def on_market_event(self, event, context=None):
            # Check VIX from context
            vix = context.get("VIX", 0.0) if context else 0.0

            # Only trade when VIX < 25
            if vix > 25:
                return

            # ML signals
            prediction = event.signals.get("prediction", 0.0)
            confidence = event.signals.get("confidence", 0.0)

            if prediction > 0.6 and confidence > 0.75:
                self.buy_percent("TEST", 0.15, event.close)

    feed = ParquetDataFeed(
        path=data_path,
        asset_id="TEST",
        timestamp_column="timestamp",
        signal_columns=["prediction", "confidence"],
    )

    engine = BacktestEngine(
        data_feed=feed,
        strategy=VIXFilteredStrategy(),
        context_data=context,  # ← Pass context!
        initial_capital=100000.0,
    )

    results = engine.run()
    assert results["events_processed"] > 0
```

## Test All Scenarios (Parameterized)

```python
def test_strategy_robustness(ml_data_scenario):
    """Test runs 6 times - once per scenario."""
    from ml4t.backtest import BacktestEngine, Strategy
    from ml4t.backtest.data.feed import ParquetDataFeed

    data_path, context, scenario_name = ml_data_scenario

    print(f"\nTesting scenario: {scenario_name}")

    class RobustStrategy(Strategy):
        def on_event(self, event):
            pass

        def on_market_event(self, event, context=None):
            prediction = event.signals.get("prediction", 0.0)
            if prediction > 0.65:
                self.buy_percent("TEST", 0.10, event.close)

    feed = ParquetDataFeed(
        path=data_path,
        asset_id="TEST",
        timestamp_column="timestamp",
        signal_columns=["prediction", "confidence"],
    )

    engine = BacktestEngine(
        data_feed=feed,
        strategy=RobustStrategy(),
        context_data=context,
        initial_capital=100000.0,
    )

    results = engine.run()

    # Strategy should work in all scenarios
    assert results["events_processed"] > 0
    assert results["final_value"] > 0

    print(f"{scenario_name}: return = {results['total_return']:.2f}%")
```

## Run Tests

```bash
# Run all fixture tests
pytest tests/unit/test_ml_fixtures.py -v

# Run your strategy test
pytest tests/unit/test_my_strategy.py -v

# Run parameterized test (see all scenarios)
pytest tests/unit/test_my_strategy.py::test_strategy_robustness -v -s
```

## Available Fixtures

- `ml_signal_data` - Neutral market (basic)
- `context_data` - Market context (VIX, regime)
- `bull_market_data` - Bull market (positive drift, low VIX)
- `bear_market_data` - Bear market (negative drift, high VIX)
- `high_volatility_data` - High volatility (VIX > 30)
- `low_volatility_data` - Low volatility (VIX < 15)
- `trending_market_data` - Strong trend
- `mean_reverting_data` - Choppy/oscillating
- `ml_data_scenario` - Parameterized (all scenarios)

See `README.md` for complete documentation.
