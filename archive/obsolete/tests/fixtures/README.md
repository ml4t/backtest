# ML Signal Test Fixtures

Reusable pytest fixtures for testing ML-based trading strategies.

## Overview

This package provides comprehensive test fixtures that generate realistic market data with ML predictions, confidence scores, and market context (VIX, regime indicators). These fixtures accelerate ML strategy testing by providing standardized, parameterized test data.

## Quick Start

```python
# tests/unit/test_my_ml_strategy.py

def test_my_strategy(ml_signal_data):
    """Basic usage: get data path and DataFrame."""
    data_path, df = ml_signal_data

    # data_path: Path to parquet file
    # df: Polars DataFrame with OHLCV + prediction + confidence

    assert "prediction" in df.columns
    assert "confidence" in df.columns
```

## Available Fixtures

### Core Fixtures

#### `ml_signal_data`
Basic ML signal data with neutral market conditions.

**Returns:** `tuple[Path, pl.DataFrame]`

**Characteristics:**
- 252 trading days (1 year)
- Neutral drift (slight positive bias)
- Moderate volatility (1.5% daily)
- ML accuracy: 75%

```python
def test_basic(ml_signal_data):
    data_path, df = ml_signal_data
    # Use in your test...
```

#### `context_data`
Market-wide context data (VIX, regime indicators).

**Returns:** `dict[datetime, dict]`

**Structure:**
```python
{
    datetime(2024, 1, 2, 9, 30): {
        "VIX": 20.5,
        "regime": "bull"
    },
    ...
}
```

```python
def test_with_context(ml_signal_data, context_data):
    data_path, _ = ml_signal_data
    # context_data provides VIX and regime for each timestamp
```

### Scenario-Specific Fixtures

All scenario fixtures return: `tuple[Path, dict[datetime, dict]]`

#### `bull_market_data`
Bull market conditions.

**Characteristics:**
- Positive drift: 0.08% daily (~20% annualized)
- Lower volatility: 1.2% daily
- High ML accuracy: 80%
- Low VIX: avg 15

```python
def test_bull_strategy(bull_market_data):
    data_path, context = bull_market_data
    # Test strategy in bull market...
```

#### `bear_market_data`
Bear market conditions.

**Characteristics:**
- Negative drift: -0.03% daily (~-7.5% annualized)
- Higher volatility: 2.5% daily
- Lower ML accuracy: 65%
- High VIX: avg 28

```python
def test_bear_strategy(bear_market_data):
    data_path, context = bear_market_data
    # Test strategy in bear market...
```

#### `high_volatility_data`
High volatility regime.

**Characteristics:**
- Neutral drift
- Very high volatility: 3.5% daily
- ML accuracy suffers: 60%
- VIX consistently > 30

```python
def test_high_vol(high_volatility_data):
    data_path, context = high_volatility_data
    # Test during volatile conditions...
```

#### `low_volatility_data`
Low volatility regime.

**Characteristics:**
- Minimal drift
- Very low volatility: 0.8% daily
- High ML accuracy: 85%
- VIX consistently < 15

```python
def test_low_vol(low_volatility_data):
    data_path, context = low_volatility_data
    # Test during calm conditions...
```

#### `trending_market_data`
Strongly trending market.

**Characteristics:**
- Strong positive drift: 1% daily
- Moderate volatility
- High ML accuracy on trend: 82%
- Directional price action

```python
def test_trending(trending_market_data):
    data_path, context = trending_market_data
    # Test trend-following strategy...
```

#### `mean_reverting_data`
Mean-reverting (choppy) market.

**Characteristics:**
- No drift (oscillates around mean)
- Moderate volatility
- Lower ML accuracy: 68%
- Multiple regime changes

```python
def test_mean_reversion(mean_reverting_data):
    data_path, context = mean_reverting_data
    # Test mean-reversion strategy...
```

### Parameterized Fixture

#### `ml_data_scenario`
Automatically runs your test with ALL scenarios (6 times).

**Returns:** `tuple[Path, dict[datetime, dict], str]`

```python
def test_all_scenarios(ml_data_scenario):
    """This test runs 6 times, once per scenario."""
    data_path, context, scenario_name = ml_data_scenario

    print(f"Testing scenario: {scenario_name}")
    # scenario_name will be: 'bull', 'bear', 'high_volatility', etc.

    # Your test logic here...
```

**When to use:**
- Testing strategy robustness across different market conditions
- Ensuring strategy doesn't break in any scenario
- Comparative performance analysis

## Complete Usage Examples

### Example 1: Basic ML Strategy Test

```python
def test_ml_momentum_strategy(ml_signal_data):
    """Test ML momentum strategy with basic data."""
    from ml4t.backtest import BacktestEngine, Strategy
    from ml4t.backtest.data.feed import ParquetDataFeed

    data_path, df = ml_signal_data

    class MLMomentumStrategy(Strategy):
        def on_event(self, event):
            pass

        def on_market_event(self, event, context=None):
            prediction = event.signals.get("prediction", 0.0)
            confidence = event.signals.get("confidence", 0.0)

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
        strategy=MLMomentumStrategy(),
        initial_capital=100000.0,
    )

    results = engine.run()
    assert results["final_value"] > 0
```

### Example 2: Strategy with Context

```python
def test_ml_strategy_with_vix_filter(bull_market_data):
    """Test ML strategy that filters trades by VIX."""
    from ml4t.backtest import BacktestEngine, Strategy
    from ml4t.backtest.data.feed import ParquetDataFeed

    data_path, context = bull_market_data

    class VIXFilteredStrategy(Strategy):
        def on_event(self, event):
            pass

        def on_market_event(self, event, context=None):
            # Get VIX from context
            vix = context.get("VIX", 0.0) if context else 0.0

            # Only trade when VIX < 25
            if vix > 25:
                return

            prediction = event.signals.get("prediction", 0.0)
            if prediction > 0.6:
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
        context_data=context,  # Pass context to engine
        initial_capital=100000.0,
    )

    results = engine.run()
    assert results["events_processed"] > 0
```

### Example 3: Parameterized Testing

```python
def test_strategy_robustness(ml_data_scenario):
    """Test strategy works in all market conditions."""
    from ml4t.backtest import BacktestEngine, Strategy
    from ml4t.backtest.data.feed import ParquetDataFeed

    data_path, context, scenario_name = ml_data_scenario

    class RobustStrategy(Strategy):
        def on_event(self, event):
            pass

        def on_market_event(self, event, context=None):
            prediction = event.signals.get("prediction", 0.0)
            confidence = event.signals.get("confidence", 0.0)

            if prediction > 0.65 and confidence > 0.75:
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

    # Strategy should complete without errors in all scenarios
    assert results["events_processed"] > 0
    assert results["final_value"] > 0

    print(f"{scenario_name}: return = {results['total_return']:.2f}%")
```

### Example 4: Comparing Scenarios

```python
@pytest.mark.parametrize("fixture_name", [
    "bull_market_data",
    "bear_market_data",
    "trending_market_data",
])
def test_strategy_performance(fixture_name, request):
    """Compare strategy performance across specific scenarios."""
    from ml4t.backtest import BacktestEngine, Strategy
    from ml4t.backtest.data.feed import ParquetDataFeed

    # Get the fixture dynamically
    data_path, context = request.getfixturevalue(fixture_name)

    class TestStrategy(Strategy):
        def on_event(self, event):
            pass

        def on_market_event(self, event, context=None):
            prediction = event.signals.get("prediction", 0.0)
            if prediction > 0.7:
                self.buy_percent("TEST", 0.20, event.close)

    feed = ParquetDataFeed(
        path=data_path,
        asset_id="TEST",
        timestamp_column="timestamp",
        signal_columns=["prediction", "confidence"],
    )

    engine = BacktestEngine(
        data_feed=feed,
        strategy=TestStrategy(),
        context_data=context,
        initial_capital=100000.0,
    )

    results = engine.run()

    # Strategy should be profitable in bull/trending markets
    if fixture_name in ["bull_market_data", "trending_market_data"]:
        assert results["total_return"] > 0, f"Expected profit in {fixture_name}"
```

## Data Structure

### OHLCV DataFrame Schema

```python
{
    "timestamp": datetime,      # Trading timestamp
    "open": float,              # Open price
    "high": float,              # High price (>= open, close, low)
    "low": float,               # Low price (<= open, close, high)
    "close": float,             # Close price
    "volume": int,              # Volume
    "prediction": float,        # ML prediction (0-1 probability)
    "confidence": float,        # ML confidence (0-1)
}
```

### Context Dictionary Schema

```python
{
    datetime(2024, 1, 2, 9, 30): {
        "VIX": float,           # Volatility index (10-80)
        "regime": str,          # Market regime ("bull" or "bear")
    },
    ...
}
```

## Best Practices

### 1. Use Parameterized Fixtures for Robustness Testing

```python
# Good: Test runs across all scenarios
def test_strategy_robustness(ml_data_scenario):
    data_path, context, scenario = ml_data_scenario
    # Test logic...

# Less comprehensive: Only tests one scenario
def test_strategy(ml_signal_data):
    data_path, df = ml_signal_data
    # Test logic...
```

### 2. Combine Fixtures for Complex Tests

```python
def test_multiple_scenarios(
    bull_market_data,
    bear_market_data,
    high_volatility_data,
):
    """Test strategy transitions between different market conditions."""
    # Use multiple fixtures in one test for custom scenario analysis
```

### 3. Use Context Data for Realistic Strategies

```python
def test_with_context(bull_market_data):
    data_path, context = bull_market_data

    # Always pass context to engine for realistic testing
    engine = BacktestEngine(
        data_feed=feed,
        strategy=strategy,
        context_data=context,  # ✅ Include context
        initial_capital=100000.0,
    )
```

### 4. Validate ML Signal Quality

```python
def test_signal_quality(ml_signal_data):
    """Verify ML signals are realistic before testing strategy."""
    _, df = ml_signal_data

    # Check prediction distribution
    pred_mean = df["prediction"].mean()
    assert 0.4 < pred_mean < 0.6, "Predictions should be roughly balanced"

    # Check confidence distribution
    conf_mean = df["confidence"].mean()
    assert conf_mean > 0.6, "Confidence should be generally high"
```

## Running Tests

```bash
# Run all fixture validation tests
pytest tests/unit/test_ml_fixtures.py -v

# Run tests using a specific scenario
pytest tests/unit/ -k "bull_market" -v

# Run parameterized tests (shows all scenarios)
pytest tests/unit/ -k "ml_data_scenario" -v

# Run with output to see scenario names
pytest tests/unit/test_ml_fixtures.py::test_all_scenarios_have_data -v -s
```

## Extending the Fixtures

To add a new scenario:

1. Add helper function in `ml_signal_data.py`:
```python
@pytest.fixture
def custom_scenario_data(tmp_path: Path) -> tuple[Path, dict[datetime, dict]]:
    """Your custom market scenario."""
    # Generate data with custom characteristics
    prices = generate_price_series(
        n_days=252,
        drift=YOUR_DRIFT,
        volatility=YOUR_VOL,
    )
    # ... rest of implementation
    return path, context
```

2. Add to parameterized fixture:
```python
@pytest.fixture(params=[
    "bull", "bear", ..., "your_custom_scenario"
])
def ml_data_scenario(request, tmp_path, ..., custom_scenario_data):
    scenario_map = {
        # ...
        "your_custom_scenario": custom_scenario_data,
    }
    # ...
```

3. Export in `__init__.py` and `conftest.py`.

## See Also

- `examples/ml_strategy_example.py` - Complete ML strategy example
- `tests/unit/test_strategy_helpers.py` - Strategy helper method tests
- `src/ml4t/backtest/data/feed.py` - ParquetDataFeed documentation
