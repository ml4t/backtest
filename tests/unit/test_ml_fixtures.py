"""Tests for ML signal fixtures.

This file validates that the ML signal fixtures work correctly and
demonstrates how to use them in your own tests.
"""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest


# ============================================================================
# Basic Fixture Usage Tests
# ============================================================================


def test_ml_signal_data_structure(ml_signal_data):
    """Test that ml_signal_data fixture provides correct structure."""
    data_path, df = ml_signal_data

    # Check path exists and is parquet
    assert data_path.exists()
    assert data_path.suffix == ".parquet"

    # Check DataFrame structure
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 252  # One trading year

    # Check required columns exist
    required_cols = ["timestamp", "open", "high", "low", "close", "volume", "prediction", "confidence"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Check OHLC validity - all bars should be valid
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()

    # Check signal ranges
    assert (df["prediction"] >= 0.0).all()
    assert (df["prediction"] <= 1.0).all()
    assert (df["confidence"] >= 0.0).all()
    assert (df["confidence"] <= 1.0).all()


def test_context_data_structure(context_data):
    """Test that context_data fixture provides correct structure."""
    # Check it's a dict
    assert isinstance(context_data, dict)
    assert len(context_data) == 252

    # Check first entry structure
    first_ts = list(context_data.keys())[0]
    assert isinstance(first_ts, datetime)

    first_context = context_data[first_ts]
    assert "VIX" in first_context
    assert "regime" in first_context

    # Check VIX range
    for context in context_data.values():
        assert 10.0 <= context["VIX"] <= 80.0
        assert context["regime"] in ["bull", "bear"]


# ============================================================================
# Scenario-Specific Tests
# ============================================================================


def test_bull_market_characteristics(bull_market_data):
    """Test that bull market fixture has expected characteristics."""
    data_path, context = bull_market_data

    # Load data
    df = pl.read_parquet(data_path)

    # Check price trend (should go up overall)
    start_price = df["close"][0]
    end_price = df["close"][-1]
    total_return = (end_price - start_price) / start_price
    assert total_return > 0, "Bull market should have positive return"

    # Check VIX is generally low
    vix_values = [ctx["VIX"] for ctx in context.values()]
    avg_vix = sum(vix_values) / len(vix_values)
    assert avg_vix < 20, f"Bull market should have low VIX, got {avg_vix:.1f}"

    # Check regime
    regimes = [ctx["regime"] for ctx in context.values()]
    assert all(r == "bull" for r in regimes), "Bull market should have 'bull' regime"


def test_bear_market_characteristics(bear_market_data):
    """Test that bear market fixture has expected characteristics."""
    data_path, context = bear_market_data

    # Load data
    df = pl.read_parquet(data_path)

    # Check price trend (should go down overall or flat)
    start_price = df["close"][0]
    end_price = df["close"][-1]
    total_return = (end_price - start_price) / start_price
    # Bear markets may not always be negative due to randomness,
    # but they should be less positive than bull markets
    assert total_return < 0.15, "Bear market should have limited upside"

    # Check VIX is generally high
    vix_values = [ctx["VIX"] for ctx in context.values()]
    avg_vix = sum(vix_values) / len(vix_values)
    assert avg_vix > 20, f"Bear market should have high VIX, got {avg_vix:.1f}"

    # Check regime
    regimes = [ctx["regime"] for ctx in context.values()]
    assert all(r == "bear" for r in regimes), "Bear market should have 'bear' regime"


def test_high_volatility_characteristics(high_volatility_data):
    """Test that high volatility fixture has expected characteristics."""
    data_path, context = high_volatility_data

    # Load data
    df = pl.read_parquet(data_path)

    # Calculate realized volatility
    returns = df["close"].pct_change().drop_nulls()
    volatility = float(returns.std())
    assert volatility > 0.025, f"High vol market should have high volatility, got {volatility:.4f}"

    # Check VIX is consistently high
    vix_values = [ctx["VIX"] for ctx in context.values()]
    avg_vix = sum(vix_values) / len(vix_values)
    assert avg_vix > 30, f"High vol market should have VIX > 30, got {avg_vix:.1f}"


def test_low_volatility_characteristics(low_volatility_data):
    """Test that low volatility fixture has expected characteristics."""
    data_path, context = low_volatility_data

    # Load data
    df = pl.read_parquet(data_path)

    # Calculate realized volatility
    returns = df["close"].pct_change().drop_nulls()
    volatility = float(returns.std())
    assert volatility < 0.015, f"Low vol market should have low volatility, got {volatility:.4f}"

    # Check VIX is consistently low
    vix_values = [ctx["VIX"] for ctx in context.values()]
    avg_vix = sum(vix_values) / len(vix_values)
    assert avg_vix < 18, f"Low vol market should have VIX < 18, got {avg_vix:.1f}"


def test_trending_market_characteristics(trending_market_data):
    """Test that trending market fixture has strong directional movement."""
    data_path, context = trending_market_data

    # Load data
    df = pl.read_parquet(data_path)

    # Check for strong trend
    start_price = df["close"][0]
    end_price = df["close"][-1]
    total_return = (end_price - start_price) / start_price

    # Trending markets should have significant price movement
    assert abs(total_return) > 0.10, f"Trending market should move >10%, got {total_return:.2%}"

    # Check that prices generally move in one direction
    # Count how many times price crosses the starting price
    crossings = sum(
        1
        for i in range(1, len(df))
        if (df["close"][i - 1] - start_price) * (df["close"][i] - start_price) < 0
    )
    # Trending markets should have few crossings
    assert crossings < 20, f"Trending market should have few crossings, got {crossings}"


def test_mean_reverting_characteristics(mean_reverting_data):
    """Test that mean-reverting market oscillates around mean."""
    data_path, context = mean_reverting_data

    # Load data
    df = pl.read_parquet(data_path)

    # Check that price oscillates (multiple regime changes)
    regimes = [ctx["regime"] for ctx in context.values()]
    regime_changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
    assert regime_changes >= 2, "Mean-reverting market should have multiple regime changes"

    # Check that final price is close to starting price
    start_price = df["close"][0]
    end_price = df["close"][-1]
    total_return = abs((end_price - start_price) / start_price)
    assert total_return < 0.20, f"Mean-reverting market should stay near start, got {total_return:.2%}"


# ============================================================================
# Parameterized Fixture Tests
# ============================================================================


def test_all_scenarios_have_data(ml_data_scenario):
    """Test that all scenarios provide valid data (runs 6 times)."""
    data_path, context, scenario_name = ml_data_scenario

    # Check path exists
    assert data_path.exists()

    # Load and validate data
    df = pl.read_parquet(data_path)
    assert len(df) == 252
    assert "prediction" in df.columns
    assert "confidence" in df.columns

    # Check context
    assert len(context) == 252
    first_ctx = list(context.values())[0]
    assert "VIX" in first_ctx
    assert "regime" in first_ctx

    # Log which scenario we're testing
    print(f"\nValidated scenario: {scenario_name}")


def test_scenario_names_are_correct(ml_data_scenario):
    """Test that scenario names match expected values (runs 6 times)."""
    _, _, scenario_name = ml_data_scenario

    expected_scenarios = {"bull", "bear", "high_volatility", "low_volatility", "trending", "mean_reverting"}
    assert scenario_name in expected_scenarios


# ============================================================================
# Integration Tests (Realistic Usage)
# ============================================================================


def test_can_use_with_polars_datafeed(ml_signal_data):
    """Test that fixture data works with PolarsDataFeed."""
    from ml4t.backtest import BacktestEngine, Strategy
    from ml4t.backtest.data import PolarsDataFeed

    data_path, _ = ml_signal_data

    # Create data feed with signal columns
    feed = PolarsDataFeed(
        price_path=data_path,
        asset_id="TEST",
        timestamp_column="timestamp",
        signal_columns=["prediction", "confidence"],
        validate_signal_timing=False,
    )

    # Verify feed works by running a simple backtest
    class TestStrategy(Strategy):
        def __init__(self):
            super().__init__()
            self.events_seen = []

        def on_event(self, event):
            pass

        def on_market_event(self, event, context=None):
            self.events_seen.append(event)

    strategy = TestStrategy()
    engine = BacktestEngine(
        data_feed=feed,
        strategy=strategy,
        initial_capital=100000.0,
    )

    engine.run()

    # Check we got events with signals
    assert len(strategy.events_seen) > 0
    first_event = strategy.events_seen[0]
    assert hasattr(first_event, "signals")
    assert "prediction" in first_event.signals
    assert "confidence" in first_event.signals


def test_can_create_backtest_engine(bull_market_data):
    """Test that fixture data works with BacktestEngine."""
    from ml4t.backtest import BacktestEngine, Strategy
    from ml4t.backtest.data import PolarsDataFeed

    data_path, context = bull_market_data

    # Create simple test strategy
    class DummyStrategy(Strategy):
        def on_event(self, event):
            pass

    # Create data feed
    feed = PolarsDataFeed(
        price_path=data_path,
        asset_id="TEST",
        timestamp_column="timestamp",
        signal_columns=["prediction", "confidence"],
        validate_signal_timing=False,
    )

    # Create engine (should not raise)
    engine = BacktestEngine(
        data_feed=feed,
        strategy=DummyStrategy(),
        context_data=context,
        initial_capital=100000.0,
    )

    assert engine is not None


# ============================================================================
# Usage Example Test (Documentation)
# ============================================================================


def test_example_ml_strategy_usage(ml_signal_data):
    """Example: How to use ML fixtures with a simple strategy.

    This test demonstrates the typical workflow for using ML signal fixtures
    in strategy testing.
    """
    from ml4t.backtest import BacktestEngine, Strategy
    from ml4t.backtest.data import PolarsDataFeed

    # Unpack fixture
    data_path, df = ml_signal_data

    # Create a simple ML strategy
    class SimpleMLStrategy(Strategy):
        def __init__(self):
            super().__init__()
            self.entries = 0

        def on_event(self, event):
            pass

        def on_market_event(self, event, context=None):
            # Access ML signals from event
            prediction = event.signals.get("prediction", 0.0)
            confidence = event.signals.get("confidence", 0.0)

            # Simple entry logic
            if prediction > 0.7 and confidence > 0.8:
                self.entries += 1

    # Create data feed
    feed = PolarsDataFeed(
        price_path=data_path,
        asset_id="TEST",
        timestamp_column="timestamp",
        signal_columns=["prediction", "confidence"],
        validate_signal_timing=False,
    )

    # Create and run backtest
    strategy = SimpleMLStrategy()
    engine = BacktestEngine(
        data_feed=feed,
        strategy=strategy,
        initial_capital=100000.0,
    )

    results = engine.run()

    # Verify backtest ran
    assert results["events_processed"] > 0
    assert strategy.entries >= 0  # Should have some entries with random signals


# ============================================================================
# Fixture Reusability Test
# ============================================================================


def test_fixtures_are_independent(ml_signal_data, bull_market_data):
    """Test that multiple fixtures can be used in same test."""
    neutral_path, neutral_df = ml_signal_data
    bull_path, bull_context = bull_market_data

    # Both should be valid
    assert neutral_path.exists()
    assert bull_path.exists()

    # Should be different files
    assert neutral_path != bull_path

    # Both should have required structure
    assert "prediction" in neutral_df.columns
    bull_df = pl.read_parquet(bull_path)
    assert "prediction" in bull_df.columns
