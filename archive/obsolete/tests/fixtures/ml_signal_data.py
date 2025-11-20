"""Pytest fixtures for ML signal data generation.

This module provides reusable fixtures for testing ML-based trading strategies.
Fixtures generate realistic market data with ML predictions, confidence scores,
and market context (VIX, regime indicators).

Usage Example:
    ```python
    def test_ml_strategy(ml_signal_data):
        data_path, signals_df = ml_signal_data
        # Use in your test...

    def test_with_scenario(ml_data_scenario):
        # Automatically runs test with all scenarios
        data_path, context, scenario_name = ml_data_scenario
        # scenario_name will be: 'bull', 'bear', 'high_vol', etc.
    ```
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import pytest


# ============================================================================
# Market Scenario Definitions
# ============================================================================

MarketScenario = Literal[
    "bull",  # Bull market: positive drift, lower volatility
    "bear",  # Bear market: negative drift, higher volatility
    "high_volatility",  # High volatility regime (VIX > 30)
    "low_volatility",  # Low volatility regime (VIX < 15)
    "trending",  # Strong directional trend
    "mean_reverting",  # Oscillating/choppy market
]


# ============================================================================
# Helper Functions for Data Generation
# ============================================================================


def generate_price_series(
    n_days: int,
    initial_price: float = 100.0,
    drift: float = 0.0005,
    volatility: float = 0.015,
    trend_strength: float = 0.0,
    mean_reversion: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate realistic price series with configurable characteristics.

    Args:
        n_days: Number of trading days
        initial_price: Starting price
        drift: Daily drift (mean return)
        volatility: Daily volatility (std of returns)
        trend_strength: Strength of trending behavior (0-1)
        mean_reversion: Strength of mean reversion (0-1)
        seed: Random seed for reproducibility

    Returns:
        Array of close prices
    """
    np.random.seed(seed)
    prices = [initial_price]

    for i in range(1, n_days):
        # Base random return
        daily_return = np.random.normal(drift, volatility)

        # Add trend component
        if trend_strength > 0:
            trend = trend_strength * drift * 2  # Amplify drift for trending
            daily_return += trend

        # Add mean reversion component
        if mean_reversion > 0:
            deviation_from_mean = (prices[-1] - initial_price) / initial_price
            reversion = -mean_reversion * deviation_from_mean
            daily_return += reversion

        prices.append(prices[-1] * (1 + daily_return))

    return np.array(prices)


def generate_ohlcv_from_close(
    close_prices: np.ndarray,
    base_volume: int = 10_000_000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate OHLCV data from close prices.

    Args:
        close_prices: Array of close prices
        base_volume: Base volume level
        seed: Random seed

    Returns:
        Dictionary with open, high, low, close, volume arrays
    """
    np.random.seed(seed)
    n = len(close_prices)

    # Generate open prices
    opens = close_prices * (1 + np.random.normal(0, 0.003, n))

    # Generate highs and lows that respect OHLC constraints
    # High = max(open, close) + some random amount
    # Low = min(open, close) - some random amount
    highs = np.maximum(opens, close_prices) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = np.minimum(opens, close_prices) * (1 - np.abs(np.random.normal(0, 0.005, n)))

    volumes = np.random.exponential(base_volume, n).astype(int)

    return {
        "open": opens,
        "high": highs,
        "low": lows,
        "close": close_prices,
        "volume": volumes,
    }


def generate_ml_predictions(
    prices: np.ndarray,
    forward_window: int = 5,
    accuracy: float = 0.75,
    noise_level: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ML predictions and confidence scores.

    Models a realistic ML predictor that forecasts forward returns
    with specified accuracy and noise characteristics.

    Args:
        prices: Array of prices
        forward_window: Days ahead to predict
        accuracy: Model accuracy (0-1)
        noise_level: Noise in predictions
        seed: Random seed

    Returns:
        Tuple of (predictions, confidence_scores)
        - predictions: Probability of positive return (0-1)
        - confidence: Model confidence (0-1)
    """
    np.random.seed(seed)
    n = len(prices)

    # Calculate true forward returns
    true_returns = []
    for i in range(n):
        if i + forward_window < n:
            ret = (prices[i + forward_window] - prices[i]) / prices[i]
            true_returns.append(ret)
        else:
            true_returns.append(0.0)

    true_returns = np.array(true_returns)

    # Generate predictions with noise
    predictions = []
    confidences = []

    for i in range(n):
        true_ret = true_returns[i]

        # Add noise to prediction
        noise = np.random.normal(0, noise_level)

        # Convert return to probability (sigmoid-like transformation)
        # Higher accuracy means stronger signal
        scaling = 10 * accuracy
        pred = 1 / (1 + np.exp(-scaling * (true_ret + noise)))
        predictions.append(pred)

        # Confidence correlates with prediction strength and accuracy
        # Strong predictions get high confidence, weak ones get low confidence
        conf = min(0.95, max(0.5, accuracy * (0.5 + abs(pred - 0.5))))
        confidences.append(conf)

    return np.array(predictions), np.array(confidences)


def generate_vix_series(
    n_days: int,
    base_vix: float = 20.0,
    volatility: float = 3.0,
    spikes: list[int] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate VIX (volatility index) time series.

    Args:
        n_days: Number of days
        base_vix: Average VIX level
        volatility: Volatility of VIX itself
        spikes: Optional list of days with VIX spikes
        seed: Random seed

    Returns:
        Array of VIX values
    """
    np.random.seed(seed)
    vix_values = base_vix + np.random.normal(0, volatility, n_days)

    # Add spikes
    if spikes:
        for spike_day in spikes:
            if 0 <= spike_day < n_days:
                vix_values[spike_day] += np.random.uniform(10, 25)

    # Clamp to realistic range
    vix_values = np.clip(vix_values, 10.0, 80.0)

    return vix_values


def generate_regime_indicators(
    n_days: int,
    regime_changes: list[int] | None = None,
    initial_regime: str = "bull",
) -> list[str]:
    """Generate market regime indicators.

    Args:
        n_days: Number of days
        regime_changes: List of days when regime changes
        initial_regime: Starting regime ('bull' or 'bear')

    Returns:
        List of regime labels
    """
    if regime_changes is None:
        regime_changes = []

    regimes = []
    current_regime = initial_regime

    for i in range(n_days):
        # Check for regime change
        if i in regime_changes:
            current_regime = "bear" if current_regime == "bull" else "bull"
        regimes.append(current_regime)

    return regimes


# ============================================================================
# Core Fixtures
# ============================================================================


@pytest.fixture
def ml_signal_data(tmp_path: Path) -> tuple[Path, pl.DataFrame]:
    """Generate basic ML signal data (neutral market).

    Returns:
        Tuple of (parquet_path, dataframe)
        DataFrame has columns: timestamp, open, high, low, close, volume,
                               prediction, confidence

    Example:
        ```python
        def test_strategy(ml_signal_data):
            data_path, df = ml_signal_data
            assert "prediction" in df.columns
            assert "confidence" in df.columns
        ```
    """
    n_days = 252  # One trading year

    # Generate timestamps
    timestamps = [
        datetime(2024, 1, 2, 9, 30) + timedelta(days=i) for i in range(n_days)
    ]

    # Generate price data (neutral market)
    prices = generate_price_series(
        n_days=n_days,
        initial_price=100.0,
        drift=0.0003,  # Slight positive drift
        volatility=0.015,  # Moderate volatility
    )

    # Generate OHLCV
    ohlcv = generate_ohlcv_from_close(prices)

    # Generate ML predictions
    predictions, confidences = generate_ml_predictions(
        prices=prices,
        forward_window=5,
        accuracy=0.75,
        noise_level=0.03,
    )

    # Create DataFrame
    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["TEST"] * n_days,  # Required by PolarsDataFeed
            "asset_id": ["TEST"] * n_days,  # Required by PolarsDataFeed
            "open": ohlcv["open"],
            "high": ohlcv["high"],
            "low": ohlcv["low"],
            "close": ohlcv["close"],
            "volume": ohlcv["volume"],
            "prediction": predictions,
            "confidence": confidences,
        }
    )

    # Save to parquet
    path = tmp_path / "ml_signal_data.parquet"
    df.write_parquet(str(path))

    return path, df


@pytest.fixture
def context_data() -> dict[datetime, dict]:
    """Generate basic market context data.

    Returns:
        Dictionary mapping timestamps to context dicts with VIX and regime

    Example:
        ```python
        def test_with_context(context_data):
            first_ts = list(context_data.keys())[0]
            assert "VIX" in context_data[first_ts]
            assert "regime" in context_data[first_ts]
        ```
    """
    n_days = 252
    timestamps = [
        datetime(2024, 1, 2, 9, 30) + timedelta(days=i) for i in range(n_days)
    ]

    # Generate VIX and regime
    vix_values = generate_vix_series(n_days, base_vix=20.0)
    regimes = generate_regime_indicators(n_days, regime_changes=[126])  # Mid-year change

    context = {}
    for i, ts in enumerate(timestamps):
        context[ts] = {
            "VIX": float(vix_values[i]),
            "regime": regimes[i],
        }

    return context


# ============================================================================
# Scenario-Specific Fixtures
# ============================================================================


@pytest.fixture
def bull_market_data(tmp_path: Path) -> tuple[Path, dict[datetime, dict]]:
    """Generate bull market data with ML signals.

    Characteristics:
    - Positive drift (0.08% daily = ~20% annualized)
    - Lower volatility (1.2% daily)
    - High ML accuracy (80%)
    - Low VIX (avg 15)

    Returns:
        Tuple of (parquet_path, context_dict)

    Example:
        ```python
        def test_bull_strategy(bull_market_data):
            data_path, context = bull_market_data
            # Test strategy in bull market conditions
        ```
    """
    n_days = 252
    timestamps = [
        datetime(2024, 1, 2, 9, 30) + timedelta(days=i) for i in range(n_days)
    ]

    # Bull market price series
    prices = generate_price_series(
        n_days=n_days,
        initial_price=100.0,
        drift=0.0008,  # Strong positive drift
        volatility=0.012,  # Lower volatility
    )

    ohlcv = generate_ohlcv_from_close(prices)

    # Higher accuracy in bull markets
    predictions, confidences = generate_ml_predictions(
        prices=prices,
        accuracy=0.80,
        noise_level=0.02,
    )

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["TEST"] * n_days,  # Required by PolarsDataFeed
            **ohlcv,
            "prediction": predictions,
            "confidence": confidences,
        }
    )

    path = tmp_path / "bull_market_data.parquet"
    df.write_parquet(str(path))

    # Bull market context: low VIX, bull regime
    vix_values = generate_vix_series(n_days, base_vix=15.0, volatility=2.0)
    regimes = generate_regime_indicators(n_days, initial_regime="bull")

    context = {}
    for i, ts in enumerate(timestamps):
        context[ts] = {"VIX": float(vix_values[i]), "regime": regimes[i]}

    return path, context


@pytest.fixture
def bear_market_data(tmp_path: Path) -> tuple[Path, dict[datetime, dict]]:
    """Generate bear market data with ML signals.

    Characteristics:
    - Negative drift (-0.03% daily = ~-7.5% annualized)
    - Higher volatility (2.5% daily)
    - Lower ML accuracy (65%)
    - High VIX (avg 28)

    Returns:
        Tuple of (parquet_path, context_dict)
    """
    n_days = 252
    timestamps = [
        datetime(2024, 1, 2, 9, 30) + timedelta(days=i) for i in range(n_days)
    ]

    # Bear market price series
    prices = generate_price_series(
        n_days=n_days,
        initial_price=100.0,
        drift=-0.0003,  # Negative drift
        volatility=0.025,  # Higher volatility
    )

    ohlcv = generate_ohlcv_from_close(prices)

    # Lower accuracy in bear markets
    predictions, confidences = generate_ml_predictions(
        prices=prices,
        accuracy=0.65,
        noise_level=0.04,
    )

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["TEST"] * n_days,  # Required by PolarsDataFeed
            **ohlcv,
            "prediction": predictions,
            "confidence": confidences,
        }
    )

    path = tmp_path / "bear_market_data.parquet"
    df.write_parquet(str(path))

    # Bear market context: high VIX, bear regime
    vix_values = generate_vix_series(n_days, base_vix=28.0, volatility=4.0)
    regimes = generate_regime_indicators(n_days, initial_regime="bear")

    context = {}
    for i, ts in enumerate(timestamps):
        context[ts] = {"VIX": float(vix_values[i]), "regime": regimes[i]}

    return path, context


@pytest.fixture
def high_volatility_data(tmp_path: Path) -> tuple[Path, dict[datetime, dict]]:
    """Generate high volatility market data.

    Characteristics:
    - Neutral drift
    - Very high volatility (3.5% daily)
    - ML accuracy suffers (60%)
    - VIX consistently > 30

    Returns:
        Tuple of (parquet_path, context_dict)
    """
    n_days = 252
    timestamps = [
        datetime(2024, 1, 2, 9, 30) + timedelta(days=i) for i in range(n_days)
    ]

    # High volatility price series
    prices = generate_price_series(
        n_days=n_days,
        initial_price=100.0,
        drift=0.0,
        volatility=0.035,  # Very high volatility
    )

    ohlcv = generate_ohlcv_from_close(prices)

    # Lower accuracy in volatile markets
    predictions, confidences = generate_ml_predictions(
        prices=prices,
        accuracy=0.60,
        noise_level=0.05,
    )

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["TEST"] * n_days,  # Required by PolarsDataFeed
            **ohlcv,
            "prediction": predictions,
            "confidence": confidences,
        }
    )

    path = tmp_path / "high_volatility_data.parquet"
    df.write_parquet(str(path))

    # High VIX context
    vix_values = generate_vix_series(n_days, base_vix=35.0, volatility=5.0)
    regimes = generate_regime_indicators(n_days, initial_regime="bear")

    context = {}
    for i, ts in enumerate(timestamps):
        context[ts] = {"VIX": float(vix_values[i]), "regime": regimes[i]}

    return path, context


@pytest.fixture
def low_volatility_data(tmp_path: Path) -> tuple[Path, dict[datetime, dict]]:
    """Generate low volatility market data.

    Characteristics:
    - Minimal drift
    - Very low volatility (0.8% daily)
    - High ML accuracy (85%)
    - VIX consistently < 15

    Returns:
        Tuple of (parquet_path, context_dict)
    """
    n_days = 252
    timestamps = [
        datetime(2024, 1, 2, 9, 30) + timedelta(days=i) for i in range(n_days)
    ]

    # Low volatility price series
    prices = generate_price_series(
        n_days=n_days,
        initial_price=100.0,
        drift=0.0002,
        volatility=0.008,  # Very low volatility
    )

    ohlcv = generate_ohlcv_from_close(prices)

    # Higher accuracy in calm markets
    predictions, confidences = generate_ml_predictions(
        prices=prices,
        accuracy=0.85,
        noise_level=0.015,
    )

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["TEST"] * n_days,  # Required by PolarsDataFeed
            **ohlcv,
            "prediction": predictions,
            "confidence": confidences,
        }
    )

    path = tmp_path / "low_volatility_data.parquet"
    df.write_parquet(str(path))

    # Low VIX context
    vix_values = generate_vix_series(n_days, base_vix=12.0, volatility=1.5)
    regimes = generate_regime_indicators(n_days, initial_regime="bull")

    context = {}
    for i, ts in enumerate(timestamps):
        context[ts] = {"VIX": float(vix_values[i]), "regime": regimes[i]}

    return path, context


@pytest.fixture
def trending_market_data(tmp_path: Path) -> tuple[Path, dict[datetime, dict]]:
    """Generate strongly trending market data.

    Characteristics:
    - Strong positive drift (1% daily)
    - Moderate volatility
    - High ML accuracy on trend (82%)
    - Directional price action

    Returns:
        Tuple of (parquet_path, context_dict)
    """
    n_days = 252
    timestamps = [
        datetime(2024, 1, 2, 9, 30) + timedelta(days=i) for i in range(n_days)
    ]

    # Trending price series
    prices = generate_price_series(
        n_days=n_days,
        initial_price=100.0,
        drift=0.001,  # Strong trend
        volatility=0.015,
        trend_strength=0.8,  # High trend strength
    )

    ohlcv = generate_ohlcv_from_close(prices)

    # High accuracy on trending data
    predictions, confidences = generate_ml_predictions(
        prices=prices,
        accuracy=0.82,
        noise_level=0.02,
    )

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["TEST"] * n_days,  # Required by PolarsDataFeed
            **ohlcv,
            "prediction": predictions,
            "confidence": confidences,
        }
    )

    path = tmp_path / "trending_market_data.parquet"
    df.write_parquet(str(path))

    # Trending market context
    vix_values = generate_vix_series(n_days, base_vix=18.0, volatility=2.5)
    regimes = generate_regime_indicators(n_days, initial_regime="bull")

    context = {}
    for i, ts in enumerate(timestamps):
        context[ts] = {"VIX": float(vix_values[i]), "regime": regimes[i]}

    return path, context


@pytest.fixture
def mean_reverting_data(tmp_path: Path) -> tuple[Path, dict[datetime, dict]]:
    """Generate mean-reverting (choppy) market data.

    Characteristics:
    - No drift (oscillating around mean)
    - Moderate volatility
    - Lower ML accuracy (68%)
    - Price oscillates rather than trends

    Returns:
        Tuple of (parquet_path, context_dict)
    """
    n_days = 252
    timestamps = [
        datetime(2024, 1, 2, 9, 30) + timedelta(days=i) for i in range(n_days)
    ]

    # Mean-reverting price series
    prices = generate_price_series(
        n_days=n_days,
        initial_price=100.0,
        drift=0.0,
        volatility=0.018,
        mean_reversion=0.15,  # Strong mean reversion
    )

    ohlcv = generate_ohlcv_from_close(prices)

    # Lower accuracy on choppy markets
    predictions, confidences = generate_ml_predictions(
        prices=prices,
        accuracy=0.68,
        noise_level=0.035,
    )

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["TEST"] * n_days,  # Required by PolarsDataFeed
            **ohlcv,
            "prediction": predictions,
            "confidence": confidences,
        }
    )

    path = tmp_path / "mean_reverting_data.parquet"
    df.write_parquet(str(path))

    # Choppy market context
    vix_values = generate_vix_series(n_days, base_vix=22.0, volatility=3.5)
    regimes = generate_regime_indicators(
        n_days, regime_changes=[63, 126, 189]  # Multiple regime flips
    )

    context = {}
    for i, ts in enumerate(timestamps):
        context[ts] = {"VIX": float(vix_values[i]), "regime": regimes[i]}

    return path, context


# ============================================================================
# Parameterized Fixture for Multiple Scenarios
# ============================================================================


@pytest.fixture(
    params=["bull", "bear", "high_volatility", "low_volatility", "trending", "mean_reverting"]
)
def ml_data_scenario(
    request,
    tmp_path: Path,
    bull_market_data,
    bear_market_data,
    high_volatility_data,
    low_volatility_data,
    trending_market_data,
    mean_reverting_data,
) -> tuple[Path, dict[datetime, dict], str]:
    """Parameterized fixture that runs test with all market scenarios.

    This fixture automatically runs your test 6 times, once for each scenario.

    Returns:
        Tuple of (data_path, context_dict, scenario_name)

    Example:
        ```python
        def test_strategy_all_scenarios(ml_data_scenario):
            data_path, context, scenario = ml_data_scenario
            print(f"Testing scenario: {scenario}")
            # Your test logic here...
            # This test will run 6 times, once per scenario
        ```
    """
    scenario_map = {
        "bull": bull_market_data,
        "bear": bear_market_data,
        "high_volatility": high_volatility_data,
        "low_volatility": low_volatility_data,
        "trending": trending_market_data,
        "mean_reverting": mean_reverting_data,
    }

    scenario_name = request.param
    data_path, context = scenario_map[scenario_name]

    return data_path, context, scenario_name
