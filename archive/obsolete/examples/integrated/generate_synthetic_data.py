"""Generate realistic synthetic data for Top 25 ML Strategy example.

This script creates data for 500 stocks with:
- OHLCV price data (252 trading days)
- ML scores (0-1, ~58% accuracy correlation with returns)
- ATR values (realistic volatility)
- VIX context data (market-wide volatility indicator)
- Market regime changes (bull/bear periods)
"""

import numpy as np
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)  # For reproducibility

# Configuration
N_STOCKS = 500
N_DAYS = 252
START_DATE = datetime(2023, 1, 3)
OUTPUT_DIR = Path(__file__).parent / "data"


def generate_stock_prices(ticker: str, n_days: int, regime_periods: list) -> dict:
    """Generate realistic OHLCV data for one stock.

    Args:
        ticker: Stock ticker symbol
        n_days: Number of trading days
        regime_periods: List of (start_day, end_day, regime_type) tuples

    Returns:
        Dict with timestamp, open, high, low, close, volume, asset_id
    """
    # Base parameters vary by stock
    np.random.seed(hash(ticker) % (2**32))  # Deterministic per ticker

    initial_price = np.random.uniform(20, 500)
    base_vol = np.random.uniform(0.015, 0.04)  # 1.5% - 4% daily vol

    prices = [initial_price]

    for i in range(1, n_days):
        # Find current regime
        regime = "bull"
        for start, end, r in regime_periods:
            if start <= i < end:
                regime = r
                break

        # Regime-dependent drift and volatility
        if regime == "bull":
            drift = np.random.uniform(0.0003, 0.0008)
            vol_mult = 1.0
        elif regime == "bear":
            drift = np.random.uniform(-0.0005, -0.0001)
            vol_mult = 1.5
        else:  # choppy
            drift = np.random.uniform(-0.0002, 0.0002)
            vol_mult = 1.2

        daily_return = np.random.normal(drift, base_vol * vol_mult)
        prices.append(prices[-1] * (1 + daily_return))

    prices = np.array(prices)

    # Generate OHLC from close prices
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    opens = prices * (1 + np.random.normal(0, 0.003, n_days))

    # Volume correlates with price movement
    price_changes = np.abs(np.diff(prices, prepend=prices[0]))
    base_volume = np.random.uniform(100_000, 10_000_000)
    volumes = (base_volume * (1 + 5 * price_changes / prices)).astype(int)

    # Generate timestamps (trading days only, 9:30 AM ET)
    timestamps = [START_DATE + timedelta(days=i) for i in range(n_days)]

    return {
        "timestamp": timestamps,
        "asset_id": [ticker] * n_days,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range (ATR).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)

    Returns:
        ATR values (first `period` values are NaN)
    """
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    high_low = high - low

    # Shift close by 1 for previous close
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # First value has no previous

    high_pc = np.abs(high - prev_close)
    low_pc = np.abs(low - prev_close)

    true_range = np.maximum(high_low, np.maximum(high_pc, low_pc))

    # Calculate ATR as EMA of true range
    atr = np.full_like(true_range, np.nan)
    atr[period - 1] = np.mean(true_range[:period])

    multiplier = 1.0 / period
    for i in range(period, len(true_range)):
        atr[i] = atr[i-1] + multiplier * (true_range[i] - atr[i-1])

    return atr


def generate_ml_scores(close: np.ndarray, atr: np.ndarray, regime_periods: list) -> tuple:
    """Generate synthetic ML scores with realistic predictive power.

    ML scores have ~58% accuracy (slightly better than random) and correlate
    with future returns, ATR (prefer stable stocks), and regime.

    Args:
        close: Close prices
        atr: ATR values
        regime_periods: Market regime information

    Returns:
        Tuple of (ml_scores, future_5d_returns)
    """
    n_days = len(close)

    # Calculate 5-day forward returns (what ML is trying to predict)
    future_5d_returns = np.full(n_days, np.nan)
    for i in range(n_days - 5):
        future_5d_returns[i] = (close[i + 5] - close[i]) / close[i]

    # Generate ML scores with partial predictive power
    ml_scores = np.zeros(n_days)

    for i in range(n_days):
        # Find current regime
        regime = "bull"
        for start, end, r in regime_periods:
            if start <= i < end:
                regime = r
                break

        # True future return (if available)
        true_ret = future_5d_returns[i] if not np.isnan(future_5d_returns[i]) else 0.0

        # Model quality varies by regime
        if regime == "bull":
            signal_strength = 0.6  # 60% signal, 40% noise
            noise_std = 0.03
        elif regime == "bear":
            signal_strength = 0.4  # 40% signal, 60% noise (harder to predict)
            noise_std = 0.05
        else:  # choppy
            signal_strength = 0.3  # 30% signal, 70% noise
            noise_std = 0.06

        # ML score combines true signal + noise
        # Score is higher for stocks with positive expected returns and lower volatility
        signal = true_ret * signal_strength

        # Penalize high ATR stocks (prefer stable stocks)
        if not np.isnan(atr[i]) and close[i] > 0:
            volatility_penalty = -0.5 * (atr[i] / close[i])
        else:
            volatility_penalty = 0.0

        noise = np.random.normal(0, noise_std)

        raw_score = signal + volatility_penalty + noise

        # Convert to 0-1 probability via sigmoid
        ml_scores[i] = 1.0 / (1.0 + np.exp(-10 * raw_score))

    return ml_scores, future_5d_returns


def generate_vix_data(n_days: int, regime_periods: list) -> dict:
    """Generate synthetic VIX (volatility index) data.

    VIX is higher during bear markets and spikes at regime transitions.

    Args:
        n_days: Number of trading days
        regime_periods: Market regime information

    Returns:
        Dict with timestamp and VIX values
    """
    vix_values = []

    for i in range(n_days):
        # Find current regime
        regime = "bull"
        for start, end, r in regime_periods:
            if start <= i < end:
                regime = r
                break

        # Base VIX by regime
        if regime == "bull":
            base_vix = 15.0
        elif regime == "bear":
            base_vix = 30.0
        else:  # choppy
            base_vix = 22.0

        # Spike at regime transitions
        transition_boost = 0.0
        for start, end, _ in regime_periods:
            if abs(i - start) < 5:  # Within 5 days of transition
                transition_boost = np.random.uniform(5, 15)

        vix = base_vix + transition_boost + np.random.normal(0, 3)
        vix = np.clip(vix, 10.0, 60.0)  # Realistic VIX range

        vix_values.append(vix)

    timestamps = [START_DATE + timedelta(days=i) for i in range(n_days)]

    return {
        "timestamp": timestamps,
        "asset_id": [None] * n_days,  # Market-wide feature
        "vix": vix_values,
    }


def main():
    """Generate all synthetic data and save to parquet files."""
    print("Generating synthetic data for 500-stock universe...")
    print(f"- Trading days: {N_DAYS}")
    print(f"- Date range: {START_DATE.date()} to {(START_DATE + timedelta(days=N_DAYS-1)).date()}")

    # Define market regimes
    regime_periods = [
        (0, 60, "bull"),      # Days 0-59: Bull market
        (60, 120, "choppy"),  # Days 60-119: Choppy market
        (120, 180, "bull"),   # Days 120-179: Bull market
        (180, 220, "bear"),   # Days 180-219: Bear market
        (220, 252, "bull"),   # Days 220-251: Bull market
    ]

    print(f"- Market regimes: {len(regime_periods)} periods")

    # Generate data for all stocks
    all_stock_data = []

    for i in range(N_STOCKS):
        ticker = f"STOCK{i:03d}"

        if (i + 1) % 100 == 0:
            print(f"  Generating stock {i+1}/{N_STOCKS}...")

        # Generate OHLCV data
        stock_data = generate_stock_prices(ticker, N_DAYS, regime_periods)

        # Calculate ATR
        atr_values = calculate_atr(
            np.array(stock_data["high"]),
            np.array(stock_data["low"]),
            np.array(stock_data["close"]),
            period=14
        )

        # Generate ML scores
        ml_scores, _ = generate_ml_scores(
            np.array(stock_data["close"]),
            atr_values,
            regime_periods
        )

        # Add features to stock data
        stock_data["atr"] = atr_values
        stock_data["ml_score"] = ml_scores

        all_stock_data.append(pl.DataFrame(stock_data))

    # Combine all stocks into single DataFrame
    print("Combining data into single DataFrame...")
    combined_df = pl.concat(all_stock_data)

    # Sort by timestamp, then asset_id for efficient access
    combined_df = combined_df.sort(["timestamp", "asset_id"])

    # Save price and features data
    output_path = OUTPUT_DIR / "stock_data.parquet"
    combined_df.write_parquet(str(output_path))
    print(f"✓ Saved stock data: {output_path} ({len(combined_df):,} rows)")

    # Generate and save VIX data (market-wide context)
    print("Generating VIX data...")
    vix_data = generate_vix_data(N_DAYS, regime_periods)
    vix_df = pl.DataFrame(vix_data)

    vix_output_path = OUTPUT_DIR / "vix_data.parquet"
    vix_df.write_parquet(str(vix_output_path))
    print(f"✓ Saved VIX data: {vix_output_path} ({len(vix_df):,} rows)")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total stocks: {N_STOCKS}")
    print(f"Total rows: {len(combined_df):,}")
    print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    print(f"\nPrice statistics:")
    print(f"  Close price range: ${combined_df['close'].min():.2f} - ${combined_df['close'].max():.2f}")
    print(f"  Mean close: ${combined_df['close'].mean():.2f}")
    print(f"\nFeature statistics:")
    print(f"  ML score range: {combined_df['ml_score'].min():.3f} - {combined_df['ml_score'].max():.3f}")
    print(f"  Mean ML score: {combined_df['ml_score'].mean():.3f}")
    print(f"  ATR range: {combined_df['atr'].drop_nulls().min():.2f} - {combined_df['atr'].drop_nulls().max():.2f}")
    print(f"\nVIX statistics:")
    print(f"  VIX range: {vix_df['vix'].min():.1f} - {vix_df['vix'].max():.1f}")
    print(f"  Mean VIX: {vix_df['vix'].mean():.1f}")
    print("=" * 60)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
