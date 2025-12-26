"""ML Strategy: Linear Regression for Return Prediction.

This example demonstrates:
- Feature engineering from price data (momentum, volatility, volume)
- Training a simple linear regression to predict next-day returns
- Walk-forward (rolling) model updates to avoid look-ahead bias
- Backtesting the ML-based strategy
- Analyzing trade performance and feature importance

Uses real ETF data from ~/ml4t/data/etfs/

Usage:
    uv run python examples/analysis/03_ml_linear_regression.py
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from ml4t.backtest import DataFeed, Engine, ExecutionMode, OrderSide, Strategy
from ml4t.backtest.analysis import BacktestAnalyzer

# Path to ETF data
DATA_DIR = Path.home() / "ml4t" / "data" / "etfs" / "ohlcv_1d"

# Feature and model parameters
FEATURE_WINDOWS = [5, 10, 21, 63]  # Short to medium-term features
LOOKBACK_WINDOW = 252  # 1 year of training data
RETRAIN_FREQUENCY = 21  # Retrain monthly


def load_etf_data(
    ticker: str, start_date: str | None = None, end_date: str | None = None
) -> pl.DataFrame:
    """Load ETF OHLCV data from Hive-partitioned parquet."""
    path = DATA_DIR / f"ticker={ticker}" / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")

    df = pl.read_parquet(path)

    # Normalize column name if needed
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename({"date": "timestamp"})

    df = df.with_columns(pl.lit(ticker).alias("asset"))

    if start_date:
        df = df.filter(pl.col("timestamp") >= datetime.fromisoformat(start_date))
    if end_date:
        df = df.filter(pl.col("timestamp") <= datetime.fromisoformat(end_date))

    return df.sort("timestamp")


def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute ML features from OHLCV data.

    Features:
    - Momentum: returns over various windows
    - Volatility: rolling std of returns
    - Volume: relative volume
    - Price position: close relative to high/low range
    """
    # Returns at various horizons
    close = df["close"]

    features = {"timestamp": df["timestamp"]}

    for window in FEATURE_WINDOWS:
        # Momentum (past returns)
        features[f"momentum_{window}d"] = (close / close.shift(window) - 1).fill_null(0)

        # Volatility (rolling std of daily returns)
        daily_ret = (close / close.shift(1) - 1).fill_null(0)
        features[f"volatility_{window}d"] = daily_ret.rolling_std(window).fill_null(0)

    # Volume features
    volume = df["volume"]
    features["volume_ratio_21d"] = (volume / volume.rolling_mean(21)).fill_null(1)

    # Price position (where close is in the high-low range)
    high = df["high"]
    low = df["low"]
    features["price_position"] = ((close - low) / (high - low + 1e-8)).fill_null(0.5)

    # Target: next-day return (forward-looking, for training only)
    features["target"] = (close.shift(-1) / close - 1).fill_null(0)

    return pl.DataFrame(features)


class LinearRegressionStrategy(Strategy):
    """ML strategy using linear regression for return prediction.

    - Uses walk-forward training (retrain every N bars)
    - Goes long when predicted return > threshold
    - Goes flat when predicted return < -threshold
    - Position sizing based on prediction magnitude
    """

    def __init__(
        self,
        asset: str,
        features_df: pl.DataFrame,
        lookback: int = 252,
        retrain_freq: int = 21,
        prediction_threshold: float = 0.001,
        max_position_pct: float = 0.95,
    ):
        """Initialize ML strategy.

        Args:
            asset: Asset symbol to trade
            features_df: Pre-computed features DataFrame
            lookback: Training window size
            retrain_freq: Bars between model retraining
            prediction_threshold: Minimum predicted return to trade
            max_position_pct: Maximum portfolio fraction to invest
        """
        self.asset = asset
        self.lookback = lookback
        self.retrain_freq = retrain_freq
        self.prediction_threshold = prediction_threshold
        self.max_position_pct = max_position_pct

        # Convert features to dict for fast timestamp lookup
        self._features_df = features_df
        self._feature_cols = [c for c in features_df.columns if c not in ["timestamp", "target"]]
        self._feature_data = {
            row["timestamp"]: {col: row[col] for col in self._feature_cols}
            for row in features_df.iter_rows(named=True)
        }
        self._target_data = {
            row["timestamp"]: row["target"] for row in features_df.iter_rows(named=True)
        }

        # Model state
        self.model = Ridge(alpha=1.0)
        self.bar_count = 0
        self.history_timestamps: list = []
        self.is_trained = False

        # Track feature importance
        self.feature_importances: dict[str, float] = {}

    def _get_training_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get training data from history."""
        if len(self.history_timestamps) < self.lookback:
            return None

        # Use most recent lookback timestamps
        train_timestamps = self.history_timestamps[-self.lookback :]

        X = []
        y = []
        for ts in train_timestamps:
            if ts in self._feature_data and ts in self._target_data:
                features = self._feature_data[ts]
                X.append([features[col] for col in self._feature_cols])
                y.append(self._target_data[ts])

        if len(X) < self.lookback // 2:  # Require at least half the data
            return None

        return np.array(X), np.array(y)

    def _train_model(self):
        """Train/retrain the model on recent data."""
        data = self._get_training_data()
        if data is None:
            return

        X, y = data

        # Clip extreme targets to reduce impact of outliers
        y = np.clip(y, -0.10, 0.10)

        self.model.fit(X, y)
        self.is_trained = True

        # Store feature importance (coefficients)
        self.feature_importances = {
            col: abs(coef) for col, coef in zip(self._feature_cols, self.model.coef_)
        }

    def _predict(self, timestamp) -> float | None:
        """Get prediction for current bar."""
        if not self.is_trained or timestamp not in self._feature_data:
            return None

        features = self._feature_data[timestamp]
        X = np.array([[features[col] for col in self._feature_cols]])
        return float(self.model.predict(X)[0])

    def on_data(self, timestamp, data, context, broker):
        """Process bar and generate trading signals."""
        self.bar_count += 1

        # Record timestamp for training history
        self.history_timestamps.append(timestamp)

        # Retrain model periodically
        if self.bar_count % self.retrain_freq == 0:
            self._train_model()

        # Get prediction
        prediction = self._predict(timestamp)
        if prediction is None:
            return

        # Get current state
        pos = broker.get_position(self.asset)
        has_position = pos is not None and pos.quantity > 0
        equity = broker.get_account_value()
        price = data[self.asset]["close"]

        # Trading logic
        if prediction > self.prediction_threshold:
            # Strong positive signal - go long
            target_value = equity * self.max_position_pct
            target_qty = target_value / price

            if not has_position:
                broker.submit_order(self.asset, target_qty, OrderSide.BUY)
            elif pos.quantity < target_qty * 0.95:
                # Scale up
                diff = target_qty - pos.quantity
                broker.submit_order(self.asset, diff, OrderSide.BUY)

        elif prediction < -self.prediction_threshold:
            # Negative signal - exit position
            if has_position:
                broker.submit_order(self.asset, pos.quantity, OrderSide.SELL)


def main():
    """Run ML strategy on SPY and analyze performance."""
    print("=" * 70)
    print("ML Strategy: Linear Regression Return Prediction")
    print("=" * 70)

    # Load data
    print("\nLoading SPY data...")
    df = load_etf_data("SPY", start_date="2010-01-01", end_date="2023-12-31")
    print(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Compute features
    print("\nComputing features...")
    features_df = compute_features(df)
    print(f"Features: {[c for c in features_df.columns if c not in ['timestamp', 'target']]}")

    # Create combined data for backtest
    prices_df = df.select(["timestamp", "asset", "open", "high", "low", "close", "volume"])
    feed = DataFeed(prices_df=prices_df)

    # Create ML strategy
    strategy = LinearRegressionStrategy(
        asset="SPY",
        features_df=features_df,
        lookback=LOOKBACK_WINDOW,
        retrain_freq=RETRAIN_FREQUENCY,
        prediction_threshold=0.001,
    )

    # Run backtest
    print("\nRunning backtest...")
    initial_cash = 100_000.0
    engine = Engine(
        feed=feed,
        strategy=strategy,
        initial_cash=initial_cash,
        execution_mode=ExecutionMode.NEXT_BAR,
    )
    result = engine.run()

    # Results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"Initial Capital:  ${initial_cash:,.2f}")
    print(f"Final Value:      ${result['final_value']:,.2f}")
    print(f"Total Return:     {result['total_return']:.2%}")
    print(f"Sharpe Ratio:     {result['sharpe']:.2f}")
    print(f"Max Drawdown:     {result['max_drawdown_pct']:.2%}")

    # Trade analysis
    print("\n" + "=" * 70)
    print("TRADE ANALYSIS")
    print("=" * 70)

    analyzer = BacktestAnalyzer(engine)
    stats = analyzer.trade_statistics()
    print(stats.summary())

    # Feature importance from final model
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE (from final model)")
    print("-" * 70)

    if strategy.feature_importances:
        sorted_importance = sorted(
            strategy.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for feature, importance in sorted_importance:
            print(f"  {feature:20s}: {importance:.6f}")

    # Detailed trade analysis
    trades_df = analyzer.get_trades_dataframe()

    if len(trades_df) > 0:
        print("\n" + "-" * 70)
        print("TRADE DURATION ANALYSIS")
        print("-" * 70)

        # Group trades by holding period
        trades_with_duration = trades_df.with_columns(
            pl.when(pl.col("bars_held") <= 5)
            .then(pl.lit("1-5 days"))
            .when(pl.col("bars_held") <= 21)
            .then(pl.lit("1-4 weeks"))
            .when(pl.col("bars_held") <= 63)
            .then(pl.lit("1-3 months"))
            .otherwise(pl.lit("3+ months"))
            .alias("duration_bucket")
        )

        duration_stats = (
            trades_with_duration.group_by("duration_bucket")
            .agg(
                pl.len().alias("trades"),
                pl.col("pnl").sum().alias("total_pnl"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
                pl.col("pnl_percent").mean().alias("avg_return"),
            )
            .sort("trades", descending=True)
        )

        print(duration_stats)

        print("\n" + "-" * 70)
        print("BEST AND WORST TRADES")
        print("-" * 70)

        print("\nTop 5 Best:")
        for row in trades_df.sort("pnl", descending=True).head(5).iter_rows(named=True):
            print(
                f"  {row['entry_time'].date()} → {row['exit_time'].date()} "
                f"({row['bars_held']} bars): ${row['pnl']:,.2f} ({row['pnl_percent']:.2%})"
            )

        print("\nTop 5 Worst:")
        for row in trades_df.sort("pnl").head(5).iter_rows(named=True):
            print(
                f"  {row['entry_time'].date()} → {row['exit_time'].date()} "
                f"({row['bars_held']} bars): ${row['pnl']:,.2f} ({row['pnl_percent']:.2%})"
            )

    # Diagnostic integration
    print("\n" + "=" * 70)
    print("ML DIAGNOSTIC INTEGRATION")
    print("=" * 70)
    print("""
For deeper ML model analysis with ml4t.diagnostic:

    from ml4t.backtest.analysis import BacktestAnalyzer
    from ml4t.diagnostic.evaluation import TradeShapAnalyzer

    # Get trade records with feature context
    analyzer = BacktestAnalyzer(engine)
    records = analyzer.get_trade_records()

    # Enrich with feature values at entry/exit
    enriched_records = enrich_with_features(records, features_df)

    # Use SHAP to explain worst trades
    shap_analyzer = TradeShapAnalyzer(
        model=strategy.model,
        features_df=features_df,
        shap_values=shap_values  # From shap.Explainer
    )
    patterns = shap_analyzer.explain_worst_trades(worst_20)
    # → Identifies which features drove bad predictions

    # Example insight:
    # "High volatility periods have high momentum_5d but model
    #  overweights this feature leading to false signals"
""")

    # Export
    if len(trades_df) > 0:
        output_path = Path(__file__).parent / "ml_linreg_trades.parquet"
        trades_df.write_parquet(output_path)
        print(f"\nTrades exported to: {output_path}")


if __name__ == "__main__":
    main()
