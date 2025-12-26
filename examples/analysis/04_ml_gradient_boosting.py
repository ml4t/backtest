"""ML Strategy: Gradient Boosting with SHAP-style Analysis.

This example demonstrates:
- Rich feature engineering for ML models
- Gradient boosting classification for directional prediction
- Walk-forward training with proper purging
- SHAP-style feature contribution analysis
- Integration with diagnostic library for error analysis

Uses sklearn's HistGradientBoostingClassifier (no extra dependencies).
For LightGBM/XGBoost, replace the model and use shap.TreeExplainer.

Uses real ETF data from ~/ml4t/data/etfs/

Usage:
    uv run python examples/analysis/04_ml_gradient_boosting.py
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from ml4t.backtest import DataFeed, Engine, ExecutionMode, OrderSide, Strategy
from ml4t.backtest.analysis import BacktestAnalyzer

# Path to ETF data
DATA_DIR = Path.home() / "ml4t" / "data" / "etfs" / "ohlcv_1d"

# Configuration
LOOKBACK_WINDOW = 252  # 1 year training
RETRAIN_FREQUENCY = 63  # Quarterly retraining
PURGE_WINDOW = 5  # Days to purge between train/test


def load_etf_data(
    ticker: str, start_date: str | None = None, end_date: str | None = None
) -> pl.DataFrame:
    """Load ETF OHLCV data."""
    path = DATA_DIR / f"ticker={ticker}" / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")

    df = pl.read_parquet(path)

    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename({"date": "timestamp"})

    df = df.with_columns(pl.lit(ticker).alias("asset"))

    if start_date:
        df = df.filter(pl.col("timestamp") >= datetime.fromisoformat(start_date))
    if end_date:
        df = df.filter(pl.col("timestamp") <= datetime.fromisoformat(end_date))

    return df.sort("timestamp")


def compute_rich_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute comprehensive ML features.

    Feature categories:
    1. Price momentum at multiple horizons
    2. Volatility and risk measures
    3. Volume signals
    4. Technical indicators (simplified)
    5. Regime indicators
    """
    # Build expressions for lazy evaluation
    exprs = [pl.col("timestamp")]

    # === 1. Momentum features ===
    for window in [1, 5, 10, 21, 63, 126]:
        exprs.append(
            ((pl.col("close") / pl.col("close").shift(window)) - 1)
            .fill_null(0)
            .alias(f"ret_{window}d")
        )

    # === 2. Volatility features ===
    daily_ret = (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0)
    for window in [5, 10, 21, 63]:
        exprs.append(daily_ret.rolling_std(window).fill_null(0).alias(f"vol_{window}d"))
        exprs.append(daily_ret.rolling_skew(window).fill_null(0).alias(f"vol_skew_{window}d"))

    # Parkinson volatility
    hl_ratio = (pl.col("high") / pl.col("low")).log().fill_null(0)
    parkinson = (hl_ratio**2).rolling_mean(21).fill_null(0).sqrt() / (4 * np.log(2)) ** 0.5
    exprs.append(parkinson.alias("parkinson_vol_21d"))

    # === 3. Volume features ===
    vol_ma = pl.col("volume").rolling_mean(21).fill_null(pl.col("volume"))
    exprs.append((pl.col("volume") / vol_ma).fill_null(1).alias("volume_ma_ratio"))
    exprs.append(
        (pl.col("volume").rolling_std(21).fill_null(0) / vol_ma.fill_null(1)).alias("volume_std")
    )

    # Price-volume divergence
    price_up = (pl.col("close") > pl.col("close").shift(1)).cast(pl.Int32).fill_null(0)
    vol_up = (pl.col("volume") > pl.col("volume").shift(1)).cast(pl.Int32).fill_null(0)
    exprs.append(
        (price_up != vol_up).cast(pl.Float64).rolling_mean(10).fill_null(0.5).alias("pv_divergence")
    )

    # === 4. Technical indicators ===
    # RSI approximation
    delta = (pl.col("close") - pl.col("close").shift(1)).fill_null(0)
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)
    avg_gain = gain.rolling_mean(14).fill_null(0)
    avg_loss = loss.rolling_mean(14).fill_null(1e-8)
    rs = avg_gain / avg_loss.clip(1e-8, None)
    exprs.append((100 - (100 / (1 + rs))).fill_null(50).alias("rsi_14d"))

    # Bollinger position
    sma_20 = pl.col("close").rolling_mean(20).fill_null(pl.col("close"))
    std_20 = pl.col("close").rolling_std(20).fill_null(1)
    exprs.append(
        ((pl.col("close") - sma_20) / (2 * std_20.clip(0.01, None)))
        .fill_null(0)
        .alias("bb_position")
    )

    # MACD signal
    ema_12 = pl.col("close").ewm_mean(span=12).fill_null(pl.col("close"))
    ema_26 = pl.col("close").ewm_mean(span=26).fill_null(pl.col("close"))
    macd = ema_12 - ema_26
    exprs.append((macd / pl.col("close")).fill_null(0).alias("macd_norm"))

    # === 5. Regime indicators ===
    high_63 = pl.col("high").rolling_max(63).fill_null(pl.col("high"))
    low_63 = pl.col("low").rolling_min(63).fill_null(pl.col("low"))
    exprs.append(((high_63 - pl.col("close")) / high_63).fill_null(0).alias("dist_from_high"))
    exprs.append(((pl.col("close") - low_63) / low_63).fill_null(0).alias("dist_from_low"))

    # Trend strength
    sma_50 = pl.col("close").rolling_mean(50).fill_null(pl.col("close"))
    sma_200 = pl.col("close").rolling_mean(200).fill_null(pl.col("close"))
    exprs.append(((sma_50 - sma_200) / sma_200).fill_null(0).alias("trend_50_200"))

    # === Target: next N-day return ===
    forward_ret = pl.col("close").shift(-5) / pl.col("close") - 1
    exprs.append(forward_ret.fill_null(0).alias("target"))
    exprs.append((forward_ret > 0).cast(pl.Int32).fill_null(0).alias("target_binary"))

    return df.select(exprs)


class GradientBoostingStrategy(Strategy):
    """ML strategy using gradient boosting for direction prediction.

    - Binary classification: predict up/down over next 5 days
    - Position sizing based on prediction probability
    - Tracks feature contributions for analysis
    """

    def __init__(
        self,
        asset: str,
        features_df: pl.DataFrame,
        lookback: int = 252,
        retrain_freq: int = 63,
        purge_window: int = 5,
        prob_threshold: float = 0.55,
        max_position_pct: float = 0.95,
    ):
        """Initialize gradient boosting strategy.

        Args:
            asset: Asset symbol
            features_df: Pre-computed features
            lookback: Training window
            retrain_freq: Bars between retraining
            purge_window: Gap between train and prediction
            prob_threshold: Minimum probability to enter
            max_position_pct: Maximum position size
        """
        self.asset = asset
        self.lookback = lookback
        self.retrain_freq = retrain_freq
        self.purge_window = purge_window
        self.prob_threshold = prob_threshold
        self.max_position_pct = max_position_pct

        # Feature data
        self._features_df = features_df
        self._feature_cols = [
            c for c in features_df.columns if c not in ["timestamp", "target", "target_binary"]
        ]
        self._feature_data = {
            row["timestamp"]: {col: row[col] for col in self._feature_cols}
            for row in features_df.iter_rows(named=True)
        }
        self._target_data = {
            row["timestamp"]: row["target_binary"] for row in features_df.iter_rows(named=True)
        }

        # Model
        self.model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42,
        )
        self.bar_count = 0
        self.history_timestamps: list = []
        self.is_trained = False

        # Analysis tracking
        self.predictions: list[dict] = []
        self.feature_importances: dict[str, float] = {}

    def _get_training_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get training data with purging."""
        if len(self.history_timestamps) < self.lookback + self.purge_window:
            return None

        # Get timestamps: use [:-purge_window] for training
        train_timestamps = self.history_timestamps[
            -(self.lookback + self.purge_window) : -self.purge_window
        ]

        X, y = [], []
        for ts in train_timestamps:
            if ts in self._feature_data and ts in self._target_data:
                features = self._feature_data[ts]
                X.append([features[col] for col in self._feature_cols])
                y.append(self._target_data[ts])

        if len(X) < self.lookback // 2:
            return None

        return np.array(X), np.array(y)

    def _train_model(self):
        """Train/retrain the model."""
        data = self._get_training_data()
        if data is None:
            return

        X, y = data

        # Handle class imbalance by checking if both classes exist
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return

        self.model.fit(X, y)
        self.is_trained = True

        # Compute permutation importance (slower but more reliable)
        # Using a subset for speed
        n_samples = min(100, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        result = permutation_importance(
            self.model, X[indices], y[indices], n_repeats=5, random_state=42, n_jobs=-1
        )
        self.feature_importances = {
            col: float(imp) for col, imp in zip(self._feature_cols, result.importances_mean)
        }

    def _predict(self, timestamp) -> tuple[float, float] | None:
        """Get prediction probability and class."""
        if not self.is_trained or timestamp not in self._feature_data:
            return None

        features = self._feature_data[timestamp]
        X = np.array([[features[col] for col in self._feature_cols]])
        proba = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        return pred_class, confidence

    def on_data(self, timestamp, data, context, broker):
        """Process bar and generate signals."""
        self.bar_count += 1
        self.history_timestamps.append(timestamp)

        # Retrain periodically
        if self.bar_count % self.retrain_freq == 0:
            self._train_model()

        # Get prediction
        prediction = self._predict(timestamp)
        if prediction is None:
            return

        pred_class, confidence = prediction

        # Track predictions for analysis
        self.predictions.append(
            {
                "timestamp": timestamp,
                "prediction": pred_class,
                "confidence": confidence,
            }
        )

        # Get current state
        pos = broker.get_position(self.asset)
        has_position = pos is not None and pos.quantity > 0
        equity = broker.get_account_value()
        price = data[self.asset]["close"]

        # Trading logic
        if pred_class == 1 and confidence > self.prob_threshold:
            # High confidence bullish prediction
            # Scale position by confidence
            confidence_scale = (confidence - 0.5) * 2  # 0-1 scale
            target_value = equity * self.max_position_pct * confidence_scale
            target_qty = target_value / price

            if not has_position:
                broker.submit_order(self.asset, target_qty, OrderSide.BUY)
            elif pos.quantity < target_qty * 0.95:
                diff = target_qty - pos.quantity
                broker.submit_order(self.asset, diff, OrderSide.BUY)

        elif pred_class == 0 or confidence < 0.50:
            # Bearish or low confidence - exit
            if has_position:
                broker.submit_order(self.asset, pos.quantity, OrderSide.SELL)


def main():
    """Run gradient boosting strategy with analysis."""
    print("=" * 70)
    print("ML Strategy: Gradient Boosting with Feature Analysis")
    print("=" * 70)

    # Load data
    print("\nLoading SPY data...")
    df = load_etf_data("SPY", start_date="2008-01-01", end_date="2023-12-31")
    print(f"Loaded {len(df)} bars")

    # Compute features
    print("\nComputing features...")
    features_df = compute_rich_features(df)
    feature_cols = [
        c for c in features_df.columns if c not in ["timestamp", "target", "target_binary"]
    ]
    print(f"Total features: {len(feature_cols)}")
    print("Feature categories: momentum, volatility, volume, technical, regime")

    # Create backtest
    prices_df = df.select(["timestamp", "asset", "open", "high", "low", "close", "volume"])
    feed = DataFeed(prices_df=prices_df)

    strategy = GradientBoostingStrategy(
        asset="SPY",
        features_df=features_df,
        lookback=LOOKBACK_WINDOW,
        retrain_freq=RETRAIN_FREQUENCY,
        purge_window=PURGE_WINDOW,
        prob_threshold=0.55,
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

    # Feature importance
    print("\n" + "-" * 70)
    print("FEATURE IMPORTANCE (Permutation-based)")
    print("-" * 70)

    if strategy.feature_importances:
        sorted_imp = sorted(
            strategy.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:15]  # Top 15

        max_imp = max(v for _, v in sorted_imp) if sorted_imp else 1
        for feature, imp in sorted_imp:
            bar_len = int(30 * imp / max_imp) if max_imp > 0 else 0
            bar = "█" * bar_len
            print(f"  {feature:20s}: {bar} {imp:.4f}")

    # Prediction analysis
    print("\n" + "-" * 70)
    print("PREDICTION ANALYSIS")
    print("-" * 70)

    if strategy.predictions:
        preds_df = pl.DataFrame(strategy.predictions)

        # Confidence distribution
        print("\nConfidence distribution:")
        print(f"  Mean:   {preds_df['confidence'].mean():.2%}")
        print(f"  Median: {preds_df['confidence'].median():.2%}")
        print(f"  Std:    {preds_df['confidence'].std():.2%}")

        # Prediction balance
        pred_counts = preds_df.group_by("prediction").agg(pl.len().alias("count"))
        print("\nPrediction balance:")
        for row in pred_counts.iter_rows(named=True):
            label = "Bullish" if row["prediction"] == 1 else "Bearish"
            print(f"  {label}: {row['count']} ({row['count'] / len(preds_df):.1%})")

    # Trades analysis
    trades_df = analyzer.get_trades_dataframe()

    if len(trades_df) > 0:
        print("\n" + "-" * 70)
        print("TRADE TIMING ANALYSIS")
        print("-" * 70)

        # By year
        yearly = (
            trades_df.with_columns(pl.col("exit_time").dt.year().alias("year"))
            .group_by("year")
            .agg(
                pl.len().alias("trades"),
                pl.col("pnl").sum().alias("pnl"),
                (pl.col("pnl") > 0).mean().alias("win_rate"),
            )
            .sort("year")
        )

        print("\nPerformance by year:")
        print(yearly)

    # SHAP integration guidance
    print("\n" + "=" * 70)
    print("SHAP INTEGRATION (for deeper analysis)")
    print("=" * 70)
    print("""
For production SHAP analysis with LightGBM/XGBoost:

    # Install: pip install lightgbm shap

    import lightgbm as lgb
    import shap

    # Replace sklearn model with LightGBM
    model = lgb.LGBMClassifier(...)
    model.fit(X_train, y_train)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Analyze worst trades
    from ml4t.diagnostic.evaluation import TradeShapAnalyzer
    from ml4t.diagnostic.config import TradeConfig

    config = TradeConfig(n_worst=50, n_clusters=5)
    shap_analyzer = TradeShapAnalyzer(
        model=model,
        features=features_df,
        shap_values=shap_values,
        config=config
    )

    # Get SHAP-based error patterns
    result = shap_analyzer.analyze_worst_trades(trades_df)

    # Actionable insights:
    # - Which features drove bad predictions?
    # - Are there clusters of similar failures?
    # - What conditions led to false signals?

    print(result.error_patterns)
    print(result.improvement_hypotheses)
""")

    # Export
    if len(trades_df) > 0:
        output_path = Path(__file__).parent / "ml_gbm_trades.parquet"
        trades_df.write_parquet(output_path)
        print(f"\nTrades exported to: {output_path}")


if __name__ == "__main__":
    main()
