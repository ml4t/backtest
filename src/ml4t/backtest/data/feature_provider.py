"""Feature provider interface for per-asset signals and market-wide context.

The FeatureProvider abstraction enables pluggable feature computation/retrieval.
It provides two types of data:

1. **Per-asset signals** (get_features): All numerical features for one asset
   - ML predictions, technical indicators, computed features
   - Goes into MarketEvent.signals dict
   - Used for: entry/exit decisions, position sizing, risk management

2. **Market context** (get_market_features): Market-wide data shared across all assets
   - VIX, SPY returns, sector indices, regime indicators
   - Goes into MarketEvent.context dict
   - Used for: regime filtering, volatility adjustment, correlation

Design Principles:
    - Unified signals model: ML scores and indicators treated identically
    - Point-in-time correctness - only return data available at decision time
    - Supports both precomputed (fast) and on-the-fly (flexible) patterns

Examples:
    >>> # Precomputed features (common for backtesting)
    >>> features_df = pl.read_parquet("features.parquet")
    >>> provider = PrecomputedFeatureProvider(features_df)
    >>> signals = provider.get_features("AAPL", timestamp)
    >>> # signals = {'ml_score': 0.85, 'atr_20': 2.5, 'rsi_14': 65}
    >>>
    >>> # Callable features (on-the-fly computation)
    >>> def compute_signals(asset_id, timestamp):
    ...     # Custom computation logic (ML inference, indicators, etc.)
    ...     return {'custom_signal': value, 'atr': atr_value}
    >>> provider = CallableFeatureProvider(compute_signals)
    >>>
    >>> # Market-wide context
    >>> context = provider.get_market_features(timestamp)
    >>> # context = {'vix': 18.5, 'spy_return': 0.005}
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable

import polars as pl

from ml4t.backtest.core.types import AssetId


class FeatureProvider(ABC):
    """Abstract base for signal and context computation/retrieval.

    Provides two types of data:
    1. **Per-asset signals**: All numerical features for one asset (ML + indicators)
    2. **Market context**: Market-wide data shared across all assets (VIX, SPY, etc.)

    Implementations must ensure point-in-time correctness - only return data
    available at the requested timestamp.
    """

    @abstractmethod
    def get_features(
        self, asset_id: AssetId, timestamp: datetime
    ) -> dict[str, float]:
        """Get per-asset signals at specific timestamp.

        This includes ALL per-asset numerical features: ML predictions,
        technical indicators, computed features, etc. The user's strategy
        code decides how to use them (entry, exit, sizing, risk management).

        Args:
            asset_id: Asset identifier (e.g., "AAPL", "BTC-USD")
            timestamp: Point in time for feature retrieval

        Returns:
            Dictionary of signal name → value pairs
            Empty dict if no signals available

        Examples:
            >>> signals = provider.get_features("AAPL", timestamp)
            >>> # All numerical features in one dict:
            >>> atr = signals.get('atr_20', 0.0)
            >>> rsi = signals.get('rsi_14', 50.0)
            >>> ml_score = signals.get('ml_score', 0.0)
            >>> momentum = signals.get('momentum_20', 0.0)

        Note:
            Must respect point-in-time correctness - only return features
            that were available at `timestamp`, not future data.
        """
        pass

    @abstractmethod
    def get_market_features(self, timestamp: datetime) -> dict[str, float]:
        """Get market-wide features at specific timestamp.

        Market features are shared across all assets and represent market conditions,
        regimes, or macro indicators.

        Args:
            timestamp: Point in time for feature retrieval

        Returns:
            Dictionary of feature name → value pairs
            Empty dict if no features available

        Examples:
            >>> market_features = provider.get_market_features(timestamp)
            >>> vix = market_features.get('vix', 15.0)
            >>> spy_return = market_features.get('spy_return', 0.0)
            >>> regime = market_features.get('market_regime', 0.0)

        Use Cases:
            - VIX filtering: "Don't trade if VIX > 30"
            - Market regime detection: Different strategies for trending/mean-reverting
            - Sector rotation: Adjust positions based on sector performance
            - Correlation: Multi-asset strategies need market-wide context
        """
        pass


class PrecomputedFeatureProvider(FeatureProvider):
    """Feature provider for precomputed features stored in DataFrame.

    Efficient for backtesting where all features are computed ahead of time.
    Uses Polars for fast lookups.

    Expected schema:
        - timestamp: datetime (when features are valid)
        - asset_id: str (optional, None for market-wide features)
        - feature columns: float (ATR, RSI, ML scores, etc.)

    Args:
        features_df: Polars DataFrame with precomputed features
        timestamp_col: Column name for timestamp (default: 'timestamp')
        asset_col: Column name for asset_id (default: 'asset_id')

    Examples:
        >>> # Per-asset features
        >>> features_df = pl.DataFrame({
        ...     'timestamp': [...],
        ...     'asset_id': ['AAPL', 'AAPL', 'MSFT', ...],
        ...     'atr': [2.5, 2.6, 3.1, ...],
        ...     'rsi': [65, 70, 45, ...],
        ...     'ml_score': [0.8, 0.9, 0.3, ...]
        ... })
        >>> provider = PrecomputedFeatureProvider(features_df)
        >>>
        >>> # Market-wide features (asset_id = None)
        >>> market_df = pl.DataFrame({
        ...     'timestamp': [...],
        ...     'asset_id': [None, None, ...],
        ...     'vix': [15.2, 18.5, ...],
        ...     'spy_return': [0.01, -0.02, ...]
        ... })
        >>> provider = PrecomputedFeatureProvider(market_df)
    """

    def __init__(
        self,
        features_df: pl.DataFrame,
        timestamp_col: str = "timestamp",
        asset_col: str = "asset_id",
    ):
        self.features_df = features_df
        self.timestamp_col = timestamp_col
        self.asset_col = asset_col

        # Identify feature columns (exclude timestamp and asset_id)
        self.feature_cols = [
            col
            for col in features_df.columns
            if col not in [timestamp_col, asset_col]
        ]

    def get_features(
        self, asset_id: AssetId, timestamp: datetime
    ) -> dict[str, float]:
        """Get per-asset features from precomputed DataFrame."""
        row = self.features_df.filter(
            (pl.col(self.timestamp_col) == timestamp)
            & (pl.col(self.asset_col) == asset_id)
        )

        if row.height == 0:
            return {}

        # Convert first row to dict, excluding timestamp and asset_id
        features = row.select(self.feature_cols).to_dicts()[0]
        return {k: float(v) if v is not None else 0.0 for k, v in features.items()}

    def get_market_features(self, timestamp: datetime) -> dict[str, float]:
        """Get market-wide features from precomputed DataFrame.

        Looks for rows where asset_id is None (market-wide data).
        """
        row = self.features_df.filter(
            (pl.col(self.timestamp_col) == timestamp)
            & (pl.col(self.asset_col).is_null())
        )

        if row.height == 0:
            return {}

        features = row.select(self.feature_cols).to_dicts()[0]
        return {k: float(v) if v is not None else 0.0 for k, v in features.items()}


class CallableFeatureProvider(FeatureProvider):
    """Feature provider for on-the-fly computation via callable.

    Useful for:
    - Real-time trading (compute features from live data)
    - Complex features requiring external services (ML inference APIs)
    - Features depending on dynamic state

    Args:
        compute_fn: Callable taking (asset_id, timestamp) → dict[str, float]
        compute_market_fn: Optional callable for market features taking
                          timestamp → dict[str, float]. If None, returns empty dict.

    Examples:
        >>> # Simple on-the-fly computation
        >>> def compute_atr(asset_id, timestamp):
        ...     # Fetch recent prices and compute ATR
        ...     prices = get_recent_prices(asset_id, timestamp, lookback=14)
        ...     atr = compute_atr_from_prices(prices)
        ...     return {'atr': atr}
        >>>
        >>> provider = CallableFeatureProvider(compute_atr)
        >>>
        >>> # With market features
        >>> def compute_market(timestamp):
        ...     vix = fetch_vix(timestamp)
        ...     return {'vix': vix}
        >>>
        >>> provider = CallableFeatureProvider(compute_atr, compute_market)
    """

    def __init__(
        self,
        compute_fn: Callable[[AssetId, datetime], dict[str, float]],
        compute_market_fn: Callable[[datetime], dict[str, float]] | None = None,
    ):
        self.compute_fn = compute_fn
        self.compute_market_fn = compute_market_fn

    def get_features(
        self, asset_id: AssetId, timestamp: datetime
    ) -> dict[str, float]:
        """Compute per-asset features on-the-fly."""
        try:
            return self.compute_fn(asset_id, timestamp)
        except Exception as e:
            # Log error and return empty dict to avoid breaking backtest
            # In production, might want more sophisticated error handling
            print(f"Error computing features for {asset_id} at {timestamp}: {e}")
            return {}

    def get_market_features(self, timestamp: datetime) -> dict[str, float]:
        """Compute market-wide features on-the-fly."""
        if self.compute_market_fn is None:
            return {}

        try:
            return self.compute_market_fn(timestamp)
        except Exception as e:
            print(f"Error computing market features at {timestamp}: {e}")
            return {}
