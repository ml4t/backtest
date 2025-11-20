"""Unit tests for FeatureProvider interface and implementations."""

from datetime import datetime, timezone

import polars as pl
import pytest

from ml4t.backtest.data.feature_provider import (
    CallableFeatureProvider,
    FeatureProvider,
    PrecomputedFeatureProvider,
)


class TestPrecomputedFeatureProvider:
    """Tests for PrecomputedFeatureProvider."""

    @pytest.fixture
    def features_df(self) -> pl.DataFrame:
        """Sample features DataFrame with per-asset and market-wide data."""
        return pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, tzinfo=timezone.utc),
                    # Market-wide features (asset_id = None)
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 1, 2, tzinfo=timezone.utc),
                ],
                "asset_id": ["AAPL", "MSFT", "AAPL", "MSFT", None, None],
                "atr": [2.5, 3.1, 2.6, 3.2, None, None],
                "rsi": [65.0, 45.0, 70.0, 50.0, None, None],
                "ml_score": [0.8, 0.3, 0.9, 0.4, None, None],
                "vix": [None, None, None, None, 15.2, 18.5],
                "spy_return": [None, None, None, None, 0.01, -0.02],
            }
        )

    @pytest.fixture
    def provider(self, features_df: pl.DataFrame) -> PrecomputedFeatureProvider:
        """Create PrecomputedFeatureProvider instance."""
        return PrecomputedFeatureProvider(features_df)

    def test_initialization(self, provider: PrecomputedFeatureProvider):
        """Test provider initialization and feature column detection."""
        assert provider.timestamp_col == "timestamp"
        assert provider.asset_col == "asset_id"
        assert set(provider.feature_cols) == {
            "atr",
            "rsi",
            "ml_score",
            "vix",
            "spy_return",
        }

    def test_get_features_existing_data(
        self, provider: PrecomputedFeatureProvider
    ):
        """Test retrieving features for existing asset and timestamp."""
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        features = provider.get_features("AAPL", timestamp)

        assert features["atr"] == 2.5
        assert features["rsi"] == 65.0
        assert features["ml_score"] == 0.8
        # Market features should be None for per-asset rows, converted to 0.0
        assert features["vix"] == 0.0
        assert features["spy_return"] == 0.0

    def test_get_features_different_asset(
        self, provider: PrecomputedFeatureProvider
    ):
        """Test retrieving features for different asset."""
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        features = provider.get_features("MSFT", timestamp)

        assert features["atr"] == 3.1
        assert features["rsi"] == 45.0
        assert features["ml_score"] == 0.3

    def test_get_features_different_timestamp(
        self, provider: PrecomputedFeatureProvider
    ):
        """Test retrieving features for different timestamp."""
        timestamp = datetime(2024, 1, 2, tzinfo=timezone.utc)
        features = provider.get_features("AAPL", timestamp)

        assert features["atr"] == 2.6
        assert features["rsi"] == 70.0
        assert features["ml_score"] == 0.9

    def test_get_features_missing_data(
        self, provider: PrecomputedFeatureProvider
    ):
        """Test retrieving features for non-existent asset/timestamp."""
        timestamp = datetime(2024, 1, 3, tzinfo=timezone.utc)
        features = provider.get_features("AAPL", timestamp)

        assert features == {}

    def test_get_features_missing_asset(
        self, provider: PrecomputedFeatureProvider
    ):
        """Test retrieving features for non-existent asset."""
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        features = provider.get_features("GOOG", timestamp)

        assert features == {}

    def test_get_market_features_existing_data(
        self, provider: PrecomputedFeatureProvider
    ):
        """Test retrieving market-wide features."""
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        market_features = provider.get_market_features(timestamp)

        assert market_features["vix"] == 15.2
        assert market_features["spy_return"] == 0.01
        # Per-asset features should be None for market rows, converted to 0.0
        assert market_features["atr"] == 0.0
        assert market_features["rsi"] == 0.0
        assert market_features["ml_score"] == 0.0

    def test_get_market_features_different_timestamp(
        self, provider: PrecomputedFeatureProvider
    ):
        """Test retrieving market features for different timestamp."""
        timestamp = datetime(2024, 1, 2, tzinfo=timezone.utc)
        market_features = provider.get_market_features(timestamp)

        assert market_features["vix"] == 18.5
        assert market_features["spy_return"] == -0.02

    def test_get_market_features_missing_data(
        self, provider: PrecomputedFeatureProvider
    ):
        """Test retrieving market features for non-existent timestamp."""
        timestamp = datetime(2024, 1, 3, tzinfo=timezone.utc)
        market_features = provider.get_market_features(timestamp)

        assert market_features == {}

    def test_custom_column_names(self, features_df: pl.DataFrame):
        """Test provider with custom column names."""
        # Rename columns
        renamed_df = features_df.rename(
            {"timestamp": "ts", "asset_id": "symbol"}
        )
        provider = PrecomputedFeatureProvider(
            renamed_df, timestamp_col="ts", asset_col="symbol"
        )

        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
        features = provider.get_features("AAPL", timestamp)

        assert features["atr"] == 2.5
        assert features["rsi"] == 65.0


class TestCallableFeatureProvider:
    """Tests for CallableFeatureProvider."""

    def test_get_features_simple_callable(self):
        """Test retrieving features via simple callable."""

        def compute_fn(asset_id: str, timestamp: datetime) -> dict[str, float]:
            # Simple mock computation
            return {
                "atr": 2.5 if asset_id == "AAPL" else 3.1,
                "rsi": 65.0 if asset_id == "AAPL" else 45.0,
            }

        provider = CallableFeatureProvider(compute_fn)
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

        features_aapl = provider.get_features("AAPL", timestamp)
        assert features_aapl["atr"] == 2.5
        assert features_aapl["rsi"] == 65.0

        features_msft = provider.get_features("MSFT", timestamp)
        assert features_msft["atr"] == 3.1
        assert features_msft["rsi"] == 45.0

    def test_get_features_with_timestamp(self):
        """Test callable receives correct timestamp."""
        received_timestamps = []

        def compute_fn(asset_id: str, timestamp: datetime) -> dict[str, float]:
            received_timestamps.append(timestamp)
            return {"feature": 1.0}

        provider = CallableFeatureProvider(compute_fn)
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 2, tzinfo=timezone.utc)

        provider.get_features("AAPL", ts1)
        provider.get_features("AAPL", ts2)

        assert received_timestamps == [ts1, ts2]

    def test_get_features_error_handling(self):
        """Test callable error handling returns empty dict."""

        def compute_fn(asset_id: str, timestamp: datetime) -> dict[str, float]:
            raise ValueError("Computation failed")

        provider = CallableFeatureProvider(compute_fn)
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

        features = provider.get_features("AAPL", timestamp)
        assert features == {}

    def test_get_market_features_with_callable(self):
        """Test retrieving market features via callable."""

        def compute_asset_fn(
            asset_id: str, timestamp: datetime
        ) -> dict[str, float]:
            return {"atr": 2.5}

        def compute_market_fn(timestamp: datetime) -> dict[str, float]:
            return {"vix": 15.2, "spy_return": 0.01}

        provider = CallableFeatureProvider(compute_asset_fn, compute_market_fn)
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

        market_features = provider.get_market_features(timestamp)
        assert market_features["vix"] == 15.2
        assert market_features["spy_return"] == 0.01

    def test_get_market_features_no_callable(self):
        """Test market features returns empty dict when no callable provided."""

        def compute_fn(asset_id: str, timestamp: datetime) -> dict[str, float]:
            return {"atr": 2.5}

        provider = CallableFeatureProvider(compute_fn)
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

        market_features = provider.get_market_features(timestamp)
        assert market_features == {}

    def test_get_market_features_error_handling(self):
        """Test market features error handling returns empty dict."""

        def compute_fn(asset_id: str, timestamp: datetime) -> dict[str, float]:
            return {}

        def compute_market_fn(timestamp: datetime) -> dict[str, float]:
            raise ValueError("Market data unavailable")

        provider = CallableFeatureProvider(compute_fn, compute_market_fn)
        timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)

        market_features = provider.get_market_features(timestamp)
        assert market_features == {}


class TestFeatureProviderInterface:
    """Tests for FeatureProvider abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that FeatureProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FeatureProvider()  # type: ignore

    def test_concrete_implementations_satisfy_interface(self):
        """Test that concrete implementations satisfy the interface."""
        # PrecomputedFeatureProvider
        df = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
                "asset_id": ["AAPL"],
                "atr": [2.5],
            }
        )
        provider1 = PrecomputedFeatureProvider(df)
        assert isinstance(provider1, FeatureProvider)

        # CallableFeatureProvider
        def compute_fn(asset_id: str, timestamp: datetime) -> dict[str, float]:
            return {}

        provider2 = CallableFeatureProvider(compute_fn)
        assert isinstance(provider2, FeatureProvider)
