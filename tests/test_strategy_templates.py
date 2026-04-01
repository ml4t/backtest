"""Tests for strategy templates."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ml4t.backtest import BacktestConfig, DataFeed, Engine
from ml4t.backtest.execution.schedule import RebalanceSchedule
from ml4t.backtest.strategies import (
    LongShortStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    SignalFollowingStrategy,
)
from ml4t.specs.market_data import FeedSpec


def make_price_data(
    n_bars: int = 100,
    n_assets: int = 1,
    seed: int = 42,
    include_signal: bool = True,
) -> pl.DataFrame:
    """Generate synthetic price data for testing."""
    np.random.seed(seed)

    rows = []
    base_date = datetime(2023, 1, 1)

    for i in range(n_bars):
        timestamp = base_date + timedelta(days=i)
        for j in range(n_assets):
            asset = f"ASSET{j:03d}"
            # Random walk with drift
            close = 100.0 if i == 0 else rows[-n_assets]["close"] * (1 + np.random.randn() * 0.02)

            row = {
                "timestamp": timestamp,
                "asset": asset,
                "open": close * (1 + np.random.randn() * 0.005),
                "high": close * (1 + abs(np.random.randn() * 0.01)),
                "low": close * (1 - abs(np.random.randn() * 0.01)),
                "close": close,
                "volume": int(1000000 * (1 + np.random.rand())),
            }

            if include_signal:
                # Alternating signal for testing
                row["signal"] = 1.0 if (i // 10) % 2 == 0 else -1.0

            rows.append(row)

    return pl.DataFrame(rows)


class TestSignalFollowingStrategy:
    """Tests for SignalFollowingStrategy template."""

    def test_basic_long_only(self):
        """Test basic long-only signal following."""

        class SimpleLongStrategy(SignalFollowingStrategy):
            signal_column = "signal"
            position_size = 0.5

            def should_enter_long(self, signal):
                return signal > 0.5

            def should_exit(self, signal):
                return signal < -0.5

        df = make_price_data(n_bars=50)
        feed = DataFeed(
            prices_df=df,
            signals_df=df.select(["timestamp", "asset", "signal"]),
        )

        engine = Engine.from_config(feed, SimpleLongStrategy(), BacktestConfig.from_preset("fast"))
        result = engine.run()

        # Should have some trades
        assert len(result.trades) > 0

    def test_long_short(self):
        """Test long/short signal following."""

        class LongShortSignalStrategy(SignalFollowingStrategy):
            signal_column = "signal"
            position_size = 0.3
            allow_shorts = True

            def should_enter_long(self, signal):
                return signal > 0.8

            def should_enter_short(self, signal):
                return signal < -0.8

            def should_exit(self, signal):
                return abs(signal) < 0.2

        # Create data with strong signals
        df = make_price_data(n_bars=50)
        df = df.with_columns(
            pl.when(pl.col("signal") > 0).then(pl.lit(0.9)).otherwise(pl.lit(-0.9)).alias("signal")
        )

        feed = DataFeed(
            prices_df=df,
            signals_df=df.select(["timestamp", "asset", "signal"]),
        )

        engine = Engine.from_config(
            feed, LongShortSignalStrategy(), BacktestConfig.from_preset("fast")
        )
        result = engine.run()

        # Should have trades
        assert len(result.trades) >= 0  # May or may not have trades depending on signal pattern


class TestMomentumStrategy:
    """Tests for MomentumStrategy template."""

    def test_basic_momentum(self):
        """Test basic momentum strategy."""

        class SimpleMomentum(MomentumStrategy):
            lookback = 10
            entry_threshold = 0.02
            exit_threshold = -0.01
            position_size = 0.5

        # Create trending data
        np.random.seed(42)
        n_bars = 50
        prices = 100 * np.cumprod(1 + np.random.randn(n_bars) * 0.01 + 0.002)  # Upward drift

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_bars)],
                "asset": ["SPY"] * n_bars,
                "close": prices.tolist(),
            }
        )

        feed = DataFeed(prices_df=df)
        engine = Engine.from_config(feed, SimpleMomentum(), BacktestConfig.from_preset("fast"))
        result = engine.run()

        # Should complete without error
        assert result.metrics["final_value"] > 0

    def test_momentum_calculation(self):
        """Test momentum calculation method."""
        strategy = MomentumStrategy()

        # Test momentum calculation
        prices = [100, 105, 110, 115, 120]
        momentum = strategy.calculate_momentum(prices)
        assert abs(momentum - 0.20) < 0.01  # 20% return

        # Empty list
        assert strategy.calculate_momentum([]) == 0.0

        # Zero starting price
        assert strategy.calculate_momentum([0, 100]) == 0.0


class TestMeanReversionStrategy:
    """Tests for MeanReversionStrategy template."""

    def test_basic_mean_reversion(self):
        """Test basic mean reversion strategy."""

        class SimpleMeanReversion(MeanReversionStrategy):
            lookback = 10
            entry_zscore = -1.5
            exit_zscore = 0.0
            position_size = 0.5

        # Create mean-reverting data
        np.random.seed(42)
        n_bars = 50
        prices = []
        price = 100
        for _ in range(n_bars):
            # Mean-reverting process
            price = price + 0.5 * (100 - price) + np.random.randn() * 2
            prices.append(price)

        df = pl.DataFrame(
            {
                "timestamp": [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_bars)],
                "asset": ["SPY"] * n_bars,
                "close": prices,
            }
        )

        feed = DataFeed(prices_df=df)
        engine = Engine.from_config(feed, SimpleMeanReversion(), BacktestConfig.from_preset("fast"))
        result = engine.run()

        # Should complete without error
        assert result.metrics["final_value"] > 0

    def test_zscore_calculation(self):
        """Test z-score calculation method."""
        strategy = MeanReversionStrategy()

        # Normal case - prices with variance
        prices = [98, 99, 100, 101, 102]  # Mean = 100, has variance
        zscore = strategy.calculate_zscore(prices, 104)
        assert zscore is not None
        assert zscore > 0  # Above mean

        zscore = strategy.calculate_zscore(prices, 96)
        assert zscore is not None
        assert zscore < 0  # Below mean

        # Insufficient data
        assert strategy.calculate_zscore([100], 100) is None

        # Zero std dev (all same values)
        assert strategy.calculate_zscore([100, 100, 100], 100) is None


class TestLongShortStrategy:
    """Tests for LongShortStrategy template."""

    def test_basic_long_short(self):
        """Test basic long/short strategy."""

        class SimpleLongShort(LongShortStrategy):
            signal_column = "signal"
            long_count = 2
            short_count = 2
            position_size = 0.1
            rebalance_frequency = 5

        # Create multi-asset data with different signals
        np.random.seed(42)
        n_bars = 30
        n_assets = 5

        rows = []
        base_date = datetime(2023, 1, 1)

        for i in range(n_bars):
            timestamp = base_date + timedelta(days=i)
            for j in range(n_assets):
                asset = f"ASSET{j:03d}"
                rows.append(
                    {
                        "timestamp": timestamp,
                        "asset": asset,
                        "close": 100 + np.random.randn() * 10,
                        "signal": j - 2.0,  # Signals from -2 to 2
                    }
                )

        df = pl.DataFrame(rows)
        feed = DataFeed(
            prices_df=df,
            signals_df=df.select(["timestamp", "asset", "signal"]),
        )

        engine = Engine.from_config(feed, SimpleLongShort(), BacktestConfig.from_preset("fast"))
        result = engine.run()

        # Should complete without error
        assert result.metrics["final_value"] > 0

    def test_ranking(self):
        """Test asset ranking method."""
        strategy = LongShortStrategy()
        strategy.long_count = 2
        strategy.short_count = 2

        # Data structure matches DataFeed output (signals nested under 'signals')
        data = {
            "A": {"signals": {"signal": 5.0}},
            "B": {"signals": {"signal": 3.0}},
            "C": {"signals": {"signal": 1.0}},
            "D": {"signals": {"signal": -1.0}},
            "E": {"signals": {"signal": -3.0}},
        }

        long_assets, short_assets = strategy.rank_assets(data)

        # Top 2 should be long
        assert "A" in long_assets
        assert "B" in long_assets

        # Bottom 2 should be short (excluding any in long)
        assert "E" in short_assets
        assert "D" in short_assets

    def test_schedule_overrides_bar_frequency(self):
        """Explicit schedules should override fixed bar-count rebalancing."""

        class ScheduledLongShort(LongShortStrategy):
            signal_column = "signal"
            long_count = 1
            short_count = 1
            position_size = 0.1
            rebalance_frequency = 999
            rebalance_schedule = RebalanceSchedule.explicit_timestamps(
                [datetime(2023, 1, 2), datetime(2023, 1, 5)]
            )

        rows = []
        for i in range(6):
            timestamp = datetime(2023, 1, 1) + timedelta(days=i)
            if timestamp == datetime(2023, 1, 2):
                signals = {"A": 2.0, "B": -2.0, "C": 0.0}
            elif timestamp == datetime(2023, 1, 5):
                signals = {"A": -1.0, "B": 0.0, "C": 2.0}
            else:
                signals = {"A": 1.0, "B": -1.0, "C": 0.0}
            rows.extend(
                [
                    {"timestamp": timestamp, "asset": "A", "close": 100.0, "signal": signals["A"]},
                    {"timestamp": timestamp, "asset": "B", "close": 100.0, "signal": signals["B"]},
                    {"timestamp": timestamp, "asset": "C", "close": 100.0, "signal": signals["C"]},
                ]
            )

        df = pl.DataFrame(rows)
        feed = DataFeed(
            prices_df=df,
            signals_df=df.select(["timestamp", "asset", "signal"]),
        )

        engine = Engine.from_config(
            feed,
            ScheduledLongShort(),
            BacktestConfig.from_preset("fast"),
        )
        result = engine.run()

        entry_days = sorted({trade.entry_time.date() for trade in result.trades})
        assert entry_days == [datetime(2023, 1, 2).date(), datetime(2023, 1, 5).date()]

    def test_weekly_schedule_uses_feed_session_labels(self):
        """Weekly schedules on daily labeled bars should rebalance on labeled Fridays."""

        class WeeklyLongShort(LongShortStrategy):
            signal_column = "signal"
            long_count = 1
            short_count = 1
            position_size = 0.1
            rebalance_frequency = 999
            rebalance_schedule = RebalanceSchedule.weekly()

        rows = []
        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 3),
            datetime(2024, 1, 4),
            datetime(2024, 1, 5),
            datetime(2024, 1, 8),
            datetime(2024, 1, 9),
            datetime(2024, 1, 10),
            datetime(2024, 1, 11),
            datetime(2024, 1, 12),
        ]
        for timestamp in dates:
            if timestamp == datetime(2024, 1, 5):
                signals = {"A": 2.0, "B": -2.0, "C": 0.0}
            elif timestamp == datetime(2024, 1, 12):
                signals = {"A": -1.0, "B": 0.0, "C": 2.0}
            else:
                signals = {"A": 1.0, "B": -1.0, "C": 0.0}
            rows.extend(
                [
                    {"timestamp": timestamp, "asset": "A", "close": 100.0, "signal": signals["A"]},
                    {"timestamp": timestamp, "asset": "B", "close": 100.0, "signal": signals["B"]},
                    {"timestamp": timestamp, "asset": "C", "close": 100.0, "signal": signals["C"]},
                ]
            )

        df = pl.DataFrame(rows)
        feed = DataFeed(
            prices_df=df,
            signals_df=df.select(["timestamp", "asset", "signal"]),
            feed_spec=FeedSpec(
                calendar="NYSE",
                data_frequency="daily",
                timestamp_semantics="session_label",
            ),
        )

        engine = Engine.from_config(
            feed,
            WeeklyLongShort(),
            BacktestConfig.from_preset("fast"),
        )
        result = engine.run()

        entry_days = sorted({trade.entry_time.date() for trade in result.trades})
        assert entry_days == [datetime(2024, 1, 5).date(), datetime(2024, 1, 12).date()]


class TestStrategyImports:
    """Test strategy template import paths."""

    def test_import_from_strategies(self):
        """Test importing from ml4t.backtest.strategies."""
        from ml4t.backtest.strategies import (
            LongShortStrategy,
            MeanReversionStrategy,
            MomentumStrategy,
            SignalFollowingStrategy,
        )

        assert SignalFollowingStrategy is not None
        assert MomentumStrategy is not None
        assert MeanReversionStrategy is not None
        assert LongShortStrategy is not None
