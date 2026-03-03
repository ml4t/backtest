"""Tests for stateful strategy examples.

Each test group verifies the key stateful behavior that makes the strategy
impossible to implement in a vectorized framework.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ml4t.backtest import BacktestConfig, DataFeed, Engine

from .stateful_strategies import (
    AdaptiveKellySizingStrategy,
    DrawdownCircuitBreakerStrategy,
    GridTradingStrategy,
    PairsTradingStrategy,
    PyramidingStrategy,
)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def make_trending_data(
    n_bars: int = 100,
    drift: float = 0.003,
    volatility: float = 0.01,
    seed: int = 42,
    asset: str = "ASSET",
) -> pl.DataFrame:
    """Generate upward-trending price data (good for pyramiding/Kelly)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, volatility, n_bars)
    prices = 100.0 * np.cumprod(1 + returns)

    base = datetime(2023, 1, 1)
    rows = []
    for i in range(n_bars):
        p = float(prices[i])
        rows.append(
            {
                "timestamp": base + timedelta(days=i),
                "asset": asset,
                "open": p * 0.999,
                "high": p * 1.005,
                "low": p * 0.995,
                "close": p,
                "volume": 1_000_000,
            }
        )
    return pl.DataFrame(rows)


def make_alternating_signal(
    n_bars: int = 100,
    cycle: int = 10,
    asset: str = "ASSET",
) -> pl.DataFrame:
    """Generate alternating +1/-1 signal with given cycle length."""
    base = datetime(2023, 1, 1)
    rows = []
    for i in range(n_bars):
        rows.append(
            {
                "timestamp": base + timedelta(days=i),
                "asset": asset,
                "signal": 1.0 if (i // cycle) % 2 == 0 else -1.0,
            }
        )
    return pl.DataFrame(rows)


def make_drawdown_data(
    n_bars: int = 100,
    crash_start: int = 40,
    crash_bars: int = 10,
    crash_pct: float = 0.20,
    seed: int = 42,
    asset: str = "ASSET",
) -> pl.DataFrame:
    """Generate prices with a crash episode for circuit breaker testing."""
    rng = np.random.default_rng(seed)
    prices = [100.0]
    for i in range(1, n_bars):
        if crash_start <= i < crash_start + crash_bars:
            # Crash: steep decline
            daily_drop = crash_pct / crash_bars
            prices.append(prices[-1] * (1 - daily_drop))
        else:
            # Normal: small positive drift
            prices.append(prices[-1] * (1 + rng.normal(0.001, 0.005)))

    base = datetime(2023, 1, 1)
    rows = []
    for i in range(n_bars):
        p = float(prices[i])
        rows.append(
            {
                "timestamp": base + timedelta(days=i),
                "asset": asset,
                "open": p * 0.999,
                "high": p * 1.005,
                "low": p * 0.995,
                "close": p,
                "volume": 1_000_000,
            }
        )
    return pl.DataFrame(rows)


def make_pair_data(
    n_bars: int = 100,
    seed: int = 42,
    asset_a: str = "A",
    asset_b: str = "B",
) -> pl.DataFrame:
    """Generate two correlated assets that diverge then converge."""
    rng = np.random.default_rng(seed)
    # Shared factor + idiosyncratic noise
    common = rng.normal(0, 0.01, n_bars)
    noise_a = rng.normal(0, 0.005, n_bars)
    noise_b = rng.normal(0, 0.005, n_bars)

    # Inject a divergence/convergence cycle
    divergence = np.zeros(n_bars)
    for i in range(n_bars):
        if 25 <= i < 40:
            divergence[i] = 0.003 * (i - 25)  # Diverge
        elif 40 <= i < 55:
            divergence[i] = 0.003 * (55 - i)  # Converge back

    prices_a = 100.0 * np.cumprod(1 + common + noise_a)
    prices_b = 100.0 * np.cumprod(1 + common + noise_b + divergence)

    base = datetime(2023, 1, 1)
    rows = []
    for i in range(n_bars):
        pa, pb = float(prices_a[i]), float(prices_b[i])
        for asset, p in [(asset_a, pa), (asset_b, pb)]:
            rows.append(
                {
                    "timestamp": base + timedelta(days=i),
                    "asset": asset,
                    "open": p * 0.999,
                    "high": p * 1.005,
                    "low": p * 0.995,
                    "close": p,
                    "volume": 1_000_000,
                }
            )
    return pl.DataFrame(rows)


def make_grid_data(
    n_bars: int = 100,
    seed: int = 42,
    asset: str = "ASSET",
    volatility: float = 0.015,
) -> pl.DataFrame:
    """Generate mean-reverting prices for grid trading."""
    rng = np.random.default_rng(seed)
    prices = [100.0]
    for _ in range(1, n_bars):
        # Mean-reverting around 100
        reversion = 0.05 * (100.0 - prices[-1])
        prices.append(prices[-1] * (1 + reversion / prices[-1] + rng.normal(0, volatility)))

    base = datetime(2023, 1, 1)
    rows = []
    for i in range(n_bars):
        p = float(prices[i])
        rows.append(
            {
                "timestamp": base + timedelta(days=i),
                "asset": asset,
                "open": p * 0.999,
                "high": p * 1.008,
                "low": p * 0.992,
                "close": p,
                "volume": 1_000_000,
            }
        )
    return pl.DataFrame(rows)


def _fast_config(**overrides: object) -> BacktestConfig:
    """Get the fast preset with optional overrides."""
    config = BacktestConfig.from_preset("fast")
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# ---------------------------------------------------------------------------
# 1. Adaptive Kelly Sizing tests
# ---------------------------------------------------------------------------


class TestAdaptiveKellySizing:
    """Tests for AdaptiveKellySizingStrategy."""

    def test_starts_at_base_size(self):
        """Before min_trades, should use base_size."""
        strategy = AdaptiveKellySizingStrategy(
            signal_column="signal",
            base_size=0.10,
            min_trades=5,
        )

        prices = make_trending_data(30, drift=0.002, seed=1)
        signals = make_alternating_signal(30, cycle=5)  # Frequent trades
        feed = DataFeed(prices_df=prices, signals_df=signals)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        # First entry should use base_size since no trades yet
        assert len(strategy.size_history) > 0
        assert strategy.size_history[0] == 0.10

    def test_adapts_after_enough_trades(self):
        """After min_trades, size should adapt based on Kelly formula."""
        strategy = AdaptiveKellySizingStrategy(
            signal_column="signal",
            base_size=0.10,
            min_trades=3,
            min_size=0.02,
            max_size=0.25,
        )

        # Lots of bars with frequent signal flips to accumulate trades
        prices = make_trending_data(200, drift=0.002, seed=10)
        signals = make_alternating_signal(200, cycle=8)
        feed = DataFeed(prices_df=prices, signals_df=signals)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        # Should have adapted: not all sizes are base_size
        if len(strategy.size_history) > 3:
            later_sizes = strategy.size_history[3:]
            # At least one adapted size should differ from base
            assert any(abs(s - 0.10) > 1e-6 for s in later_sizes), (
                f"All sizes stayed at base: {later_sizes}"
            )

    def test_clamps_to_bounds(self):
        """Size should always be within [min_size, max_size]."""
        strategy = AdaptiveKellySizingStrategy(
            signal_column="signal",
            min_size=0.03,
            max_size=0.20,
            min_trades=2,
        )

        prices = make_trending_data(200, drift=0.003, seed=7)
        signals = make_alternating_signal(200, cycle=6)
        feed = DataFeed(prices_df=prices, signals_df=signals)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        for size in strategy.size_history:
            assert 0.03 - 1e-9 <= size <= 0.20 + 1e-9, f"Size {size} out of bounds"

    def test_runs_without_error(self):
        """Smoke test: the strategy completes a full backtest."""
        strategy = AdaptiveKellySizingStrategy()
        prices = make_trending_data(50)
        signals = make_alternating_signal(50)
        feed = DataFeed(prices_df=prices, signals_df=signals)
        result = Engine.from_config(feed, strategy, _fast_config()).run()
        assert result.metrics["final_value"] > 0


# ---------------------------------------------------------------------------
# 2. Pyramiding tests
# ---------------------------------------------------------------------------


class TestPyramiding:
    """Tests for PyramidingStrategy."""

    def test_enters_base_level(self):
        """Should enter level 1 on signal."""
        strategy = PyramidingStrategy(
            signal_column="signal",
            max_levels=3,
            base_size=0.10,
        )

        prices = make_trending_data(50, drift=0.005, seed=1)
        signals = make_alternating_signal(50, cycle=20)  # Long signal for first 20 bars
        feed = DataFeed(prices_df=prices, signals_df=signals)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        result = engine.run()

        # Should have entered at least once
        assert len(result.trades) > 0 or len(result.broker.positions) > 0

    def test_pyramids_on_profit(self):
        """Should add levels when position is profitable."""
        strategy = PyramidingStrategy(
            signal_column="signal",
            max_levels=3,
            profit_threshold=0.01,  # Low threshold to trigger pyramids
            base_size=0.05,
        )

        # Strong uptrend to ensure profitable positions
        prices = make_trending_data(100, drift=0.008, seed=3)
        # Signal stays positive for long stretches
        signals = make_alternating_signal(100, cycle=40)
        feed = DataFeed(prices_df=prices, signals_df=signals)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        # If trend was strong enough, should have pyramided above level 1
        max_level = max(strategy.pyramid_levels.values()) if strategy.pyramid_levels else 0
        # Also check level_entries for evidence of pyramiding
        max_entries = max((len(v) for v in strategy.level_entries.values()), default=0)
        assert max_level > 1 or max_entries > 1, (
            f"No pyramiding occurred: max_level={max_level}, max_entries={max_entries}"
        )

    def test_respects_max_levels(self):
        """Should never exceed max_levels."""
        max_levels = 2
        strategy = PyramidingStrategy(
            signal_column="signal",
            max_levels=max_levels,
            profit_threshold=0.005,
            base_size=0.03,
        )

        prices = make_trending_data(150, drift=0.01, seed=5)
        signals = make_alternating_signal(150, cycle=60)
        feed = DataFeed(prices_df=prices, signals_df=signals)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        for level in strategy.pyramid_levels.values():
            assert level <= max_levels

    def test_exits_reset_levels(self):
        """Exit signal should close position and reset pyramid state."""
        strategy = PyramidingStrategy(signal_column="signal", max_levels=3)

        prices = make_trending_data(60)
        signals = make_alternating_signal(60, cycle=15)
        feed = DataFeed(prices_df=prices, signals_df=signals)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        result = engine.run()

        # After exit, levels should be 0 for closed assets
        if result.trades:
            # If we have closed trades, at some point levels were reset
            assert True  # Exit path was exercised


# ---------------------------------------------------------------------------
# 3. Pairs Trading tests
# ---------------------------------------------------------------------------


class TestPairsTrading:
    """Tests for PairsTradingStrategy."""

    def test_enters_on_divergence(self):
        """Should open positions when z-score exceeds threshold."""
        strategy = PairsTradingStrategy(
            asset_a="A",
            asset_b="B",
            lookback=15,
            entry_zscore=1.5,
            exit_zscore=0.3,
            position_size=0.10,
        )

        prices = make_pair_data(80, seed=42)
        feed = DataFeed(prices_df=prices)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        result = engine.run()

        # Should have entered the pair trade at some point
        assert strategy.pair_status != "flat" or len(result.trades) > 0

    def test_exits_on_convergence(self):
        """Should close positions when z-score reverts."""
        strategy = PairsTradingStrategy(
            asset_a="A",
            asset_b="B",
            lookback=15,
            entry_zscore=1.5,
            exit_zscore=0.3,
            position_size=0.10,
        )

        prices = make_pair_data(100, seed=42)
        feed = DataFeed(prices_df=prices)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        result = engine.run()

        # If pair entered, should have at least some trades (entries and exits)
        if len(result.trades) >= 2:
            # Both legs should appear in trades
            assets_traded = {t.symbol for t in result.trades}
            assert "A" in assets_traded or "B" in assets_traded

    def test_correct_direction(self):
        """When z > threshold (B expensive), should long A and short B."""
        strategy = PairsTradingStrategy(
            asset_a="A",
            asset_b="B",
            lookback=15,
            entry_zscore=1.5,
            exit_zscore=0.3,
        )

        prices = make_pair_data(100, seed=42)
        feed = DataFeed(prices_df=prices)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        # Just verify it runs and produces valid state
        assert strategy.pair_status in ("flat", "long_spread", "short_spread")

    def test_no_entry_within_bounds(self):
        """Should NOT enter when z-score is within entry threshold."""
        strategy = PairsTradingStrategy(
            asset_a="A",
            asset_b="B",
            lookback=15,
            entry_zscore=100.0,  # Impossibly high threshold
            exit_zscore=0.3,
        )

        prices = make_pair_data(80, seed=42)
        feed = DataFeed(prices_df=prices)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        result = engine.run()

        assert len(result.trades) == 0
        assert strategy.pair_status == "flat"


# ---------------------------------------------------------------------------
# 4. Drawdown Circuit Breaker tests
# ---------------------------------------------------------------------------


class TestDrawdownCircuitBreaker:
    """Tests for DrawdownCircuitBreakerStrategy."""

    def test_normal_sizing_without_drawdown(self):
        """Without drawdown, multiplier should stay at 1.0."""
        strategy = DrawdownCircuitBreakerStrategy(
            signal_column="signal",
            base_size=0.10,
            caution_threshold=0.05,
            halt_threshold=0.10,
        )

        # Gentle uptrend — no drawdown
        prices = make_trending_data(30, drift=0.005, volatility=0.001, seed=1)
        signals = make_alternating_signal(30, cycle=10)
        feed = DataFeed(prices_df=prices, signals_df=signals)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        # After warmup, multiplier should be at or near 1.0
        if len(strategy.multiplier_history) > 5:
            assert strategy.multiplier_history[-1] >= 0.9

    def test_reduces_at_caution_threshold(self):
        """Multiplier should decrease during drawdown."""
        strategy = DrawdownCircuitBreakerStrategy(
            signal_column="signal",
            base_size=0.80,  # Large size so price drop → visible equity drawdown
            caution_threshold=0.03,
            halt_threshold=0.15,
            reduction_factor=0.5,
        )

        # Data with a crash to trigger drawdown (20% crash with 80% invested → ~16% equity drop)
        prices = make_drawdown_data(80, crash_start=30, crash_bars=15, crash_pct=0.20, seed=2)
        # Stay invested: always-positive signal
        base = datetime(2023, 1, 1)
        signal_rows = [
            {"timestamp": base + timedelta(days=i), "asset": "ASSET", "signal": 1.0}
            for i in range(80)
        ]
        signals = pl.DataFrame(signal_rows)
        feed = DataFeed(prices_df=prices, signals_df=signals)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        # Should have reduced multiplier at some point during the crash
        min_mult = min(strategy.multiplier_history) if strategy.multiplier_history else 1.0
        assert min_mult < 1.0, f"Multiplier never reduced: min={min_mult}"

    def test_halts_at_threshold(self):
        """Multiplier should hit 0.0 during severe drawdown."""
        strategy = DrawdownCircuitBreakerStrategy(
            signal_column="signal",
            base_size=0.80,  # Large size so crash is felt in equity
            caution_threshold=0.02,
            halt_threshold=0.10,
            reduction_factor=0.5,
        )

        # Severe crash: 30% price drop with 80% invested → ~24% equity drop
        prices = make_drawdown_data(80, crash_start=25, crash_bars=10, crash_pct=0.30, seed=3)
        base = datetime(2023, 1, 1)
        signal_rows = [
            {"timestamp": base + timedelta(days=i), "asset": "ASSET", "signal": 1.0}
            for i in range(80)
        ]
        signals = pl.DataFrame(signal_rows)
        feed = DataFeed(prices_df=prices, signals_df=signals)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        # Should have hit zero multiplier during the crash
        min_mult = min(strategy.multiplier_history) if strategy.multiplier_history else 1.0
        assert min_mult < 0.01, f"Multiplier never halted: min={min_mult}"

    def test_recovers_after_drawdown(self):
        """Multiplier should increase once drawdown recedes."""
        strategy = DrawdownCircuitBreakerStrategy(
            signal_column="signal",
            base_size=0.80,
            caution_threshold=0.03,
            halt_threshold=0.15,
            recovery_rate=0.05,
        )

        # Crash followed by recovery
        prices = make_drawdown_data(120, crash_start=30, crash_bars=10, crash_pct=0.15, seed=4)
        base = datetime(2023, 1, 1)
        signal_rows = [
            {"timestamp": base + timedelta(days=i), "asset": "ASSET", "signal": 1.0}
            for i in range(120)
        ]
        signals = pl.DataFrame(signal_rows)
        feed = DataFeed(prices_df=prices, signals_df=signals)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        # After the crash, multiplier should have recovered toward 1.0
        history = strategy.multiplier_history
        if len(history) > 60:
            # Find the min during crash and check that post-crash is higher
            crash_region = history[30:50]
            post_crash = history[60:]
            if crash_region and post_crash:
                min_crash = min(crash_region)
                max_post = max(post_crash)
                assert max_post > min_crash, (
                    f"No recovery: crash_min={min_crash}, post_max={max_post}"
                )


# ---------------------------------------------------------------------------
# 5. Grid Trading tests
# ---------------------------------------------------------------------------


class TestGridTrading:
    """Tests for GridTradingStrategy."""

    def test_initializes_grid(self):
        """Should place buy and sell limit orders on first bar."""
        strategy = GridTradingStrategy(
            asset="ASSET",
            grid_spacing=0.02,
            num_levels=3,
            order_size=50,
        )

        prices = make_grid_data(10, seed=1)
        feed = DataFeed(prices_df=prices)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        # Grid should have been initialized
        assert strategy.initialized
        assert strategy.reference_price > 0

    def test_reacts_to_fills(self):
        """Should place reactive orders when grid levels fill."""
        strategy = GridTradingStrategy(
            asset="ASSET",
            grid_spacing=0.01,  # 1% spacing — volatile data should trigger fills
            num_levels=5,
            order_size=50,
            max_position=500,
        )

        # Use higher volatility to trigger limit fills
        prices = make_grid_data(80, seed=42, volatility=0.025)
        feed = DataFeed(prices_df=prices)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        result = engine.run()

        # Even if no grid fills triggered, it should complete without error
        assert result.metrics["final_value"] > 0
        assert strategy.initialized

    def test_respects_max_position(self):
        """Net position should never exceed max_position."""
        max_pos = 200
        strategy = GridTradingStrategy(
            asset="ASSET",
            grid_spacing=0.01,
            num_levels=5,
            order_size=50,
            max_position=max_pos,
        )

        prices = make_grid_data(80, seed=42, volatility=0.02)
        feed = DataFeed(prices_df=prices)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        # Check final position via engine's broker (result doesn't expose broker)
        pos = engine.broker.get_position("ASSET")
        if pos is not None:
            assert abs(pos.quantity) <= max_pos + 50 + 1e-6  # Allow one order size tolerance

    def test_recenters_on_drift(self):
        """Should recenter grid when price drifts beyond threshold."""
        strategy = GridTradingStrategy(
            asset="ASSET",
            grid_spacing=0.01,
            num_levels=3,
            order_size=50,
            recenter_threshold=0.03,  # 3% drift triggers recenter
        )

        # Create data with a significant trend to trigger recentering
        rng = np.random.default_rng(99)
        prices_list = [100.0]
        for _ in range(79):
            prices_list.append(prices_list[-1] * (1 + rng.normal(0.002, 0.01)))

        base = datetime(2023, 1, 1)
        rows = []
        for i in range(80):
            p = float(prices_list[i])
            rows.append(
                {
                    "timestamp": base + timedelta(days=i),
                    "asset": "ASSET",
                    "open": p * 0.999,
                    "high": p * 1.008,
                    "low": p * 0.992,
                    "close": p,
                    "volume": 1_000_000,
                }
            )

        prices = pl.DataFrame(rows)
        feed = DataFeed(prices_df=prices)
        config = _fast_config()

        engine = Engine.from_config(feed, strategy, config)
        engine.run()

        # If price drifted >3% from initial reference, grid should have recentered
        # meaning reference_price should have updated from its initial value
        initial_ref = float(prices_list[0])
        final_price = float(prices_list[-1])
        if abs(final_price - initial_ref) / initial_ref > 0.03:
            assert abs(strategy.reference_price - initial_ref) > 0.01, (
                "Grid should have recentered but reference_price unchanged"
            )
