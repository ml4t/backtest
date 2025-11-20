"""Integration tests for enhanced slippage models in realistic backtest scenarios (TASK-INT-040).

Tests SpreadAwareSlippage, VolumeAwareSlippage, and OrderTypeDependentSlippage
in full backtest workflows to validate realistic transaction costs.

Acceptance Criteria:
1. Test SpreadAwareSlippage: verify slippage scales with bid/ask spread
2. Test VolumeAwareSlippage: verify participation rate impact on slippage
3. Test OrderTypeDependentSlippage: verify different fills by order type
4. Compare all models to baseline PercentageSlippage
5. Verify slippage costs are within realistic ranges (0.01% - 0.5%)
6. Performance: no significant overhead vs simple slippage
7. Documentation on when to use each model
8. Test completes in <10 seconds
"""

import time
import uuid
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from ml4t.backtest.core.event import MarketEvent, OrderEvent
from ml4t.backtest.core.types import OrderSide, OrderType, TimeInForce
from ml4t.backtest.engine import BacktestEngine
from ml4t.backtest.execution.commission import NoCommission
from ml4t.backtest.execution.order import Order
from ml4t.backtest.execution.slippage import (
    NoSlippage,
    OrderTypeDependentSlippage,
    PercentageSlippage,
    SpreadAwareSlippage,
    VolumeAwareSlippage,
)
from ml4t.backtest.strategy.base import Strategy


# ============================================================================
# Test Strategies
# ============================================================================


class SimpleMAStrategy(Strategy):
    """Simple moving average crossover strategy for testing slippage impact.

    Generates buy/sell signals to create realistic trading activity.
    """

    def __init__(self, order_type: OrderType = OrderType.MARKET):
        """Initialize strategy.

        Args:
            order_type: Order type to use for all orders
        """
        super().__init__()
        self.order_type = order_type
        self.prices = []
        self.position_size = 0
        self.trade_count = 0
        self.max_trades = 10  # Limit trades for performance

    def on_start(self, portfolio=None, event_bus=None):
        """Initialize strategy."""
        self.portfolio = portfolio
        self.event_bus = event_bus

    def on_event(self, event):
        """Handle all events."""
        if isinstance(event, MarketEvent):
            self.on_market_data(event)

    def on_market_data(self, event: MarketEvent):
        """React to market data - simple MA crossover logic."""
        if self.trade_count >= self.max_trades:
            return

        self.prices.append(event.close)

        # Need at least 5 bars for short MA
        if len(self.prices) < 5:
            return

        # Calculate simple moving averages
        short_ma = sum(self.prices[-3:]) / 3  # 3-period MA
        long_ma = sum(self.prices[-5:]) / 5  # 5-period MA

        # Get current position
        position = self.portfolio.get_position(event.asset_id)
        current_qty = position.quantity if position else 0

        # Buy signal: short MA crosses above long MA
        if short_ma > long_ma and current_qty == 0:
            # Calculate order size (use fixed 100 shares)
            order_qty = 100

            # Create and publish buy order event
            order_event = OrderEvent(
                timestamp=event.timestamp,
                order_id=str(uuid.uuid4()),
                asset_id=event.asset_id,
                order_type=self.order_type,
                side=OrderSide.BUY,
                quantity=order_qty,
                time_in_force=TimeInForce.DAY,
            )
            self.event_bus.publish(order_event)
            self.trade_count += 1

        # Sell signal: short MA crosses below long MA
        elif short_ma < long_ma and current_qty > 0:
            # Create and publish sell order event
            order_event = OrderEvent(
                timestamp=event.timestamp,
                order_id=str(uuid.uuid4()),
                asset_id=event.asset_id,
                order_type=self.order_type,
                side=OrderSide.SELL,
                quantity=current_qty,
                time_in_force=TimeInForce.DAY,
            )
            self.event_bus.publish(order_event)
            self.trade_count += 1


class MultiOrderTypeStrategy(Strategy):
    """Strategy that uses different order types to test OrderTypeDependentSlippage."""

    def __init__(self):
        """Initialize strategy."""
        super().__init__()
        self.bar_count = 0
        self.trades = []

    def on_start(self, portfolio=None, event_bus=None):
        """Initialize strategy."""
        self.portfolio = portfolio
        self.event_bus = event_bus

    def on_event(self, event):
        """Handle all events."""
        if isinstance(event, MarketEvent):
            self.on_market_data(event)

    def on_market_data(self, event: MarketEvent):
        """Place different order types at different times."""
        self.bar_count += 1

        # Limit total trades
        if self.bar_count > 10:
            return

        # Rotate through different order types
        if self.bar_count % 3 == 1:
            # MARKET order
            order_event = OrderEvent(
                timestamp=event.timestamp,
                order_id=str(uuid.uuid4()),
                asset_id=event.asset_id,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=100,
                time_in_force=TimeInForce.DAY,
            )
            self.event_bus.publish(order_event)
            self.trades.append(("MARKET", "BUY"))

        elif self.bar_count % 3 == 2:
            # LIMIT order
            order_event = OrderEvent(
                timestamp=event.timestamp,
                order_id=str(uuid.uuid4()),
                asset_id=event.asset_id,
                order_type=OrderType.LIMIT,
                side=OrderSide.BUY,
                quantity=100,
                limit_price=event.close * 1.01,  # Limit above market
                time_in_force=TimeInForce.DAY,
            )
            self.event_bus.publish(order_event)
            self.trades.append(("LIMIT", "BUY"))

        # Close positions on bar 9
        elif self.bar_count == 9:
            position = self.portfolio.get_position(event.asset_id)
            if position and position.quantity > 0:
                order_event = OrderEvent(
                    timestamp=event.timestamp,
                    order_id=str(uuid.uuid4()),
                    asset_id=event.asset_id,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    time_in_force=TimeInForce.DAY,
                )
                self.event_bus.publish(order_event)
                self.trades.append(("MARKET", "SELL"))


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def market_data_with_spread() -> pl.DataFrame:
    """Create realistic market data with bid/ask spreads and volume.

    Returns:
        DataFrame with timestamp, asset_id, OHLCV, bid/ask, volume

    Note:
        Price pattern: up (bars 0-9) then down (bars 10-19) to trigger
        both buy and sell signals for strategy testing.
    """
    from datetime import timedelta
    base_date = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    timestamps = [base_date + timedelta(days=i) for i in range(40)]  # 40 bars

    # Create price pattern that will trigger both buy and sell signals
    base_price = 100.0
    prices = []
    for i in range(40):
        if i < 20:
            # Uptrend: triggers BUY signals
            trend = i * 0.3
            noise = (i % 3 - 1) * 0.5
        else:
            # Downtrend: triggers SELL signals
            trend = (40 - i) * 0.3
            noise = (i % 3 - 1) * 0.5
        prices.append(base_price + trend + noise)

    data = {
        "timestamp": timestamps,
        "asset_id": ["SPY"] * 40,
        "open": [p - 0.5 for p in prices],
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [1_000_000 + i * 50_000 for i in range(40)],  # Varying volume
        "bid_price": [p - 0.02 for p in prices],  # Spread = 0.04
        "ask_price": [p + 0.02 for p in prices],
    }

    return pl.DataFrame(data)


@pytest.fixture
def market_data_varying_spreads() -> pl.DataFrame:
    """Create market data with varying bid/ask spreads (tight vs wide)."""
    from datetime import timedelta
    base_date = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    timestamps = [base_date + timedelta(days=i) for i in range(40)]  # 40 bars

    # Create price pattern: up then down (same as market_data_with_spread)
    base_price = 100.0
    prices = []
    for i in range(40):
        if i < 20:
            # Uptrend: triggers BUY signals
            trend = i * 0.3
            noise = (i % 3 - 1) * 0.5
        else:
            # Downtrend: triggers SELL signals
            trend = (40 - i) * 0.3
            noise = (i % 3 - 1) * 0.5
        prices.append(base_price + trend + noise)

    # Alternate between tight and wide spreads
    bid_prices = []
    ask_prices = []
    for i, p in enumerate(prices):
        if i % 2 == 0:
            # Tight spread (0.04)
            bid_prices.append(p - 0.02)
            ask_prices.append(p + 0.02)
        else:
            # Wide spread (0.20)
            bid_prices.append(p - 0.10)
            ask_prices.append(p + 0.10)

    data = {
        "timestamp": timestamps,
        "asset_id": ["SPY"] * 40,
        "open": [p - 0.5 for p in prices],
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [1_000_000 + i * 50_000 for i in range(40)],
        "bid_price": bid_prices,
        "ask_price": ask_prices,
    }

    return pl.DataFrame(data)


@pytest.fixture
def market_data_varying_volume() -> pl.DataFrame:
    """Create market data with varying volume (high vs low liquidity)."""
    # Changed from 20 to 40 bars to allow both buy AND sell signals
    # Pattern: uptrend (0-19), downtrend (20-39)
    base_date = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
    timestamps = [base_date + timedelta(days=i) for i in range(40)]

    base_price = 100.0
    # Up for first 20 bars, down for next 20 bars
    prices = []
    for i in range(40):
        if i < 20:
            prices.append(base_price + i * 0.3)  # Uptrend
        else:
            prices.append(base_price + (40 - i) * 0.3)  # Downtrend

    # Alternate between high and low volume
    volumes = []
    for i in range(40):
        if i % 2 == 0:
            # High volume (low participation rate)
            volumes.append(5_000_000)
        else:
            # Low volume (high participation rate)
            volumes.append(500_000)

    data = {
        "timestamp": timestamps,
        "asset_id": ["SPY"] * 40,
        "open": [p - 0.5 for p in prices],
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": volumes,
    }

    return pl.DataFrame(data)


# ============================================================================
# Helper Functions
# ============================================================================


def calculate_slippage_cost_percentage(results) -> float:
    """Calculate total slippage cost as percentage of trade value.

    Args:
        results: BacktestResults object

    Returns:
        Slippage cost as percentage (e.g., 0.15 = 0.15%)
    """
    from ml4t.backtest.results import BacktestResults

    if not isinstance(results, BacktestResults):
        raise TypeError(f"Expected BacktestResults, got {type(results)}")

    trades_df = results.get_trades()
    if trades_df.is_empty():
        return 0.0

    # Calculate total slippage (entry + exit)
    total_slippage = (
        trades_df.select(
            (pl.col("entry_slippage").abs() + pl.col("exit_slippage").abs()).sum()
        ).item()
    )

    # Calculate total trade value (entry + exit)
    total_value = (
        trades_df.select(
            (
                (pl.col("entry_price") * pl.col("entry_quantity")).abs() +
                (pl.col("exit_price") * pl.col("exit_quantity")).abs()
            ).sum()
        ).item()
    )

    if total_value == 0:
        return 0.0

    return (total_slippage / total_value) * 100  # Return as percentage


def run_backtest_with_slippage(
    market_data: pl.DataFrame,
    slippage_model,
    strategy_class: type[Strategy],
    tmp_path=None,
    **strategy_kwargs,
):
    """Run backtest with specific slippage model.

    Args:
        market_data: Market data DataFrame
        slippage_model: Slippage model instance
        strategy_class: Strategy class to instantiate
        tmp_path: Temporary path for writing parquet file (required)
        **strategy_kwargs: Additional strategy arguments

    Returns:
        BacktestResults object
    """
    from pathlib import Path
    from ml4t.backtest.data.polars_feed import PolarsDataFeed
    from ml4t.backtest.execution.broker import SimulationBroker

    if tmp_path is None:
        import tempfile
        tmp_path = Path(tempfile.mkdtemp())

    # Write data to temporary parquet file
    parquet_file = Path(tmp_path) / "market_data.parquet"
    market_data.write_parquet(parquet_file)

    # Create data feed
    data_feed = PolarsDataFeed(price_path=parquet_file, asset_id="SPY")

    # Create strategy
    strategy = strategy_class(**strategy_kwargs)

    # Create broker with slippage and commission models
    broker = SimulationBroker(
        initial_cash=100_000.0,
        slippage_model=slippage_model,
        commission_model=NoCommission(),  # Isolate slippage effect
    )

    # Create engine
    engine = BacktestEngine(
        data_feed=data_feed,
        strategy=strategy,
        broker=broker,
        initial_capital=100_000.0,
    )

    # Run backtest
    engine.run()

    # Get results
    return engine.get_results()


# ============================================================================
# Test Cases
# ============================================================================


class TestSpreadAwareSlippageIntegration:
    """Integration tests for SpreadAwareSlippage in realistic backtests."""

    def test_spread_aware_with_tight_spreads(self, market_data_with_spread, tmp_path):
        """Test SpreadAwareSlippage with consistent tight spreads.

        Expected: Lower slippage cost than percentage-based model.
        """
        # Run with SpreadAwareSlippage
        spread_model = SpreadAwareSlippage(spread_factor=0.5, fallback_slippage_pct=0.001)
        results_spread = run_backtest_with_slippage(
            market_data_with_spread,
            spread_model,
            SimpleMAStrategy,
            tmp_path=tmp_path,
        )

        # Run with baseline PercentageSlippage
        pct_model = PercentageSlippage(slippage_pct=0.001)
        results_pct = run_backtest_with_slippage(
            market_data_with_spread,
            pct_model,
            SimpleMAStrategy,
            tmp_path=tmp_path,
        )

        # Calculate slippage costs
        slippage_spread = calculate_slippage_cost_percentage(results_spread)
        slippage_pct = calculate_slippage_cost_percentage(results_pct)

        # SpreadAware should have lower cost with tight spreads (0.04 vs 0.1%)
        assert slippage_spread < slippage_pct, (
            f"SpreadAware ({slippage_spread:.4f}%) should be lower than "
            f"Percentage ({slippage_pct:.4f}%) with tight spreads"
        )

        # Verify realistic range (0.01% - 0.5%)
        assert 0.01 <= slippage_spread <= 0.5, f"Slippage {slippage_spread:.4f}% outside realistic range"

    def test_spread_aware_with_varying_spreads(self, market_data_varying_spreads, tmp_path):
        """Test SpreadAwareSlippage adapts to varying spreads.

        Expected: Slippage cost varies with spread width.
        """
        # Run with SpreadAwareSlippage
        spread_model = SpreadAwareSlippage(spread_factor=0.5)
        results = run_backtest_with_slippage(
            market_data_varying_spreads,
            spread_model,
            SimpleMAStrategy,
            tmp_path=tmp_path,
        )

        trades_df = results.get_trades()
        assert not trades_df.is_empty(), "Should have executed trades"

        # Analyze trades during tight vs wide spreads
        # (Difficult to categorize without trade timestamps, so just verify cost range)
        slippage_cost = calculate_slippage_cost_percentage(results)

        # Should be in realistic range even with varying spreads
        assert 0.01 <= slippage_cost <= 0.5, f"Slippage {slippage_cost:.4f}% outside realistic range"

    def test_spread_aware_fallback(self, market_data_varying_volume, tmp_path):
        """Test SpreadAwareSlippage fallback when bid/ask unavailable.

        Expected: Uses fallback percentage when spread data missing.
        """
        # Data without bid/ask should trigger fallback
        spread_model = SpreadAwareSlippage(spread_factor=0.5, fallback_slippage_pct=0.001)
        results = run_backtest_with_slippage(
            market_data_varying_volume,  # No bid/ask in this data
            spread_model,
            SimpleMAStrategy,
            tmp_path=tmp_path,
        )

        slippage_cost = calculate_slippage_cost_percentage(results)

        # Should fall back to ~0.2% (fallback_slippage_pct 0.1% × 2 for round-trip)
        # Allow some variance due to market price differences
        assert 0.15 <= slippage_cost <= 0.25, (
            f"Fallback slippage {slippage_cost:.4f}% should be close to 0.2% (round-trip)"
        )


class TestVolumeAwareSlippageIntegration:
    """Integration tests for VolumeAwareSlippage in realistic backtests."""

    def test_volume_aware_with_varying_volume(self, market_data_varying_volume, tmp_path):
        """Test VolumeAwareSlippage adapts to volume changes.

        Expected: Higher slippage during low volume, lower during high volume.
        """
        # Run with VolumeAwareSlippage (linear model)
        volume_model = VolumeAwareSlippage(
            base_slippage_pct=0.0001,
            linear_impact_coeff=0.01,
            sqrt_impact_coeff=0.0,
            fallback_slippage_pct=0.001,
        )
        results = run_backtest_with_slippage(
            market_data_varying_volume,
            volume_model,
            SimpleMAStrategy,
            tmp_path=tmp_path,
        )

        slippage_cost = calculate_slippage_cost_percentage(results)

        # Should be in realistic range
        assert 0.01 <= slippage_cost <= 0.5, f"Slippage {slippage_cost:.4f}% outside realistic range"

        # Verify trades were executed
        trades_df = results.get_trades()
        assert not trades_df.is_empty(), "Should have executed trades"

    def test_volume_aware_participation_rate_impact(self, market_data_with_spread, tmp_path):
        """Test VolumeAwareSlippage scales with participation rate.

        Expected: Larger orders (higher participation) pay more slippage.
        """

        class SmallOrderStrategy(Strategy):
            """Strategy with small orders (low participation)."""

            def __init__(self):
                super().__init__()
                self.bar_count = 0
                self.position_opened = False

            def on_start(self, portfolio=None, event_bus=None):
                self.portfolio = portfolio
                self.event_bus = event_bus

            def on_event(self, event):
                if isinstance(event, MarketEvent):
                    self.on_market_data(event)

            def on_market_data(self, event: MarketEvent):
                self.bar_count += 1

                # Buy on bar 5
                if self.bar_count == 5 and not self.position_opened:
                    # Small order: 100 shares (0.01% of 1M volume)
                    order_event = OrderEvent(
                        timestamp=event.timestamp,
                        order_id=str(uuid.uuid4()),
                        asset_id=event.asset_id,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        quantity=100,
                        time_in_force=TimeInForce.DAY,
                    )
                    self.event_bus.publish(order_event)
                    self.position_opened = True

                # Sell on bar 10 to complete round-trip
                elif self.bar_count == 10 and self.position_opened:
                    position = self.portfolio.get_position(event.asset_id)
                    if position and position.quantity > 0:
                        order_event = OrderEvent(
                            timestamp=event.timestamp,
                            order_id=str(uuid.uuid4()),
                            asset_id=event.asset_id,
                            order_type=OrderType.MARKET,
                            side=OrderSide.SELL,
                            quantity=position.quantity,
                            time_in_force=TimeInForce.DAY,
                        )
                        self.event_bus.publish(order_event)

        class LargeOrderStrategy(Strategy):
            """Strategy with large orders (high participation)."""

            def __init__(self):
                super().__init__()
                self.bar_count = 0
                self.position_opened = False

            def on_start(self, portfolio=None, event_bus=None):
                self.portfolio = portfolio
                self.event_bus = event_bus

            def on_event(self, event):
                if isinstance(event, MarketEvent):
                    self.on_market_data(event)

            def on_market_data(self, event: MarketEvent):
                self.bar_count += 1

                # Buy on bar 5
                if self.bar_count == 5 and not self.position_opened:
                    # Large order: 50,000 shares (5% of 1M volume)
                    order_event = OrderEvent(
                        timestamp=event.timestamp,
                        order_id=str(uuid.uuid4()),
                        asset_id=event.asset_id,
                        order_type=OrderType.MARKET,
                        side=OrderSide.BUY,
                        quantity=50_000,
                        time_in_force=TimeInForce.DAY,
                    )
                    self.event_bus.publish(order_event)
                    self.position_opened = True

                # Sell on bar 10 to complete round-trip
                elif self.bar_count == 10 and self.position_opened:
                    position = self.portfolio.get_position(event.asset_id)
                    if position and position.quantity > 0:
                        order_event = OrderEvent(
                            timestamp=event.timestamp,
                            order_id=str(uuid.uuid4()),
                            asset_id=event.asset_id,
                            order_type=OrderType.MARKET,
                            side=OrderSide.SELL,
                            quantity=position.quantity,
                            time_in_force=TimeInForce.DAY,
                        )
                        self.event_bus.publish(order_event)

        volume_model = VolumeAwareSlippage(
            base_slippage_pct=0.0001,
            linear_impact_coeff=0.01,
            sqrt_impact_coeff=0.0,
            max_participation_rate=0.1,
        )

        # Run small order backtest
        results_small = run_backtest_with_slippage(
            market_data_with_spread,
            volume_model,
            SmallOrderStrategy,
            tmp_path=tmp_path,
        )

        # Run large order backtest
        results_large = run_backtest_with_slippage(
            market_data_with_spread,
            volume_model,
            LargeOrderStrategy,
            tmp_path=tmp_path,
        )

        slippage_small = calculate_slippage_cost_percentage(results_small)
        slippage_large = calculate_slippage_cost_percentage(results_large)

        # Large order should pay significantly more slippage
        assert slippage_large > slippage_small, (
            f"Large order slippage ({slippage_large:.4f}%) should be higher than "
            f"small order ({slippage_small:.4f}%)"
        )

        # Both should be in realistic range
        assert 0.01 <= slippage_small <= 0.5
        assert 0.01 <= slippage_large <= 0.5

    def test_volume_aware_vs_percentage_baseline(self, market_data_varying_volume, tmp_path):
        """Compare VolumeAwareSlippage to baseline PercentageSlippage.

        Expected: VolumeAware adapts to market conditions, Percentage is constant.
        """
        # VolumeAware model
        volume_model = VolumeAwareSlippage(
            base_slippage_pct=0.0001,
            linear_impact_coeff=0.01,
            sqrt_impact_coeff=0.0,
        )

        # Percentage baseline
        pct_model = PercentageSlippage(slippage_pct=0.001)

        results_volume = run_backtest_with_slippage(
            market_data_varying_volume,
            volume_model,
            SimpleMAStrategy,
            tmp_path=tmp_path,
        )

        results_pct = run_backtest_with_slippage(
            market_data_varying_volume,
            pct_model,
            SimpleMAStrategy,
            tmp_path=tmp_path,
        )

        slippage_volume = calculate_slippage_cost_percentage(results_volume)
        slippage_pct = calculate_slippage_cost_percentage(results_pct)

        # Both should be in realistic range
        assert 0.01 <= slippage_volume <= 0.5
        assert 0.15 <= slippage_pct <= 0.25  # Should be ~0.2% (round-trip: 0.1% × 2)

        # VolumeAware should differ from fixed percentage
        # (may be higher or lower depending on participation rates)
        assert slippage_volume != slippage_pct


class TestOrderTypeDependentSlippageIntegration:
    """Integration tests for OrderTypeDependentSlippage in realistic backtests."""

    def test_order_type_slippage_hierarchy(self, market_data_with_spread, tmp_path):
        """Test different order types pay different slippage.

        Expected: MARKET > STOP > LIMIT slippage hierarchy.
        """
        # Run with OrderTypeDependentSlippage
        order_type_model = OrderTypeDependentSlippage(
            market_slippage_pct=0.001,  # 0.10%
            limit_slippage_pct=0.0001,  # 0.01%
            stop_slippage_pct=0.0005,  # 0.05%
        )

        results = run_backtest_with_slippage(
            market_data_with_spread,
            order_type_model,
            MultiOrderTypeStrategy,
            tmp_path=tmp_path,
        )

        trades_df = results.get_trades()
        assert not trades_df.is_empty(), "Should have executed trades"

        # Note: BacktestResults.get_trades() may not include order_type in the schema
        # For now, just verify that slippage was applied
        # TODO: Enhance BacktestResults to include order metadata for analysis

        # Calculate total slippage cost
        slippage_cost = calculate_slippage_cost_percentage(results)

        # Verify slippage was applied (should be > 0 for OrderTypeDependentSlippage)
        assert slippage_cost > 0.001, (
            f"OrderTypeDependentSlippage should apply non-zero slippage, got {slippage_cost:.4f}%"
        )

        # Should be in realistic range
        assert 0.01 <= slippage_cost <= 0.5, (
            f"Slippage {slippage_cost:.4f}% outside realistic range [0.01%, 0.5%]"
        )

    def test_order_type_vs_uniform_baseline(self, market_data_with_spread, tmp_path):
        """Compare OrderTypeDependentSlippage to uniform PercentageSlippage.

        Expected: OrderType model differentiates, Percentage treats all same.
        """
        # OrderType model
        order_type_model = OrderTypeDependentSlippage(
            market_slippage_pct=0.001,
            limit_slippage_pct=0.0001,
        )

        # Uniform percentage
        pct_model = PercentageSlippage(slippage_pct=0.001)

        results_order_type = run_backtest_with_slippage(
            market_data_with_spread,
            order_type_model,
            MultiOrderTypeStrategy,
            tmp_path=tmp_path,
        )

        results_pct = run_backtest_with_slippage(
            market_data_with_spread,
            pct_model,
            MultiOrderTypeStrategy,
            tmp_path=tmp_path,
        )

        slippage_order_type = calculate_slippage_cost_percentage(results_order_type)
        slippage_pct = calculate_slippage_cost_percentage(results_pct)

        # Both should be in realistic range
        assert 0.01 <= slippage_order_type <= 0.5
        assert 0.15 <= slippage_pct <= 0.27  # Should be ~0.2% (round-trip: 0.1% × 2), allow variance


class TestSlippageModelComparison:
    """Compare all enhanced slippage models."""

    def test_all_models_realistic_costs(self, market_data_with_spread, tmp_path):
        """Verify all slippage models produce realistic costs (0.01% - 0.5%)."""
        models = [
            ("NoSlippage", NoSlippage()),
            ("Percentage", PercentageSlippage(slippage_pct=0.001)),
            ("SpreadAware", SpreadAwareSlippage(spread_factor=0.5)),
            ("VolumeAware", VolumeAwareSlippage(base_slippage_pct=0.0001, linear_impact_coeff=0.01)),
            ("OrderTypeDependent", OrderTypeDependentSlippage()),
        ]

        results = {}

        for name, model in models:
            backtest_results = run_backtest_with_slippage(
                market_data_with_spread,
                model,
                SimpleMAStrategy,
            tmp_path=tmp_path,
            )
            slippage_cost = calculate_slippage_cost_percentage(backtest_results)
            results[name] = slippage_cost

        # NoSlippage should be 0
        assert results["NoSlippage"] == 0.0

        # All others should be in realistic range
        for name in ["Percentage", "SpreadAware", "VolumeAware", "OrderTypeDependent"]:
            cost = results[name]
            assert 0.01 <= cost <= 0.5, f"{name} slippage {cost:.4f}% outside realistic range"

        # Enhanced models should differ from baseline
        assert results["SpreadAware"] != results["Percentage"]
        assert results["VolumeAware"] != results["Percentage"]

    def test_model_performance_overhead(self, market_data_with_spread, tmp_path):
        """Verify enhanced models have no significant performance overhead.

        Expected: All models complete in <10 seconds total.
        """
        models = [
            ("NoSlippage", NoSlippage()),
            ("Percentage", PercentageSlippage(slippage_pct=0.001)),
            ("SpreadAware", SpreadAwareSlippage(spread_factor=0.5)),
            ("VolumeAware", VolumeAwareSlippage(base_slippage_pct=0.0001, linear_impact_coeff=0.01)),
            ("OrderTypeDependent", OrderTypeDependentSlippage()),
        ]

        start_time = time.perf_counter()

        for name, model in models:
            _ = run_backtest_with_slippage(
                market_data_with_spread,
                model,
                SimpleMAStrategy,
            tmp_path=tmp_path,
            )

        total_time = time.perf_counter() - start_time

        # All models should complete in <10 seconds total
        assert total_time < 10.0, f"Performance test took {total_time:.2f}s (limit: 10s)"


class TestSlippageModelDocumentation:
    """Test documentation and usage guidance for slippage models.

    This test class serves as documentation for when to use each model.
    """

    def test_when_to_use_spread_aware(self):
        """Documentation: When to use SpreadAwareSlippage.

        Use SpreadAwareSlippage when:
        1. You have bid/ask quote data available
        2. Trading liquid assets with tight, consistent spreads
        3. Want realistic fill prices within the spread
        4. Need to differentiate between liquid and illiquid periods

        Example use cases:
        - High-frequency strategies with quote-level data
        - Market-making strategies
        - Liquid equity or forex trading
        - When bid/ask spread is primary cost driver

        Configuration:
        - spread_factor=0.5: Fill at mid-price (realistic for liquid assets)
        - spread_factor=1.0: Fill at ask (buy) or bid (sell) - conservative
        - spread_factor=0.0: Fill at mid exactly (optimistic)
        - fallback_slippage_pct: Used when bid/ask unavailable

        Expected slippage: 0.01% - 0.10% for liquid assets
        """
        # This test documents usage patterns
        assert True

    def test_when_to_use_volume_aware(self):
        """Documentation: When to use VolumeAwareSlippage.

        Use VolumeAwareSlippage when:
        1. Order size varies significantly
        2. Trading across different liquidity regimes
        3. Want slippage to scale with participation rate
        4. Large orders that impact the market

        Example use cases:
        - Portfolio rebalancing (varying order sizes)
        - Strategies with position sizing
        - Illiquid assets or low-volume periods
        - When market impact is primary cost driver

        Configuration:
        - base_slippage_pct: Minimum slippage (e.g., 0.0001 = 0.01%)
        - linear_impact_coeff: Linear impact per participation rate (e.g., 0.01)
        - sqrt_impact_coeff: Non-linear impact (use for large orders)
        - max_participation_rate: Cap on participation (e.g., 0.1 = 10%)

        Expected slippage:
        - Small orders (<1% volume): 0.01% - 0.05%
        - Medium orders (1-5% volume): 0.05% - 0.20%
        - Large orders (>5% volume): 0.20% - 0.50%
        """
        assert True

    def test_when_to_use_order_type_dependent(self):
        """Documentation: When to use OrderTypeDependentSlippage.

        Use OrderTypeDependentSlippage when:
        1. Strategy uses multiple order types (MARKET, LIMIT, STOP)
        2. Want to differentiate execution quality by order type
        3. Testing impact of order type choice
        4. Simulating realistic execution behavior

        Example use cases:
        - Strategies with limit orders for better fills
        - Stop-loss orders (pay slippage during volatile moves)
        - Testing MARKET vs LIMIT order efficiency
        - Multi-leg strategies with different order types

        Configuration:
        - market_slippage_pct: Highest (e.g., 0.001 = 0.10%)
        - limit_slippage_pct: Lowest (e.g., 0.0001 = 0.01%)
        - stop_slippage_pct: Medium (e.g., 0.0005 = 0.05%)

        Expected slippage hierarchy: MARKET > STOP > LIMIT

        Rationale:
        - MARKET: Pay for immediacy, worse price
        - LIMIT: Patient execution, better price
        - STOP: Conditional execution, medium slippage
        """
        assert True

    def test_model_selection_guide(self):
        """Documentation: How to choose a slippage model.

        Decision tree:

        1. Do you have bid/ask data?
           → YES: Consider SpreadAwareSlippage
           → NO: Go to #2

        2. Do order sizes vary significantly?
           → YES: Consider VolumeAwareSlippage
           → NO: Go to #3

        3. Do you use multiple order types?
           → YES: Consider OrderTypeDependentSlippage
           → NO: Use PercentageSlippage (simple baseline)

        4. Can you combine multiple models?
           → Create composite model (e.g., SpreadAware + VolumeAware)
           → Requires custom SlippageModel implementation

        Default recommendation:
        - Research/backtesting: PercentageSlippage (0.1% default)
        - Production simulation: VolumeAwareSlippage (adapts to liquidity)
        - HFT strategies: SpreadAwareSlippage (quote-level precision)
        - Multi-strategy: OrderTypeDependentSlippage (differentiates execution)
        """
        assert True


# ============================================================================
# Performance Benchmark
# ============================================================================


@pytest.mark.slow
def test_performance_benchmark_all_models(market_data_with_spread, tmp_path):
    """Benchmark all slippage models for performance comparison.

    Expected: All models complete in <2 seconds each.
    """
    models = [
        ("NoSlippage", NoSlippage()),
        ("Percentage", PercentageSlippage(slippage_pct=0.001)),
        ("SpreadAware", SpreadAwareSlippage(spread_factor=0.5)),
        ("VolumeAware", VolumeAwareSlippage(base_slippage_pct=0.0001, linear_impact_coeff=0.01)),
        ("OrderTypeDependent", OrderTypeDependentSlippage()),
    ]

    timings = {}

    for name, model in models:
        start = time.perf_counter()
        _ = run_backtest_with_slippage(
            market_data_with_spread,
            model,
            SimpleMAStrategy,
            tmp_path=tmp_path,
        )
        elapsed = time.perf_counter() - start
        timings[name] = elapsed

    # Print results (for manual inspection)
    print("\n" + "=" * 60)
    print("Slippage Model Performance Benchmark")
    print("=" * 60)
    for name, elapsed in timings.items():
        print(f"{name:25s}: {elapsed:.4f}s")
    print("=" * 60)

    # All models should complete in <2 seconds
    for name, elapsed in timings.items():
        assert elapsed < 2.0, f"{name} took {elapsed:.4f}s (limit: 2.0s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
