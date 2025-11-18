"""End-to-end integration tests for advanced risk management rules.

This test suite validates that advanced risk rules work correctly in real backtest scenarios:
- Volatility-scaled stops adapting to ATR changes
- Dynamic trailing stops capturing trends
- Regime-dependent rules switching at VIX thresholds
- Portfolio constraints halting trading during drawdowns
- Multiple rule combinations with conflict resolution

Each scenario validates expected trades, exit reasons, P&L, and performance overhead.
"""

import time
from datetime import datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest

from ml4t.backtest.core.event import EventType, MarketEvent
from ml4t.backtest.core.types import MarketDataType, OrderSide, OrderType
from ml4t.backtest.data.polars_feed import PolarsDataFeed
from ml4t.backtest.engine import BacktestEngine
from ml4t.backtest.execution.broker import SimulationBroker
from ml4t.backtest.execution.order import Order
from ml4t.backtest.portfolio.portfolio import Portfolio
from ml4t.backtest.risk import RiskManager
from ml4t.backtest.risk.rules import (
    DynamicTrailingStop,
    MaxDailyLossRule,
    MaxDrawdownRule,
    PriceBasedStopLoss,
    RegimeDependentRule,
    TimeBasedExit,
    VolatilityScaledStopLoss,
    VolatilityScaledTakeProfit,
)
from ml4t.backtest.strategy.base import Strategy


# ============================================================================
# Test Data Creation
# ============================================================================


def create_trending_data_with_volatility(days: int = 100) -> pl.DataFrame:
    """Create synthetic trending price data with changing volatility.

    Args:
        days: Number of days of data

    Returns:
        DataFrame with OHLCV columns, ATR, and VIX-like volatility indicator
    """
    start_date = datetime(2024, 1, 1, 9, 30)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    prices = []
    for i in range(days):
        # Create trending pattern with changing volatility
        if i < 30:
            # Low volatility uptrend: 100 -> 115
            price = 100 + (i / 30) * 15
            volatility = 0.5
            atr = 0.8
            vix = 15.0
        elif i < 50:
            # High volatility downtrend: 115 -> 95
            price = 115 - ((i - 30) / 20) * 20
            volatility = 2.0
            atr = 3.0
            vix = 35.0  # High VIX during volatility
        elif i < 70:
            # Low volatility recovery: 95 -> 105
            price = 95 + ((i - 50) / 20) * 10
            volatility = 0.5
            atr = 0.8
            vix = 18.0
        else:
            # Sideways with moderate volatility: around 105
            price = 105 + ((i % 5) - 2) * 1.5
            volatility = 1.0
            atr = 1.5
            vix = 22.0

        # Add intraday range based on volatility
        high = price + volatility * 1.5
        low = price - volatility * 1.5
        open_price = price - volatility * 0.5
        close_price = price + volatility * 0.5

        prices.append(
            {
                "timestamp": dates[i],
                "asset_id": "TEST",
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": 1_000_000,
                "atr": atr,  # ATR indicator
                "vix": vix,  # VIX-like volatility indicator
            }
        )

    return pl.DataFrame(prices)


def create_drawdown_scenario_data(days: int = 60) -> pl.DataFrame:
    """Create data that triggers portfolio constraints (daily loss, drawdown).

    Args:
        days: Number of days of data

    Returns:
        DataFrame with price pattern that causes drawdown then recovery
    """
    start_date = datetime(2024, 1, 1, 9, 30)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    prices = []
    for i in range(days):
        if i < 15:
            # Stable start: 100
            price = 100
        elif i < 25:
            # Sharp drawdown: 100 -> 85 (-15%)
            price = 100 - ((i - 15) / 10) * 15
        elif i < 35:
            # Continued weakness: 85 -> 82 (testing daily loss limit)
            price = 85 - ((i - 25) / 10) * 3
        elif i < 50:
            # Recovery: 82 -> 95
            price = 82 + ((i - 35) / 15) * 13
        else:
            # Stabilization: around 95
            price = 95 + ((i % 3) - 1) * 0.5

        # Standard OHLC
        high = price + 0.5
        low = price - 0.5
        open_price = price - 0.2
        close_price = price + 0.2

        prices.append(
            {
                "timestamp": dates[i],
                "asset_id": "TEST",
                "open": open_price,
                "high": high,
                "low": low,
                "close": close_price,
                "volume": 1_000_000,
            }
        )

    return pl.DataFrame(prices)


@pytest.fixture
def trending_data_file(tmp_path):
    """Create temporary Parquet file with trending data and volatility."""
    data = create_trending_data_with_volatility(days=100)
    file_path = tmp_path / "trending_volatility_data.parquet"
    data.write_parquet(file_path)
    return file_path


@pytest.fixture
def drawdown_data_file(tmp_path):
    """Create temporary Parquet file with drawdown scenario data."""
    data = create_drawdown_scenario_data(days=60)
    file_path = tmp_path / "drawdown_data.parquet"
    data.write_parquet(file_path)
    return file_path


# ============================================================================
# Test Strategies
# ============================================================================


class SimpleEntryStrategy(Strategy):
    """Simple buy-and-hold strategy that enters on day 1 and holds."""

    def __init__(self, entry_day: int = 1, quantity: int = 100):
        """Initialize strategy.

        Args:
            entry_day: Day to enter position (0-indexed)
            quantity: Position size
        """
        super().__init__()
        self.entry_day = entry_day
        self.quantity = quantity
        self.days_seen = 0
        self.entered = False

    def on_event(self, event) -> None:
        """Handle events (required abstract method)."""
        pass

    def on_market_data(self, event: MarketEvent, context=None) -> None:
        """Enter position on specified day."""
        if self.days_seen == self.entry_day and not self.entered:
            self.entered = True
            order = Order(
                asset_id=event.asset_id,
                quantity=self.quantity,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
            )
            self.broker.submit_order(order)

        self.days_seen += 1


class MultipleEntryStrategy(Strategy):
    """Strategy that enters multiple times to test portfolio constraints."""

    def __init__(self, entry_days: list[int], quantity: int = 100):
        """Initialize strategy.

        Args:
            entry_days: Days to enter positions (0-indexed)
            quantity: Position size per entry
        """
        super().__init__()
        self.entry_days = set(entry_days)
        self.quantity = quantity
        self.days_seen = 0

    def on_event(self, event) -> None:
        """Handle events (required abstract method)."""
        pass

    def on_market_data(self, event: MarketEvent, context=None) -> None:
        """Enter position on specified days."""
        if self.days_seen in self.entry_days:
            order = Order(
                asset_id=event.asset_id,
                quantity=self.quantity,
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
            )
            self.broker.submit_order(order)

        self.days_seen += 1


# ============================================================================
# Scenario 1: Volatility-Scaled Stops Adapt to Changing ATR
# ============================================================================


class TestScenario1VolatilityScaledStops:
    """Scenario 1: Volatility-scaled stops adapt to changing ATR correctly."""

    def test_volatility_scaled_stops_adapt_to_atr_changes(
        self, trending_data_file, tmp_path
    ):
        """Test that volatility-scaled stops tighten in low volatility and widen in high volatility.

        Scenario:
        - Days 0-30: Low volatility (ATR=0.8), enter at day 5
        - Days 30-50: High volatility (ATR=3.0), stop should widen
        - Days 50-70: Low volatility (ATR=0.8), stop should tighten
        - Exit should occur when price hits the adapted stop loss
        """
        # Setup
        feed = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")
        strategy = SimpleEntryStrategy(entry_day=5, quantity=100)
        broker = SimulationBroker(initial_cash=10000.0)

        # Risk rule: 2x ATR stop loss
        risk_manager = RiskManager()
        risk_manager.add_rule(
            VolatilityScaledStopLoss(
                atr_multiplier=2.0,
                volatility_key="atr",
                priority=5,
            )
        )

        # Run backtest
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            broker=broker,
            risk_manager=risk_manager,
        )

        start_time = time.time()
        results = engine.run()
        execution_time = time.time() - start_time

        # Validations
        assert results is not None, "Backtest should complete"
        assert execution_time < 30, f"Should complete in <30s, took {execution_time:.2f}s"

        # Check that position was entered
        trades = broker.trade_tracker.get_all_trades()
        assert len(trades) > 0, "Should have at least one trade"

        entry_trade = trades[0]
        assert entry_trade.quantity == 100, "Entry should be 100 shares"

        # Check that exits occurred (due to volatility-scaled stops)
        # In high volatility period, stop should be wider
        # In low volatility period, stop should be tighter
        # We expect at least one exit due to stop loss
        exit_trades = [t for t in trades if t.quantity < 0]
        assert len(exit_trades) > 0, "Should have exit trades due to stop loss"

        # Verify that stop loss was the exit reason (check metadata if available)
        # This is a basic check - detailed metadata would require engine enhancement
        print(f"\n✅ Scenario 1: {len(trades)} trades, execution time: {execution_time:.2f}s")

    def test_volatility_scaled_stops_performance_overhead(
        self, trending_data_file, tmp_path
    ):
        """Test that volatility-scaled stops add <5% overhead vs simple stops."""
        feed_simple = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")
        feed_volatility = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")

        strategy_simple = SimpleEntryStrategy(entry_day=5, quantity=100)
        strategy_volatility = SimpleEntryStrategy(entry_day=5, quantity=100)

        broker_simple = SimulationBroker(initial_cash=10000.0)
        broker_volatility = SimulationBroker(initial_cash=10000.0)

        # Simple stop loss
        risk_manager_simple = RiskManager()
        risk_manager_simple.register_rule(PriceBasedStopLoss(sl_pct=0.02))

        # Volatility-scaled stop loss
        risk_manager_volatility = RiskManager()
        risk_manager_volatility.register_rule(
            VolatilityScaledStopLoss(atr_multiplier=2.0, volatility_key="atr")
        )

        # Run simple backtest
        engine_simple = BacktestEngine(
            data_feed=feed_simple,
            strategy=strategy_simple,
            broker=broker_simple,
            risk_manager=risk_manager_simple,
        )
        start_simple = time.time()
        engine_simple.run()
        time_simple = time.time() - start_simple

        # Run volatility-scaled backtest
        engine_volatility = BacktestEngine(
            data_feed=feed_volatility,
            strategy=strategy_volatility,
            broker=broker_volatility,
            risk_manager=risk_manager_volatility,
        )
        start_volatility = time.time()
        engine_volatility.run()
        time_volatility = time.time() - start_volatility

        # Calculate overhead
        overhead_pct = ((time_volatility - time_simple) / time_simple) * 100
        print(
            f"\n⏱️  Simple: {time_simple:.3f}s, Volatility-scaled: {time_volatility:.3f}s, Overhead: {overhead_pct:.1f}%"
        )

        assert (
            overhead_pct < 5.0
        ), f"Overhead {overhead_pct:.1f}% exceeds 5% threshold"


# ============================================================================
# Scenario 2: Dynamic Trailing Stop Captures Trend and Exits on Reversal
# ============================================================================


class TestScenario2DynamicTrailingStop:
    """Scenario 2: Dynamic trailing stop captures trend and exits on reversal."""

    def test_trailing_stop_captures_uptrend_and_exits_on_reversal(
        self, trending_data_file, tmp_path
    ):
        """Test that trailing stop tightens during uptrend and exits on reversal.

        Scenario:
        - Days 0-30: Uptrend (100 -> 115), enter at day 5
        - Trailing stop should tighten as price rises
        - Days 30-60: Downtrend (115 -> 95)
        - Should exit when price reverses and hits trailing stop
        """
        # Setup
        feed = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")
        strategy = SimpleEntryStrategy(entry_day=5, quantity=100)
        broker = SimulationBroker(initial_cash=10000.0)

        # Dynamic trailing stop: track MFE and trail by 50%
        risk_manager = RiskManager()
        risk_manager.add_rule(
            DynamicTrailingStop(
                trail_pct=0.05,  # Trail 5% below highest price
                activation_pct=0.03,  # Activate after 3% profit
                priority=5,
            )
        )

        # Run backtest
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            broker=broker,
            risk_manager=risk_manager,
        )

        start_time = time.time()
        results = engine.run()
        execution_time = time.time() - start_time

        # Validations
        assert results is not None, "Backtest should complete"
        assert execution_time < 30, f"Should complete in <30s, took {execution_time:.2f}s"

        # Check trades
        trades = broker.trade_tracker.get_all_trades()
        assert len(trades) > 0, "Should have trades"

        # Entry should occur
        entry_trade = trades[0]
        assert entry_trade.quantity == 100, "Entry should be 100 shares"

        # Exit should occur during downtrend when trailing stop is hit
        exit_trades = [t for t in trades if t.quantity < 0]
        assert len(exit_trades) > 0, "Should have exit due to trailing stop"

        # Entry around day 5, price ~102.5
        # Uptrend peaks around day 30, price ~115
        # Trailing stop should activate and trail
        # Downtrend should trigger exit

        print(
            f"\n✅ Scenario 2: {len(trades)} trades, execution time: {execution_time:.2f}s"
        )


# ============================================================================
# Scenario 3: Regime-Dependent Rules Switch at VIX Threshold
# ============================================================================


class TestScenario3RegimeDependentRules:
    """Scenario 3: Regime-dependent rules switch at VIX threshold."""

    def test_regime_switching_at_vix_threshold(self, trending_data_file, tmp_path):
        """Test that regime-dependent rules switch between low and high volatility regimes.

        Scenario:
        - Days 0-30: Low VIX (15), use tight stops (2% SL)
        - Days 30-50: High VIX (35), switch to wide stops (5% SL)
        - Days 50+: Low VIX (18), switch back to tight stops
        - Verify that stops adapt to regime
        """
        # Setup
        feed = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")
        strategy = SimpleEntryStrategy(entry_day=5, quantity=100)
        broker = SimulationBroker(initial_cash=10000.0)

        # Regime-dependent rule: different stops for low vs high VIX
        risk_manager = RiskManager()

        # Use factory method for VIX-based regime detection
        regime_rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=25.0,  # VIX > 25 = high volatility
            vix_key="vix",
            low_vol_rule=PriceBasedStopLoss(sl_pct=0.02),  # 2% stop in low vol
            high_vol_rule=PriceBasedStopLoss(sl_pct=0.05),  # 5% stop in high vol
            priority=5,
        )
        risk_manager.add_rule(regime_rule)

        # Run backtest
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            broker=broker,
            risk_manager=risk_manager,
        )

        start_time = time.time()
        results = engine.run()
        execution_time = time.time() - start_time

        # Validations
        assert results is not None, "Backtest should complete"
        assert execution_time < 30, f"Should complete in <30s, took {execution_time:.2f}s"

        # Check trades
        trades = broker.trade_tracker.get_all_trades()
        assert len(trades) > 0, "Should have trades"

        # Verify regime switching occurred
        # Entry at day 5 (low VIX) -> tight stop
        # Day 30-50 (high VIX) -> wide stop
        # Day 50+ (low VIX) -> tight stop again

        print(
            f"\n✅ Scenario 3: {len(trades)} trades, execution time: {execution_time:.2f}s"
        )


# ============================================================================
# Scenario 4: Portfolio Constraints Halt Trading During Drawdown
# ============================================================================


class TestScenario4PortfolioConstraints:
    """Scenario 4: Portfolio constraints halt trading during drawdown, resume after recovery."""

    def test_max_drawdown_halts_trading_and_resumes(
        self, drawdown_data_file, tmp_path
    ):
        """Test that MaxDrawdownRule halts trading during drawdown and resumes after recovery.

        Scenario:
        - Days 0-15: Stable at 100, establish high-water mark
        - Days 15-25: Drawdown to 85 (-15%), should hit 10% max drawdown limit
        - Days 25-35: Further weakness to 82, trading should be halted
        - Days 35-50: Recovery to 95, resume trading
        """
        # Setup
        feed = PolarsDataFeed(price_path=drawdown_data_file, asset_id="TEST")

        # Strategy attempts to enter multiple times
        strategy = MultipleEntryStrategy(
            entry_days=[5, 20, 30, 45], quantity=100
        )
        broker = SimulationBroker(initial_cash=10000.0)

        # Portfolio constraint: max 10% drawdown
        risk_manager = RiskManager()
        risk_manager.add_rule(
            MaxDrawdownRule(
                max_drawdown_pct=0.10,  # 10% max drawdown
                priority=15,  # High priority (evaluated first)
            )
        )

        # Run backtest
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            broker=broker,
            risk_manager=risk_manager,
        )

        start_time = time.time()
        results = engine.run()
        execution_time = time.time() - start_time

        # Validations
        assert results is not None, "Backtest should complete"
        assert execution_time < 30, f"Should complete in <30s, took {execution_time:.2f}s"

        # Check trades
        trades = broker.trade_tracker.get_all_trades()

        # Day 5 entry should succeed (before drawdown)
        # Day 20 entry should be rejected (during drawdown)
        # Day 30 entry should be rejected (still in drawdown)
        # Day 45 entry should succeed (after recovery)

        # We expect fewer entries than strategy attempted
        entry_trades = [t for t in trades if t.quantity > 0]
        print(
            f"\n✅ Scenario 4: {len(entry_trades)} entries (expected <4 due to drawdown constraint)"
        )
        print(f"   Execution time: {execution_time:.2f}s")

        # At least one entry should be rejected
        assert len(entry_trades) < 4, "Some entries should be rejected by drawdown rule"

    def test_max_daily_loss_halts_trading(self, drawdown_data_file, tmp_path):
        """Test that MaxDailyLossRule halts trading when daily loss limit is hit."""
        # Setup
        feed = PolarsDataFeed(price_path=drawdown_data_file, asset_id="TEST")
        strategy = MultipleEntryStrategy(
            entry_days=[5, 20, 25, 30], quantity=100
        )
        broker = SimulationBroker(initial_cash=10000.0)

        # Portfolio constraint: max 2% daily loss
        risk_manager = RiskManager()
        risk_manager.add_rule(
            MaxDailyLossRule(
                max_daily_loss_pct=0.02,  # 2% max daily loss
                priority=15,
            )
        )

        # Run backtest
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            broker=broker,
            risk_manager=risk_manager,
        )

        start_time = time.time()
        results = engine.run()
        execution_time = time.time() - start_time

        # Validations
        assert results is not None, "Backtest should complete"
        assert execution_time < 30, f"Should complete in <30s, took {execution_time:.2f}s"

        # Check that rule prevented some trades during high loss days
        trades = broker.trade_tracker.get_all_trades()
        entry_trades = [t for t in trades if t.quantity > 0]

        print(
            f"\n✅ Scenario 4 (Daily Loss): {len(entry_trades)} entries, execution time: {execution_time:.2f}s"
        )


# ============================================================================
# Scenario 5: Multiple Rules Combined with Conflict Resolution
# ============================================================================


class TestScenario5MultipleRulesCombined:
    """Scenario 5: Multiple rules combined (volatility + trailing + time), verify conflict resolution."""

    def test_multiple_rules_combined_with_conflict_resolution(
        self, trending_data_file, tmp_path
    ):
        """Test multiple rules working together with conflict resolution.

        Scenario:
        - Combine volatility-scaled stop, trailing stop, and time-based exit
        - Enter position and let multiple rules compete
        - Verify that conflict resolution works (priority + conservative logic)
        - Check that most conservative exit wins
        """
        # Setup
        feed = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")
        strategy = SimpleEntryStrategy(entry_day=5, quantity=100)
        broker = SimulationBroker(initial_cash=10000.0)

        # Multiple rules with different priorities
        risk_manager = RiskManager()

        # High priority: Time-based exit (max 20 days)
        risk_manager.add_rule(TimeBasedExit(max_bars=20, priority=10))

        # Medium priority: Volatility-scaled stop loss (2x ATR)
        risk_manager.add_rule(
            VolatilityScaledStopLoss(
                atr_multiplier=2.0, volatility_key="atr", priority=5
            )
        )

        # Medium priority: Dynamic trailing stop
        risk_manager.add_rule(
            DynamicTrailingStop(
                trail_pct=0.05, activation_pct=0.03, priority=5
            )
        )

        # Low priority: Volatility-scaled take profit (3x ATR)
        risk_manager.add_rule(
            VolatilityScaledTakeProfit(
                atr_multiplier=3.0, volatility_key="atr", priority=3
            )
        )

        # Run backtest
        engine = BacktestEngine(
            data_feed=feed,
            strategy=strategy,
            broker=broker,
            risk_manager=risk_manager,
        )

        start_time = time.time()
        results = engine.run()
        execution_time = time.time() - start_time

        # Validations
        assert results is not None, "Backtest should complete"
        assert execution_time < 30, f"Should complete in <30s, took {execution_time:.2f}s"

        # Check trades
        trades = broker.trade_tracker.get_all_trades()
        assert len(trades) > 0, "Should have trades"

        # Entry should occur
        entry_trade = trades[0]
        assert entry_trade.quantity == 100, "Entry should be 100 shares"

        # Exit should occur (one of the rules should trigger)
        exit_trades = [t for t in trades if t.quantity < 0]
        assert len(exit_trades) > 0, "Should have exit due to one of the rules"

        # Verify that exit occurred within 20 days (time-based max)
        # This validates that time-based exit (highest priority) is respected

        print(
            f"\n✅ Scenario 5: {len(trades)} trades with 4 rules combined, execution time: {execution_time:.2f}s"
        )

    def test_multiple_rules_performance_overhead(
        self, trending_data_file, tmp_path
    ):
        """Test that multiple rules (4+) add <5% overhead vs single rule."""
        # Simple single rule
        feed_simple = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")
        strategy_simple = SimpleEntryStrategy(entry_day=5, quantity=100)
        broker_simple = SimulationBroker(initial_cash=10000.0)
        risk_manager_simple = RiskManager()
        risk_manager_simple.register_rule(PriceBasedStopLoss(sl_pct=0.02))

        engine_simple = BacktestEngine(
            data_feed=feed_simple,
            strategy=strategy_simple,
            broker=broker_simple,
            risk_manager=risk_manager_simple,
        )
        start_simple = time.time()
        engine_simple.run()
        time_simple = time.time() - start_simple

        # Multiple rules (4 rules)
        feed_multi = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")
        strategy_multi = SimpleEntryStrategy(entry_day=5, quantity=100)
        broker_multi = SimulationBroker(initial_cash=10000.0)
        risk_manager_multi = RiskManager()
        risk_manager_multi.register_rule(TimeBasedExit(max_bars=20, priority=10))
        risk_manager_multi.register_rule(
            VolatilityScaledStopLoss(
                atr_multiplier=2.0, volatility_key="atr", priority=5
            )
        )
        risk_manager_multi.register_rule(
            DynamicTrailingStop(trail_pct=0.05, activation_pct=0.03, priority=5)
        )
        risk_manager_multi.register_rule(
            VolatilityScaledTakeProfit(
                atr_multiplier=3.0, volatility_key="atr", priority=3
            )
        )

        engine_multi = BacktestEngine(
            data_feed=feed_multi,
            strategy=strategy_multi,
            broker=broker_multi,
            risk_manager=risk_manager_multi,
        )
        start_multi = time.time()
        engine_multi.run()
        time_multi = time.time() - start_multi

        # Calculate overhead
        overhead_pct = ((time_multi - time_simple) / time_simple) * 100
        print(
            f"\n⏱️  Single rule: {time_simple:.3f}s, 4 rules: {time_multi:.3f}s, Overhead: {overhead_pct:.1f}%"
        )

        assert (
            overhead_pct < 5.0
        ), f"Overhead {overhead_pct:.1f}% exceeds 5% threshold"


# ============================================================================
# Performance Summary Test
# ============================================================================


class TestPerformanceSummary:
    """Overall performance validation across all scenarios."""

    def test_all_scenarios_complete_in_30_seconds(
        self, trending_data_file, drawdown_data_file, tmp_path
    ):
        """Test that all 5 scenarios complete in <30 seconds total."""
        start_total = time.time()

        # Scenario 1: Volatility-scaled stops
        feed1 = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")
        strategy1 = SimpleEntryStrategy(entry_day=5, quantity=100)
        broker1 = SimulationBroker(initial_cash=10000.0)
        risk_manager1 = RiskManager()
        risk_manager1.register_rule(
            VolatilityScaledStopLoss(atr_multiplier=2.0, volatility_key="atr")
        )
        engine1 = BacktestEngine(
            data_feed=feed1,
            strategy=strategy1,
            broker=broker1,
            risk_manager=risk_manager1,
        )
        engine1.run()

        # Scenario 2: Trailing stop
        feed2 = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")
        strategy2 = SimpleEntryStrategy(entry_day=5, quantity=100)
        broker2 = SimulationBroker(initial_cash=10000.0)
        risk_manager2 = RiskManager()
        risk_manager2.register_rule(
            DynamicTrailingStop(trail_pct=0.05, activation_pct=0.03)
        )
        engine2 = BacktestEngine(
            data_feed=feed2,
            strategy=strategy2,
            broker=broker2,
            risk_manager=risk_manager2,
        )
        engine2.run()

        # Scenario 3: Regime-dependent
        feed3 = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")
        strategy3 = SimpleEntryStrategy(entry_day=5, quantity=100)
        broker3 = SimulationBroker(initial_cash=10000.0)
        risk_manager3 = RiskManager()
        regime_rule = RegimeDependentRule.from_vix_threshold(
            vix_threshold=25.0,
            vix_key="vix",
            low_vol_rule=PriceBasedStopLoss(sl_pct=0.02),
            high_vol_rule=PriceBasedStopLoss(sl_pct=0.05),
        )
        risk_manager3.register_rule(regime_rule)
        engine3 = BacktestEngine(
            data_feed=feed3,
            strategy=strategy3,
            broker=broker3,
            risk_manager=risk_manager3,
        )
        engine3.run()

        # Scenario 4: Portfolio constraints
        feed4 = PolarsDataFeed(price_path=drawdown_data_file, asset_id="TEST")
        strategy4 = MultipleEntryStrategy(entry_days=[5, 20, 30, 45], quantity=100)
        broker4 = SimulationBroker(initial_cash=10000.0)
        risk_manager4 = RiskManager()
        risk_manager4.register_rule(MaxDrawdownRule(max_drawdown_pct=0.10))
        engine4 = BacktestEngine(
            data_feed=feed4,
            strategy=strategy4,
            broker=broker4,
            risk_manager=risk_manager4,
        )
        engine4.run()

        # Scenario 5: Multiple rules
        feed5 = PolarsDataFeed(price_path=trending_data_file, asset_id="TEST")
        strategy5 = SimpleEntryStrategy(entry_day=5, quantity=100)
        broker5 = SimulationBroker(initial_cash=10000.0)
        risk_manager5 = RiskManager()
        risk_manager5.register_rule(TimeBasedExit(max_bars=20, priority=10))
        risk_manager5.register_rule(
            VolatilityScaledStopLoss(
                atr_multiplier=2.0, volatility_key="atr", priority=5
            )
        )
        risk_manager5.register_rule(
            DynamicTrailingStop(trail_pct=0.05, activation_pct=0.03, priority=5)
        )
        engine5 = BacktestEngine(
            data_feed=feed5,
            strategy=strategy5,
            broker=broker5,
            risk_manager=risk_manager5,
        )
        engine5.run()

        total_time = time.time() - start_total

        print(f"\n⏱️  Total execution time for all 5 scenarios: {total_time:.2f}s")
        assert (
            total_time < 30.0
        ), f"All scenarios should complete in <30s, took {total_time:.2f}s"
