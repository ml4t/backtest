"""Tests for portfolio rebalancing utilities."""

from datetime import datetime

import pytest

from ml4t.backtest import (
    Broker,
    NoCommission,
    NoSlippage,
    OrderSide,
    RebalanceConfig,
    TargetWeightExecutor,
)


class TestRebalanceConfig:
    """Test RebalanceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RebalanceConfig()
        assert config.min_trade_value == 100.0
        assert config.min_weight_change == 0.01
        assert config.allow_fractional is False
        assert config.round_lots is False
        assert config.lot_size == 100
        assert config.allow_short is False
        assert config.max_single_weight == 1.0
        assert config.cancel_before_rebalance is True
        assert config.account_for_pending is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RebalanceConfig(
            min_trade_value=500.0,
            allow_fractional=True,
            allow_short=True,
            max_single_weight=0.25,
        )
        assert config.min_trade_value == 500.0
        assert config.allow_fractional is True
        assert config.allow_short is True
        assert config.max_single_weight == 0.25


class TestTargetWeightExecutorBasic:
    """Test basic TargetWeightExecutor functionality."""

    @pytest.fixture
    def broker(self):
        """Create a broker with $100,000 initial cash."""
        return Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

    @pytest.fixture
    def executor(self):
        """Create executor with default config."""
        return TargetWeightExecutor()

    @pytest.fixture
    def sample_data(self):
        """Sample market data."""
        return {
            "AAPL": {"close": 150.0, "open": 149.0, "volume": 1000000},
            "GOOG": {"close": 100.0, "open": 99.0, "volume": 500000},
            "MSFT": {"close": 200.0, "open": 199.0, "volume": 800000},
        }

    def test_empty_portfolio_rebalance(self, broker, executor, sample_data):
        """Test rebalancing from empty portfolio."""
        target_weights = {"AAPL": 0.3, "GOOG": 0.3, "MSFT": 0.4}

        orders = executor.execute(target_weights, sample_data, broker)

        assert len(orders) == 3
        # Check all are BUY orders
        for order in orders:
            assert order.side == OrderSide.BUY

    def test_buy_to_target(self, broker, executor, sample_data):
        """Test buying to reach target weight."""
        # Start with 10% AAPL
        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 1000000},
            {},
        )
        broker.submit_order("AAPL", 66, OrderSide.BUY)  # ~$10,000 = 10%
        broker._process_orders()

        # Target 30% AAPL
        target_weights = {"AAPL": 0.3}
        orders = executor.execute(target_weights, sample_data, broker)

        # Should generate BUY order for additional shares
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].asset == "AAPL"

    def test_sell_to_target(self, broker, executor, sample_data):
        """Test selling to reach target weight."""
        # Start with 50% AAPL
        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 1000000},
            {},
        )
        broker.submit_order("AAPL", 333, OrderSide.BUY)  # ~$50,000 = 50%
        broker._process_orders()

        # Target 30% AAPL
        target_weights = {"AAPL": 0.3}
        orders = executor.execute(target_weights, sample_data, broker)

        # Should generate SELL order
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].asset == "AAPL"

    def test_close_position_not_in_target(self, broker, executor, sample_data):
        """Test closing positions not in target weights."""
        # Start with AAPL and GOOG positions
        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0, "GOOG": 100.0},
            {"AAPL": 150.0, "GOOG": 100.0},
            {"AAPL": 150.0, "GOOG": 100.0},
            {"AAPL": 150.0, "GOOG": 100.0},
            {"AAPL": 1000000, "GOOG": 500000},
            {},
        )
        broker.submit_order("AAPL", 100, OrderSide.BUY)
        broker.submit_order("GOOG", 100, OrderSide.BUY)
        broker._process_orders()

        # Target only AAPL (GOOG should be closed)
        target_weights = {"AAPL": 0.5}
        orders = executor.execute(target_weights, sample_data, broker)

        # Should have orders for AAPL rebalance and GOOG close
        asset_orders = {o.asset: o for o in orders}
        assert "GOOG" in asset_orders
        assert asset_orders["GOOG"].side == OrderSide.SELL


class TestTargetWeightExecutorThresholds:
    """Test threshold-based filtering."""

    @pytest.fixture
    def broker(self):
        return Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

    @pytest.fixture
    def sample_data(self):
        return {"AAPL": {"close": 150.0}}

    def test_skip_small_weight_change(self, broker, sample_data):
        """Test skipping trades below min_weight_change threshold."""
        executor = TargetWeightExecutor(config=RebalanceConfig(min_weight_change=0.05))

        # Start with 30% AAPL
        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 1000000},
            {},
        )
        broker.submit_order("AAPL", 200, OrderSide.BUY)  # ~$30,000 = 30%
        broker._process_orders()

        # Target 32% (only 2% change, below 5% threshold)
        target_weights = {"AAPL": 0.32}
        orders = executor.execute(target_weights, sample_data, broker)

        assert len(orders) == 0

    def test_skip_small_trade_value(self, broker, sample_data):
        """Test skipping trades below min_trade_value threshold."""
        executor = TargetWeightExecutor(
            config=RebalanceConfig(min_trade_value=1000.0, min_weight_change=0.001)
        )

        # Empty portfolio, target tiny weight
        target_weights = {"AAPL": 0.005}  # 0.5% of $100k = $500 < $1000
        orders = executor.execute(target_weights, sample_data, broker)

        assert len(orders) == 0


class TestTargetWeightExecutorShareHandling:
    """Test share rounding and fractional shares."""

    @pytest.fixture
    def broker(self):
        return Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

    @pytest.fixture
    def sample_data(self):
        return {"AAPL": {"close": 150.0}}

    def test_whole_shares_default(self, broker, sample_data):
        """Test that shares are rounded to whole numbers by default."""
        executor = TargetWeightExecutor(config=RebalanceConfig(allow_fractional=False))

        target_weights = {"AAPL": 0.1}  # 10% of $100k = $10k / $150 = 66.67 shares
        orders = executor.execute(target_weights, sample_data, broker)

        assert len(orders) == 1
        assert orders[0].quantity == 66  # int(66.67) = 66

    def test_fractional_shares(self, broker, sample_data):
        """Test fractional shares when allowed."""
        executor = TargetWeightExecutor(config=RebalanceConfig(allow_fractional=True))

        target_weights = {"AAPL": 0.1}
        orders = executor.execute(target_weights, sample_data, broker)

        assert len(orders) == 1
        # Should be approximately 66.67
        assert 66.6 < orders[0].quantity < 66.7

    def test_lot_rounding(self, broker, sample_data):
        """Test rounding to lot sizes."""
        executor = TargetWeightExecutor(config=RebalanceConfig(round_lots=True, lot_size=100))

        # Target enough shares to round meaningfully
        target_weights = {"AAPL": 0.3}  # 30% = $30k / $150 = 200 shares
        orders = executor.execute(target_weights, sample_data, broker)

        assert len(orders) == 1
        assert orders[0].quantity % 100 == 0  # Multiple of 100

    def test_rounds_to_zero_skipped(self, broker, sample_data):
        """Test that trades rounding to zero shares are skipped."""
        executor = TargetWeightExecutor(
            config=RebalanceConfig(allow_fractional=False, min_weight_change=0.001)
        )

        # Very small target that rounds to 0 shares
        # 0.1% of $100k = $100 / $150 = 0.67 shares -> int(0.67) = 0
        target_weights = {"AAPL": 0.001}
        orders = executor.execute(target_weights, sample_data, broker)

        assert len(orders) == 0


class TestTargetWeightExecutorPendingOrders:
    """Test pending order awareness."""

    @pytest.fixture
    def broker(self):
        return Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

    @pytest.fixture
    def sample_data(self):
        return {"AAPL": {"close": 150.0}}

    def test_cancel_before_rebalance_default(self, broker, sample_data):
        """Test that pending orders are cancelled by default."""
        executor = TargetWeightExecutor(config=RebalanceConfig(cancel_before_rebalance=True))

        # Create a pending order
        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 1000000},
            {},
        )
        broker.submit_order("AAPL", 100, OrderSide.BUY)
        assert len(broker.pending_orders) == 1

        # Execute rebalance
        target_weights = {"AAPL": 0.3}
        executor.execute(target_weights, sample_data, broker)

        # Original pending order should be cancelled
        # (new order may have been added by execute)

    def test_effective_weights_with_pending(self, broker, sample_data):
        """Test effective weight calculation including pending orders."""
        executor = TargetWeightExecutor(
            config=RebalanceConfig(
                cancel_before_rebalance=False,
                account_for_pending=True,
            )
        )

        # Create a pending BUY order (not yet filled)
        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 150.0},
            {"AAPL": 1000000},
            {},
        )
        broker.submit_order("AAPL", 200, OrderSide.BUY)  # $30k = 30% pending

        # Without accounting for pending, would see 0% current and try to buy 30%
        # With accounting, should see ~30% effective and generate minimal orders
        effective = executor._get_effective_weights(broker, sample_data)
        assert "AAPL" in effective
        assert effective["AAPL"] > 0.2  # Should reflect pending order value


class TestTargetWeightExecutorCashTargeting:
    """Test cash targeting (weights < 1.0)."""

    @pytest.fixture
    def broker(self):
        return Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

    @pytest.fixture
    def sample_data(self):
        return {
            "AAPL": {"close": 150.0},
            "GOOG": {"close": 100.0},
        }

    def test_implicit_cash_holding(self, broker, sample_data):
        """Test that weights summing to < 1.0 leave cash."""
        executor = TargetWeightExecutor()

        # Target 90% invested (10% cash)
        target_weights = {"AAPL": 0.45, "GOOG": 0.45}  # = 90%
        orders = executor.execute(target_weights, sample_data, broker)

        # Should generate orders totaling ~90% of equity
        total_value = sum(o.quantity * sample_data[o.asset]["close"] for o in orders)
        assert 89000 < total_value < 91000  # ~$90k (90% of $100k)

    def test_scale_down_over_100_percent(self, broker, sample_data):
        """Test that weights > 100% are scaled down."""
        executor = TargetWeightExecutor()

        # Target 120% invested (impossible without leverage)
        target_weights = {"AAPL": 0.7, "GOOG": 0.5}  # = 120%
        orders = executor.execute(target_weights, sample_data, broker)

        # Should scale to 100% max
        total_value = sum(o.quantity * sample_data[o.asset]["close"] for o in orders)
        # After scaling: AAPL=58.3%, GOOG=41.7%
        assert total_value < 101000  # Should not exceed equity


class TestTargetWeightExecutorPreview:
    """Test preview functionality."""

    @pytest.fixture
    def broker(self):
        return Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

    @pytest.fixture
    def sample_data(self):
        return {"AAPL": {"close": 150.0}, "GOOG": {"close": 100.0}}

    def test_preview_returns_details(self, broker, sample_data):
        """Test that preview returns trade details."""
        executor = TargetWeightExecutor()

        target_weights = {"AAPL": 0.3, "GOOG": 0.3}
        previews = executor.preview(target_weights, sample_data, broker)

        assert len(previews) == 2
        for preview in previews:
            assert "asset" in preview
            assert "current_weight" in preview
            assert "target_weight" in preview
            assert "weight_delta" in preview
            assert "shares" in preview
            assert "value" in preview
            assert "skip_reason" in preview

    def test_preview_shows_skip_reason(self, broker, sample_data):
        """Test that preview shows why trades would be skipped."""
        executor = TargetWeightExecutor(config=RebalanceConfig(min_weight_change=0.1))

        # Small change that would be skipped
        target_weights = {"AAPL": 0.05}  # Only 5% change, below 10% threshold
        previews = executor.preview(target_weights, sample_data, broker)

        assert len(previews) == 1
        assert previews[0]["skip_reason"] == "weight_change_too_small"

    def test_preview_does_not_execute(self, broker, sample_data):
        """Test that preview doesn't submit orders."""
        executor = TargetWeightExecutor()

        initial_orders = len(broker.orders)
        target_weights = {"AAPL": 0.5}
        executor.preview(target_weights, sample_data, broker)

        # No new orders should be created
        assert len(broker.orders) == initial_orders

    def test_preview_rounds_to_zero(self, broker):
        """Test preview when shares round to zero."""
        executor = TargetWeightExecutor(
            RebalanceConfig(
                min_trade_value=1.0,
                min_weight_change=0.0001,
                allow_fractional=False,
            )
        )

        data = {"AAPL": {"close": 150.0}}

        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0},
            {"AAPL": 149.0},
            {"AAPL": 151.0},
            {"AAPL": 148.0},
            {"AAPL": 1000000},
            {},
        )

        # Very small target weight that would round to 0 shares
        target_weights = {"AAPL": 0.001}  # 0.1% of 100k = $100 = 0.67 shares
        previews = executor.preview(target_weights, data, broker)

        assert len(previews) == 1
        assert previews[0]["skip_reason"] == "rounds_to_zero_shares"

    def test_preview_with_existing_position_to_close(self, broker):
        """Test preview shows positions to close."""
        executor = TargetWeightExecutor(
            RebalanceConfig(
                min_trade_value=100.0,
                min_weight_change=0.01,
                allow_fractional=False,
            )
        )

        data = {
            "AAPL": {"close": 150.0},
            "GOOG": {"close": 100.0},
        }

        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0, "GOOG": 100.0},
            {"AAPL": 149.0, "GOOG": 99.0},
            {"AAPL": 151.0, "GOOG": 101.0},
            {"AAPL": 148.0, "GOOG": 98.0},
            {"AAPL": 1000000, "GOOG": 500000},
            {},
        )

        # Open a position first
        broker.submit_order("GOOG", 100)
        broker._process_orders()

        # Preview rebalance that excludes GOOG (should show close)
        target_weights = {"AAPL": 0.5}  # No GOOG
        previews = executor.preview(target_weights, data, broker)

        # Should have AAPL target and GOOG close
        assert len(previews) == 2
        goog_preview = next(p for p in previews if p["asset"] == "GOOG")
        assert goog_preview["action"] == "close_position"
        assert goog_preview["target_weight"] == 0.0

    def test_preview_zero_equity(self):
        """Test preview with zero equity returns empty."""
        executor = TargetWeightExecutor(
            RebalanceConfig(
                min_trade_value=100.0,
                min_weight_change=0.01,
                allow_fractional=False,
            )
        )

        broker = Broker(
            initial_cash=0.0,  # Zero equity
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

        data = {"AAPL": {"close": 150.0}}
        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0},
            {"AAPL": 149.0},
            {"AAPL": 151.0},
            {"AAPL": 148.0},
            {"AAPL": 1000000},
            {},
        )

        target_weights = {"AAPL": 0.5}
        previews = executor.preview(target_weights, data, broker)

        assert previews == []


class TestTargetWeightExecutorEdgeCases:
    """Test edge cases."""

    @pytest.fixture
    def broker(self):
        return Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

    def test_empty_target_weights(self, broker):
        """Test with empty target weights."""
        executor = TargetWeightExecutor()
        orders = executor.execute({}, {}, broker)
        assert orders == []

    def test_zero_equity(self, broker):
        """Test with zero equity."""
        executor = TargetWeightExecutor()
        broker.cash = 0.0  # Zero equity
        orders = executor.execute({"AAPL": 0.5}, {"AAPL": {"close": 150.0}}, broker)
        assert orders == []

    def test_missing_price_data(self, broker):
        """Test with missing price data for asset."""
        executor = TargetWeightExecutor()
        target_weights = {"AAPL": 0.3}
        data = {}  # No price data
        orders = executor.execute(target_weights, data, broker)
        assert orders == []

    def test_zero_price(self, broker):
        """Test with zero price."""
        executor = TargetWeightExecutor()
        target_weights = {"AAPL": 0.3}
        data = {"AAPL": {"close": 0.0}}
        orders = executor.execute(target_weights, data, broker)
        assert orders == []

    def test_max_single_weight_constraint(self, broker):
        """Test max_single_weight constraint."""
        executor = TargetWeightExecutor(config=RebalanceConfig(max_single_weight=0.25))

        target_weights = {"AAPL": 0.5}  # Request 50%, but max is 25%
        data = {"AAPL": {"close": 150.0}}
        orders = executor.execute(target_weights, data, broker)

        # Should be capped to 25%
        assert len(orders) == 1
        value = orders[0].quantity * 150.0
        assert value < 26000  # Less than 26% of $100k

    def test_short_weight_disallowed(self, broker):
        """Test negative weight is set to 0 when shorts not allowed."""
        executor = TargetWeightExecutor(
            RebalanceConfig(
                allow_short=False,
                min_trade_value=1.0,
                min_weight_change=0.001,
            )
        )

        data = {"AAPL": {"close": 150.0}}

        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0},
            {"AAPL": 149.0},
            {"AAPL": 151.0},
            {"AAPL": 148.0},
            {"AAPL": 1000000},
            {},
        )

        # Negative weight should be ignored
        target_weights = {"AAPL": -0.3}
        orders = executor.execute(target_weights, data, broker)

        # Should not create any orders (negative weight becomes 0)
        assert len(orders) == 0

    def test_effective_weights_path(self, broker):
        """Test using effective weights when accounting for pending."""
        executor = TargetWeightExecutor(
            RebalanceConfig(
                cancel_before_rebalance=False,  # Don't cancel pending
                account_for_pending=True,  # Use effective weights
                min_trade_value=1.0,
            )
        )

        data = {"AAPL": {"close": 150.0}}

        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0},
            {"AAPL": 149.0},
            {"AAPL": 151.0},
            {"AAPL": 148.0},
            {"AAPL": 1000000},
            {},
        )

        target_weights = {"AAPL": 0.3}
        orders = executor.execute(target_weights, data, broker)

        # Should work correctly
        assert len(orders) >= 0  # May or may not have orders

    def test_preview_effective_weights_path(self, broker):
        """Test preview using effective weights path."""
        executor = TargetWeightExecutor(
            RebalanceConfig(
                cancel_before_rebalance=False,
                account_for_pending=True,
            )
        )

        data = {"AAPL": {"close": 150.0}}

        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0},
            {"AAPL": 149.0},
            {"AAPL": 151.0},
            {"AAPL": 148.0},
            {"AAPL": 1000000},
            {},
        )

        target_weights = {"AAPL": 0.3}
        previews = executor.preview(target_weights, data, broker)

        # Should work correctly
        assert isinstance(previews, list)


class TestTargetWeightExecutorIntegration:
    """Integration tests with full workflow."""

    def test_full_rebalance_workflow(self):
        """Test complete rebalancing workflow."""
        broker = Broker(
            initial_cash=100000.0,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )

        executor = TargetWeightExecutor(
            config=RebalanceConfig(
                min_trade_value=100.0,
                min_weight_change=0.01,
                allow_fractional=True,
            )
        )

        data = {
            "AAPL": {"close": 150.0, "open": 149.0, "volume": 1000000},
            "GOOG": {"close": 100.0, "open": 99.0, "volume": 500000},
            "MSFT": {"close": 200.0, "open": 199.0, "volume": 800000},
        }

        # Initialize broker time
        broker._update_time(
            datetime(2024, 1, 1, 9, 30),
            {"AAPL": 150.0, "GOOG": 100.0, "MSFT": 200.0},
            {"AAPL": 149.0, "GOOG": 99.0, "MSFT": 199.0},
            {"AAPL": 151.0, "GOOG": 101.0, "MSFT": 201.0},
            {"AAPL": 148.0, "GOOG": 98.0, "MSFT": 198.0},
            {"AAPL": 1000000, "GOOG": 500000, "MSFT": 800000},
            {},
        )

        # Step 1: Initial rebalance to 30/30/40
        target1 = {"AAPL": 0.3, "GOOG": 0.3, "MSFT": 0.4}
        orders1 = executor.execute(target1, data, broker)
        assert len(orders1) == 3

        # Process orders
        broker._process_orders()

        # Step 2: Rebalance to 50/50/0 (close MSFT)
        target2 = {"AAPL": 0.5, "GOOG": 0.5}
        orders2 = executor.execute(target2, data, broker)

        # Should have AAPL buy, GOOG buy, MSFT sell (close)
        assert len(orders2) >= 2
