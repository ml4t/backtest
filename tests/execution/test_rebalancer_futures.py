"""Tests for TargetWeightExecutor with futures contract specs (multiplier awareness).

Verifies that weight calculation and share sizing correctly account for
contract multipliers. Without this fix, a 30% target weight in ES futures
(multiplier=50) would compute 50x too many contracts.
"""

from datetime import datetime

import pytest

from ml4t.backtest import Broker, OrderSide
from ml4t.backtest.execution.rebalancer import RebalanceConfig, TargetWeightExecutor
from ml4t.backtest.models import NoCommission, NoSlippage
from ml4t.backtest.types import AssetClass, ContractSpec


# --- Fixtures ---

ES_SPEC = ContractSpec(symbol="ES", asset_class=AssetClass.FUTURE, multiplier=50.0)
CL_SPEC = ContractSpec(symbol="CL", asset_class=AssetClass.FUTURE, multiplier=1000.0)
GC_SPEC = ContractSpec(symbol="GC", asset_class=AssetClass.FUTURE, multiplier=100.0)

DEMO_SPECS = {"ES": ES_SPEC, "CL": CL_SPEC, "GC": GC_SPEC}


def _make_broker(initial_cash: float = 1_000_000, specs: dict | None = None) -> Broker:
    return Broker(
        initial_cash=initial_cash,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        contract_specs=specs or DEMO_SPECS,
        allow_short_selling=True,
        allow_leverage=True,
    )


def _init_prices(broker: Broker, prices: dict[str, float]) -> None:
    """Set up broker with current prices."""
    broker._update_time(
        datetime(2024, 1, 2, 9, 30),
        prices,  # close
        prices,  # open
        prices,  # high
        prices,  # low
        {a: 100_000 for a in prices},  # volume
        {},
    )


class TestFuturesWeightCalculation:
    """Verify weight computation includes multiplier."""

    def test_current_weight_includes_multiplier(self):
        """Position weight should reflect notional = qty * price * multiplier."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"ES": {"close": 5000.0}}
        _init_prices(broker, {"ES": 5000.0})

        # Buy 2 ES contracts: notional = 2 * 5000 * 50 = $500,000
        broker.submit_order("ES", 2, OrderSide.BUY)
        broker._process_orders()

        executor = TargetWeightExecutor()
        weights = executor._get_current_weights(broker, data)

        # Equity = cash + positions = $500,000 + $500,000 = $1,000,000
        # Weight should be $500,000 / $1,000,000 = 0.50
        assert "ES" in weights
        assert abs(weights["ES"] - 0.50) < 0.02

    def test_current_weight_without_multiplier_would_be_wrong(self):
        """Without multiplier, 2 ES at $5000 would show as 1% weight instead of 50%."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"ES": {"close": 5000.0}}
        _init_prices(broker, {"ES": 5000.0})

        broker.submit_order("ES", 2, OrderSide.BUY)
        broker._process_orders()

        executor = TargetWeightExecutor()
        weights = executor._get_current_weights(broker, data)

        # The weight should NOT be 2 * 5000 / 1_000_000 = 0.01
        assert weights["ES"] > 0.4  # Must be much larger than 1%


class TestFuturesShareSizing:
    """Verify share sizing accounts for multiplier."""

    def test_target_weight_produces_correct_contracts(self):
        """30% of $1M in ES (mult=50, price=$5000) = $300K / $250K per contract ≈ 1.2."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"ES": {"close": 5000.0}}
        _init_prices(broker, {"ES": 5000.0})

        executor = TargetWeightExecutor(
            config=RebalanceConfig(allow_fractional=True, min_weight_change=0.001)
        )
        orders = executor.execute({"ES": 0.30}, data, broker)

        assert len(orders) == 1
        assert orders[0].asset == "ES"
        assert orders[0].side == OrderSide.BUY

        # target_value = 0.30 * $1M = $300,000
        # notional_per_contract = 5000 * 50 = $250,000
        # qty = $300,000 / $250,000 = 1.2 contracts
        assert abs(orders[0].quantity - 1.2) < 0.01

    def test_without_multiplier_would_buy_50x_too_many(self):
        """Sanity check: qty = $300K / $5000 = 60 contracts (wrong, should be ~1.2)."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"ES": {"close": 5000.0}}
        _init_prices(broker, {"ES": 5000.0})

        executor = TargetWeightExecutor(
            config=RebalanceConfig(allow_fractional=True, min_weight_change=0.001)
        )
        orders = executor.execute({"ES": 0.30}, data, broker)

        # Should NOT be 60 contracts
        assert orders[0].quantity < 5  # Must be in the 1-2 range, not 60

    def test_high_multiplier_product(self):
        """CL (mult=1000): 10% of $1M at $70 = $100K / $70K per contract ≈ 1.43."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"CL": {"close": 70.0}}
        _init_prices(broker, {"CL": 70.0})

        executor = TargetWeightExecutor(
            config=RebalanceConfig(allow_fractional=True, min_weight_change=0.001)
        )
        orders = executor.execute({"CL": 0.10}, data, broker)

        assert len(orders) == 1
        # target = 0.10 * $1M = $100,000
        # per_contract = 70 * 1000 = $70,000
        # qty = $100,000 / $70,000 ≈ 1.43
        assert abs(orders[0].quantity - 100_000 / 70_000) < 0.01

    def test_whole_contract_rounding(self):
        """With allow_fractional=False, contracts round to integers."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"ES": {"close": 5000.0}}
        _init_prices(broker, {"ES": 5000.0})

        executor = TargetWeightExecutor(
            config=RebalanceConfig(allow_fractional=False, min_weight_change=0.001)
        )
        orders = executor.execute({"ES": 0.30}, data, broker)

        assert len(orders) == 1
        # 1.2 rounds to 1 with int()
        assert orders[0].quantity == 1
        assert isinstance(orders[0].quantity, int)


class TestFuturesMultiAssetRebalance:
    """Test rebalancing across multiple futures products."""

    def test_equal_weight_three_futures(self):
        """Equal-weight across ES, CL, GC should produce correct contract counts."""
        broker = _make_broker(initial_cash=1_000_000)
        prices = {"ES": 5000.0, "CL": 70.0, "GC": 2000.0}
        data = {a: {"close": p} for a, p in prices.items()}
        _init_prices(broker, prices)

        executor = TargetWeightExecutor(
            config=RebalanceConfig(allow_fractional=True, min_weight_change=0.001)
        )
        orders = executor.execute({"ES": 0.33, "CL": 0.33, "GC": 0.33}, data, broker)

        assert len(orders) == 3
        order_map = {o.asset: o for o in orders}

        # ES: $330K / (5000 * 50) = 1.32 contracts
        assert abs(order_map["ES"].quantity - 330_000 / 250_000) < 0.01
        # CL: $330K / (70 * 1000) = 4.71 contracts
        assert abs(order_map["CL"].quantity - 330_000 / 70_000) < 0.01
        # GC: $330K / (2000 * 100) = 1.65 contracts
        assert abs(order_map["GC"].quantity - 330_000 / 200_000) < 0.01

    def test_rebalance_from_existing_position(self):
        """Rebalancing should compute delta using multiplier-correct weights."""
        broker = _make_broker(initial_cash=1_000_000)
        prices = {"ES": 5000.0, "GC": 2000.0}
        data = {a: {"close": p} for a, p in prices.items()}
        _init_prices(broker, prices)

        executor = TargetWeightExecutor(
            config=RebalanceConfig(allow_fractional=True, min_weight_change=0.001)
        )

        # First: buy 2 ES contracts (notional = $500K = 50% of $1M)
        broker.submit_order("ES", 2, OrderSide.BUY)
        broker._process_orders()

        # Now rebalance to 25% ES, 25% GC
        orders = executor.execute({"ES": 0.25, "GC": 0.25}, data, broker)

        order_map = {o.asset: o for o in orders}

        # ES: currently 50%, target 25% → sell ~1 contract
        assert "ES" in order_map
        assert order_map["ES"].side == OrderSide.SELL
        # delta_value = (0.25 - 0.50) * $1M = -$250K
        # delta_contracts = -$250K / (5000 * 50) = -1.0
        assert abs(order_map["ES"].quantity - 1.0) < 0.05

        # GC: currently 0%, target 25% → buy
        assert "GC" in order_map
        assert order_map["GC"].side == OrderSide.BUY

    def test_close_position_not_in_target(self):
        """Closing a futures position should work correctly."""
        broker = _make_broker(initial_cash=1_000_000)
        prices = {"ES": 5000.0, "GC": 2000.0}
        data = {a: {"close": p} for a, p in prices.items()}
        _init_prices(broker, prices)

        # Buy ES
        broker.submit_order("ES", 2, OrderSide.BUY)
        broker._process_orders()

        executor = TargetWeightExecutor(
            config=RebalanceConfig(allow_fractional=True, min_weight_change=0.001)
        )

        # Target only GC — ES should be closed
        orders = executor.execute({"GC": 0.25}, data, broker)

        order_map = {o.asset: o for o in orders}
        assert "ES" in order_map
        assert order_map["ES"].side == OrderSide.SELL
        assert order_map["ES"].quantity == 2  # Close all 2 contracts


class TestFuturesShortPositions:
    """Test short positions with futures multipliers."""

    def test_negative_weight_creates_short(self):
        """Negative target weight should create short position."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"ES": {"close": 5000.0}}
        _init_prices(broker, {"ES": 5000.0})

        executor = TargetWeightExecutor(
            config=RebalanceConfig(
                allow_fractional=True, allow_short=True, min_weight_change=0.001
            )
        )
        orders = executor.execute({"ES": -0.25}, data, broker)

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        # -$250K / (5000 * 50) = -1.0 → sell 1 contract
        assert abs(orders[0].quantity - 1.0) < 0.01


class TestFuturesPreview:
    """Test preview with futures multipliers."""

    def test_preview_shows_correct_shares_with_multiplier(self):
        """Preview should compute shares using multiplier."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"ES": {"close": 5000.0}}
        _init_prices(broker, {"ES": 5000.0})

        executor = TargetWeightExecutor(
            config=RebalanceConfig(allow_fractional=True, min_weight_change=0.001)
        )
        previews = executor.preview({"ES": 0.30}, data, broker)

        assert len(previews) == 1
        p = previews[0]
        assert p["asset"] == "ES"
        assert abs(p["target_weight"] - 0.30) < 0.001
        # shares = $300K / (5000 * 50) = 1.2
        assert abs(p["shares"] - 1.2) < 0.01

    def test_preview_close_shows_multiplied_value(self):
        """Preview close position should show notional value with multiplier."""
        broker = _make_broker(initial_cash=1_000_000)
        prices = {"ES": 5000.0, "GC": 2000.0}
        data = {a: {"close": p} for a, p in prices.items()}
        _init_prices(broker, prices)

        # Buy 2 ES
        broker.submit_order("ES", 2, OrderSide.BUY)
        broker._process_orders()

        executor = TargetWeightExecutor(
            config=RebalanceConfig(allow_fractional=True, min_weight_change=0.001)
        )
        # Target only GC — ES will be shown as close
        previews = executor.preview({"GC": 0.25}, data, broker)

        es_preview = next(p for p in previews if p["asset"] == "ES")
        assert es_preview["action"] == "close_position"
        # value = -2 * 5000 * 50 = -$500,000
        assert abs(es_preview["value"] - (-500_000)) < 100


class TestFuturesEffectiveWeights:
    """Test effective weights (pending orders) with multipliers."""

    def test_effective_weight_includes_multiplier(self):
        """Pending orders should be valued with multiplier."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"ES": {"close": 5000.0}}
        _init_prices(broker, {"ES": 5000.0})

        # Submit but don't process — order stays pending
        broker.submit_order("ES", 2, OrderSide.BUY)

        executor = TargetWeightExecutor(
            config=RebalanceConfig(cancel_before_rebalance=False, account_for_pending=True)
        )
        weights = executor._get_effective_weights(broker, data)

        # Pending 2 ES: 2 * 5000 * 50 = $500K → ~50% of $1M
        assert "ES" in weights
        assert weights["ES"] > 0.4  # Must reflect multiplied value


class TestEquityBackwardCompatibility:
    """Verify equities (multiplier=1) still work identically."""

    def test_equity_weight_unchanged(self):
        """Without contract specs, weight = qty * price / equity (multiplier=1)."""
        broker = Broker(
            initial_cash=100_000,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )
        data = {"AAPL": {"close": 150.0}}
        _init_prices(broker, {"AAPL": 150.0})

        broker.submit_order("AAPL", 200, OrderSide.BUY)
        broker._process_orders()

        executor = TargetWeightExecutor()
        weights = executor._get_current_weights(broker, data)

        # 200 * 150 = $30,000 out of $100,000 = 30%
        assert abs(weights["AAPL"] - 0.30) < 0.01

    def test_equity_share_sizing_unchanged(self):
        """Without contract specs, shares = delta_value / price (multiplier=1)."""
        broker = Broker(
            initial_cash=100_000,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
        )
        data = {"AAPL": {"close": 150.0}}
        _init_prices(broker, {"AAPL": 150.0})

        executor = TargetWeightExecutor(
            config=RebalanceConfig(allow_fractional=True, min_weight_change=0.001)
        )
        orders = executor.execute({"AAPL": 0.30}, data, broker)

        assert len(orders) == 1
        # $30,000 / $150 = 200 shares
        assert abs(orders[0].quantity - 200.0) < 0.1


class TestContractSpecMarginWiring:
    """Verify ContractSpec.margin auto-populates fixed_margin_schedule."""

    def test_margin_from_contract_spec(self):
        """ContractSpec.margin should be wired into the broker's margin schedule."""
        es_spec = ContractSpec(
            symbol="ES", asset_class=AssetClass.FUTURE, multiplier=50.0, margin=15_000.0
        )
        broker = Broker(
            initial_cash=100_000,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
            contract_specs={"ES": es_spec},
            allow_leverage=True,
        )

        # The margin schedule should have been auto-populated from ContractSpec.margin
        policy = broker.account.policy
        assert policy.fixed_margin_schedule is not None
        assert "ES" in policy.fixed_margin_schedule
        # Initial margin = spec.margin, maintenance = 50% of initial
        im, mm = policy.fixed_margin_schedule["ES"]
        assert im == 15_000.0
        assert mm == 7_500.0

    def test_explicit_schedule_takes_precedence(self):
        """Explicit fixed_margin_schedule should override ContractSpec.margin."""
        es_spec = ContractSpec(
            symbol="ES", asset_class=AssetClass.FUTURE, multiplier=50.0, margin=15_000.0
        )
        broker = Broker(
            initial_cash=100_000,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
            contract_specs={"ES": es_spec},
            fixed_margin_schedule={"ES": (20_000.0, 10_000.0)},
            allow_leverage=True,
        )

        policy = broker.account.policy
        im, mm = policy.fixed_margin_schedule["ES"]
        # Explicit value should win over ContractSpec.margin
        assert im == 20_000.0
        assert mm == 10_000.0

    def test_margin_enables_leveraged_futures(self):
        """With margin set, futures portfolio can exceed 1.0 gross weight."""
        es_spec = ContractSpec(
            symbol="ES", asset_class=AssetClass.FUTURE, multiplier=50.0, margin=15_000.0
        )
        broker = Broker(
            initial_cash=100_000,
            commission_model=NoCommission(),
            slippage_model=NoSlippage(),
            contract_specs={"ES": es_spec},
            allow_leverage=True,
        )
        data = {"ES": {"close": 5000.0}}
        _init_prices(broker, {"ES": 5000.0})

        executor = TargetWeightExecutor(
            config=RebalanceConfig(
                allow_fractional=True,
                min_weight_change=0.001,
                max_single_weight=10.0,  # Allow leveraged weights
            )
        )

        # Target 200% weight (2x leverage) — 1 ES contract = $250K notional
        # With $100K cash and $15K margin per contract, we can afford ~6 contracts
        # 200% of $100K = $200K notional = 0.8 contracts
        orders = executor.execute({"ES": 2.0}, data, broker)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        # 2.0 * $100K / ($5000 * 50) = 0.8 contracts
        assert abs(orders[0].quantity - 0.8) < 0.01


class TestMaxGrossLeverage:
    """Test the max_gross_leverage safety guardrail."""

    def test_cap_scales_weights(self):
        """max_gross_leverage should scale down weights proportionally."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"ES": {"close": 5000.0}, "CL": {"close": 70.0}}
        _init_prices(broker, {"ES": 5000.0, "CL": 70.0})

        executor = TargetWeightExecutor(
            config=RebalanceConfig(
                allow_fractional=True,
                min_weight_change=0.001,
                max_single_weight=10.0,  # Allow leveraged weights
                max_gross_leverage=3.0,
            )
        )

        # Target 4.0 gross weight → should scale to 3.0
        orders = executor.execute({"ES": 2.0, "CL": 2.0}, data, broker)
        assert len(orders) == 2

        # After scaling: each should be 1.5 (3.0/4.0 * 2.0)
        order_map = {o.asset: o for o in orders}
        # ES: 1.5 * $1M / (5000 * 50) = 6.0 contracts
        assert abs(order_map["ES"].quantity - 6.0) < 0.1
        # CL: 1.5 * $1M / (70 * 1000) = 21.43 contracts
        assert abs(order_map["CL"].quantity - 1_500_000 / 70_000) < 0.1

    def test_no_cap_passes_through(self):
        """Without max_gross_leverage, all weights pass through."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"ES": {"close": 5000.0}}
        _init_prices(broker, {"ES": 5000.0})

        executor = TargetWeightExecutor(
            config=RebalanceConfig(
                allow_fractional=True,
                min_weight_change=0.001,
                max_single_weight=10.0,  # Allow leveraged weights
                # max_gross_leverage=None (default)
            )
        )

        # Target 300% — no cap
        orders = executor.execute({"ES": 3.0}, data, broker)
        assert len(orders) == 1
        # 3.0 * $1M / (5000 * 50) = 12.0 contracts
        assert abs(orders[0].quantity - 12.0) < 0.1

    def test_cap_handles_long_short(self):
        """max_gross_leverage should use absolute weights for long-short portfolios."""
        broker = _make_broker(initial_cash=1_000_000)
        data = {"ES": {"close": 5000.0}, "CL": {"close": 70.0}}
        _init_prices(broker, {"ES": 5000.0, "CL": 70.0})

        executor = TargetWeightExecutor(
            config=RebalanceConfig(
                allow_fractional=True,
                allow_short=True,
                min_weight_change=0.001,
                max_single_weight=10.0,  # Allow leveraged weights
                max_gross_leverage=2.0,
            )
        )

        # Gross = |1.5| + |-1.5| = 3.0, exceeds cap of 2.0
        # Scale: 2.0/3.0 = 0.667 → ES=1.0, CL=-1.0
        orders = executor.execute({"ES": 1.5, "CL": -1.5}, data, broker)
        assert len(orders) == 2

        order_map = {o.asset: o for o in orders}
        assert order_map["ES"].side == OrderSide.BUY
        assert order_map["CL"].side == OrderSide.SELL
        # ES: 1.0 * $1M / (5000 * 50) = 4.0 contracts
        assert abs(order_map["ES"].quantity - 4.0) < 0.1
