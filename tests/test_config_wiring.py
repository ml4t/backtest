"""Tests for config fields that are wired to runtime behavior.

Validates that BacktestConfig fields actually affect execution:
- share_type: INTEGER rounds shares at fill time
- reject_on_insufficient_cash: False allows skipping instead of rejecting
- cash_buffer_pct: reserves cash from available buying power
- partial_fills_allowed: fills max affordable when cash is insufficient
- fill_ordering: EXIT_FIRST vs FIFO processing order
- Preset round-trip: presets produce correct field values
"""

from datetime import datetime

import pytest

from ml4t.backtest import (
    BacktestConfig,
    Broker,
    ExecutionMode,
)
from ml4t.backtest.config import CommissionModel, FillOrdering, ShareType, SlippageModel
from ml4t.backtest.models import (
    CombinedCommission,
    NoCommission,
    NoSlippage,
    PerShareCommission,
    TieredCommission,
    VolumeShareSlippage,
)
from ml4t.backtest.types import OrderSide

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_broker(**kwargs) -> Broker:
    """Create a Broker with sensible defaults, overriding with kwargs."""
    defaults = {
        "initial_cash": 100_000.0,
        "commission_model": NoCommission(),
        "slippage_model": NoSlippage(),
        "execution_mode": ExecutionMode.SAME_BAR,
        "allow_short_selling": True,
        "allow_leverage": False,
    }
    defaults.update(kwargs)
    return Broker(**defaults)


def _set_prices(broker: Broker, prices: dict[str, float], ts=None):
    """Set current prices on broker for order processing."""
    if ts is None:
        ts = datetime(2024, 1, 1)
    broker._current_time = ts
    broker._current_prices = prices
    broker._current_opens = prices
    broker._current_highs = prices
    broker._current_lows = prices


# ---------------------------------------------------------------------------
# share_type enforcement
# ---------------------------------------------------------------------------


class TestShareType:
    """share_type=INTEGER should round order quantities at fill time."""

    def test_integer_share_type_rounds_quantity(self):
        broker = _make_broker(share_type=ShareType.INTEGER)
        _set_prices(broker, {"AAPL": 150.0})

        # Submit order with fractional quantity
        broker.submit_order("AAPL", 10.7, OrderSide.BUY)
        broker._process_orders()

        # Should have been rounded to 10 shares
        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 10.0

    def test_fractional_share_type_preserves_quantity(self):
        broker = _make_broker(share_type=ShareType.FRACTIONAL)
        _set_prices(broker, {"AAPL": 150.0})

        broker.submit_order("AAPL", 10.7, OrderSide.BUY)
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 10.7

    def test_integer_rounds_to_zero_rejects(self):
        broker = _make_broker(share_type=ShareType.INTEGER)
        _set_prices(broker, {"AAPL": 150.0})

        broker.submit_order("AAPL", 0.5, OrderSide.BUY)
        broker._process_orders()

        # Should be rejected (rounds to 0)
        pos = broker.get_position("AAPL")
        assert pos is None

    def test_from_config_propagates_share_type(self):
        config = BacktestConfig(share_type=ShareType.INTEGER)
        broker = Broker.from_config(config)
        assert broker.share_type == ShareType.INTEGER


# ---------------------------------------------------------------------------
# reject_on_insufficient_cash
# ---------------------------------------------------------------------------


class TestRejectOnInsufficientCash:
    """reject_on_insufficient_cash=False should skip (not reject) unaffordable orders."""

    def test_default_rejects_unaffordable(self):
        broker = _make_broker(initial_cash=1000.0, reject_on_insufficient_cash=True)
        _set_prices(broker, {"AAPL": 150.0})

        broker.submit_order("AAPL", 100, OrderSide.BUY)  # costs $15,000
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos is None
        # Order should be rejected
        rejected = [o for o in broker.orders if o.rejection_reason]
        assert len(rejected) == 1

    def test_permissive_skips_unaffordable(self):
        broker = _make_broker(initial_cash=1000.0, reject_on_insufficient_cash=False)
        _set_prices(broker, {"AAPL": 150.0})

        broker.submit_order("AAPL", 100, OrderSide.BUY)  # costs $15,000
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos is None
        # Order should NOT have rejection_reason set (silently skipped)
        rejected = [o for o in broker.orders if o.rejection_reason]
        assert len(rejected) == 0

    def test_from_config_propagates(self):
        config = BacktestConfig.from_preset("vectorbt")
        assert config.reject_on_insufficient_cash is False
        broker = Broker.from_config(config)
        assert broker.reject_on_insufficient_cash is False


# ---------------------------------------------------------------------------
# cash_buffer_pct
# ---------------------------------------------------------------------------


class TestCashBufferPct:
    """cash_buffer_pct should reserve a fraction of cash from buying power."""

    def test_buffer_reduces_buying_power(self):
        # With 2% buffer, $100k cash → $98k available
        broker = _make_broker(initial_cash=100_000.0, cash_buffer_pct=0.02)
        _set_prices(broker, {"AAPL": 100.0})

        # Try to buy exactly $99,000 worth = 990 shares
        # Available is $98,000, so 990 shares ($99k) should be rejected
        broker.submit_order("AAPL", 990, OrderSide.BUY)
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos is None  # rejected due to buffer

    def test_buffer_allows_within_limit(self):
        broker = _make_broker(initial_cash=100_000.0, cash_buffer_pct=0.02)
        _set_prices(broker, {"AAPL": 100.0})

        # Buy $97,000 worth = 970 shares (within $98k available)
        broker.submit_order("AAPL", 970, OrderSide.BUY)
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 970

    def test_zero_buffer_allows_full_cash(self):
        broker = _make_broker(initial_cash=100_000.0, cash_buffer_pct=0.0)
        _set_prices(broker, {"AAPL": 100.0})

        broker.submit_order("AAPL", 1000, OrderSide.BUY)  # exactly $100k
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos is not None

    def test_from_config_propagates(self):
        config = BacktestConfig.from_preset("realistic")
        assert config.cash_buffer_pct == 0.02
        broker = Broker.from_config(config)
        assert broker.cash_buffer_pct == 0.02
        assert broker.gatekeeper.cash_buffer_pct == 0.02


# ---------------------------------------------------------------------------
# partial_fills_allowed
# ---------------------------------------------------------------------------


class TestPartialFills:
    """partial_fills_allowed=True should fill max affordable quantity."""

    def test_partial_fill_on_insufficient_cash(self):
        broker = _make_broker(
            initial_cash=5_000.0,
            partial_fills_allowed=True,
            reject_on_insufficient_cash=True,
        )
        _set_prices(broker, {"AAPL": 100.0})

        # Try to buy 100 shares ($10k) but only have $5k
        broker.submit_order("AAPL", 100, OrderSide.BUY)
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos is not None
        # Should have filled ~50 shares (max affordable)
        assert pos.quantity <= 50
        assert pos.quantity > 0

    def test_no_partial_fill_when_disabled(self):
        broker = _make_broker(
            initial_cash=5_000.0,
            partial_fills_allowed=False,
            reject_on_insufficient_cash=True,
        )
        _set_prices(broker, {"AAPL": 100.0})

        broker.submit_order("AAPL", 100, OrderSide.BUY)
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos is None

    def test_partial_fill_with_integer_shares(self):
        broker = _make_broker(
            initial_cash=5_250.0,
            partial_fills_allowed=True,
            share_type=ShareType.INTEGER,
        )
        _set_prices(broker, {"AAPL": 100.0})

        broker.submit_order("AAPL", 100, OrderSide.BUY)
        broker._process_orders()

        pos = broker.get_position("AAPL")
        assert pos is not None
        # Should be integer shares
        assert pos.quantity == int(pos.quantity)
        assert pos.quantity == 52.0  # floor(5250/100)


# ---------------------------------------------------------------------------
# fill_ordering
# ---------------------------------------------------------------------------


class TestFillOrdering:
    """fill_ordering controls order processing sequence."""

    def test_exit_first_frees_capital(self):
        """EXIT_FIRST processes exits before entries, freeing cash."""
        broker = _make_broker(
            initial_cash=10_000.0,
            fill_ordering=FillOrdering.EXIT_FIRST,
        )
        _set_prices(broker, {"AAPL": 100.0, "GOOG": 100.0})

        # Buy AAPL first (use all cash)
        broker.submit_order("AAPL", 100, OrderSide.BUY)
        broker._process_orders()
        assert broker.get_position("AAPL") is not None

        # Now submit exit AAPL + entry GOOG in same bar
        _set_prices(broker, {"AAPL": 100.0, "GOOG": 100.0})
        broker.submit_order("AAPL", 100, OrderSide.SELL)
        broker.submit_order("GOOG", 100, OrderSide.BUY)
        broker._process_orders()

        # EXIT_FIRST: AAPL sell frees $10k, then GOOG buy succeeds
        assert broker.get_position("AAPL") is None
        assert broker.get_position("GOOG") is not None

    def test_fifo_processes_in_submission_order(self):
        """FIFO processes orders in submission order."""
        broker = _make_broker(
            initial_cash=10_000.0,
            fill_ordering=FillOrdering.FIFO,
        )
        _set_prices(broker, {"AAPL": 100.0, "GOOG": 100.0})

        # Buy AAPL first
        broker.submit_order("AAPL", 100, OrderSide.BUY)
        broker._process_orders()

        # Submit sell AAPL + buy GOOG
        _set_prices(broker, {"AAPL": 100.0, "GOOG": 100.0})
        broker.submit_order("AAPL", 100, OrderSide.SELL)
        broker.submit_order("GOOG", 100, OrderSide.BUY)
        broker._process_orders()

        # FIFO: sell AAPL first (frees cash via mark-to-market), then buy GOOG
        assert broker.get_position("AAPL") is None
        assert broker.get_position("GOOG") is not None

    def test_from_config_backtrader_uses_fifo(self):
        config = BacktestConfig.from_preset("backtrader")
        assert config.fill_ordering == FillOrdering.FIFO

    def test_from_config_vectorbt_uses_exit_first(self):
        config = BacktestConfig.from_preset("vectorbt")
        assert config.fill_ordering == FillOrdering.EXIT_FIRST

    def test_from_config_default_uses_exit_first(self):
        config = BacktestConfig.from_preset("default")
        assert config.fill_ordering == FillOrdering.EXIT_FIRST


# ---------------------------------------------------------------------------
# Preset round-trip
# ---------------------------------------------------------------------------


class TestPresetRoundTrip:
    """Presets should produce correct field values."""

    @pytest.mark.parametrize(
        "preset_name", ["default", "backtrader", "vectorbt", "zipline", "realistic"]
    )
    def test_preset_creates_valid_config(self, preset_name):
        config = BacktestConfig.from_preset(preset_name)
        assert config.preset_name == preset_name
        assert isinstance(config.share_type, ShareType)
        assert isinstance(config.fill_ordering, FillOrdering)

    def test_backtrader_preset_values(self):
        config = BacktestConfig.from_preset("backtrader")
        assert config.share_type == ShareType.INTEGER
        assert config.fill_ordering == FillOrdering.FIFO
        assert config.reject_on_insufficient_cash is True

    def test_vectorbt_preset_values(self):
        config = BacktestConfig.from_preset("vectorbt")
        assert config.share_type == ShareType.FRACTIONAL
        assert config.fill_ordering == FillOrdering.EXIT_FIRST
        assert config.reject_on_insufficient_cash is False
        assert config.partial_fills_allowed is True

    def test_realistic_preset_values(self):
        config = BacktestConfig.from_preset("realistic")
        assert config.share_type == ShareType.INTEGER
        assert config.cash_buffer_pct == 0.02

    def test_to_dict_from_dict_roundtrip(self):
        config = BacktestConfig.from_preset("backtrader")
        d = config.to_dict()
        restored = BacktestConfig.from_dict(d)
        assert restored.fill_ordering == config.fill_ordering
        assert restored.share_type == config.share_type
        assert restored.cash_buffer_pct == config.cash_buffer_pct
        assert restored.reject_on_insufficient_cash == config.reject_on_insufficient_cash
        assert restored.partial_fills_allowed == config.partial_fills_allowed

    def test_sizing_method_removed_from_fields(self):
        """sizing_method was removed from BacktestConfig fields."""
        config = BacktestConfig()
        assert not hasattr(config, "sizing_method")

    def test_allow_negative_cash_removed(self):
        """allow_negative_cash was removed from BacktestConfig fields."""
        config = BacktestConfig()
        assert not hasattr(config, "allow_negative_cash")


class TestConfigModelWiring:
    """All commission/slippage enum choices should map to model instances."""

    def test_per_trade_commission_maps_to_combined_commission(self):
        broker = Broker.from_config(
            BacktestConfig(
                commission_model=CommissionModel.PER_TRADE,
                commission_per_trade=2.5,
            )
        )
        assert isinstance(broker.commission_model, CombinedCommission)
        assert broker.commission_model.fixed == 2.5

    def test_tiered_commission_maps_to_tiered_commission(self):
        broker = Broker.from_config(
            BacktestConfig(
                commission_model=CommissionModel.TIERED,
                commission_rate=0.0012,
            )
        )
        assert isinstance(broker.commission_model, TieredCommission)
        assert broker.commission_model.tiers == [(float("inf"), 0.0012)]

    def test_volume_based_slippage_maps_to_volume_share_slippage(self):
        broker = Broker.from_config(
            BacktestConfig(
                slippage_model=SlippageModel.VOLUME_BASED,
                slippage_rate=0.25,
            )
        )
        assert isinstance(broker.slippage_model, VolumeShareSlippage)
        assert broker.slippage_model.impact_factor == 0.25

    def test_per_share_commission_still_maps_correctly(self):
        broker = Broker.from_config(
            BacktestConfig(
                commission_model=CommissionModel.PER_SHARE,
                commission_per_share=0.01,
                commission_minimum=1.0,
            )
        )
        assert isinstance(broker.commission_model, PerShareCommission)
        assert broker.commission_model.per_share == 0.01
