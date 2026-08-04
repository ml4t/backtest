"""Scale-relative quantity-zero tolerance.

Covers the primitive in ``ml4t.backtest.core.shared`` and every call site that was
migrated onto it, plus the sites deliberately left alone.

Expected boundaries here are built independently of the production helper. For a
float64 ``x`` in the binade ``[2**e, 2**(e+1))`` the spacing is exactly
``2**(e-52)``; ``_ulp_from_binade`` reconstructs that from ``frexp`` alone, using
neither ``math.ulp`` nor ``quantity_zero_tolerance``. A test that asked the
production code for its own expected answer would pass no matter what the rule
said.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest

from ml4t.backtest.broker import Broker
from ml4t.backtest.config import ShareType
from ml4t.backtest.core.order_book import OrderBook
from ml4t.backtest.core.shared import (
    QTY_ZERO_FLOOR,
    QTY_ZERO_ULPS,
    quantity_zero_tolerance,
)
from ml4t.backtest.models import NoCommission, NoSlippage
from ml4t.backtest.types import OrderSide, Position

TS = datetime(2024, 1, 2, 16, 0)
PRICE = 10.0

# The residual observed in the field, on a short cover of ~10,927 shares.
OBSERVED_RESIDUAL = 2.0**-39
OBSERVED_QUANTITY = 10927.322882


def _ulp_from_binade(x: float) -> float:
    """Spacing of float64 at ``x``, derived from the exponent alone.

    ``frexp`` returns ``(m, e)`` with ``0.5 <= |m| < 1`` and ``x = m * 2**e``, so
    ``x`` lies in the binade ``[2**(e-1), 2**e)`` and the spacing there is
    ``2**(e-1-52)``. Independent of ``math.ulp`` and of the production helper.
    """
    _, exponent = math.frexp(abs(x))
    return 2.0 ** (exponent - 1 - 52)


def _expected_tolerance(*operands: float) -> float:
    """Reference implementation of the contract, written from the spec."""
    finite = [abs(o) for o in operands if math.isfinite(o)]
    scale = max(finite) if finite else 0.0
    if scale == 0.0:
        return QTY_ZERO_FLOOR
    return max(QTY_ZERO_FLOOR, QTY_ZERO_ULPS * _ulp_from_binade(scale))


def _broker() -> Broker:
    return Broker(
        initial_cash=100_000_000.0,
        commission_model=NoCommission(),
        slippage_model=NoSlippage(),
        allow_short_selling=True,
        allow_leverage=True,
        share_type=ShareType.FRACTIONAL,
    )


def _seed(broker: Broker, asset: str, quantity: float) -> None:
    broker.positions[asset] = Position(
        asset=asset,
        quantity=quantity,
        entry_price=PRICE,
        current_price=PRICE,
        entry_time=TS,
    )


def _fill(broker: Broker, asset: str, signed_qty: float) -> None:
    """Submit and process one market order of ``signed_qty`` shares."""
    broker._update_time(
        timestamp=TS,
        prices={asset: PRICE},
        opens={asset: PRICE},
        highs={asset: PRICE},
        lows={asset: PRICE},
        volumes={asset: 1e12},
        signals={},
    )
    side = OrderSide.BUY if signed_qty > 0 else OrderSide.SELL
    broker.submit_order(asset, abs(signed_qty), side)
    broker._process_orders()


def _close_with_residual(position_qty: float, residual_ulps: int) -> Broker:
    """Seed ``position_qty`` and fill the offsetting order, off by ``residual_ulps``.

    The offsetting quantity is nudged so the sum lands exactly ``residual_ulps``
    units in the last place away from zero. Both operands sit within a factor of
    two of each other, so by Sterbenz's lemma the addition is exact and the
    residual is precisely what was constructed.
    """
    broker = _broker()
    _seed(broker, "AAA", position_qty)
    magnitude = abs(position_qty)
    offset = residual_ulps * _ulp_from_binade(magnitude)
    closing = math.copysign(magnitude + offset, -position_qty)
    _fill(broker, "AAA", closing)
    return broker


# --------------------------------------------------------------------------
# The primitive
# --------------------------------------------------------------------------


class TestTolerancePrimitive:
    def test_floor_applies_where_spacing_is_finer(self):
        # 16 ULP of 1.0 is 3.55e-15, far below the floor.
        assert quantity_zero_tolerance(1.0) == QTY_ZERO_FLOOR

    @pytest.mark.parametrize("magnitude", [4096.0, 8191.0, 8192.0, 8192.5, 10927.322882, 874442.6])
    def test_matches_independent_reference(self, magnitude):
        assert quantity_zero_tolerance(magnitude) == _expected_tolerance(magnitude)
        assert quantity_zero_tolerance(-magnitude, magnitude) == _expected_tolerance(magnitude)

    def test_binade_boundary_is_8192(self):
        """The absolute 1e-12 rule failed at exactly 2**13; document the crossing."""
        assert _ulp_from_binade(8191.0) < 1e-12
        assert _ulp_from_binade(8192.0) > 1e-12
        assert _ulp_from_binade(8192.0) == OBSERVED_RESIDUAL

    def test_scale_is_the_largest_operand(self):
        assert quantity_zero_tolerance(3.0, -20000.0) == quantity_zero_tolerance(20000.0)
        assert quantity_zero_tolerance(-20000.0, 3.0) == quantity_zero_tolerance(20000.0)

    def test_symmetric_in_sign(self):
        for magnitude in (1.0, 8192.0, OBSERVED_QUANTITY, 874442.6):
            assert quantity_zero_tolerance(magnitude) == quantity_zero_tolerance(-magnitude)
            assert quantity_zero_tolerance(magnitude, -magnitude) == quantity_zero_tolerance(
                -magnitude, magnitude
            )

    def test_zero_scale_falls_back_to_floor(self):
        assert quantity_zero_tolerance() == QTY_ZERO_FLOOR
        assert quantity_zero_tolerance(0.0) == QTY_ZERO_FLOOR
        assert quantity_zero_tolerance(0.0, -0.0) == QTY_ZERO_FLOOR

    def test_subnormal_operands_fall_back_to_floor(self):
        assert quantity_zero_tolerance(5e-324) == QTY_ZERO_FLOOR
        assert quantity_zero_tolerance(1e-320, -1e-322) == QTY_ZERO_FLOOR

    @pytest.mark.parametrize(
        "operands",
        [
            (float("inf"),),
            (float("-inf"),),
            (float("nan"),),
            (float("nan"), float("inf")),
        ],
    )
    def test_non_finite_only_operands_fall_back_to_floor(self, operands):
        result = quantity_zero_tolerance(*operands)
        assert result == QTY_ZERO_FLOOR
        assert math.isfinite(result)

    def test_non_finite_operands_are_ignored_not_propagated(self):
        result = quantity_zero_tolerance(float("nan"), -8192.0, float("inf"))
        assert math.isfinite(result)
        assert result == _expected_tolerance(8192.0)

    def test_tolerance_never_reaches_the_operand_scale(self):
        """A single-operand call can only fire through the floor, never the ULP term."""
        for magnitude in (1e-300, 1e-10, 1.0, 8192.0, 1e10, 1e300):
            assert quantity_zero_tolerance(magnitude) < max(magnitude, QTY_ZERO_FLOOR * 2)

    def test_single_operand_is_equivalent_to_the_old_absolute_rule(self):
        """Sites that only ask 'is this book quantity zero?' keep prior behaviour.

        The engine's own boundary convention is ``<=``; against the old ``<= 1e-12``
        sites the migration is exactly behaviour-preserving.
        """
        for magnitude in (0.0, 5e-324, 1e-13, 1e-12, 2e-12, 1.0, 8192.0, 1e9):
            old = abs(magnitude) <= 1e-12
            new = abs(magnitude) <= quantity_zero_tolerance(magnitude)
            assert old == new, magnitude


# --------------------------------------------------------------------------
# Position closure through the real fill path
# --------------------------------------------------------------------------


class TestExactClose:
    @pytest.mark.parametrize(
        "quantity",
        [
            4096.5,  # below the old failure point
            8191.9,  # immediately below 2**13
            8192.0,  # exactly at 2**13
            math.nextafter(8192.0, math.inf),  # immediately above 2**13
            OBSERVED_QUANTITY,  # the case observed in the field
            16384.0,  # 2**14
            1048576.0,  # 2**20
        ],
    )
    def test_bitwise_exact_close_removes_the_key(self, quantity):
        for signed in (quantity, -quantity):
            broker = _close_with_residual(signed, residual_ulps=0)
            assert "AAA" not in broker.positions
            assert len(broker.positions) == 0


class TestResidualClose:
    @pytest.mark.parametrize("ulps", [1, 2, 8, QTY_ZERO_ULPS - 1, QTY_ZERO_ULPS])
    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_residue_inside_the_bound_closes(self, ulps, sign):
        broker = _close_with_residual(sign * OBSERVED_QUANTITY, residual_ulps=ulps)
        assert "AAA" not in broker.positions

    @pytest.mark.parametrize("ulps", [QTY_ZERO_ULPS + 1, QTY_ZERO_ULPS + 2, 64])
    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_residue_outside_the_bound_is_retained(self, ulps, sign):
        broker = _close_with_residual(sign * OBSERVED_QUANTITY, residual_ulps=ulps)
        assert "AAA" in broker.positions
        assert abs(broker.positions["AAA"].quantity) > 0.0

    def test_long_and_short_boundaries_are_identical(self):
        """The last closing and first retained ULP count must match across signs."""
        edges = {}
        for sign in (1.0, -1.0):
            closed = [
                n
                for n in range(0, 2 * QTY_ZERO_ULPS + 2)
                if "AAA" not in _close_with_residual(sign * OBSERVED_QUANTITY, n).positions
            ]
            edges[sign] = (max(closed), len(closed))
        assert edges[1.0] == edges[-1.0]
        # The predicate is |q| <= tol, so the tolerance itself still closes.
        assert edges[1.0][0] == QTY_ZERO_ULPS

    def test_the_observed_field_case(self):
        """Short 10,927.322882 covered exactly, residual 2**-39, must close."""
        broker = _broker()
        _seed(broker, "AAA", -OBSERVED_QUANTITY)
        closing = math.nextafter(OBSERVED_QUANTITY, math.inf)
        # Sterbenz: the operands are within a factor of two, so this is exact.
        assert -OBSERVED_QUANTITY + closing == OBSERVED_RESIDUAL
        assert OBSERVED_RESIDUAL > 1e-12  # the old absolute rule could not fire
        _fill(broker, "AAA", closing)
        assert "AAA" not in broker.positions
        assert len(broker.positions) == 0


class TestGenuineSmallPositions:
    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_genuine_small_position_survives_reduction_of_a_large_one(self, sign):
        """Trading 10,000 down to 1e-9 leaves a real position, not residue."""
        broker = _broker()
        _seed(broker, "AAA", sign * 10_000.0)
        genuine = 1e-9
        _fill(broker, "AAA", -sign * (10_000.0 - genuine))
        assert "AAA" in broker.positions
        assert broker.positions["AAA"].quantity == pytest.approx(sign * genuine, rel=1e-6)

    @pytest.mark.parametrize("sign", [1.0, -1.0])
    def test_genuine_small_position_opened_at_its_own_scale_survives(self, sign):
        broker = _broker()
        _fill(broker, "AAA", sign * 1e-7)
        assert "AAA" in broker.positions
        assert broker.positions["AAA"].quantity == pytest.approx(sign * 1e-7, rel=1e-9)

    def test_a_genuine_position_at_the_residual_magnitude_is_not_erased(self):
        """1e-9 shares is 550x the tolerance at scale 10,000 - it must persist."""
        broker = _broker()
        _seed(broker, "AAA", 10_000.0)
        _fill(broker, "AAA", -9_999.999999999)
        assert "AAA" in broker.positions
        assert 0.0 < broker.positions["AAA"].quantity < 1e-8


class TestPartialFills:
    def test_partial_long_reduction(self):
        broker = _broker()
        _seed(broker, "AAA", 20_000.0)
        _fill(broker, "AAA", -7_500.0)
        assert broker.positions["AAA"].quantity == pytest.approx(12_500.0)

    def test_partial_short_cover(self):
        broker = _broker()
        _seed(broker, "AAA", -20_000.0)
        _fill(broker, "AAA", 7_500.0)
        assert broker.positions["AAA"].quantity == pytest.approx(-12_500.0)

    def test_repeated_fills_ending_flat_close_the_key(self):
        broker = _broker()
        _seed(broker, "AAA", 12_000.0)
        for _ in range(4):
            _fill(broker, "AAA", -3_000.0)
        assert "AAA" not in broker.positions

    def test_repeated_fills_leaving_a_position_keep_the_key(self):
        broker = _broker()
        _seed(broker, "AAA", 12_000.0)
        for _ in range(3):
            _fill(broker, "AAA", -3_000.0)
        assert broker.positions["AAA"].quantity == pytest.approx(3_000.0)


class TestPositionCountConsistency:
    def test_book_holds_no_residual_keys_after_a_close(self):
        """n_positions is len(broker.positions); a residual key inflates it."""
        broker = _broker()
        for asset, qty in (("AAA", -OBSERVED_QUANTITY), ("BBB", 5_000.0), ("CCC", 30_000.0)):
            _seed(broker, asset, qty)
        _fill(broker, "AAA", math.nextafter(OBSERVED_QUANTITY, math.inf))
        assert set(broker.positions) == {"BBB", "CCC"}
        assert len(broker.positions) == 2
        assert all(p.quantity != 0.0 for p in broker.positions.values())


# --------------------------------------------------------------------------
# Cross-site consistency
# --------------------------------------------------------------------------


class TestMigratedSiteConsistency:
    @pytest.mark.parametrize("ulps", [0, 1, QTY_ZERO_ULPS, QTY_ZERO_ULPS + 1, 64])
    def test_precheck_simulation_agrees_with_the_executor(self, ulps):
        """OrderBook._simulate_position_update models the same closure decision."""
        magnitude = OBSERVED_QUANTITY
        offset = ulps * _ulp_from_binade(magnitude)
        old_qty = -magnitude
        size = magnitude + offset

        new_qty, _price, _opened, _closed = OrderBook._simulate_position_update(
            old_qty, PRICE, size, PRICE
        )
        simulated_closed = new_qty == 0.0

        broker = _close_with_residual(old_qty, residual_ulps=ulps)
        executed_closed = "AAA" not in broker.positions

        assert simulated_closed == executed_closed

    @pytest.mark.parametrize("ulps", [0, 1, QTY_ZERO_ULPS, QTY_ZERO_ULPS + 1])
    def test_shadow_queue_commit_agrees_with_the_executor(self, ulps):
        """ExecutionEngine._commit_shadow_queue_fill drops the same keys."""
        magnitude = OBSERVED_QUANTITY
        offset = ulps * _ulp_from_binade(magnitude)
        broker = _broker()
        broker._update_time(
            timestamp=TS,
            prices={"AAA": PRICE},
            opens={"AAA": PRICE},
            highs={"AAA": PRICE},
            lows={"AAA": PRICE},
            volumes={"AAA": 1e12},
            signals={},
        )
        shadow = {
            "AAA": Position(
                asset="AAA",
                quantity=-magnitude,
                entry_price=PRICE,
                current_price=PRICE,
                entry_time=TS,
            )
        }
        order = broker.submit_order("AAA", magnitude + offset, OrderSide.BUY)
        assert order is not None
        broker._execution_engine._commit_shadow_queue_fill(
            order=order, fill_price=PRICE, shadow_cash=0.0, shadow_positions=shadow
        )
        shadow_closed = "AAA" not in shadow

        executed_closed = "AAA" not in _close_with_residual(-magnitude, ulps).positions
        assert shadow_closed == executed_closed


class TestRetainedSites:
    def test_minimum_order_size_is_unchanged(self):
        """_MIN_ORDER_SIZE is an economic policy, not a residue rule."""
        assert OrderBook._MIN_ORDER_SIZE == 1e-8

    def test_orders_at_or_below_the_minimum_are_still_refused(self):
        broker = _broker()
        broker._update_time(
            timestamp=TS,
            prices={"AAA": PRICE},
            opens={"AAA": PRICE},
            highs={"AAA": PRICE},
            lows={"AAA": PRICE},
            volumes={"AAA": 1e12},
            signals={},
        )
        assert broker.submit_order("AAA", 1e-8, OrderSide.BUY) is None
        assert broker.submit_order("AAA", 1e-9, OrderSide.BUY) is None
        assert broker.submit_order("AAA", 1e-7, OrderSide.BUY) is not None

    def test_tolerance_stays_below_the_minimum_order_size_in_practice(self):
        """Largest operation scale seen in the field is ~874k shares."""
        assert quantity_zero_tolerance(874_442.6149965813) < OrderBook._MIN_ORDER_SIZE
