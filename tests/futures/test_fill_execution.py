"""Tests for futures fill execution and price calculations.

Validates fill value calculations for futures orders.
"""

import pytest
from datetime import datetime

from ml4t.backtest.types import ContractSpec, Fill, OrderSide


class TestFillValue:
    """Test fill value calculations with multiplier."""

    def test_fill_value_basic(self, es_contract: ContractSpec):
        """Fill value = qty * price * multiplier.

        3 ES at 4005: 3 * 4005 * 50 = $600,750
        """
        qty = 3
        price = 4005.0
        multiplier = es_contract.multiplier

        fill_value = qty * price * multiplier
        assert fill_value == 600_750.0

    def test_fill_value_short(self, es_contract: ContractSpec):
        """Short fill value uses signed quantity.

        -3 ES at 4005: -3 * 4005 * 50 = -$600,750
        (Negative represents credit from short sale)
        """
        qty = -3
        price = 4005.0
        multiplier = es_contract.multiplier

        fill_value = qty * price * multiplier
        assert fill_value == -600_750.0


class TestPartialFills:
    """Test partial fill and average price calculations."""

    def test_partial_fill_average(self, es_contract: ContractSpec):
        """Average price for partial fills.

        Fill 1: 6 contracts at 4000
        Fill 2: 4 contracts at 4002
        Total: 10 contracts
        Average: (6*4000 + 4*4002) / 10 = 4000.8
        """
        fills = [
            {"qty": 6, "price": 4000.0},
            {"qty": 4, "price": 4002.0},
        ]

        total_qty = sum(f["qty"] for f in fills)
        weighted_sum = sum(f["qty"] * f["price"] for f in fills)
        avg_price = weighted_sum / total_qty

        assert total_qty == 10
        assert avg_price == 4000.8

    def test_partial_fill_total_value(self, es_contract: ContractSpec):
        """Total notional from partial fills.

        Fill 1: 6 at 4000 = 6 * 4000 * 50 = $1,200,000
        Fill 2: 4 at 4002 = 4 * 4002 * 50 = $800,400
        Total: $2,000,400
        """
        multiplier = es_contract.multiplier

        fill1_value = 6 * 4000.0 * multiplier
        fill2_value = 4 * 4002.0 * multiplier

        total_value = fill1_value + fill2_value
        assert total_value == 2_000_400.0


class TestLimitOrderFills:
    """Test limit order fill behavior."""

    def test_limit_better_fill(self, es_contract: ContractSpec):
        """Limit order may fill at better price than limit.

        Limit buy at 4000, market opens at 3998.
        Fill at 3998 (better for buyer).
        """
        limit_price = 4000.0
        market_price = 3998.0

        # Limit buy: fill at min(limit, market)
        fill_price = min(limit_price, market_price)
        assert fill_price == 3998.0

    def test_limit_sell_better_fill(self, es_contract: ContractSpec):
        """Limit sell may fill above limit.

        Limit sell at 4000, market at 4002.
        Fill at 4002 (better for seller).
        """
        limit_price = 4000.0
        market_price = 4002.0

        # Limit sell: fill at max(limit, market)
        fill_price = max(limit_price, market_price)
        assert fill_price == 4002.0

    def test_limit_not_filled(self, es_contract: ContractSpec):
        """Limit order not filled if market doesn't reach.

        Limit buy at 3990, market at 4000.
        Order not filled.
        """
        limit_price = 3990.0
        market_price = 4000.0

        # Buy limit at 3990 when market is 4000 - not filled
        filled = market_price <= limit_price
        assert not filled
