"""Tests for ContractSpec and futures contract specifications.

Validates that ContractSpec correctly stores and exposes
futures contract characteristics for P&L calculation.
"""

import pytest

from ml4t.backtest.types import AssetClass, ContractSpec


class TestContractSpecBasics:
    """Test ContractSpec creation and attributes."""

    def test_es_multiplier(self, es_contract: ContractSpec):
        """ES futures: multiplier = 50 (each point = $50)."""
        assert es_contract.multiplier == 50.0
        assert es_contract.symbol == "ES"
        assert es_contract.asset_class == AssetClass.FUTURE

    def test_cl_multiplier(self, cl_contract: ContractSpec):
        """CL futures: multiplier = 1000 (each $1 move = $1000)."""
        assert cl_contract.multiplier == 1000.0
        assert cl_contract.symbol == "CL"

    def test_gc_multiplier(self, gc_contract: ContractSpec):
        """GC futures: multiplier = 100 (100 oz per contract)."""
        assert gc_contract.multiplier == 100.0
        assert gc_contract.symbol == "GC"

    def test_zn_multiplier(self, zn_contract: ContractSpec):
        """ZN futures: multiplier = 1000 ($100k face value)."""
        assert zn_contract.multiplier == 1000.0
        assert zn_contract.symbol == "ZN"

    def test_nq_multiplier(self, nq_contract: ContractSpec):
        """NQ futures: multiplier = 20 (each point = $20)."""
        assert nq_contract.multiplier == 20.0
        assert nq_contract.symbol == "NQ"


class TestContractSpecCurrencyValue:
    """Test point value calculations."""

    def test_es_point_value(self, es_contract: ContractSpec):
        """1 point move on ES = $50.

        PySystemTrade formula: pnl = qty * price_change * multiplier
        """
        qty = 1
        price_change = 1.0  # 1 point
        expected_pnl = qty * price_change * es_contract.multiplier
        assert expected_pnl == 50.0

    def test_cl_dollar_value(self, cl_contract: ContractSpec):
        """$1 move on CL = $1000 per contract.

        CL is priced in $/barrel, 1000 barrels per contract.
        """
        qty = 1
        price_change = 1.0  # $1 per barrel
        expected_pnl = qty * price_change * cl_contract.multiplier
        assert expected_pnl == 1000.0

    def test_gc_dollar_value(self, gc_contract: ContractSpec):
        """$1 move on GC = $100 per contract.

        GC is priced in $/oz, 100 oz per contract.
        """
        qty = 1
        price_change = 1.0  # $1 per oz
        expected_pnl = qty * price_change * gc_contract.multiplier
        assert expected_pnl == 100.0


class TestContractSpecDefaults:
    """Test default values for equity-style assets."""

    def test_equity_defaults(self):
        """Equity defaults: multiplier=1, tick_size=0.01."""
        spec = ContractSpec(symbol="AAPL")
        assert spec.multiplier == 1.0
        assert spec.tick_size == 0.01
        assert spec.asset_class == AssetClass.EQUITY
        assert spec.currency == "USD"

    def test_explicit_asset_class(self):
        """Asset class can be set explicitly."""
        spec = ContractSpec(
            symbol="ES",
            asset_class=AssetClass.FUTURE,
            multiplier=50.0,
        )
        assert spec.asset_class == AssetClass.FUTURE

    def test_margin_optional(self):
        """Margin is optional (None by default)."""
        spec = ContractSpec(symbol="AAPL")
        assert spec.margin is None

    def test_margin_for_futures(self, es_contract: ContractSpec):
        """Futures typically have explicit margin requirements."""
        assert es_contract.margin == 15000.0
