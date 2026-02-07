"""Shared fixtures for futures validation tests.

Provides standard contract specifications and commission models
based on PySystemTrade reference implementations.
"""

import pytest

from ml4t.backtest.models import FuturesCommission, FuturesSlippage
from ml4t.backtest.types import AssetClass, ContractSpec


# === Standard Contract Specifications ===
# Reference: CME Group product specifications


@pytest.fixture
def es_contract() -> ContractSpec:
    """E-mini S&P 500 futures specification.

    - Multiplier: $50 per index point
    - Tick size: 0.25 points ($12.50)
    - Initial margin: ~$15,000 per contract (varies)
    """
    return ContractSpec(
        symbol="ES",
        asset_class=AssetClass.FUTURE,
        multiplier=50.0,
        tick_size=0.25,
        margin=15000.0,
        currency="USD",
    )


@pytest.fixture
def cl_contract() -> ContractSpec:
    """Crude Oil WTI futures specification.

    - Multiplier: $1,000 per barrel (1000 barrels per contract)
    - Tick size: $0.01 ($10)
    - Initial margin: ~$6,000 per contract (varies)
    """
    return ContractSpec(
        symbol="CL",
        asset_class=AssetClass.FUTURE,
        multiplier=1000.0,
        tick_size=0.01,
        margin=6000.0,
        currency="USD",
    )


@pytest.fixture
def gc_contract() -> ContractSpec:
    """Gold futures specification.

    - Multiplier: $100 per troy ounce (100 oz per contract)
    - Tick size: $0.10 ($10)
    - Initial margin: ~$9,500 per contract (varies)
    """
    return ContractSpec(
        symbol="GC",
        asset_class=AssetClass.FUTURE,
        multiplier=100.0,
        tick_size=0.10,
        margin=9500.0,
        currency="USD",
    )


@pytest.fixture
def zn_contract() -> ContractSpec:
    """10-Year Treasury Note futures specification.

    - Multiplier: $1,000 per point (face value $100k)
    - Tick size: 1/64 of a point
    - Initial margin: ~$2,000 per contract (varies)
    """
    return ContractSpec(
        symbol="ZN",
        asset_class=AssetClass.FUTURE,
        multiplier=1000.0,
        tick_size=0.015625,  # 1/64
        margin=2000.0,
        currency="USD",
    )


@pytest.fixture
def nq_contract() -> ContractSpec:
    """E-mini NASDAQ-100 futures specification.

    - Multiplier: $20 per index point
    - Tick size: 0.25 points ($5)
    - Initial margin: ~$18,000 per contract (varies)
    """
    return ContractSpec(
        symbol="NQ",
        asset_class=AssetClass.FUTURE,
        multiplier=20.0,
        tick_size=0.25,
        margin=18000.0,
        currency="USD",
    )


@pytest.fixture
def btc_contract() -> ContractSpec:
    """Bitcoin futures (CME) specification.

    - Multiplier: 5 BTC per contract
    - Tick size: $5 per BTC ($25 per contract)
    - Initial margin: varies significantly
    """
    return ContractSpec(
        symbol="BTC",
        asset_class=AssetClass.FUTURE,
        multiplier=5.0,
        tick_size=5.0,
        margin=100000.0,  # Highly variable
        currency="USD",
    )


# === Commission Models ===


@pytest.fixture
def ib_commission() -> FuturesCommission:
    """Interactive Brokers style commission.

    Fixed per-contract cost, typical for retail.
    ES/NQ: $2.25 per contract (each way)
    """
    return FuturesCommission(per_block=2.25)


@pytest.fixture
def pysystemtrade_commission() -> FuturesCommission:
    """PySystemTrade default commission model.

    Uses MAX(per_trade, per_block, percentage) formula.
    """
    return FuturesCommission(
        per_trade=5.0,  # $5 minimum per trade
        per_block=2.50,  # $2.50 per contract
        percentage=0.0001,  # 1 basis point on notional
    )


@pytest.fixture
def percentage_commission() -> FuturesCommission:
    """Percentage-only commission for institutional trading.

    1 basis point on notional value.
    """
    return FuturesCommission(percentage=0.0001)


# === Slippage Models ===


@pytest.fixture
def es_slippage() -> FuturesSlippage:
    """ES futures slippage: 0.25 tick (1 tick = 0.25 points).

    Cost per contract: 0.25 * 50 = $12.50
    """
    return FuturesSlippage(slippage_points=0.25)


@pytest.fixture
def cl_slippage() -> FuturesSlippage:
    """CL futures slippage: 0.02 (2 ticks, tick = $0.01).

    Cost per contract: 0.02 * 1000 = $20
    """
    return FuturesSlippage(slippage_points=0.02)
