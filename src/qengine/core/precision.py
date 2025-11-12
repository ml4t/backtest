"""Precision management for consistent rounding across asset classes.

This module ensures that all numerical calculations (positions, prices, cash)
are rounded consistently to match real-world trading constraints and eliminate
float arithmetic mismatches.

Key principles:
- Round immediately after calculation, before storing
- Use the same PrecisionManager instance everywhere for consistency
- Rounding rules are driven by asset class (equities: whole shares, crypto: 8 decimals)
- All monetary values round to cents (2 decimals) for USD
"""

from dataclasses import dataclass
from enum import Enum

from qengine.core.types import Price, Quantity, Cash


class PositionPrecision(Enum):
    """Position rounding behavior per asset class."""

    INTEGER = 0  # Whole units only (equities, futures, options)
    CRYPTO = 8  # Satoshi precision (BTC standard)
    FRACTIONAL = 6  # Some brokers offer fractional shares
    FX_STANDARD = 5  # Standard FX lot sizes (0.00001)


@dataclass
class AssetClassDefaults:
    """Default rounding precision per asset class.

    These values can be overridden per asset via AssetSpec.
    """

    position_decimals: int  # 0 for equities, 8 for crypto
    price_decimals: int  # 2 for USD-quoted, 8 for crypto
    cash_decimals: int = 2  # Always cents for USD currency


# Default configurations per asset class
PRECISION_DEFAULTS = {
    "EQUITY": AssetClassDefaults(
        position_decimals=0,  # Whole shares only
        price_decimals=2,  # Penny prices ($XX.XX)
        cash_decimals=2,  # Cent precision
    ),
    "CRYPTO": AssetClassDefaults(
        position_decimals=8,  # Satoshi precision (BTC)
        price_decimals=8,  # 8 decimal places
        cash_decimals=2,  # Commission still in USD cents
    ),
    "FUTURE": AssetClassDefaults(
        position_decimals=0,  # Whole contracts
        price_decimals=2,  # Tick size (varies by contract)
        cash_decimals=2,
    ),
    "OPTION": AssetClassDefaults(
        position_decimals=0,  # Whole contracts
        price_decimals=2,  # Dollar prices
        cash_decimals=2,
    ),
    "FX": AssetClassDefaults(
        position_decimals=5,  # Standard lot sizes
        price_decimals=5,  # Pip precision (0.00001)
        cash_decimals=2,
    ),
    "BOND": AssetClassDefaults(
        position_decimals=0,  # Whole bonds
        price_decimals=3,  # 32nds converted to decimal
        cash_decimals=2,
    ),
    "COMMODITY": AssetClassDefaults(
        position_decimals=0,  # Whole contracts
        price_decimals=2,  # Varies by commodity
        cash_decimals=2,
    ),
}


class PrecisionManager:
    """Manages consistent rounding for an asset.

    This class enforces rounding rules to match real-world trading constraints
    and eliminate float arithmetic mismatches between strategy and broker.

    Example:
        >>> from qengine.core.assets import AssetSpec, AssetClass
        >>> spec = AssetSpec(asset_id="AAPL", asset_class=AssetClass.EQUITY)
        >>> pm = PrecisionManager.from_asset_spec(spec)
        >>> pm.round_quantity(100.999)  # Equities: whole shares
        100.0
        >>> pm.round_price(123.456)  # USD: penny precision
        123.46
        >>> pm.round_cash(10.999)  # Commission: cents
        11.00

        >>> spec = AssetSpec(asset_id="BTC", asset_class=AssetClass.CRYPTO)
        >>> pm = PrecisionManager.from_asset_spec(spec)
        >>> pm.round_quantity(3.123456789)  # Crypto: 8 decimals
        3.12345679
    """

    def __init__(
        self,
        position_decimals: int,
        price_decimals: int,
        cash_decimals: int = 2,
    ):
        """Initialize with explicit decimal places.

        Args:
            position_decimals: Decimal places for position quantities
                (0 for equities, 8 for crypto)
            price_decimals: Decimal places for prices
                (2 for USD, 8 for crypto)
            cash_decimals: Decimal places for cash/commission
                (2 for USD cents)
        """
        self.position_decimals = position_decimals
        self.price_decimals = price_decimals
        self.cash_decimals = cash_decimals

    @classmethod
    def from_asset_spec(cls, asset_spec: "AssetSpec") -> "PrecisionManager":
        """Create PrecisionManager from AssetSpec.

        Uses asset class defaults, but allows per-asset overrides via
        AssetSpec attributes.

        Args:
            asset_spec: Asset specification with optional precision overrides

        Returns:
            PrecisionManager configured for this asset
        """
        # Get defaults for this asset class
        defaults = PRECISION_DEFAULTS.get(
            asset_spec.asset_class.value.upper(),
            PRECISION_DEFAULTS["EQUITY"],  # Fallback
        )

        # Check for per-asset overrides - use default if None
        position_decimals = (
            asset_spec.position_decimals
            if asset_spec.position_decimals is not None
            else defaults.position_decimals
        )
        price_decimals = (
            asset_spec.price_decimals
            if asset_spec.price_decimals is not None
            else defaults.price_decimals
        )
        cash_decimals = (
            asset_spec.cash_decimals
            if asset_spec.cash_decimals is not None
            else defaults.cash_decimals
        )

        return cls(
            position_decimals=position_decimals,
            price_decimals=price_decimals,
            cash_decimals=cash_decimals,
        )

    @classmethod
    def from_asset_class(cls, asset_class: str) -> "PrecisionManager":
        """Create PrecisionManager from asset class string.

        Args:
            asset_class: Asset class name (e.g., "EQUITY", "CRYPTO")

        Returns:
            PrecisionManager with default settings for this asset class
        """
        defaults = PRECISION_DEFAULTS.get(
            asset_class.upper(),
            PRECISION_DEFAULTS["EQUITY"],
        )
        return cls(
            position_decimals=defaults.position_decimals,
            price_decimals=defaults.price_decimals,
            cash_decimals=defaults.cash_decimals,
        )

    def round_quantity(self, qty: float) -> float:
        """Round position quantity to valid precision.

        For equities: Truncates to whole shares (no fractional shares)
        For crypto: Rounds to 8 decimal places (satoshi precision)

        Args:
            qty: Raw quantity value

        Returns:
            Rounded quantity following asset class rules

        Example:
            >>> pm = PrecisionManager(position_decimals=0, price_decimals=2)
            >>> pm.round_quantity(100.7)  # Equity
            100.0
            >>> pm = PrecisionManager(position_decimals=8, price_decimals=8)
            >>> pm.round_quantity(3.123456789)  # Crypto
            3.12345679
        """
        if self.position_decimals == 0:
            # Truncate to whole units (equities, futures, options)
            return float(int(qty))
        return round(qty, self.position_decimals)

    def round_price(self, price: float) -> float:
        """Round price to valid tick size.

        Args:
            price: Raw price value

        Returns:
            Rounded price to asset's tick size

        Example:
            >>> pm = PrecisionManager(position_decimals=0, price_decimals=2)
            >>> pm.round_price(123.456)
            123.46
        """
        return round(price, self.price_decimals)

    def round_cash(self, amount: float) -> float:
        """Round monetary amount (commission, P&L) to currency precision.

        For USD: Rounds to cents (2 decimal places)

        Args:
            amount: Raw cash/commission amount

        Returns:
            Rounded amount to currency precision

        Example:
            >>> pm = PrecisionManager(position_decimals=0, price_decimals=2)
            >>> pm.round_cash(10.999)
            11.00
            >>> pm.round_cash(123.456)
            123.46
        """
        return round(amount, self.cash_decimals)

    def is_position_zero(self, qty: float, tolerance: float | None = None) -> bool:
        """Check if position is effectively zero after rounding.

        Args:
            qty: Position quantity to check
            tolerance: Optional custom tolerance (default: 10^(-position_decimals))

        Returns:
            True if position rounds to zero

        Example:
            >>> pm = PrecisionManager(position_decimals=0, price_decimals=2)
            >>> pm.is_position_zero(0.4)  # Equity: rounds to 0
            True
            >>> pm.is_position_zero(0.6)  # Equity: rounds to 1
            False
            >>> pm = PrecisionManager(position_decimals=8, price_decimals=8)
            >>> pm.is_position_zero(0.000000001)  # Crypto: below precision
            True
        """
        if tolerance is None:
            tolerance = 10 ** (-self.position_decimals) if self.position_decimals > 0 else 0.5

        return abs(qty) < tolerance

    def __repr__(self) -> str:
        return (
            f"PrecisionManager(position_decimals={self.position_decimals}, "
            f"price_decimals={self.price_decimals}, "
            f"cash_decimals={self.cash_decimals})"
        )
