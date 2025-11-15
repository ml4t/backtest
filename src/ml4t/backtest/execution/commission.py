"""Commission models for realistic cost simulation."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml4t.backtest.core.types import Price, Quantity
    from ml4t.backtest.execution.order import Order


class CommissionModel(ABC):
    """Abstract base class for commission models."""

    @abstractmethod
    def calculate(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        fill_price: "Price",
    ) -> float:
        """Calculate commission for a filled order.

        Args:
            order: The order being filled
            fill_quantity: Quantity of the fill
            fill_price: Price at which the order was filled

        Returns:
            Commission amount in currency terms (rounded to cents for USD)
        """

    def _round_commission(self, commission: float) -> float:
        """Round commission to cents (2 decimal places for USD).

        This ensures commission values match real broker behavior where
        commissions are always charged in whole cents.

        Args:
            commission: Raw commission amount

        Returns:
            Commission rounded to nearest cent
        """
        return round(commission, 2)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


class NoCommission(CommissionModel):
    """No commission model for testing."""

    def calculate(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        fill_price: "Price",
    ) -> float:
        """Calculate zero commission."""
        return 0.0


class FlatCommission(CommissionModel):
    """Flat commission per trade."""

    def __init__(self, commission: float = 1.0):
        """Initialize flat commission model.

        Args:
            commission: Flat fee per trade (default $1)
        """
        if commission < 0:
            raise ValueError("Commission cannot be negative")
        self.commission = commission

    def calculate(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        fill_price: "Price",
    ) -> float:
        """Calculate flat commission."""
        return self.commission

    def __repr__(self) -> str:
        """String representation."""
        return f"FlatCommission(commission={self.commission})"


class PercentageCommission(CommissionModel):
    """Percentage-based commission on trade value."""

    def __init__(self, rate: float = 0.001):
        """Initialize percentage commission model.

        Args:
            rate: Commission rate as decimal (0.001 = 0.1% = 10bps)
        """
        if rate < 0:
            raise ValueError("Commission rate cannot be negative")
        if rate > 0.1:  # 10% cap as sanity check
            raise ValueError("Commission rate too high (>10%)")
        self.rate = rate

    def calculate(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        fill_price: "Price",
    ) -> float:
        """Calculate percentage-based commission."""
        notional = fill_quantity * fill_price
        commission = notional * self.rate
        return self._round_commission(commission)

    def __repr__(self) -> str:
        """String representation."""
        return f"PercentageCommission(rate={self.rate})"


class PerShareCommission(CommissionModel):
    """Per-share commission model."""

    def __init__(self, commission_per_share: float = 0.005):
        """Initialize per-share commission model.

        Args:
            commission_per_share: Commission per share (default $0.005)
        """
        if commission_per_share < 0:
            raise ValueError("Per-share commission cannot be negative")
        self.commission_per_share = commission_per_share

    def calculate(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        fill_price: "Price",
    ) -> float:
        """Calculate per-share commission."""
        return fill_quantity * self.commission_per_share

    def __repr__(self) -> str:
        """String representation."""
        return f"PerShareCommission(commission_per_share={self.commission_per_share})"


class TieredCommission(CommissionModel):
    """Tiered commission based on trade size."""

    def __init__(
        self,
        tiers: list[tuple[float, float]] | None = None,
        minimum: float = 1.0,
    ):
        """Initialize tiered commission model.

        Args:
            tiers: List of (threshold, rate) tuples in ascending order
                   Default: [(10000, 0.0010), (50000, 0.0008), (100000, 0.0005)]
            minimum: Minimum commission per trade
        """
        if tiers is None:
            # Default tiers: better rates for larger trades
            tiers = [
                (10_000, 0.0010),  # 10 bps for trades < $10k
                (50_000, 0.0008),  # 8 bps for trades $10k-$50k
                (100_000, 0.0005),  # 5 bps for trades $50k-$100k
                (float("inf"), 0.0003),  # 3 bps for trades > $100k
            ]

        # Validate tiers
        prev_threshold = 0
        for threshold, rate in tiers:
            if threshold <= prev_threshold:
                raise ValueError("Tiers must be in ascending order")
            if rate < 0:
                raise ValueError("Commission rates cannot be negative")
            prev_threshold = threshold

        if minimum < 0:
            raise ValueError("Minimum commission cannot be negative")

        self.tiers = tiers
        self.minimum = minimum

    def calculate(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        fill_price: "Price",
    ) -> float:
        """Calculate tiered commission based on notional value."""
        notional = fill_quantity * fill_price

        # Find applicable tier
        rate = self.tiers[-1][1]  # Default to highest tier
        for threshold, tier_rate in self.tiers:
            if notional < threshold:
                rate = tier_rate
                break

        commission = notional * rate
        return max(commission, self.minimum)

    def __repr__(self) -> str:
        """String representation."""
        return f"TieredCommission(tiers={self.tiers}, minimum={self.minimum})"


class MakerTakerCommission(CommissionModel):
    """Maker-taker commission model (exchanges)."""

    def __init__(
        self,
        maker_rate: float = -0.0002,  # Maker rebate
        taker_rate: float = 0.0003,  # Taker fee
    ):
        """Initialize maker-taker commission model.

        Args:
            maker_rate: Maker fee rate (negative for rebate)
            taker_rate: Taker fee rate
        """
        if taker_rate < 0:
            raise ValueError("Taker rate should be positive")
        if maker_rate > taker_rate:
            raise ValueError("Maker rate should not exceed taker rate")

        self.maker_rate = maker_rate
        self.taker_rate = taker_rate

    def calculate(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        fill_price: "Price",
    ) -> float:
        """Calculate maker-taker commission based on order type."""
        from ml4t.backtest.execution.order import OrderType

        notional = fill_quantity * fill_price

        # Market orders always take liquidity
        # Limit orders that execute immediately also take liquidity
        # For simplicity, we assume limit orders make liquidity
        rate = self.taker_rate if order.order_type == OrderType.MARKET else self.maker_rate

        commission = notional * rate
        # Negative commission means rebate, but ensure we don't pay too much rebate
        return max(commission, -notional * 0.001)  # Cap rebate at 10bps

    def __repr__(self) -> str:
        """String representation."""
        return f"MakerTakerCommission(maker_rate={self.maker_rate}, taker_rate={self.taker_rate})"


class AssetClassCommission(CommissionModel):
    """Asset class specific commission model."""

    def __init__(
        self,
        equity_rate: float = 0.001,  # 10 bps
        futures_per_contract: float = 2.50,  # $2.50 per contract
        options_per_contract: float = 0.65,  # $0.65 per contract
        forex_rate: float = 0.0002,  # 2 bps
        crypto_rate: float = 0.002,  # 20 bps
        default_rate: float = 0.001,  # 10 bps fallback
    ):
        """Initialize asset class commission model.

        Args:
            equity_rate: Commission rate for equities
            futures_per_contract: Commission per futures contract
            options_per_contract: Commission per options contract
            forex_rate: Commission rate for forex
            crypto_rate: Commission rate for crypto
            default_rate: Default commission rate
        """
        self.equity_rate = equity_rate
        self.futures_per_contract = futures_per_contract
        self.options_per_contract = options_per_contract
        self.forex_rate = forex_rate
        self.crypto_rate = crypto_rate
        self.default_rate = default_rate

    def calculate(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        fill_price: "Price",
    ) -> float:
        """Calculate commission based on asset class."""
        # Determine asset class from symbol or metadata
        asset_class = order.metadata.get("asset_class", "equity")

        if asset_class == "futures":
            # Futures charge per contract
            return fill_quantity * self.futures_per_contract
        if asset_class == "options":
            # Options charge per contract (1 contract = 100 shares usually)
            contracts = fill_quantity / 100
            return contracts * self.options_per_contract
        if asset_class == "forex":
            notional = fill_quantity * fill_price
            return notional * self.forex_rate
        if asset_class == "crypto":
            notional = fill_quantity * fill_price
            return notional * self.crypto_rate
        if asset_class == "equity":
            notional = fill_quantity * fill_price
            return notional * self.equity_rate
        # Default rate for unknown asset classes
        notional = fill_quantity * fill_price
        return notional * self.default_rate

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AssetClassCommission("
            f"equity_rate={self.equity_rate}, "
            f"futures_per_contract={self.futures_per_contract}, "
            f"options_per_contract={self.options_per_contract}, "
            f"forex_rate={self.forex_rate}, "
            f"crypto_rate={self.crypto_rate})"
        )


class InteractiveBrokersCommission(CommissionModel):
    """Interactive Brokers tiered commission structure."""

    def __init__(self, tier: str = "fixed"):
        """Initialize IB commission model.

        Args:
            tier: Commission tier ('fixed' or 'tiered')
        """
        if tier not in ["fixed", "tiered"]:
            raise ValueError("Tier must be 'fixed' or 'tiered'")
        self.tier = tier

    def calculate(
        self,
        order: "Order",
        fill_quantity: "Quantity",
        fill_price: "Price",
    ) -> float:
        """Calculate IB commission."""
        if self.tier == "fixed":
            # Fixed pricing: $0.005 per share, $1 minimum, $1% max
            per_share = fill_quantity * 0.005
            min_commission = 1.0
            max_commission = fill_quantity * fill_price * 0.01
            return min(max(per_share, min_commission), max_commission)
        # Tiered pricing (simplified)
        fill_quantity * fill_price
        if fill_quantity <= 300:
            rate = 0.0035  # $0.0035 per share for first 300
        elif fill_quantity <= 3000:
            rate = 0.0025  # $0.0025 per share for next 2700
        else:
            rate = 0.0015  # $0.0015 per share above 3000

        commission = fill_quantity * rate
        return max(commission, 0.35)  # $0.35 minimum

    def __repr__(self) -> str:
        """String representation."""
        return f"InteractiveBrokersCommission(tier='{self.tier}')"
