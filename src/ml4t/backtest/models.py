"""Pluggable commission and slippage models."""

from typing import Protocol, runtime_checkable

# === Protocols ===


@runtime_checkable
class CommissionModel(Protocol):
    """Protocol for commission calculation."""

    def calculate(self, asset: str, quantity: float, price: float) -> float: ...


@runtime_checkable
class SlippageModel(Protocol):
    """Protocol for slippage/market impact calculation."""

    def calculate(
        self, asset: str, quantity: float, price: float, volume: float | None
    ) -> float: ...


# === Commission Models ===


class NoCommission:
    """Zero commission."""

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        return 0.0


class PercentageCommission:
    """Commission as percentage of trade value."""

    def __init__(self, rate: float = 0.001):
        self.rate = rate

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        return abs(quantity * price * self.rate)


class PerShareCommission:
    """Fixed commission per share with optional minimum."""

    def __init__(self, per_share: float = 0.005, minimum: float = 1.0):
        self.per_share = per_share
        self.minimum = minimum

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        return max(abs(quantity) * self.per_share, self.minimum)


class TieredCommission:
    """Tiered commission based on trade value."""

    def __init__(self, tiers: list[tuple[float, float]]):
        # [(threshold, rate), ...] e.g. [(10000, 0.001), (50000, 0.0008), (inf, 0.0005)]
        self.tiers = sorted(tiers, key=lambda x: x[0])

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        value = abs(quantity * price)
        for threshold, rate in self.tiers:
            if value <= threshold:
                return value * rate
        return value * self.tiers[-1][1]


class CombinedCommission:
    """Combined percentage + fixed commission per trade."""

    def __init__(self, percentage: float = 0.0, fixed: float = 0.0):
        self.percentage = percentage
        self.fixed = fixed

    def calculate(self, asset: str, quantity: float, price: float) -> float:
        value = abs(quantity * price)
        return value * self.percentage + self.fixed


class FuturesCommission:
    """PySystemTrade-compatible futures commission model.

    Calculates commission as the MAX of three cost components:
    1. per_trade: Fixed cost per trade (regardless of size)
    2. per_block: Cost per contract (block)
    3. percentage: Percentage of notional value (qty * price * multiplier)

    This matches PySystemTrade's formula:
        total = max(per_trade, per_block * abs(qty), percentage * notional)

    Reference: sysobjects/instruments.py in PySystemTrade

    Args:
        per_trade: Fixed cost per trade in currency units (default 0.0)
        per_block: Cost per contract/block (default 0.0)
        percentage: Percentage of notional as decimal (e.g., 0.0001 = 1bp)

    Example:
        # Interactive Brokers ES futures: $2.25 per contract
        comm = FuturesCommission(per_block=2.25)

        # With 10 ES at $4000: max(0, 2.25*10, 0) = $22.50
        cost = comm.calculate("ES", 10, 4000.0, multiplier=50.0)

        # Percentage-based: 1bp on notional
        comm = FuturesCommission(percentage=0.0001)
        # 10 ES at $4000 = $2M notional, 1bp = $200
        cost = comm.calculate("ES", 10, 4000.0, multiplier=50.0)
    """

    def __init__(
        self,
        per_trade: float = 0.0,
        per_block: float = 0.0,
        percentage: float = 0.0,
    ):
        self.per_trade = per_trade
        self.per_block = per_block
        self.percentage = percentage

    def calculate(
        self, asset: str, quantity: float, price: float, multiplier: float = 1.0
    ) -> float:
        """Calculate commission for a futures trade.

        Args:
            asset: Asset symbol (unused, for protocol compatibility)
            quantity: Number of contracts (signed, will use absolute value)
            price: Contract price
            multiplier: Contract multiplier (point value)

        Returns:
            Commission in currency units (MAX of the three components)
        """
        notional = abs(quantity) * price * multiplier
        return max(
            self.per_trade,
            self.per_block * abs(quantity),
            self.percentage * notional,
        )


# === Slippage Models ===


class NoSlippage:
    """Zero slippage."""

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        return 0.0


class FixedSlippage:
    """Fixed slippage per share (per-unit price adjustment).

    The amount is added to the fill price for buys and subtracted for sells.
    For example, with amount=0.01:
    - Buy at $100.00 fills at $100.01
    - Sell at $100.00 fills at $99.99
    """

    def __init__(self, amount: float = 0.01):
        self.amount = amount

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        # Return per-unit price adjustment (same as PercentageSlippage contract)
        # Broker adds this to fill price: fill = base_price ± slippage
        return self.amount


class PercentageSlippage:
    """Slippage as percentage of price (per-unit price adjustment)."""

    def __init__(self, rate: float = 0.001):
        self.rate = rate

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        # Return per-unit price adjustment (not total dollars)
        # Broker adds this to fill price: fill = base_price ± slippage
        return price * self.rate


class VolumeShareSlippage:
    """Slippage based on order size vs volume (market impact)."""

    def __init__(self, impact_factor: float = 0.1):
        self.impact_factor = impact_factor

    def calculate(self, asset: str, quantity: float, price: float, volume: float | None) -> float:
        if volume is None or volume == 0:
            return 0.0
        volume_fraction = abs(quantity) / volume
        impact = volume_fraction * self.impact_factor
        # Return per-unit price adjustment (not total dollars)
        return price * impact


class FuturesSlippage:
    """PySystemTrade-compatible futures slippage model.

    Calculates slippage in currency units based on price slippage points
    and contract multiplier. This matches PySystemTrade's formula:
        slippage = abs(qty) * price_slippage_points * multiplier

    Reference: sysobjects/instruments.py in PySystemTrade

    Note: Unlike equity slippage models that return per-share adjustments,
    this returns TOTAL slippage in currency for the entire trade.

    Args:
        slippage_points: Slippage in price points per contract (default 0.0)

    Example:
        # ES futures with 0.25 tick slippage
        slip = FuturesSlippage(slippage_points=0.25)
        # 1 contract: 1 * 0.25 * 50 = $12.50
        cost = slip.calculate("ES", 1, 4000.0, multiplier=50.0)
    """

    def __init__(self, slippage_points: float = 0.0):
        self.slippage_points = slippage_points

    def calculate(
        self,
        asset: str,
        quantity: float,
        price: float,
        volume: float | None = None,
        multiplier: float = 1.0,
    ) -> float:
        """Calculate slippage for a futures trade.

        Args:
            asset: Asset symbol (unused, for protocol compatibility)
            quantity: Number of contracts (signed, will use absolute value)
            price: Contract price (unused in this model)
            volume: Bar volume (unused, for protocol compatibility)
            multiplier: Contract multiplier (point value)

        Returns:
            Total slippage cost in currency units
        """
        return abs(quantity) * self.slippage_points * multiplier
