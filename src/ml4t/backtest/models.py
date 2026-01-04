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
