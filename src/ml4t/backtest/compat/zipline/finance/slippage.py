"""Migration impact parameters compiled to native models and execution limits."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VolumeShareSlippage:
    volume_limit: float = 0.025
    price_impact: float = 0.1


@dataclass(frozen=True)
class FixedBasisPointsSlippage:
    basis_points: float = 5.0
    volume_limit: float = 0.1


@dataclass(frozen=True)
class FixedSlippage:
    spread: float = 0.0


@dataclass(frozen=True)
class NoSlippage:
    pass
