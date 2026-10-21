"""Migration cost parameters compiled to native commission methods."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PerShare:
    cost: float = 0.001
    min_trade_cost: float = 0.0


@dataclass(frozen=True)
class PerTrade:
    cost: float = 0.0


@dataclass(frozen=True)
class PerDollar:
    cost: float = 0.0015
