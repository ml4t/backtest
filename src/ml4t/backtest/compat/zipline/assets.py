"""Immutable migration identifiers, separate from dated symbol lookup."""

from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True, slots=True, eq=False)
class Asset:
    """Retain a native account key and an explicitly supplied legacy identifier.

    Display metadata never participates in identity or hash equality. An unknown
    ticker remains unknown; it is not inferred from the native key.
    """

    key: str
    symbol: str | None = None
    legacy_sid: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Migration assets require an explicit nonempty native account key")
        if self.symbol is not None and not isinstance(self.symbol, str):
            raise TypeError("Asset symbol must be a supplied string or None")
        if self.legacy_sid is not None and (
            isinstance(self.legacy_sid, bool) or not isinstance(self.legacy_sid, int)
        ):
            raise TypeError("legacy_sid must be an explicitly supplied integer or None")

    @property
    def sid(self) -> int | str:
        return self.key if self.legacy_sid is None else self.legacy_sid

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Asset) and self.key == other.key

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Asset):
            return NotImplemented
        if self == other:
            return False
        if self.legacy_sid is not None and other.legacy_sid is not None:
            return (self.legacy_sid, self.key) < (other.legacy_sid, other.key)
        if self.legacy_sid is None and other.legacy_sid is None:
            return self.key < other.key
        return self.legacy_sid is not None

    def __int__(self) -> int:
        if self.legacy_sid is None:
            raise TypeError("No legacy integer sid was declared for this native asset")
        return self.legacy_sid


@dataclass(frozen=True, slots=True, eq=False)
class Future(Asset):
    """Explicit dated-contract facts; identity remains the native stable key."""

    root_symbol: str = field(kw_only=True)
    multiplier: float = field(kw_only=True)
    tick_size: float = field(kw_only=True)

    def __post_init__(self) -> None:
        Asset.__post_init__(self)
        if not isinstance(self.root_symbol, str) or not self.root_symbol:
            raise ValueError("A future requires its declared root_symbol")
        for value in (self.multiplier, self.tick_size):
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError("Future multiplier and tick_size must be finite and positive")
