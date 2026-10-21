"""Restriction queries compiled to the native OrderConstraints predicate boundary."""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from enum import IntEnum
from numbers import Integral
from typing import Any, NamedTuple

import pandas as pd

from ..assets import Asset


class RESTRICTION_STATES(IntEnum):
    ALLOWED = 0
    FROZEN = 1


class Restriction(NamedTuple):
    asset: Asset
    effective_date: pd.Timestamp
    state: RESTRICTION_STATES


class Restrictions(ABC):
    @abstractmethod
    def is_restricted(self, assets: Asset | Iterable[Asset], dt: Any) -> bool | pd.Series:
        """Return a Boolean or a Boolean Series for the declared decision instant."""
        raise NotImplementedError

    def __or__(self, other: "Restrictions") -> "Restrictions":
        if not isinstance(other, Restrictions):
            return NotImplemented
        return _UnionRestrictions((self, other))

    @property
    def metadata(self) -> dict[str, Any]:
        return {"model": type(self).__name__, "policy": "caller_owned_predicate"}


def _query(assets, dt, predicate):
    point = pd.Timestamp(dt)
    if pd.isna(point):
        raise ValueError("Restriction queries require a known decision instant")
    if isinstance(assets, Asset):
        return bool(predicate(assets, point))
    values = list(assets)
    if any(not isinstance(asset, Asset) for asset in values):
        raise TypeError("Restriction queries require declared Assets")
    return pd.Series([bool(predicate(asset, point)) for asset in values], index=values, dtype=bool)


class NoRestrictions(Restrictions):
    def is_restricted(self, assets: Asset | Iterable[Asset], dt: Any) -> bool | pd.Series:
        return _query(assets, dt, lambda _asset, _point: False)


class StaticRestrictions(Restrictions):
    def __init__(self, restricted_list: Iterable[Asset]) -> None:
        self._assets = frozenset(restricted_list)
        if any(not isinstance(asset, Asset) for asset in self._assets):
            raise TypeError(
                "Static restrictions require declared Assets, not inferred tickers or SIDs"
            )

    def is_restricted(self, assets: Asset | Iterable[Asset], dt: Any) -> bool | pd.Series:
        return _query(assets, dt, lambda asset, _point: asset in self._assets)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model": "StaticRestrictions",
            "assets": sorted(asset.key for asset in self._assets),
        }


class HistoricalRestrictions(Restrictions):
    def __init__(self, restrictions: Iterable[Restriction]) -> None:
        grouped = defaultdict(list)
        for item in restrictions:
            if not isinstance(item, Restriction) or not isinstance(item.asset, Asset):
                raise TypeError(
                    "Historical restrictions require Restriction records and declared Assets"
                )
            point = pd.Timestamp(item.effective_date)
            if pd.isna(point):
                raise ValueError("Restriction effective dates must be known")
            grouped[item.asset].append(
                Restriction(item.asset, point, RESTRICTION_STATES(item.state))
            )
        self._changes = {}
        for asset, items in grouped.items():
            if len({item.effective_date.tz is None for item in items}) > 1:
                raise ValueError("Restriction times for an asset must share timezone-awareness")
            self._changes[asset] = tuple(sorted(items, key=lambda item: item.effective_date))

    def _restricted(self, asset, point):
        state = RESTRICTION_STATES.ALLOWED
        for item in self._changes.get(asset, ()):
            effective = item.effective_date
            if effective.tz is None:
                effective = effective.tz_localize(point.tz)
            if effective > point:
                break
            state = item.state
        return state == RESTRICTION_STATES.FROZEN

    def is_restricted(self, assets: Asset | Iterable[Asset], dt: Any) -> bool | pd.Series:
        return _query(assets, dt, self._restricted)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "model": "HistoricalRestrictions",
            "effective_date_policy": "naive_dates_use_query_timezone",
            "changes": [
                {
                    "asset": item.asset.key,
                    "effective_date": item.effective_date.isoformat(),
                    "state": item.state.name,
                }
                for rows in self._changes.values()
                for item in rows
            ],
        }


class SecurityListRestrictions(Restrictions):
    def __init__(self, security_list_by_dt: Any) -> None:
        if not callable(getattr(security_list_by_dt, "current_securities", None)):
            raise TypeError("Supply a caller-owned offline current_securities(dt) provider")
        self.current_securities = security_list_by_dt.current_securities

    def is_restricted(self, assets: Asset | Iterable[Asset], dt: Any) -> bool | pd.Series:
        point = pd.Timestamp(dt)
        if pd.isna(point):
            raise ValueError("Restriction queries require a known decision instant")
        members = frozenset(self.current_securities(point))
        if any(
            not isinstance(member, Asset | Integral) or isinstance(member, bool)
            for member in members
        ):
            raise TypeError("Security lists require declared Assets or explicit integer SIDs")
        return _query(
            assets,
            point,
            lambda asset, _point: asset in members
            or (asset.legacy_sid is not None and asset.legacy_sid in members),
        )


class _UnionRestrictions(Restrictions):
    def __init__(self, restrictions):
        self._rules = tuple(
            child
            for rule in restrictions
            for child in (rule._rules if isinstance(rule, _UnionRestrictions) else (rule,))
            if not isinstance(child, NoRestrictions)
        )

    def is_restricted(self, assets: Asset | Iterable[Asset], dt: Any) -> bool | pd.Series:
        if isinstance(assets, Asset):
            return _query(
                assets,
                dt,
                lambda asset, point: any(rule.is_restricted(asset, point) for rule in self._rules),
            )
        chosen = list(assets)
        result = _query(chosen, dt, lambda _asset, _point: False)
        for rule in self._rules:
            values = rule.is_restricted(chosen, dt)
            if (
                not isinstance(values, pd.Series)
                or not values.index.equals(result.index)
                or values.dtype != bool
            ):
                raise ValueError("Restriction query must preserve requested Boolean Series keys")
            result = result | values
        return result

    @property
    def metadata(self) -> dict[str, Any]:
        return {"model": "union", "rules": [rule.metadata for rule in self._rules]}
