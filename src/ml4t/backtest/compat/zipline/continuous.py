"""Deliver Data's causal dated selections through familiar continuous-future reads."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

import pandas as pd
import polars as pl
from ml4t.specs import FeedSpec

from ..._temporal import temporal_rows
from .assets import Asset, Future


@dataclass(frozen=True, slots=True, eq=False)
class ContinuousFuture(Asset):
    """A research key, never a directly tradable dated contract."""

    root_symbol: str = ""
    offset: int = 0
    roll_style: str = "volume"
    adjustment: str | None = "mul"

    def __post_init__(self) -> None:
        Asset.__post_init__(self)
        if not self.root_symbol or self.legacy_sid is not None:
            raise ValueError("Continuous futures require a root and no physical legacy SID")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("Continuous offset must be a nonnegative integer")
        if self.roll_style not in {"calendar", "volume"} or self.adjustment not in {
            None,
            "add",
            "mul",
        }:
            raise ValueError("Unknown continuous roll or adjustment declaration")

    @property
    def query(self) -> tuple[str, int, str, str | None]:
        return self.root_symbol, self.offset, self.roll_style, self.adjustment


@dataclass(frozen=True)
class PreparedFutures:
    """A frozen-data provider returning Data FuturesInputs at the requested cutoff.

    Data chooses the front and owns adjustment policy. The legacy offset selects
    the next dated maturity after that front, rather than a volume rank. Register
    prepared continuous histories separately through the existing history_sources.
    """

    handle: ContinuousFuture
    provider: Callable[..., Any]
    contracts: Mapping[str | int, Future]
    exchange: str
    source_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.handle, ContinuousFuture) or not callable(self.provider):
            raise TypeError("Prepared futures require a continuous handle and causal provider")
        if not self.exchange or not self.source_id or not self.contracts:
            raise ValueError("Declare exchange, source identity and dated contract mapping")
        contracts = dict(self.contracts)
        if any(
            not isinstance(asset, Future) or asset.root_symbol != self.handle.root_symbol
            for asset in contracts.values()
        ):
            raise ValueError(
                "Continuous contracts require explicitly mapped Futures of the declared root"
            )
        if len({asset.key for asset in contracts.values()}) != len(contracts):
            raise ValueError("Continuous source IDs cannot alias a dated native contract")
        object.__setattr__(self, "contracts", MappingProxyType(contracts))

    def at(
        self, as_of: datetime, *, before_observation: bool
    ) -> tuple[Future | None, list[Future]]:
        prepared = self.provider(as_of=as_of)
        metadata = prepared.metadata
        cutoff = pd.Timestamp(metadata.get("as_of"))
        if (
            pd.isna(cutoff)
            or cutoff != pd.Timestamp(as_of)
            or metadata.get("perspective") != "causal_current_decision"
        ):
            raise ValueError("Continuous provider must prepare the actual causal as_of cutoff")
        if metadata.get("selection_policy", {}).get("parameters", {}).get("rank", 0) != 0:
            raise ValueError(
                "Continuous offset follows dated maturities; prepare the front with rank zero"
            )
        method = metadata.get("selection_policy", {}).get("kind", "").rsplit(".", 1)[-1]
        methods = (
            {"HighestVolumeRoll", "VolumeBasedRoll"}
            if self.handle.roll_style == "volume"
            else {"CalendarRoll", "TimeBasedRoll", "FirstNoticeDateRoll"}
        )
        if method not in methods:
            raise ValueError("Prepared selection method conflicts with continuous roll declaration")
        chosen = prepared.selections.filter(
            (pl.col("root") == self.handle.root_symbol) & (pl.col("exchange") == self.exchange)
        )
        if chosen.height > 1:
            raise ValueError("Continuous front selection is ambiguous")
        if chosen.is_empty():
            return None, []
        selection = next(temporal_rows(chosen, named=True))
        for name in ("decision_at", "definition_available_at", "metric_available_at"):
            value = selection.get(name)
            if value is None and name == "metric_available_at":
                continue
            if not isinstance(value, datetime) or value != value or value > as_of:
                raise ValueError(
                    "Continuous selection precedes a known physical publication/decision"
                )
        observed = selection.get("metric_observed_at")
        if isinstance(observed, datetime) and (
            observed > as_of or before_observation and observed == as_of
        ):
            raise ValueError("Continuous selection uses an undelivered observation")
        chain = prepared.chain.filter(
            (pl.col("root") == self.handle.root_symbol) & (pl.col("exchange") == self.exchange)
        )
        definitions = prepared.definitions.filter(
            (pl.col("root") == self.handle.root_symbol) & (pl.col("exchange") == self.exchange)
        )
        rows = []
        for row in temporal_rows(chain, named=True):
            identity = row["asset"]
            if identity not in self.contracts:
                raise ValueError("Continuous chain contains an unmapped stable instrument ID")
            fact = definitions.filter(pl.col("asset") == identity)
            if fact.height != 1:
                raise ValueError("Continuous chain requires one known definition per instrument")
            definition = next(temporal_rows(fact, named=True))
            if (
                any(
                    not isinstance(value, datetime) or value != value or value > as_of
                    for value in (row["available_at"], definition["available_at"])
                )
                or definition["expiration"] is None
            ):
                raise ValueError("Continuous chain requires known publication and expiration facts")
            asset = self.contracts[identity]
            spec = FeedSpec.from_any(metadata.get("normalized_feed_spec"))
            if not isinstance(spec.entity_col, str):
                raise ValueError("Continuous normalized inputs require one stable entity column")
            units = (
                prepared.bars.filter(pl.col(spec.entity_col) == identity)
                .select("normalized_multiplier", "normalized_tick_size")
                .unique()
            )
            if units.height != 1 or units.row(0) != (asset.multiplier, asset.tick_size):
                raise ValueError("Normalized source units conflict with the declared dated Future")
            rows.append((definition["expiration"], str(identity), identity, asset))
        rows.sort(key=lambda row: row[:2])
        identities = [row[2] for row in rows]
        if selection["selected_instrument_id"] not in identities:
            raise ValueError("Selected continuous front is absent from the prepared chain")
        index = identities.index(selection["selected_instrument_id"]) + self.handle.offset
        assets = [row[3] for row in rows[index:]]
        return (assets[0] if assets else None), assets
