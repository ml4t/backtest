"""Select already-exported symbol ownership facts without an identity database."""

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

import pandas as pd
import polars as pl

from ..._temporal import temporal_rows
from .assets import Asset, Future


def _symbol_parts(symbol: str) -> tuple[str, str]:
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("symbol must be a nonempty string")
    parts = re.split(r"[./\-_]", symbol.upper(), maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _lookup_instant(value: date | datetime | str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("A known snapshot lookup date is required")
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tz is not None else stamp


@dataclass(frozen=True)
class _Period:
    asset: Asset
    start: pd.Timestamp
    end: pd.Timestamp
    country: str | None


class PreparedAssetLookup:
    """Read immutable IDs and ownership intervals from a verified Data InputSnapshot.

    These intervals encode the legacy bundle's lookup policy, which can extend
    beyond listing lifetimes. They do not establish tradability or historical
    metadata publication. Native instrument eligibility remains independent.
    """

    def __init__(self, snapshot: Any) -> None:
        tables = getattr(snapshot, "tables", None)
        source_id = getattr(snapshot, "snapshot_id", None)
        if not isinstance(tables, Mapping) or not isinstance(source_id, str) or not source_id:
            raise TypeError("Use a verified Data InputSnapshot with tables and snapshot_id")
        equity_required = {
            "zipline_equities": {"source_sid", "asset", "symbol", "exchange"},
            "zipline_symbol_lookup": {
                "sid",
                "company_symbol",
                "share_class_symbol",
                "start_date",
                "end_date",
            },
        }
        required = equity_required if "zipline_equities" in tables else {}
        if "zipline_futures" in tables:
            required = {
                **required,
                "zipline_futures": {
                    "source_sid",
                    "asset",
                    "symbol",
                    "root_symbol",
                    "multiplier",
                    "tick_size",
                    "exchange",
                },
            }
        if not required:
            raise ValueError("Asset lookup requires exported equities or futures")
        for name, fields in required.items():
            if not isinstance(tables.get(name), pl.DataFrame) or fields - set(tables[name].columns):
                raise ValueError(f"Symbol lookup requires exported {name} fields {sorted(fields)}")
        countries = {}
        exchanges = tables.get("zipline_exchanges")
        if isinstance(exchanges, pl.DataFrame) and {"exchange", "country_code"} <= set(
            exchanges.columns
        ):
            countries = dict(exchanges.select("exchange", "country_code").iter_rows())
        by_sid = {}
        country_by_sid = {}
        for row in temporal_rows(tables.get("zipline_equities", pl.DataFrame()), named=True):
            sid = row["source_sid"]
            if isinstance(sid, bool) or not isinstance(sid, int) or sid in by_sid:
                raise ValueError("Exported equity source SIDs must be unique integers")
            by_sid[sid] = Asset(row["asset"], row["symbol"], sid)
            country_by_sid[sid] = countries.get(row["exchange"])
        grouped = defaultdict(list)
        facts = []
        for row in temporal_rows(tables.get("zipline_futures", pl.DataFrame()), named=True):
            sid = row["source_sid"]
            if isinstance(sid, bool) or not isinstance(sid, int) or sid in by_sid:
                raise ValueError("Exported asset source SIDs must be unique integers")
            asset = Future(
                row["asset"],
                row["symbol"],
                sid,
                root_symbol=row["root_symbol"],
                multiplier=row["multiplier"],
                tick_size=row["tick_size"],
            )
            by_sid[sid] = asset
            facts.append(
                {
                    "asset": asset.key,
                    "sid": sid,
                    "symbol": asset.symbol,
                    "root_symbol": asset.root_symbol,
                    "multiplier": asset.multiplier,
                    "tick_size": asset.tick_size,
                }
            )
        if not by_sid or len({asset.key for asset in by_sid.values()}) != len(by_sid):
            raise ValueError("Exported assets require known unique stable asset keys")
        for row in temporal_rows(tables.get("zipline_symbol_lookup", pl.DataFrame()), named=True):
            sid = row["sid"]
            if (
                isinstance(sid, bool)
                or not isinstance(sid, int)
                or sid not in by_sid
                or isinstance(by_sid[sid], Future)
            ):
                raise ValueError("Ownership interval references an unexported source SID")
            company, share = row["company_symbol"], row["share_class_symbol"]
            if not isinstance(company, str) or not company or not isinstance(share, str):
                raise ValueError("Exported ownership requires known company/share-class symbols")
            start, end = pd.Timestamp(row["start_date"]), pd.Timestamp(row["end_date"])
            if (
                pd.isna(start)
                or pd.isna(end)
                or start.tz is not None
                or end.tz is not None
                or start >= end
            ):
                raise ValueError("Ownership requires ordered timezone-naive start/end dates")
            country = country_by_sid[sid]
            grouped[(company, share)].append(_Period(by_sid[sid], start, end, country))
            facts.append(
                {
                    "asset": by_sid[sid].key,
                    "sid": sid,
                    "company": company,
                    "share": share,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "country": country,
                }
            )
        self._assets = tuple(by_sid.values())
        self._periods: Mapping[tuple[str, str], tuple[_Period, ...]] = MappingProxyType(
            {key: tuple(rows) for key, rows in grouped.items()}
        )
        self._source_id = source_id
        self._facts_sha256 = hashlib.sha256(
            json.dumps(facts, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def assets(self) -> tuple[Asset, ...]:
        return self._assets

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "source_id": self._source_id,
            "lookup_sha256": self._facts_sha256,
            "method": "exported_symbol_ownership_intervals",
            "metadata_visibility": "bundle_snapshot",
            "historical_publication": "not_established",
            "listing_eligibility": "independent_native_instrument_service",
        }

    def resolve(
        self, symbol: str, on: date | datetime | str, *, country_code: str | None = None
    ) -> Asset:
        point = _lookup_instant(on)
        periods = [
            period
            for period in self._periods.get(_symbol_parts(symbol), ())
            if period.start <= point < period.end
        ]
        if country_code is not None:
            if any(period.country is None for period in periods):
                raise ValueError("Country selection requires exported exchange country metadata")
            periods = [period for period in periods if period.country == country_code]
        if len(periods) != 1:
            raise ValueError(
                f"Missing or ambiguous symbol {symbol!r} on {point} for country {country_code!r}"
            )
        return periods[0].asset
