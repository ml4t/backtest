"""Frozen, explicitly defined benchmark returns supplied by data preparation."""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from typing import Any

import pandas as pd
import polars as pl
from ml4t.specs import FeedSpec

from ._feature_signals import _pandas_frame


def _validate_benchmark_metadata(metadata: dict[str, Any] | None) -> None:
    required = {"source_id", "return_method", "reinvestment", "content_sha256"}
    if not isinstance(metadata, dict) or any(
        not isinstance(metadata.get(name), str) or not metadata[name].strip() for name in required
    ):
        raise ValueError(
            "Benchmark provenance requires its source, method, reinvestment and content identity"
        )
    if metadata.get("return_scale") != "decimal" or metadata.get("period_clock") != "physical_end":
        raise ValueError(
            "Benchmark provenance must declare decimal returns and physical period ends"
        )
    spec = FeedSpec.from_any(metadata.get("feed_spec"))
    if getattr(spec.timestamp_semantics, "value", spec.timestamp_semantics) not in {
        "event_time",
        "bar_close",
    }:
        raise ValueError("Benchmark provenance must declare a physical FeedSpec")
    json.dumps(metadata, allow_nan=False)


def _validate_benchmark_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if set(frame.columns) != {"timestamp", "benchmark_return"}:
        raise ValueError("Benchmark rows require timestamp and benchmark_return columns")
    clock = frame.schema["timestamp"]
    returns = frame["benchmark_return"]
    if not isinstance(clock, pl.Datetime) or clock.time_zone is None:
        raise ValueError("Benchmark timestamps must be timezone-aware physical instants")
    if not returns.dtype.is_numeric() or returns.null_count() or not returns.is_finite().all():
        raise ValueError("Benchmark returns must be finite decimal values, with no missing periods")
    if frame["timestamp"].null_count() or frame["timestamp"].is_duplicated().any():
        raise ValueError("Benchmark timestamps must be known and unique")
    return frame.select(
        pl.col("timestamp").dt.convert_time_zone("UTC").cast(pl.Datetime("ns", "UTC")),
        pl.col("benchmark_return").cast(pl.Float64),
    ).sort("timestamp")


class BenchmarkData:
    """Capture prepared per-period decimal returns without deriving prices or cash flows.

    The caller declares the financial method, reinvestment convention and source.
    Timestamps identify period ends on the actual observation grid; missing
    periods are errors, never zero returns or forward-filled daily percentages.
    """

    def __init__(
        self,
        frame: pl.DataFrame | pd.DataFrame,
        *,
        source_id: str,
        return_method: str,
        reinvestment: str,
        asset: str | None = None,
        feed_spec: FeedSpec | Mapping[str, Any] | None = None,
        return_col: str = "return",
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in [source_id, return_method, reinvestment]
        ):
            raise ValueError("Benchmark source, return method and reinvestment must be explicit")
        if asset is not None and (not isinstance(asset, str) or not asset):
            raise ValueError("Benchmark asset must be a stable key or None")
        spec = FeedSpec.from_any(feed_spec)
        if getattr(spec.timestamp_semantics, "value", spec.timestamp_semantics) not in {
            "event_time",
            "bar_close",
        }:
            raise ValueError("Benchmark FeedSpec must declare physical timestamp semantics")
        if isinstance(frame, pd.DataFrame):
            frame = _pandas_frame(frame)
        if not isinstance(frame, pl.DataFrame) or {spec.timestamp_col, return_col} - set(
            frame.columns
        ):
            raise ValueError("Benchmark input is missing its declared timestamp/return columns")
        if spec.timestamp_col == return_col:
            raise ValueError("Benchmark timestamp and return columns must be distinct")
        self._frame = _validate_benchmark_frame(
            frame.select(
                pl.col(spec.timestamp_col).alias("timestamp"),
                pl.col(return_col).alias("benchmark_return"),
            )
        )
        stream = BytesIO()
        self._frame.write_ipc(stream)
        self._asset = asset
        self._metadata = {
            "source_id": source_id,
            "asset": asset,
            "return_method": return_method,
            "reinvestment": reinvestment,
            "return_scale": "decimal",
            "period_clock": "physical_end",
            "feed_spec": spec.to_dict(),
            "content_sha256": sha256(stream.getvalue()).hexdigest(),
            "provenance": deepcopy(dict(provenance or {})),
        }
        _validate_benchmark_metadata(self._metadata)

    @property
    def asset(self) -> str | None:
        return self._asset

    @property
    def frame(self) -> pl.DataFrame:
        return self._frame.clone()

    @property
    def metadata(self) -> dict[str, Any]:
        return deepcopy(self._metadata)

    def select(self, timestamps: Sequence[Any], *, timezone: str = "UTC") -> pl.DataFrame:
        clock = pd.DatetimeIndex(timestamps)
        if clock.hasnans:
            raise ValueError("Benchmark alignment requires known physical instants")
        if clock.tz is None:
            clock = clock.tz_localize(timezone)
        clock = clock.tz_convert("UTC").as_unit("ns")
        query = pl.DataFrame(
            {"timestamp": pl.Series(clock.asi8).cast(pl.Datetime("ns", "UTC"))}
        ).unique()
        missing = query.join(self._frame, on="timestamp", how="anti")
        if missing.height:
            first = pd.Timestamp(missing["timestamp"].cast(pl.Int64).sort()[0], unit="ns", tz="UTC")
            raise ValueError(f"Prepared benchmark has no return for observed period {first}")
        return self._frame.join(query, on="timestamp", how="semi").sort("timestamp")
