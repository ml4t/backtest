"""Frozen, explicitly published research inputs for simulation-time queries."""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from typing import Any

import pandas as pd
import polars as pl
from ml4t.specs import FeedSpec, TimestampSemantics

from ._feature_signals import _pandas_frame


class AuxiliaryData:
    """Capture a prepared offline source without parsing dates or resolving symbols.

    The caller prepares physical observation/publication instants and stable keys.
    Every queryable field requires its own known-at declaration. Extra input columns
    remain outside the read surface. The original source identifier is an exact
    registry key, not a URL to fetch during execution.
    """

    def __init__(
        self,
        source_id: str,
        frame: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        *,
        feed_spec: FeedSpec | Any,
        known_at_cols: Mapping[str, str],
        preparation: Mapping[str, Any] | None = None,
        eligible_cols: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Auxiliary source_id must identify the original offline source")
        spec = FeedSpec.from_any(feed_spec)
        if spec.timestamp_semantics not in {
            TimestampSemantics.EVENT_TIME,
            TimestampSemantics.BAR_CLOSE,
        }:
            raise ValueError("Auxiliary observations require declared physical timestamp semantics")
        time, entity = spec.timestamp_col, spec.entity_col
        if not isinstance(entity, str) or entity == time:
            raise ValueError(
                "Auxiliary data requires distinct timestamp and single stable entity keys"
            )
        if isinstance(frame, pl.LazyFrame):
            collected = frame.collect()
            if not isinstance(collected, pl.DataFrame):
                raise TypeError("Auxiliary lazy collection must produce a DataFrame")
            frame = collected
        elif isinstance(frame, pd.DataFrame):
            frame = _pandas_frame(frame)
        if not isinstance(frame, pl.DataFrame):
            raise TypeError("Auxiliary data requires a Polars or named-column Pandas frame")
        known = dict(known_at_cols)
        if not known or any(
            not isinstance(field, str)
            or not field
            or not isinstance(column, str)
            or not column
            or field in {time, entity}
            for field, column in known.items()
        ):
            raise ValueError("Every auxiliary payload field needs an explicit known-at column")
        eligible = dict(eligible_cols or {})
        if any(
            field not in known or not isinstance(column, str) or not column
            for field, column in eligible.items()
        ):
            raise ValueError(
                "Auxiliary eligibility requires declared payload fields and Boolean columns"
            )
        required = list(dict.fromkeys([time, entity, *known, *known.values(), *eligible.values()]))
        if set(required) - set(frame.columns):
            raise ValueError(
                "Auxiliary input is missing declared key, value or publication columns"
            )
        frame = frame.select(required)
        if any(frame.schema[column] != pl.Boolean for column in eligible.values()):
            raise ValueError("Auxiliary eligibility columns must be Boolean")
        for column in [time, *known.values()]:
            dtype = frame.schema[column]
            if not isinstance(dtype, pl.Datetime) or dtype.time_zone is None:
                raise ValueError("Auxiliary observation/publication columns require aware instants")
        if frame[time].null_count() or frame[entity].null_count():
            raise ValueError("Auxiliary observation and stable entity keys cannot be null")
        if frame.select(pl.struct([time, entity]).is_duplicated().any()).item():
            raise ValueError(
                "Duplicate auxiliary keys require explicit upstream revision/event handling"
            )
        stream = BytesIO()
        frame.write_ipc(stream)
        self._frame, self._spec, self._known = frame.clone(), spec, known
        self._source_id = source_id
        self._eligible = eligible
        self._metadata = {
            "source_id": source_id,
            "feed_spec": spec.to_dict(),
            "known_at_cols": dict(known),
            "eligible_cols": dict(eligible),
            "content_sha256": sha256(stream.getvalue()).hexdigest(),
            "preparation": deepcopy(dict(preparation or {})),
        }

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def frame(self) -> pl.DataFrame:
        return self._frame.clone()

    @property
    def feed_spec(self) -> FeedSpec:
        return self._spec

    @property
    def known_at_cols(self) -> dict[str, str]:
        return dict(self._known)

    @property
    def eligible_cols(self) -> dict[str, str]:
        return dict(self._eligible)

    @property
    def metadata(self) -> dict[str, Any]:
        return deepcopy(self._metadata)
