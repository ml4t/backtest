"""Prepared factor delivery over native frozen research sources, without a DSL engine."""

from collections.abc import Mapping
from typing import Any

import pandas as pd
import polars as pl
from ml4t.specs import FeedSpec

from ..._feature_signals import _pandas_frame, feature_signals
from ..._temporal import temporal_rows, timestamp_values
from ...auxiliary import AuxiliaryData


class PreparedPipeline:
    """Capture an Engineer feature panel for the native before-trading-start clock.

    Calculations and screens are prepared upstream. Decision instants identify
    the premarket snapshots; a separate decisions table can declare empty ones.
    """

    def __init__(
        self,
        panel: pl.DataFrame | pl.LazyFrame | pd.DataFrame,
        *,
        feature_spec: Any,
        decisions: pl.DataFrame | pd.DataFrame | None = None,
        columns: Mapping[str, str] | None = None,
        source_id: str | None = None,
    ) -> None:
        target = FeedSpec(
            time_contract_version=2,
            timestamp_col="timestamp",
            entity_col="asset",
            timestamp_semantics="event_time",
            calendar=None,
        )
        projected, metadata = feature_signals(panel, feature_spec, target)
        features = metadata["spec"]["schema"]["feature_columns"]
        selected = (
            dict(columns) if columns is not None else dict(zip(features, features, strict=True))
        )
        if not selected or any(
            not isinstance(name, str) or not name or field not in features
            for name, field in selected.items()
        ):
            raise ValueError("Pipeline columns must map output names to declared feature fields")
        projection = metadata["projection"]
        self._source = AuxiliaryData(
            source_id if source_id is not None else metadata["spec"]["artifact_id"],
            projected,
            feed_spec=target,
            known_at_cols=projection["signal_known_at_cols"],
            eligible_cols=projection["signal_eligible_cols"],
            preparation=metadata,
        )
        self._columns = selected
        self._screen = metadata["spec"]["schema"]["eligible_col"]
        self._dtypes = {name: projected.schema[field] for name, field in selected.items()}
        points = projected["timestamp"].unique().sort()
        if decisions is not None:
            if isinstance(decisions, pd.DataFrame):
                decisions = _pandas_frame(decisions)
            column = metadata["spec"]["schema"]["decision_time_col"]
            if not isinstance(decisions, pl.DataFrame) or column not in decisions:
                raise ValueError("Pipeline decisions require the declared decision-time column")
            points = decisions[column].unique().sort()
        if (
            not isinstance(points.dtype, pl.Datetime)
            or points.dtype.time_zone is None
            or points.null_count()
        ):
            raise ValueError("Pipeline decision instants must be known and timezone-aware")
        self._decisions = frozenset(
            pd.Timestamp(point).tz_convert("UTC") for point in timestamp_values(points)
        )
        if any(
            pd.Timestamp(point).tz_convert("UTC") not in self._decisions
            for point in timestamp_values(projected["timestamp"])
        ):
            raise ValueError("Pipeline panel contains undeclared decision instants")

    @property
    def source(self) -> AuxiliaryData:
        return self._source

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "source": self._source.metadata,
            "columns": dict(self._columns),
            "decisions": [point.isoformat() for point in sorted(self._decisions)],
            "calculation_owner": "upstream_feature_preparation",
            "snapshot_clock": "native_before_trading_start",
            "universe_policy": "known_eligible_only",
        }


class _PipelineState:
    def __init__(self, definition: PreparedPipeline, eager: bool) -> None:
        self.definition, self.eager = definition, eager
        self._rows: list[dict[str, Any]] | None = None
        self._assets = {}
        self._frame: pd.DataFrame | None = None
        self._error: str | None = None

    def capture(self, context, runtime) -> None:
        self._frame, self._rows, self._error = None, None, None
        instant = context.get_datetime()
        if instant not in self.definition._decisions:
            self._error = (
                f"Prepared Pipeline has no declared decision at before_trading_start {instant}"
            )
            if self.eager:
                raise ValueError(self._error)
            return
        source = runtime.auxiliary(self.definition.source.source_id)
        fields = list(dict.fromkeys([*self.definition._columns.values(), self.definition._screen]))
        assets = source.assets
        self._rows = (
            [
                row
                for row in temporal_rows(source.current(assets, fields), named=True)
                if row[self.definition._screen] is True
            ]
            if assets
            else []
        )
        self._assets = dict(context._by_key)
        if self.eager:
            self.output()

    def output(self) -> pd.DataFrame:
        if self._error is not None:
            raise ValueError(self._error)
        if self._rows is None:
            raise RuntimeError("pipeline_output requires a completed before_trading_start snapshot")
        if self._frame is None:
            unknown = {row["asset"] for row in self._rows} - self._assets.keys()
            if unknown:
                raise ValueError(
                    f"Pipeline output has unmapped stable asset keys {sorted(unknown)}"
                )
            self._rows.sort(key=lambda row: self._assets[row["asset"]])
            values = {}
            for name, field in self.definition._columns.items():
                column = [row[field] for row in self._rows]
                dtype = self.definition._dtypes[name]
                if dtype.is_float():
                    target = str(dtype).lower()
                elif dtype.is_integer():
                    target = (
                        str(dtype) if any(value is None for value in column) else str(dtype).lower()
                    )
                elif dtype == pl.Boolean:
                    target = "boolean" if any(value is None for value in column) else "bool"
                else:
                    target = "object"
                values[name] = pd.Series(column, dtype=target)
            frame = pd.DataFrame(values)
            frame.index = pd.Index([self._assets[row["asset"]] for row in self._rows], dtype=object)
            self._frame = frame
        return self._frame.copy(deep=True)
