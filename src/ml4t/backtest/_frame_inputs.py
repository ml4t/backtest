"""Normalize declared input panels while preserving temporal precision and nulls."""

from typing import Literal, cast

import pandas as pd
import polars as pl


def input_frame(frame: pl.DataFrame | pl.LazyFrame | pd.DataFrame, *, label: str) -> pl.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return pandas_frame(frame, label=label)
    if isinstance(frame, pl.LazyFrame):
        collected = frame.collect()
        if not isinstance(collected, pl.DataFrame):
            raise TypeError(f"{label} requires synchronous eager collection")
        return collected
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{label} requires a Pandas or eager/lazy Polars frame")
    return frame


def pandas_frame(panel: pd.DataFrame, *, label: str = "Feature panel") -> pl.DataFrame:
    if panel.index.names != [None]:
        if any(name is None for name in panel.index.names):
            raise ValueError(f"{label} index levels must have explicit column names")
        panel = panel.reset_index()
    elif not isinstance(panel.index, pd.RangeIndex):
        raise ValueError(f"{label} requires explicit names for nontrivial index keys")
    if not panel.columns.is_unique or any(not isinstance(name, str) for name in panel.columns):
        raise ValueError(f"{label} columns must have distinct string names")
    primitive = {
        "int8": pl.Int8,
        "int16": pl.Int16,
        "int32": pl.Int32,
        "int64": pl.Int64,
        "uint8": pl.UInt8,
        "uint16": pl.UInt16,
        "uint32": pl.UInt32,
        "uint64": pl.UInt64,
        "float32": pl.Float32,
        "float64": pl.Float64,
        "bool": pl.Boolean,
        "boolean": pl.Boolean,
        "string": pl.String,
        "str": pl.String,
    }
    columns = []
    for name in panel.columns:
        values = panel[name]
        if isinstance(values.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_dtype(
            values.dtype
        ):
            clock = pd.DatetimeIndex(values)
            if clock.unit not in {"ms", "us", "ns"}:
                clock = clock.as_unit("ns")
            epochs = [
                None if missing else int(value)
                for value, missing in zip(clock.asi8, clock.isna(), strict=True)
            ]
            columns.append(
                pl.Series(name, epochs, dtype=pl.Int64).cast(
                    pl.Datetime(
                        cast(Literal["ms", "us", "ns"], clock.unit),
                        str(clock.tz) if clock.tz is not None else None,
                    )
                )
            )
        else:
            data = [
                None if value is pd.NA or value is pd.NaT else value for value in values.tolist()
            ]
            columns.append(
                pl.Series(name, data, dtype=primitive.get(str(values.dtype).lower()), strict=True)
            )
    return pl.DataFrame(columns)
