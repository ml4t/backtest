"""Preserve nanoseconds at Polars/Python boundaries without rounding event identities."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any, Literal, overload

import pandas as pd
import polars as pl


def timestamp_values(series: pl.Series) -> list[Any]:
    """Use pandas' datetime subtype where Python datetime would truncate nanoseconds."""
    dtype = series.dtype
    if not isinstance(dtype, pl.Datetime) or dtype.time_unit != "ns":
        return series.to_list()
    return [
        None if value is None else pd.Timestamp(value, unit="ns", tz=dtype.time_zone)
        for value in series.cast(pl.Int64).to_list()
    ]


def timestamp_dtype(values: Iterable[datetime | None]) -> pl.DataType:
    first = None
    nanoseconds = False
    for value in values:
        if value is None:
            continue
        if first is None:
            first = value
        if isinstance(value, pd.Timestamp) and value.unit == "ns":
            nanoseconds = True
            break
    if first is None:
        return pl.Datetime()
    dtype = pl.Series([first]).dtype
    return (
        pl.Datetime("ns", dtype.time_zone)
        if nanoseconds and isinstance(dtype, pl.Datetime)
        else dtype
    )


def temporal_row(frame: pl.DataFrame, index: int) -> dict[str, Any]:
    row = frame.row(index, named=True)
    for name, dtype in frame.schema.items():
        if isinstance(dtype, pl.Datetime) and dtype.time_unit == "ns":
            row[name] = timestamp_values(frame[name].slice(index, 1))[0]
    return row


@overload
def temporal_rows(frame: pl.DataFrame, *, named: Literal[True]) -> Iterator[dict[str, Any]]: ...


@overload
def temporal_rows(
    frame: pl.DataFrame, *, named: Literal[False] = False
) -> Iterator[tuple[Any, ...]]: ...


def temporal_rows(frame: pl.DataFrame, *, named: bool = False) -> Any:
    precise = {
        index: (name, timestamp_values(frame[name]))
        for index, (name, dtype) in enumerate(frame.schema.items())
        if isinstance(dtype, pl.Datetime) and dtype.time_unit == "ns"
    }
    rows = frame.iter_rows(named=named)
    if not precise:
        return rows

    def corrected():
        for index, row in enumerate(rows):
            if named:
                named_row = dict(row)
                for name, values in precise.values():
                    named_row[name] = values[index]
                yield named_row
            else:
                positional_row = list(row)
                for column_index, (_, values) in precise.items():
                    positional_row[column_index] = values[index]
                yield tuple(positional_row)

    return corrected()


def temporal_frame(data: Any = None, **kwargs: Any) -> pl.DataFrame:
    """Build result frames without Polars' datetime-subclass microsecond conversion."""
    frame = pl.DataFrame(data, **kwargs)
    if not frame.height or isinstance(data, pl.DataFrame):
        return frame
    columns = []
    for index, (name, dtype) in enumerate(frame.schema.items()):
        if not isinstance(dtype, pl.Datetime):
            continue
        if isinstance(data, dict):
            values = data[name]
            if isinstance(values, pl.Series):
                continue
        elif isinstance(data[0], dict):
            values = [row.get(name) for row in data]
        else:
            values = [row[index] for row in data]
        if any(isinstance(value, pd.Timestamp) and value.unit == "ns" for value in values):
            epochs = [
                None if value is None else pd.Timestamp(value).as_unit("ns").value
                for value in values
            ]
            columns.append(
                pl.Series(name, epochs, dtype=pl.Int64).cast(pl.Datetime("ns", dtype.time_zone))
            )
    return frame.with_columns(columns) if columns else frame
