# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `CSVData` class for handling CSV and TSV data files."""

from pathlib import Path

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.file import FileData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "CSVData",
]

__pdoc__ = {}

CSVDataT = tp.TypeVar("CSVDataT", bound="CSVData")


class CSVData(FileData):
    """Data class for fetching CSV and TSV data files.

    This class extends `FileData` and provides methods for verifying file types,
    resolving metadata keys, and reading CSV files using pandas.

    See:
        * `CSVData.fetch_key` for argument details.

    !!! info
        For default settings, see `custom.csv` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.csv")

    @classmethod
    def is_csv_file(cls, path: tp.PathLike) -> bool:
        """Return whether the given path is a CSV or TSV file.

        Args:
            path (PathLike): File path to check.

        Returns:
            bool: True if the file is a CSV or TSV file, False otherwise.
        """
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_file() and ".csv" in path.suffixes:
            return True
        if path.exists() and path.is_file() and ".tsv" in path.suffixes:
            return True
        return False

    @classmethod
    def is_file_match(cls, path: tp.PathLike) -> bool:
        return cls.is_csv_file(path)

    @classmethod
    def resolve_keys_meta(
        cls,
        keys: tp.Union[None, dict, tp.MaybeKeys] = None,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[None, dict, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, dict, tp.MaybeSymbols] = None,
        paths: tp.Any = None,
    ) -> tp.Kwargs:
        keys_meta = FileData.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
        )
        if keys_meta["keys"] is None and paths is None:
            keys_meta["keys"] = cls.list_paths()
        return keys_meta

    @classmethod
    def fetch_key(
        cls,
        key: tp.Key,
        path: tp.Any = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        tz: tp.TimezoneLike = None,
        start_row: tp.Optional[int] = None,
        end_row: tp.Optional[int] = None,
        header: tp.Optional[tp.MaybeSequence[int]] = None,
        index_col: tp.Optional[int] = None,
        parse_dates: tp.Optional[bool] = None,
        chunk_func: tp.Optional[tp.Callable] = None,
        squeeze: tp.Optional[bool] = None,
        **read_kwargs,
    ) -> tp.KeyData:
        """Fetch the CSV file for a given feature or symbol.

        Arguments `skiprows` and `nrows` are automatically determined based on `start_row` and `end_row`.

        If either `start` or `end` is provided, the entire CSV file is read before applying date filtering.

        For additional parameters, refer to `pd.read_csv` documentation.

        Args:
            key (Key): Feature or symbol identifier.
            path (Any): File path to the CSV file.

                If not provided, `key` is used as the file path.
            start (Optional[DatetimeLike]): Start datetime (e.g., "2024-01-01", "1 year ago").

                The object's timezone will be used. See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (Optional[DatetimeLike]): End datetime (e.g., "2025-01-01", "now").

                The object's timezone will be used. See `vectorbtpro.utils.datetime_.to_timestamp`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            start_row (Optional[int]): Starting row index (inclusive), excluding header rows.
            end_row (Optional[int]): Ending row index (exclusive), excluding header rows.
            header (Optional[Union[int, Sequence[int]]]): Row number(s) to use as the header.

                See `pd.read_csv` for details on this argument.
            index_col (Optional[int]): Column to use as the index.

                If set to False, it is treated as None. See `pd.read_csv` for details on this argument.
            parse_dates (Optional[bool]): Whether to parse dates.

                See `pd.read_csv` for details on this argument.
            chunk_func (Optional[Callable]): Function for processing and concatenating chunks from a `TextFileReader`.

                Invoked only if `iterator` or `chunksize` is specified.
            squeeze (Optional[bool]): Flag indicating whether to convert a single-column DataFrame to a Series.
            **read_kwargs: Keyword arguments for `pd.read_csv`.

                See https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html for arguments.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        from pandas.io.parsers import TextFileReader
        from pandas.api.types import is_object_dtype

        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        tz = cls.resolve_custom_setting(tz, "tz")
        start_row = cls.resolve_custom_setting(start_row, "start_row")
        if start_row is None:
            start_row = 0
        end_row = cls.resolve_custom_setting(end_row, "end_row")
        header = cls.resolve_custom_setting(header, "header")
        index_col = cls.resolve_custom_setting(index_col, "index_col")
        if index_col is False:
            index_col = None
        parse_dates = cls.resolve_custom_setting(parse_dates, "parse_dates")
        chunk_func = cls.resolve_custom_setting(chunk_func, "chunk_func")
        squeeze = cls.resolve_custom_setting(squeeze, "squeeze")
        read_kwargs = cls.resolve_custom_setting(read_kwargs, "read_kwargs", merge=True)

        if path is None:
            path = key
        if isinstance(header, int):
            header = [header]
        header_rows = header[-1] + 1
        start_row += header_rows
        if end_row is not None:
            end_row += header_rows
        skiprows = range(header_rows, start_row)
        if end_row is not None:
            nrows = end_row - start_row
        else:
            nrows = None

        sep = read_kwargs.pop("sep", None)
        if isinstance(path, (str, Path)):
            try:
                _path = Path(path)
                if _path.suffix.lower() == ".csv":
                    if sep is None:
                        sep = ","
                if _path.suffix.lower() == ".tsv":
                    if sep is None:
                        sep = "\t"
            except Exception:
                pass
        if sep is None:
            sep = ","

        obj = pd.read_csv(
            path,
            sep=sep,
            header=header,
            index_col=index_col,
            parse_dates=parse_dates,
            skiprows=skiprows,
            nrows=nrows,
            **read_kwargs,
        )
        if isinstance(obj, TextFileReader):
            if chunk_func is None:
                obj = pd.concat(list(obj), axis=0)
            else:
                obj = chunk_func(obj)
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None
        if index_col is not None and parse_dates and is_object_dtype(obj.index.dtype):
            obj.index = pd.to_datetime(obj.index, utc=True)
            if tz is not None:
                obj.index = obj.index.tz_convert(tz)
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        if start is not None or end is not None:
            if not isinstance(obj.index, pd.DatetimeIndex):
                raise TypeError("Cannot filter index that is not DatetimeIndex")
            if obj.index.tz is not None:
                if start is not None:
                    start = dt.to_tzaware_timestamp(start, naive_tz=tz, tz=obj.index.tz)
                if end is not None:
                    end = dt.to_tzaware_timestamp(end, naive_tz=tz, tz=obj.index.tz)
            else:
                if start is not None:
                    start = dt.to_naive_timestamp(start, tz=tz)
                if end is not None:
                    end = dt.to_naive_timestamp(end, tz=tz)
            mask = True
            if start is not None:
                mask &= obj.index >= start
            if end is not None:
                mask &= obj.index < end
            mask_indices = np.flatnonzero(mask)
            if len(mask_indices) == 0:
                return None
            obj = obj.iloc[mask_indices[0] : mask_indices[-1] + 1]
            start_row += mask_indices[0]
        return obj, dict(last_row=start_row - header_rows + len(obj.index) - 1, tz=tz)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Fetch the CSV file for a feature.

        Args:
            feature (Feature): Feature identifier.
            **kwargs: Keyword arguments for `CSVData.fetch_key`.

        Returns:
            FeatureData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Fetch the CSV file for a symbol.

        Args:
            symbol (Symbol): Symbol identifier.
            **kwargs: Keyword arguments for `CSVData.fetch_key`.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(symbol, **kwargs)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        """Update the CSV data for a feature or symbol.

        Args:
            key (Key): Feature or symbol identifier.
            key_is_feature (bool): Flag indicating whether the key represents a feature.
            **kwargs: Keyword arguments for `CSVData.fetch_feature` or `CSVData.fetch_symbol`.

        Returns:
            KeyData: Updated data and a metadata dictionary.
        """
        fetch_kwargs = self.select_fetch_kwargs(key)
        returned_kwargs = self.select_returned_kwargs(key)
        fetch_kwargs["start_row"] = returned_kwargs["last_row"]
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if key_is_feature:
            return self.fetch_feature(key, **kwargs)
        return self.fetch_symbol(key, **kwargs)

    def update_feature(self, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        return self.update_key(feature, key_is_feature=True, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        return self.update_key(symbol, key_is_feature=False, **kwargs)
