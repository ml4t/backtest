# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `ParquetData` class for fetching Parquet files using PyArrow or FastParquet."""

import re
from pathlib import Path

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.file import FileData
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "ParquetData",
]

__pdoc__ = {}

ParquetDataT = tp.TypeVar("ParquetDataT", bound="ParquetData")


class ParquetData(FileData):
    """Data class for fetching and processing Parquet files using PyArrow or FastParquet.

    See:
        * `ParquetData.fetch_key` for argument details.

    !!! info
        For default settings, see `custom.parquet` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.parquet")

    @classmethod
    def is_parquet_file(cls, path: tp.PathLike) -> bool:
        """Return True if the provided path is a valid Parquet file, otherwise False.

        Args:
            path (PathLike): File path to be evaluated.

        Returns:
            bool: True if the path is a valid Parquet file, False otherwise.
        """
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_file() and ".parquet" in path.suffixes:
            return True
        return False

    @classmethod
    def is_parquet_group_dir(cls, path: tp.PathLike) -> bool:
        """Return True if the provided path is a directory representing a Hive-style partition
        group of Parquet partitions, otherwise False.

        Args:
            path (PathLike): Directory path to check.

        Returns:
            bool: True if the path is a Parquet partition group directory, False otherwise.

        !!! note
            Assumes the Hive partitioning scheme.
        """
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_dir():
            partition_regex = r"^(.+)=(.+)"
            if re.match(partition_regex, path.name):
                for p in path.iterdir():
                    if cls.is_parquet_group_dir(p) or cls.is_parquet_file(p):
                        return True
        return False

    @classmethod
    def is_parquet_dir(cls, path: tp.PathLike) -> bool:
        """Return True if the provided path is a directory representing a Parquet partition
        group or contains such groups, otherwise False.

        Args:
            path (PathLike): Directory path to check.

        Returns:
            bool: True if the path is a Parquet partition directory or contains such groups, False otherwise.
        """
        if cls.is_parquet_group_dir(path):
            return True
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_dir():
            for p in path.iterdir():
                if cls.is_parquet_group_dir(p):
                    return True
        return False

    @classmethod
    def is_dir_match(cls, path: tp.PathLike) -> bool:
        return cls.is_parquet_dir(path)

    @classmethod
    def is_file_match(cls, path: tp.PathLike) -> bool:
        return cls.is_parquet_file(path)

    @classmethod
    def list_partition_cols(cls, path: tp.PathLike) -> tp.List[str]:
        """List partitioning columns derived from directory names in a Hive-partitioned structure.

        Args:
            path (PathLike): Directory path containing partition directories.

        Returns:
            List[str]: List of partition column names extracted from the directory structure.

        !!! note
            Assumes the Hive partitioning scheme.
        """
        if not isinstance(path, Path):
            path = Path(path)
        partition_cols = []
        found_last_level = False
        while not found_last_level:
            found_new_level = False
            for p in path.iterdir():
                if cls.is_parquet_group_dir(p):
                    partition_cols.append(p.name.split("=")[0])
                    path = p
                    found_new_level = True
                    break
            if not found_new_level:
                found_last_level = True
        return partition_cols

    @classmethod
    def is_default_partition_col(cls, level: str) -> bool:
        """Return True if the provided partition column name is considered a default
        partition column, otherwise False.

        Args:
            level (str): Partition column name.

        Returns:
            bool: True if the partition column name is a default partition column, False otherwise.
        """
        return re.match(r"^(\bgroup\b)|(group_\d+)", level) is not None

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
        tz: tp.TimezoneLike = None,
        squeeze: tp.Optional[bool] = None,
        keep_partition_cols: tp.Optional[bool] = None,
        engine: tp.Optional[str] = None,
        **read_kwargs,
    ) -> tp.KeyData:
        """Fetch the Parquet file corresponding to a feature or symbol.

        Args:
            key (Key): Feature or symbol identifier.
            path (Any): File path to the Parquet file.

                If None, uses `key` as the path to the Parquet file.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            squeeze (Optional[bool]): Flag indicating whether to convert a single-column DataFrame to a Series.
            keep_partition_cols (Optional[bool]): Flag to retain partition columns.

                When None, default partition columns will be removed using `ParquetData.list_partition_cols`.
            engine (Optional[str]): Parquet engine to use.

                See `pd.read_parquet` for details.
            **read_kwargs: Keyword arguments for `pd.read_parquet`.

                See https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html for arguments.

        Returns:
            KeyData: Fetched data and a metadata dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import, assert_can_import_any

        tz = cls.resolve_custom_setting(tz, "tz")
        squeeze = cls.resolve_custom_setting(squeeze, "squeeze")
        keep_partition_cols = cls.resolve_custom_setting(keep_partition_cols, "keep_partition_cols")
        engine = cls.resolve_custom_setting(engine, "engine")
        read_kwargs = cls.resolve_custom_setting(read_kwargs, "read_kwargs", merge=True)

        if engine == "pyarrow":
            assert_can_import("pyarrow")
        elif engine == "fastparquet":
            assert_can_import("fastparquet")
        elif engine == "auto":
            assert_can_import_any("pyarrow", "fastparquet")
        else:
            raise ValueError(f"Invalid engine: '{engine}'")

        if path is None:
            path = key
        obj = pd.read_parquet(path, engine=engine, **read_kwargs)

        if keep_partition_cols in (None, False):
            if cls.is_parquet_dir(path):
                drop_columns = []
                partition_cols = cls.list_partition_cols(path)
                for col in obj.columns:
                    if col in partition_cols:
                        if keep_partition_cols is False or cls.is_default_partition_col(col):
                            drop_columns.append(col)
                obj = obj.drop(drop_columns, axis=1)
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None
        return obj, dict(tz=tz)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Fetch the Parquet file corresponding to the given feature.

        Args:
            feature (Feature): Feature identifier.
            **kwargs: Keyword arguments for `ParquetData.fetch_key`.

        Returns:
            FeatureData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Fetch the Parquet file corresponding to the given symbol.

        Args:
            symbol (Symbol): Symbol identifier.
            **kwargs: Keyword arguments for `ParquetData.fetch_key`.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(symbol, **kwargs)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        """Update and return the data for a given feature or symbol.

        Args:
            key (Key): Feature or symbol identifier.
            key_is_feature (bool): Flag indicating whether the key represents a feature.
            **kwargs: Keyword arguments for `ParquetData.fetch_feature` or `ParquetData.fetch_symbol`.

        Returns:
            KeyData: Updated data and a metadata dictionary.
        """
        fetch_kwargs = self.select_fetch_kwargs(key)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if key_is_feature:
            return self.fetch_feature(key, **kwargs)
        return self.fetch_symbol(key, **kwargs)

    def update_feature(self, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        return self.update_key(feature, key_is_feature=True, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        return self.update_key(symbol, key_is_feature=False, **kwargs)
