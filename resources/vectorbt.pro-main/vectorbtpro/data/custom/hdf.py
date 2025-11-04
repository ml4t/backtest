# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `HDFData` class for fetching HDF data using PyTables."""

import re
from glob import glob
from pathlib import Path, PurePath

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.file import FileData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.parsing import get_func_arg_names

__all__ = [
    "HDFData",
]

__pdoc__ = {}


class HDFPathNotFoundError(Exception):
    """Exception raised when the path to an HDF file cannot be found."""

    pass


class HDFKeyNotFoundError(Exception):
    """Exception raised when the key to an HDF object cannot be found."""

    pass


HDFDataT = tp.TypeVar("HDFDataT", bound="HDFData")


class HDFData(FileData):
    """Data class for fetching HDF data using PyTables.

    See:
        * `HDFData.fetch_key` for argument details.

    !!! info
        For default settings, see `custom.hdf` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.hdf")

    @classmethod
    def is_hdf_file(cls, path: tp.PathLike) -> bool:
        """Return whether the provided path is an HDF file.

        Args:
            path (PathLike): File path to validate.

        Returns:
            bool: True if the path is an HDF file, False otherwise.

        !!! note
            Checks for file suffixes `.hdf`, `.hdf5`, and `.h5`.
        """
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_file() and ".hdf" in path.suffixes:
            return True
        if path.exists() and path.is_file() and ".hdf5" in path.suffixes:
            return True
        if path.exists() and path.is_file() and ".h5" in path.suffixes:
            return True
        return False

    @classmethod
    def is_file_match(cls, path: tp.PathLike) -> bool:
        return cls.is_hdf_file(path)

    @classmethod
    def split_hdf_path(
        cls,
        path: tp.PathLike,
        key: tp.Optional[str] = None,
        _full_path: tp.Optional[Path] = None,
    ) -> tp.Tuple[Path, tp.Optional[str]]:
        """Split the provided HDF path into its file and key components.

        If the given path does not immediately correspond to an existing file,
        the method recursively inspects parent directories to locate the HDF file
        and constructs the key from intermediate path segments.

        Args:
            path (PathLike): Path to the HDF object.
            key (Optional[str]): Key to identify the object within the HDF file.

                Defaults to None. When provided, it is combined with portions of the path during recursion.

        Returns:
            Tuple[Path, Optional[str]]: Tuple containing the HDF file path and the associated key.
        """
        path = Path(path)
        if _full_path is None:
            _full_path = path
        if path.exists():
            if path.is_dir():
                raise HDFPathNotFoundError(f"No HDF files could be matched with {_full_path}")
            return path, key
        new_path = path.parent
        if key is None:
            new_key = path.name
        else:
            new_key = str(Path(path.name) / key)
        return cls.split_hdf_path(new_path, new_key, _full_path=_full_path)

    @classmethod
    def match_path(
        cls,
        path: tp.PathLike,
        match_regex: tp.Optional[str] = None,
        sort_paths: bool = True,
        recursive: bool = True,
        **kwargs,
    ) -> tp.List[Path]:
        path = Path(path)
        if path.exists():
            if path.is_dir() and not cls.is_dir_match(path):
                sub_paths = []
                for p in path.iterdir():
                    if p.is_dir() and cls.is_dir_match(p):
                        sub_paths.append(p)
                    if p.is_file() and cls.is_file_match(p):
                        sub_paths.append(p)
                key_paths = [p for sub_path in sub_paths for p in cls.match_path(sub_path, sort_paths=False, **kwargs)]
            else:
                with pd.HDFStore(str(path), mode="r") as store:
                    keys = [k[1:] for k in store.keys()]
                key_paths = [path / k for k in keys]
        else:
            try:
                file_path, key = cls.split_hdf_path(path)
                with pd.HDFStore(str(file_path), mode="r") as store:
                    keys = [k[1:] for k in store.keys()]
                if key is None:
                    key_paths = [file_path / k for k in keys]
                elif key in keys:
                    key_paths = [file_path / key]
                else:
                    matching_keys = []
                    for k in keys:
                        if k.startswith(key) or PurePath("/" + str(k)).match("/" + str(key)):
                            matching_keys.append(k)
                    if len(matching_keys) == 0:
                        raise HDFKeyNotFoundError(f"No HDF keys could be matched with {key}")
                    key_paths = [file_path / k for k in matching_keys]
            except HDFPathNotFoundError:
                sub_paths = list([Path(p) for p in glob(str(path), recursive=recursive)])
                if len(sub_paths) == 0 and re.match(r".+\..+", str(path)):
                    base_path = None
                    base_ended = False
                    key_path = None
                    for part in path.parts:
                        part = Path(part)
                        if base_ended:
                            if key_path is None:
                                key_path = part
                            else:
                                key_path /= part
                        else:
                            if re.match(r".+\..+", str(part)):
                                base_ended = True
                            if base_path is None:
                                base_path = part
                            else:
                                base_path /= part
                    sub_paths = list([Path(p) for p in glob(str(base_path), recursive=recursive)])
                    if key_path is not None:
                        sub_paths = [p / key_path for p in sub_paths]
                key_paths = [p for sub_path in sub_paths for p in cls.match_path(sub_path, sort_paths=False, **kwargs)]
        if match_regex is not None:
            key_paths = [p for p in key_paths if re.match(match_regex, str(p))]
        if sort_paths:
            key_paths = sorted(key_paths)
        return key_paths

    @classmethod
    def path_to_key(cls, path: tp.PathLike, **kwargs) -> str:
        return Path(path).name

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
        chunk_func: tp.Optional[tp.Callable] = None,
        **read_kwargs,
    ) -> tp.KeyData:
        """Fetch the HDF object of a feature or symbol.

        Args:
            key (Key): Feature or symbol identifier.
            path (Any): File path to the HDF file.

                Will be resolved using `HDFData.split_hdf_path`.

                If None, `key` is used as the file path.
            start (Optional[DatetimeLike]): Start datetime (e.g., "2024-01-01", "1 year ago").

                Extracts the object's index and compares it to this date using the object's timezone.
                See `vectorbtpro.utils.datetime_.to_timestamp`.

                !!! note
                    Applicable only if the object was saved in table format.
            end (Optional[DatetimeLike]): End datetime (e.g., "2025-01-01", "now").

                Extracts the object's index and compares it to this date using the object's timezone.
                See `vectorbtpro.utils.datetime_.to_timestamp`.

                !!! note
                    Applicable only if the object was saved in table format.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            start_row (Optional[int]): Index of the starting row (inclusive).

                Also used for querying the index.
            end_row (Optional[int]): Index of the ending row (exclusive).

                Also used for querying the index.
            chunk_func (Optional[Callable]): Function for processing and concatenating chunks from a `TableIterator `.

                Invoked only if `iterator` or `chunksize` is specified.
            **read_kwargs: Keyword arguments for `pd.read_hdf`.

                See https://pandas.pydata.org/docs/reference/api/pandas.read_hdf.html for arguments.

        Returns:
            KeyData: Fetched data and a metadata dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tables")

        from pandas.io.pytables import TableIterator

        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        tz = cls.resolve_custom_setting(tz, "tz")
        start_row = cls.resolve_custom_setting(start_row, "start_row")
        if start_row is None:
            start_row = 0
        end_row = cls.resolve_custom_setting(end_row, "end_row")
        read_kwargs = cls.resolve_custom_setting(read_kwargs, "read_kwargs", merge=True)

        if path is None:
            path = key
        path = Path(path)
        file_path, file_key = cls.split_hdf_path(path)
        if file_key is not None:
            key = file_key

        if start is not None or end is not None:
            hdf_store_arg_names = get_func_arg_names(pd.HDFStore.__init__)
            hdf_store_kwargs = dict()
            for k, v in read_kwargs.items():
                if k in hdf_store_arg_names:
                    hdf_store_kwargs[k] = v
            with pd.HDFStore(str(file_path), mode="r", **hdf_store_kwargs) as store:
                index = store.select_column(key, "index", start=start_row, stop=end_row)
            if not isinstance(index, pd.Index):
                index = pd.Index(index)
            if not isinstance(index, pd.DatetimeIndex):
                raise TypeError("Cannot filter index that is not DatetimeIndex")
            if tz is None:
                tz = index.tz
            if index.tz is not None:
                if start is not None:
                    start = dt.to_tzaware_timestamp(start, naive_tz=tz, tz=index.tz)
                if end is not None:
                    end = dt.to_tzaware_timestamp(end, naive_tz=tz, tz=index.tz)
            else:
                if start is not None:
                    start = dt.to_naive_timestamp(start, tz=tz)
                if end is not None:
                    end = dt.to_naive_timestamp(end, tz=tz)
            mask = True
            if start is not None:
                mask &= index >= start
            if end is not None:
                mask &= index < end
            mask_indices = np.flatnonzero(mask)
            if len(mask_indices) == 0:
                return None
            start_row += mask_indices[0]
            end_row = start_row + mask_indices[-1] - mask_indices[0] + 1

        obj = pd.read_hdf(file_path, key=key, start=start_row, stop=end_row, **read_kwargs)
        if isinstance(obj, TableIterator):
            if chunk_func is None:
                obj = pd.concat(list(obj), axis=0)
            else:
                obj = chunk_func(obj)
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        return obj, dict(last_row=start_row + len(obj.index) - 1, tz=tz)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Fetch the HDF object for a feature.

        Args:
            feature (Feature): Feature identifier.
            **kwargs: Keyword arguments for `HDFData.fetch_key`.

        Returns:
            FeatureData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Fetch the HDF object for a symbol.

        Args:
            symbol (Symbol): Symbol identifier.
            **kwargs: Keyword arguments for `HDFData.fetch_key`.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(symbol, **kwargs)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        """Update the HDF data for a feature or symbol.

        Args:
            key (Key): Feature or symbol identifier.
            key_is_feature (bool): Flag indicating whether the key represents a feature.
            **kwargs: Keyword arguments for `HDFData.fetch_feature` or `HDFData.fetch_symbol`.

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
