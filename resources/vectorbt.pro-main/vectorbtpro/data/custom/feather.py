# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `FeatherData` class for fetching Feather data using PyArrow."""

from pathlib import Path

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.file import FileData
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "FeatherData",
]

__pdoc__ = {}

FeatherDataT = tp.TypeVar("FeatherDataT", bound="FeatherData")


class FeatherData(FileData):
    """Data class class for fetching Feather data using PyArrow.

    See:
        * `FeatherData.fetch_key` for argument details.

    !!! info
        For default settings, see `custom.feather` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.feather")

    @classmethod
    def list_paths(cls, path: tp.PathLike = ".", **match_path_kwargs) -> tp.List[Path]:
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_dir():
            path = path / "*.feather"
        return cls.match_path(path, **match_path_kwargs)

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
            keys_meta["keys"] = "*.feather"
        return keys_meta

    @classmethod
    def fetch_key(
        cls,
        key: tp.Key,
        path: tp.Any = None,
        tz: tp.TimezoneLike = None,
        index_col: tp.Optional[tp.MaybeSequence[tp.IntStr]] = None,
        squeeze: tp.Optional[bool] = None,
        **read_kwargs,
    ) -> tp.KeyData:
        """Fetch a Feather file for a given feature or symbol.

        Args:
            key (Key): Feature or symbol identifier.
            path (str): File path for the Feather file.

                If None, `key` is used as the file path.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            index_col (Optional[MaybeSequence[IntStr]]): Column position(s) or name(s) to use as the index.

                Applies if the fetched data has a default index.
            squeeze (bool): Flag indicating whether to convert a single-column DataFrame to a Series.
            **read_kwargs: Keyword arguments for `pd.read_feather`.

                See https://pandas.pydata.org/docs/reference/api/pandas.read_feather.html for arguments.

        Returns:
            KeyData: Fetched data and a metadata dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("pyarrow")

        tz = cls.resolve_custom_setting(tz, "tz")
        index_col = cls.resolve_custom_setting(index_col, "index_col")
        if index_col is False:
            index_col = None
        squeeze = cls.resolve_custom_setting(squeeze, "squeeze")
        read_kwargs = cls.resolve_custom_setting(read_kwargs, "read_kwargs", merge=True)

        if path is None:
            path = key
        obj = pd.read_feather(path, **read_kwargs)

        if isinstance(obj, pd.DataFrame) and checks.is_default_index(obj.index):
            if index_col is not None:
                if checks.is_int(index_col):
                    keys = obj.columns[index_col]
                elif isinstance(index_col, str):
                    keys = index_col
                else:
                    keys = []
                    for col in index_col:
                        if checks.is_int(col):
                            keys.append(obj.columns[col])
                        else:
                            keys.append(col)
                obj = obj.set_index(keys)
                if not isinstance(obj.index, pd.MultiIndex):
                    if obj.index.name == "index":
                        obj.index.name = None
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None
        return obj, dict(tz=tz)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Fetch a Feather file for a feature.

        Args:
            feature (Feature): Feature identifier.
            **kwargs: Keyword arguments for `FeatherData.fetch_key`.

        Returns:
            FeatureData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Fetch a Feather file for a symbol.

        Args:
            symbol (Symbol): Symbol identifier.
            **kwargs: Keyword arguments for `FeatherData.fetch_key`.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(symbol, **kwargs)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        """Update data for a given feature or symbol.

        Args:
            key (Key): Feature or symbol identifier.
            key_is_feature (bool): Flag indicating whether the key represents a feature.
            **kwargs: Keyword arguments for `FeatherData.fetch_feature` or `FeatherData.fetch_symbol`.

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
