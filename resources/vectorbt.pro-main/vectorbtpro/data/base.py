# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing base classes and dictionary types for working with data sources."""

import inspect
import string
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.indexes import stack_indexes
from vectorbtpro.base.merging import column_stack_arrays, is_merge_func_from_config
from vectorbtpro.base.reshaping import to_any_array, to_pd_array, to_1d_array, to_2d_array, broadcast_to
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.data.decorators import attach_symbol_dict_methods
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.analyzable import Analyzable
from vectorbtpro.generic.drawdowns import Drawdowns
from vectorbtpro.ohlcv.nb import mirror_ohlc_nb
from vectorbtpro.ohlcv.enums import PriceFeature
from vectorbtpro.returns.accessors import ReturnsAccessor
from vectorbtpro.utils import checks, datetime_ as dt
from vectorbtpro.utils.attr_ import get_dict_attr
from vectorbtpro.utils.base import Base
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig, copy_dict
from vectorbtpro.utils.decorators import cached_property, hybrid_method
from vectorbtpro.utils.enum_ import map_enum_fields
from vectorbtpro.utils.execution import Task, NoResult, NoResultsException, filter_out_no_results, execute
from vectorbtpro.utils.merging import MergeFunc
from vectorbtpro.utils.parsing import get_func_arg_names, extend_args
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.pickling import pdict, RecState
from vectorbtpro.utils.template import Rep, RepEval, CustomTemplate, substitute_templates
from vectorbtpro.utils.warnings_ import warn
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg

if tp.TYPE_CHECKING:
    from sqlalchemy import Engine as EngineT
else:
    EngineT = "sqlalchemy.Engine"
if tp.TYPE_CHECKING:
    from duckdb import DuckDBPyConnection as DuckDBPyConnectionT
else:
    DuckDBPyConnectionT = "duckdb.DuckDBPyConnection"

__all__ = [
    "key_dict",
    "feature_dict",
    "symbol_dict",
    "run_func_dict",
    "run_arg_dict",
    "Data",
]

__pdoc__ = {}


class key_dict(pdict):
    """Class for a dictionary that uses features or symbols as keys."""

    pass


class feature_dict(key_dict):
    """Class for a dictionary that uses features as keys."""

    pass


class symbol_dict(key_dict):
    """Class for a dictionary that uses symbols as keys."""

    pass


class run_func_dict(pdict):
    """Class for a dictionary that uses function names as keys for `Data.run`."""

    pass


class run_arg_dict(pdict):
    """Class for a dictionary that uses argument names as keys for `Data.run`."""

    pass


BaseDataMixinT = tp.TypeVar("BaseDataMixinT", bound="BaseDataMixin")


class BaseDataMixin(Base):
    """Base mixin class for working with data.

    This class provides helper properties and methods for managing feature and symbol data,
    including key normalization and index lookup.
    """

    @property
    def feature_wrapper(self) -> ArrayWrapper:
        """Column wrapper for feature data.

        Returns:
            ArrayWrapper: Column wrapper for feature data.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def symbol_wrapper(self) -> ArrayWrapper:
        """Column wrapper for symbol data.

        Returns:
            ArrayWrapper: Column wrapper for symbol data.

        !!! abstract
            This property should be overridden in a subclass.
        """
        raise NotImplementedError

    @property
    def features(self) -> tp.List[tp.Feature]:
        """List of features obtained from the feature wrapper columns.

        Returns:
            List[Feature]: List of features.
        """
        return self.feature_wrapper.columns.tolist()

    @property
    def symbols(self) -> tp.List[tp.Symbol]:
        """List of symbols obtained from the symbol wrapper columns.

        Returns:
            List[Symbol]: List of symbols.
        """
        return self.symbol_wrapper.columns.tolist()

    @classmethod
    def has_multiple_keys(cls, keys: tp.MaybeKeys) -> bool:
        """Check whether the provided keys represent multiple keys.

        Args:
            keys (MaybeKeys): Feature or symbol identifier(s).

        Returns:
            bool: True if the keys are a sequence, False if they are hashable.
        """
        if checks.is_hashable(keys):
            return False
        elif checks.is_sequence(keys):
            return True
        raise TypeError("Keys must be either a hashable or a sequence of hashable")

    @classmethod
    def prepare_key(cls, key: tp.Key) -> tp.Key:
        """Prepare a key by normalizing it.

        Transforms string keys to lowercase, strips whitespace, and replaces spaces with underscores.
        Tuples are processed recursively.

        Args:
            key (Key): Feature or symbol identifier.

        Returns:
            Key: Normalized key.
        """
        if isinstance(key, tuple):
            return tuple([cls.prepare_key(k) for k in key])
        if isinstance(key, str):
            return key.lower().strip().replace(" ", "_")
        return key

    def get_feature_idx(self, feature: tp.Feature, raise_error: bool = False) -> int:
        """Return the index of the specified feature.

        Args:
            feature (Feature): Feature identifier.
            raise_error (bool): Whether to raise an error if the feature is not found.

        Returns:
            int: Index of the feature, or -1 if not found.

        Raises:
            ValueError: If multiple features match the specified key.
        """
        # shortcut
        columns = self.feature_wrapper.columns
        if not columns.has_duplicates:
            if feature in columns:
                return columns.get_loc(feature)

        feature = self.prepare_key(feature)
        found_indices = []
        for i, c in enumerate(self.features):
            c = self.prepare_key(c)
            if feature == c:
                found_indices.append(i)
        if len(found_indices) == 0:
            if raise_error:
                raise ValueError(f"No features match the feature '{str(feature)}'")
            return -1
        if len(found_indices) == 1:
            return found_indices[0]
        raise ValueError(f"Multiple features match the feature '{str(feature)}'")

    def get_symbol_idx(self, symbol: tp.Symbol, raise_error: bool = False) -> int:
        """Return the index of the specified symbol.

        Args:
            symbol (Symbol): Symbol identifier.
            raise_error (bool): Whether to raise an error if the symbol is not found.

        Returns:
            int: Index of the symbol, or -1 if not found.

        Raises:
            ValueError: If multiple symbols match the specified key.
        """
        # shortcut
        columns = self.symbol_wrapper.columns
        if not columns.has_duplicates:
            if symbol in columns:
                return columns.get_loc(symbol)

        symbol = self.prepare_key(symbol)
        found_indices = []
        for i, c in enumerate(self.symbols):
            c = self.prepare_key(c)
            if symbol == c:
                found_indices.append(i)
        if len(found_indices) == 0:
            if raise_error:
                raise ValueError(f"No symbols match the symbol '{str(symbol)}'")
            return -1
        if len(found_indices) == 1:
            return found_indices[0]
        raise ValueError(f"Multiple symbols match the symbol '{str(symbol)}'")

    def select_feature_idxs(self: BaseDataMixinT, idxs: tp.MaybeSequence[int], **kwargs) -> BaseDataMixinT:
        """Select one or more features by their index positions.

        Args:
            idxs (MaybeSequence[int]): Index or indices of the features to select.
            **kwargs: Keyword arguments for feature selection.

        Returns:
            BaseDataMixin: New instance with the selected features.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def select_symbol_idxs(self: BaseDataMixinT, idxs: tp.MaybeSequence[int], **kwargs) -> BaseDataMixinT:
        """Select one or more symbols by their index positions.

        Args:
            idxs (MaybeSequence[int]): Index or indices of the symbols to select.
            **kwargs: Keyword arguments for symbol selection.

        Returns:
            BaseDataMixin: New instance with the selected symbols.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def select_features(self: BaseDataMixinT, features: tp.MaybeFeatures, **kwargs) -> BaseDataMixinT:
        """Select one or more features using label(s).

        Args:
            features (MaybeFeatures): Feature identifier(s).
            **kwargs: Keyword arguments for `BaseDataMixin.select_feature_idxs`.

        Returns:
            BaseDataMixin: New instance containing the selected features.
        """
        if self.has_multiple_keys(features):
            feature_idxs = [self.get_feature_idx(k, raise_error=True) for k in features]
        else:
            feature_idxs = self.get_feature_idx(features, raise_error=True)
        return self.select_feature_idxs(feature_idxs, **kwargs)

    def select_symbols(self: BaseDataMixinT, symbols: tp.MaybeSymbols, **kwargs) -> BaseDataMixinT:
        """Select one or more symbols using label(s).

        Args:
            symbols (MaybeSymbols): Symbol identifier(s).
            **kwargs: Keyword arguments for `BaseDataMixin.select_symbol_idxs`.

        Returns:
            BaseDataMixin: New instance containing the selected symbols.
        """
        if self.has_multiple_keys(symbols):
            symbol_idxs = [self.get_symbol_idx(k, raise_error=True) for k in symbols]
        else:
            symbol_idxs = self.get_symbol_idx(symbols, raise_error=True)
        return self.select_symbol_idxs(symbol_idxs, **kwargs)

    def get(
        self,
        features: tp.Optional[tp.MaybeFeatures] = None,
        symbols: tp.Optional[tp.MaybeSymbols] = None,
        feature: tp.Optional[tp.Feature] = None,
        symbol: tp.Optional[tp.Symbol] = None,
        **kwargs,
    ) -> tp.MaybeTuple[tp.SeriesFrame]:
        """Retrieve data for specified features and symbols.

        Args:
            features (Optional[MaybeFeatures]): Feature identifier(s).
            symbols (Optional[MaybeSymbols]): Symbol identifier(s).
            feature (Optional[Feature]): Feature identifier.
            symbol (Optional[Symbol]): Symbol identifier.
            **kwargs: Keyword arguments for data retrieval.

        Returns:
            MaybeTuple[SeriesFrame]: Retrieved data.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def has_feature(self, feature: tp.Feature) -> bool:
        """Check whether the specified feature exists.

        Args:
            feature (Feature): Feature identifier.

        Returns:
            bool: True if the feature exists, False otherwise.
        """
        feature_idx = self.get_feature_idx(feature, raise_error=False)
        return feature_idx != -1

    def has_symbol(self, symbol: tp.Symbol) -> bool:
        """Check whether the specified symbol exists.

        Args:
            symbol (Symbol): Symbol identifier.

        Returns:
            bool: True if the symbol exists, False otherwise.
        """
        symbol_idx = self.get_symbol_idx(symbol, raise_error=False)
        return symbol_idx != -1

    def assert_has_feature(self, feature: tp.Feature) -> None:
        """Assert that the specified feature exists.

        Args:
            feature (Feature): Feature identifier.

        Returns:
            None
        """
        self.get_feature_idx(feature, raise_error=True)

    def assert_has_symbol(self, symbol: tp.Symbol) -> None:
        """Assert that the specified symbol exists.

        Args:
            symbol (Symbol): Symbol identifier.

        Returns:
            None
        """
        self.get_symbol_idx(symbol, raise_error=True)

    def get_feature(
        self,
        feature: tp.Union[int, tp.Feature],
        raise_error: bool = False,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Retrieve data for a feature by its index or label.

        Args:
            feature (Union[int, Feature]): Index or label of the feature.
            raise_error (bool): Whether to raise an error if the feature is not found.

        Returns:
            Optional[SeriesFrame]: Data corresponding to the specified feature, or None if not found.
        """
        if checks.is_int(feature):
            return self.get(features=self.features[feature])
        feature_idx = self.get_feature_idx(feature, raise_error=raise_error)
        if feature_idx == -1:
            return None
        return self.get(features=self.features[feature_idx])

    def get_symbol(
        self,
        symbol: tp.Union[int, tp.Symbol],
        raise_error: bool = False,
    ) -> tp.Optional[tp.SeriesFrame]:
        """Retrieve data for a symbol by its index or label.

        Args:
            symbol (Union[int, Symbol]): Index or label of the symbol.
            raise_error (bool): Whether to raise an error if the symbol is not found.

        Returns:
            Optional[SeriesFrame]: Data corresponding to the specified symbol, or None if not found.
        """
        if checks.is_int(symbol):
            return self.get(symbol=self.symbols[symbol])
        symbol_idx = self.get_symbol_idx(symbol, raise_error=raise_error)
        if symbol_idx == -1:
            return None
        return self.get(symbol=self.symbols[symbol_idx])


OHLCDataMixinT = tp.TypeVar("OHLCDataMixinT", bound="OHLCDataMixin")


class OHLCDataMixin(BaseDataMixin):
    """Mixin class for handling OHLC data properties and calculations for financial time series."""

    @property
    def open(self) -> tp.Optional[tp.SeriesFrame]:
        """Series representing the open prices.

        Returns:
            Optional[SeriesFrame]: Open prices for the data if available; otherwise, None.
        """
        return self.get_feature("Open")

    @property
    def high(self) -> tp.Optional[tp.SeriesFrame]:
        """Series representing the high prices.

        Returns:
            Optional[SeriesFrame]: High prices for the data if available; otherwise, None.
        """
        return self.get_feature("High")

    @property
    def low(self) -> tp.Optional[tp.SeriesFrame]:
        """Series representing the low prices.

        Returns:
            Optional[SeriesFrame]: Low prices for the data if available; otherwise, None.
        """
        return self.get_feature("Low")

    @property
    def close(self) -> tp.Optional[tp.SeriesFrame]:
        """Series representing the close prices.

        Returns:
            Optional[SeriesFrame]: Close prices for the data if available; otherwise, None.
        """
        return self.get_feature("Close")

    @property
    def volume(self) -> tp.Optional[tp.SeriesFrame]:
        """Series representing the volume data.

        Returns:
            Optional[SeriesFrame]: Volume data for the data if available; otherwise, None.
        """
        return self.get_feature("Volume")

    @property
    def trade_count(self) -> tp.Optional[tp.SeriesFrame]:
        """Series representing the trade count.

        Returns:
            Optional[SeriesFrame]: Trade count for the data if available; otherwise, None.
        """
        return self.get_feature("Trade count")

    @property
    def vwap(self) -> tp.Optional[tp.SeriesFrame]:
        """Series representing the volume-weighted average price (VWAP) data.

        Returns:
            Optional[SeriesFrame]: Volume-weighted average price for the data if available; otherwise, None.
        """
        return self.get_feature("VWAP")

    @property
    def hlc3(self) -> tp.SeriesFrame:
        """Series computed as the arithmetic mean of the high, low, and close prices.

        Returns:
            SeriesFrame: Arithmetic mean of the high, low, and close prices.
        """
        high = self.get_feature("High", raise_error=True)
        low = self.get_feature("Low", raise_error=True)
        close = self.get_feature("Close", raise_error=True)
        return (high + low + close) / 3

    @property
    def ohlc4(self) -> tp.SeriesFrame:
        """Series computed as the arithmetic mean of the open, high, low, and close prices.

        Returns:
            SeriesFrame: Arithmetic mean of the open, high, low, and close prices.
        """
        open = self.get_feature("Open", raise_error=True)
        high = self.get_feature("High", raise_error=True)
        low = self.get_feature("Low", raise_error=True)
        close = self.get_feature("Close", raise_error=True)
        return (open + high + low + close) / 4

    @property
    def has_any_ohlc(self) -> bool:
        """Boolean flag indicating if any OHLC feature (open, high, low, or close) is present.

        Returns:
            bool: True if any OHLC feature is present, False otherwise.
        """
        return (
            self.has_feature("Open") or self.has_feature("High") or self.has_feature("Low") or self.has_feature("Close")
        )

    @property
    def has_ohlc(self) -> bool:
        """Boolean flag indicating if all OHLC features (open, high, low, and close) are present.

        Returns:
            bool: True if all OHLC features are present, False otherwise.
        """
        return (
            self.has_feature("Open")
            and self.has_feature("High")
            and self.has_feature("Low")
            and self.has_feature("Close")
        )

    @property
    def has_any_ohlcv(self) -> bool:
        """Boolean flag indicating if any OHLCV feature (OHLC or volume) is available.

        Returns:
            bool: True if any OHLCV feature is present, False otherwise.
        """
        return self.has_any_ohlc or self.has_feature("Volume")

    @property
    def has_ohlcv(self) -> bool:
        """Boolean flag indicating if all OHLCV features (open, high, low, close, and volume) are present.

        Returns:
            bool: True if all OHLCV features are present, False otherwise.
        """
        return self.has_ohlc and self.has_feature("Volume")

    @property
    def ohlc(self: OHLCDataMixinT) -> OHLCDataMixinT:
        """New `OHLCDataMixin` instance containing only OHLC features (open, high, low, and close).

        Returns:
            OHLCDataMixin: New instance containing only OHLC features.
        """
        open_idx = self.get_feature_idx("Open", raise_error=True)
        high_idx = self.get_feature_idx("High", raise_error=True)
        low_idx = self.get_feature_idx("Low", raise_error=True)
        close_idx = self.get_feature_idx("Close", raise_error=True)
        return self.select_feature_idxs([open_idx, high_idx, low_idx, close_idx])

    @property
    def ohlcv(self: OHLCDataMixinT) -> OHLCDataMixinT:
        """New `OHLCDataMixin` instance containing only OHLCV features (open, high, low, close, and volume).

        Returns:
            OHLCDataMixin: New instance containing only OHLCV features.
        """
        open_idx = self.get_feature_idx("Open", raise_error=True)
        high_idx = self.get_feature_idx("High", raise_error=True)
        low_idx = self.get_feature_idx("Low", raise_error=True)
        close_idx = self.get_feature_idx("Close", raise_error=True)
        volume_idx = self.get_feature_idx("Volume", raise_error=True)
        return self.select_feature_idxs([open_idx, high_idx, low_idx, close_idx, volume_idx])

    def get_returns_acc(self, **kwargs) -> ReturnsAccessor:
        """Return a `vectorbtpro.returns.accessors.ReturnsAccessor` constructed from the close price data.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`.

        Returns:
            ReturnsAccessor: Accessor for return calculations using the close price.
        """
        return ReturnsAccessor.from_value(
            self.get_feature("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=False,
            **kwargs,
        )

    @property
    def returns_acc(self) -> ReturnsAccessor:
        """`ReturnsAccessor` using default parameters from `OHLCDataMixin.get_returns_acc`.

        Returns:
            ReturnsAccessor: Accessor for return calculations using the close price.
        """
        return self.get_returns_acc()

    def get_returns(self, **kwargs) -> tp.SeriesFrame:
        """Return computed return values derived from the close price data.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`.

        Returns:
            SeriesFrame: Computed returns from the close price.
        """
        return ReturnsAccessor.from_value(
            self.get_feature("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=True,
            **kwargs,
        )

    @property
    def returns(self) -> tp.SeriesFrame:
        """Computed return values using default parameters from `OHLCDataMixin.get_returns`.

        Returns:
            SeriesFrame: Computed returns from the close price.
        """
        return self.get_returns()

    def get_log_returns(self, **kwargs) -> tp.SeriesFrame:
        """Return computed logarithmic return values derived from the close price data.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`.

        Returns:
            SeriesFrame: Computed logarithmic returns from the close price.
        """
        return ReturnsAccessor.from_value(
            self.get_feature("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=True,
            log_returns=True,
            **kwargs,
        )

    @property
    def log_returns(self) -> tp.SeriesFrame:
        """Logarithmic return values using default parameters from `OHLCDataMixin.get_log_returns`.

        Returns:
            SeriesFrame: Computed logarithmic returns from the close price.
        """
        return self.get_log_returns()

    def get_daily_returns(self, **kwargs) -> tp.SeriesFrame:
        """Return daily returns computed from the close price data.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`.

        Returns:
            SeriesFrame: Daily returns computed from the close price.
        """
        return ReturnsAccessor.from_value(
            self.get_feature("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=False,
            **kwargs,
        ).daily()

    @property
    def daily_returns(self) -> tp.SeriesFrame:
        """Daily computed returns using default parameters from `OHLCDataMixin.get_daily_returns`.

        Returns:
            SeriesFrame: Daily computed returns from the close price.
        """
        return self.get_daily_returns()

    def get_daily_log_returns(self, **kwargs) -> tp.SeriesFrame:
        """Return daily logarithmic returns computed from the close price data.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.returns.accessors.ReturnsAccessor.from_value`.

        Returns:
            SeriesFrame: Daily logarithmic returns computed from the close price.
        """
        return ReturnsAccessor.from_value(
            self.get_feature("Close", raise_error=True),
            wrapper=self.symbol_wrapper,
            return_values=False,
            log_returns=True,
            **kwargs,
        ).daily()

    @property
    def daily_log_returns(self) -> tp.SeriesFrame:
        """Daily computed logarithmic returns using default parameters from
        `OHLCDataMixin.get_daily_log_returns`.

        Returns:
            SeriesFrame: Daily computed logarithmic returns from the close price.
        """
        return self.get_daily_log_returns()

    def get_drawdowns(self, **kwargs) -> Drawdowns:
        """Generate drawdown records from the OHLC price data.

        Args:
            **kwargs: Keyword arguments for `vectorbtpro.generic.drawdowns.Drawdowns.from_price`.

        Returns:
            Drawdowns: Drawdown records generated using open, high, low, and close prices.
        """
        return Drawdowns.from_price(
            open=self.get_feature("Open", raise_error=True),
            high=self.get_feature("High", raise_error=True),
            low=self.get_feature("Low", raise_error=True),
            close=self.get_feature("Close", raise_error=True),
            **kwargs,
        )

    @property
    def drawdowns(self) -> Drawdowns:
        """Drawdown records using default parameters from `OHLCDataMixin.get_drawdowns`.

        Returns:
            Drawdowns: Drawdown records generated using open, high, low, and close prices.
        """
        return self.get_drawdowns()


DataT = tp.TypeVar("DataT", bound="Data")


class MetaData(type(Analyzable)):
    """Metaclass for the `Data` class, providing custom feature configuration."""

    @property
    def feature_config(cls) -> Config:
        """Feature configuration for the `Data` class.

        Returns:
            Config: Feature configuration associated with the class.
        """
        return cls._feature_config


@attach_symbol_dict_methods
class Data(Analyzable, OHLCDataMixin, metaclass=MetaData):
    """Class for downloading, updating, and managing data from a data source.

    Args:
        wrapper (ArrayWrapper): Array wrapper instance.

            See `vectorbtpro.base.wrapping.ArrayWrapper`.
        data (Union[feature_dict, symbol_dict]): Data dictionary structured as feature-oriented
            (`feature_dict`) or symbol-oriented (`symbol_dict`).
        single_key (bool): Specifies whether the instance should be treated as having a single key.

            Automatically set to False when multiple keys are present.
        classes (Union[None, feature_dict, symbol_dict]): Class definitions for keys.
        level_name (Union[None, bool, MaybeIterable[Hashable]]): Name(s) of levels for keys.
        fetch_kwargs (Union[None, feature_dict, symbol_dict]): Additional parameters for data fetching.
        returned_kwargs (Union[None, feature_dict, symbol_dict]): Keyword arguments returned from data fetching.
        last_index (Union[None, feature_dict, symbol_dict]): Container to record
            the last datetime index for each key.
        delisted (Union[None, feature_dict, symbol_dict]): Container to track
            delisted status for each key.
        tz_localize (Union[None, bool, TimezoneLike]): Flag or specification for timezone localization.

            Set to False to disable localization. See `Data.prepare_dt_index`.
        tz_convert (Union[None, bool, TimezoneLike]): Flag or specification for timezone conversion.

            Set to False to disable conversion. See `Data.prepare_dt_index`.
        missing_index (Optional[str]): Specifies how to handle missing indices when aligning data.

            See `Data.align_index`.
        missing_columns (Optional[str]): Specifies how to handle missing columns when aligning data.

            See `Data.align_columns`.
        **kwargs: Keyword arguments for `vectorbtpro.generic.analyzable.Analyzable`.

    !!! info
        For default settings, see `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(base="data")

    _writeable_attrs: tp.WriteableAttrs = {"_feature_config"}

    _feature_config: tp.ClassVar[Config] = HybridConfig()

    _key_dict_attrs = [
        "fetch_kwargs",
        "returned_kwargs",
        "last_index",
        "delisted",
        "classes",
    ]
    """Attributes that subclass either `feature_dict` or `symbol_dict`."""

    _data_dict_type_attrs = [
        "classes",
    ]
    """Attributes that subclass the data dict type."""

    _updatable_attrs = [
        "fetch_kwargs",
        "returned_kwargs",
        "classes",
    ]
    """Attributes that have a method for updating."""

    def __init__(
        self,
        wrapper: ArrayWrapper,
        data: tp.Union[feature_dict, symbol_dict],
        single_key: bool = True,
        classes: tp.Union[None, feature_dict, symbol_dict] = None,
        level_name: tp.Union[None, bool, tp.MaybeIterable[tp.Hashable]] = None,
        fetch_kwargs: tp.Union[None, feature_dict, symbol_dict] = None,
        returned_kwargs: tp.Union[None, feature_dict, symbol_dict] = None,
        last_index: tp.Union[None, feature_dict, symbol_dict] = None,
        delisted: tp.Union[None, feature_dict, symbol_dict] = None,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        **kwargs,
    ) -> None:
        Analyzable.__init__(
            self,
            wrapper,
            data=data,
            single_key=single_key,
            classes=classes,
            level_name=level_name,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            last_index=last_index,
            delisted=delisted,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            **kwargs,
        )

        if len(set(map(self.prepare_key, data.keys()))) < len(list(map(self.prepare_key, data.keys()))):
            raise ValueError("Found duplicate keys in data dictionary")
        data = self.fix_data_dict_type(data)
        for obj in data.values():
            checks.assert_meta_equal(obj, data[list(data.keys())[0]])
        if len(data) > 1:
            single_key = False

        self._data = data
        self._single_key = single_key
        self._level_name = level_name
        self._tz_localize = tz_localize
        self._tz_convert = tz_convert
        self._missing_index = missing_index
        self._missing_columns = missing_columns

        attr_kwargs = dict()
        for attr in self._key_dict_attrs:
            attr_value = locals()[attr]
            attr_kwargs[attr] = attr_value
        attr_kwargs = self.fix_dict_types_in_kwargs(type(data), **attr_kwargs)
        for k, v in attr_kwargs.items():
            setattr(self, "_" + k, v)

        # Copy writeable attrs
        self._feature_config = type(self)._feature_config.copy()

    @property
    def data(self) -> tp.Union[feature_dict, symbol_dict]:
        """Data dictionary.

        Either feature-oriented (`feature_dict`) or symbol-oriented (`symbol_dict`).

        Returns:
            Union[feature_dict, symbol_dict]: Data dictionary.
        """
        return self._data

    @property
    def dict_type(self) -> tp.Type[tp.Union[feature_dict, symbol_dict]]:
        """Data dictionary type.

        Indicates whether the data dictionary is feature-oriented (`feature_dict`) or
        symbol-oriented (`symbol_dict`).

        Returns:
            Type[Union[feature_dict, symbol_dict]]: Type of the data dictionary.
        """
        return type(self.data)

    @property
    def column_type(self) -> tp.Type[tp.Union[feature_dict, symbol_dict]]:
        """Column type.

        If the data is feature-oriented (`feature_dict`), returns `symbol_dict`;
        otherwise, returns `feature_dict`.

        Returns:
            Type[Union[feature_dict, symbol_dict]]: Column type.
        """
        if isinstance(self.data, feature_dict):
            return symbol_dict
        return feature_dict

    @property
    def feature_oriented(self) -> bool:
        """Feature-oriented flag.

        Indicates whether the data is feature-oriented, meaning keys represent features.

        Returns:
            bool: True if the data is feature-oriented, False otherwise.
        """
        return issubclass(self.dict_type, feature_dict)

    @property
    def symbol_oriented(self) -> bool:
        """Symbol-oriented flag.

        Indicates whether the data is symbol-oriented, meaning keys represent symbols.

        Returns:
            bool: True if the data is symbol-oriented, False otherwise.
        """
        return issubclass(self.dict_type, symbol_dict)

    def get_keys(self, dict_type: tp.Type[tp.Union[feature_dict, symbol_dict]]) -> tp.List[tp.Key]:
        """Return keys based on the provided dictionary type.

        Args:
            dict_type (Type[Union[feature_dict, symbol_dict]]): Data dictionary type used to
                determine which keys to return.

        Returns:
            List[Key]: List of keys corresponding to the specified dictionary type.
        """
        checks.assert_subclass_of(dict_type, (feature_dict, symbol_dict), arg_name="dict_type")
        if issubclass(dict_type, feature_dict):
            return self.features
        return self.symbols

    @property
    def keys(self) -> tp.List[tp.Union[tp.Feature, tp.Symbol]]:
        """List of keys in the data dictionary.

        These represent features for `feature_dict` data or symbols for `symbol_dict` data.

        Returns:
            List[Union[Feature, Symbol]]: List of keys in the data dictionary.
        """
        return list(self.data.keys())

    @property
    def single_key(self) -> bool:
        """Single key flag.

        Indicates whether the underlying data dictionary contains only one key.

        Returns:
            bool: True if the data dictionary has only one key, False otherwise.
        """
        return self._single_key

    @property
    def single_feature(self) -> bool:
        """Single feature flag.

        Indicates whether there is only one feature in the data. For feature-oriented data,
        this is equivalent to `single_key`; otherwise, it is determined by the dimensionality of the wrapper.

        Returns:
            bool: True if there is only one feature, False otherwise.
        """
        if self.feature_oriented:
            return self.single_key
        return self.wrapper.ndim == 1

    @property
    def single_symbol(self) -> bool:
        """Single symbol flag.

        Indicates whether there is only one symbol in the data. For symbol-oriented data,
        this equals `single_key`; otherwise, it is determined by the dimensionality of the wrapper.

        Returns:
            bool: True if there is only one symbol, False otherwise.
        """
        if self.symbol_oriented:
            return self.single_key
        return self.wrapper.ndim == 1

    @property
    def classes(self) -> tp.Union[feature_dict, symbol_dict]:
        """Class definitions for the keys in the data dictionary.

        Returns:
            Type[Union[feature_dict, symbol_dict]]: Type of the data dictionary.
        """
        return self._classes

    @property
    def feature_classes(self) -> tp.Optional[feature_dict]:
        """Feature classes.

        Returns:
            Optional[feature_dict]: Key classes if the data is feature-oriented; otherwise, returns None.
        """
        if self.feature_oriented:
            return self.classes
        return None

    @property
    def symbol_classes(self) -> tp.Optional[symbol_dict]:
        """Symbol classes.

        Returns:
            Optional[symbol_dict]: Key classes if the data is symbol-oriented; otherwise, returns None.
        """
        if self.symbol_oriented:
            return self.classes
        return None

    @hybrid_method
    def get_level_name(
        cls_or_self,
        keys: tp.Optional[tp.Keys] = None,
        level_name: tp.Union[None, bool, tp.MaybeIterable[tp.Hashable]] = None,
        feature_oriented: tp.Optional[bool] = None,
    ) -> tp.Optional[tp.MaybeIterable[tp.Hashable]]:
        """Determine level name(s) for data keys.

        Args:
            keys (Optional[Keys]): List of keys; required for class method calls.
            level_name (Union[None, bool, MaybeIterable[Hashable]]): Specification for level name(s).

                If a boolean, `False` returns None and `True` resets the level name.
            feature_oriented (Optional[bool]): Indicates whether the data is feature-oriented;
                required for class method calls.

        Returns:
            Optional[MaybeIterable[Hashable]]: Level name(s) for the keys.

                Returns a tuple when keys are multi-level.

        !!! note
            If `level_name` is boolean `False`, no level names are applied.
        """
        if isinstance(cls_or_self, type):
            checks.assert_not_none(keys, arg_name="keys")
            checks.assert_not_none(feature_oriented, arg_name="feature_oriented")
        else:
            if keys is None:
                keys = cls_or_self.keys
            if level_name is None:
                level_name = cls_or_self._level_name
            if feature_oriented is None:
                feature_oriented = cls_or_self.feature_oriented
        first_key = keys[0]
        if isinstance(level_name, bool):
            if level_name:
                level_name = None
            else:
                return None
        if feature_oriented:
            key_prefix = "feature"
        else:
            key_prefix = "symbol"
        if isinstance(first_key, tuple):
            if level_name is None:
                level_name = ["%s_%d" % (key_prefix, i) for i in range(len(first_key))]
            if not checks.is_iterable(level_name) or isinstance(level_name, str):
                raise TypeError("Level name should be list-like for a MultiIndex")
            return tuple(level_name)
        if level_name is None:
            level_name = key_prefix
        return level_name

    @property
    def level_name(self) -> tp.Optional[tp.MaybeIterable[tp.Hashable]]:
        """Level name(s) property.

        Specifies the name(s) for the keys in the data dictionary.
        For multi-level keys, it is a sequence of names; for single-level keys, it is a hashable.
        If set to `False`, no level names are used.

        Returns:
            Optional[MaybeIterable[Hashable]]: Level name(s) for the keys.
        """
        return self.get_level_name()

    @hybrid_method
    def get_key_index(
        cls_or_self,
        keys: tp.Optional[tp.Keys] = None,
        level_name: tp.Union[None, bool, tp.MaybeIterable[tp.Hashable]] = None,
        feature_oriented: tp.Optional[bool] = None,
    ) -> tp.Index:
        """Generate Pandas Index for the data keys.

        Args:
            keys (Optional[Keys]): List of keys; required for class method calls.
            level_name (Union[None, bool, MaybeIterable[Hashable]]): Specification for level name(s)
                to be used in the Index.
            feature_oriented (Optional[bool]): Indicates whether the data is feature-oriented;
                required for class method calls.

        Returns:
            Index: Pandas Index, or a MultiIndex if level names are provided as a tuple.
        """
        if isinstance(cls_or_self, type):
            checks.assert_not_none(keys, arg_name="keys")
        else:
            if keys is None:
                keys = cls_or_self.keys
        level_name = cls_or_self.get_level_name(keys=keys, level_name=level_name, feature_oriented=feature_oriented)
        if isinstance(level_name, tuple):
            return pd.MultiIndex.from_tuples(keys, names=level_name)
        return pd.Index(keys, name=level_name)

    @property
    def key_index(self) -> tp.Index:
        """Key index property.

        A Pandas Index generated from the data keys using `Data.get_key_index`.

        Returns:
            Index: Pandas Index, or a MultiIndex if level names are provided as a tuple.
        """
        return self.get_key_index()

    @property
    def fetch_kwargs(self) -> tp.Union[feature_dict, symbol_dict]:
        """Keyword arguments originally passed to `Data.fetch_symbol`, maintained as a `symbol_dict`.

        Returns:
            Union[feature_dict, symbol_dict]: Keyword arguments for fetching data.
        """
        return self._fetch_kwargs

    @property
    def returned_kwargs(self) -> tp.Union[feature_dict, symbol_dict]:
        """Keyword arguments returned by `Data.fetch_symbol`, stored as a `symbol_dict`.

        Returns:
            Union[feature_dict, symbol_dict]: Keyword arguments returned from fetching data.
        """
        return self._returned_kwargs

    @property
    def last_index(self) -> tp.Union[feature_dict, symbol_dict]:
        """Last fetched index for each symbol, maintained as a `symbol_dict`.

        Returns:
            Union[feature_dict, symbol_dict]: Last index for each symbol.
        """
        return self._last_index

    @property
    def delisted(self) -> tp.Union[feature_dict, symbol_dict]:
        """Delisting status for each symbol, stored as a `symbol_dict`.

        Returns:
            Union[feature_dict, symbol_dict]: Delisting status for each symbol.
        """
        return self._delisted

    @property
    def tz_localize(self) -> tp.Union[None, bool, tp.TimezoneLike]:
        """Timezone for localizing a datetime-naive index, initially provided to `Data.pull`.

        Returns:
            Union[None, bool, TimezoneLike]: Timezone for localizing the index.
        """
        return self._tz_localize

    @property
    def tz_convert(self) -> tp.Union[None, bool, tp.TimezoneLike]:
        """Timezone for converting a datetime-aware index, initially provided to `Data.pull`.

        Returns:
            Union[None, bool, TimezoneLike]: Timezone for converting the index.
        """
        return self._tz_convert

    @property
    def missing_index(self) -> tp.Optional[str]:
        """`missing` argument provided to `Data.align_index`.

        Returns:
            Optional[str]: Missing index argument.
        """
        return self._missing_index

    @property
    def missing_columns(self) -> tp.Optional[str]:
        """`missing` argument provided to `Data.align_columns`.

        Returns:
            Optional[str]: Missing columns argument.
        """
        return self._missing_columns

    # ############# Settings ############# #

    @classmethod
    def get_base_settings(cls, *args, **kwargs) -> dict:
        """Retrieve base settings using `CustomData.get_settings` with `path_id="base"`.

        Args:
            *args: Positional arguments for `Data.get_settings`.
            **kwargs: Keyword arguments for `Data.get_settings`.

        Returns:
            dict: Base settings.
        """
        return cls.get_settings(*args, path_id="base", **kwargs)

    @classmethod
    def has_base_settings(cls, *args, **kwargs) -> bool:
        """Determine whether base settings exist by invoking `CustomData.has_settings` with `path_id="base"`.

        Args:
            *args: Positional arguments for `Data.has_settings`.
            **kwargs: Keyword arguments for `Data.has_settings`.

        Returns:
            bool: True if base settings exist, False otherwise.
        """
        return cls.has_settings(*args, path_id="base", **kwargs)

    @classmethod
    def get_base_setting(cls, *args, **kwargs) -> tp.Any:
        """Fetch a base setting using `CustomData.get_setting` with `path_id="base"`.

        Args:
            *args: Positional arguments for `Data.get_setting`.
            **kwargs: Keyword arguments for `Data.get_setting`.

        Returns:
            Any: Requested base setting.
        """
        return cls.get_setting(*args, path_id="base", **kwargs)

    @classmethod
    def has_base_setting(cls, *args, **kwargs) -> bool:
        """Check for the presence of a base setting by using `CustomData.has_setting` with `path_id="base"`.

        Args:
            *args: Positional arguments for `Data.has_setting`.
            **kwargs: Keyword arguments for `Data.has_setting`.

        Returns:
            bool: True if the base setting exists, False otherwise.
        """
        return cls.has_setting(*args, path_id="base", **kwargs)

    @classmethod
    def resolve_base_setting(cls, *args, **kwargs) -> tp.Any:
        """Resolve a base setting by calling `CustomData.resolve_setting` with `path_id="base"`.

        Args:
            *args: Positional arguments for `Data.resolve_setting`.
            **kwargs: Keyword arguments for `Data.resolve_setting`.

        Returns:
            Any: Resolved base setting.
        """
        return cls.resolve_setting(*args, path_id="base", **kwargs)

    @classmethod
    def set_base_settings(cls, *args, **kwargs) -> None:
        """Apply base settings by invoking `CustomData.set_settings` with `path_id="base"`.

        Args:
            *args: Positional arguments for `Data.set_settings`.
            **kwargs: Keyword arguments for `Data.set_settings`.

        Returns:
            None
        """
        cls.set_settings(*args, path_id="base", **kwargs)

    # ############# Config ############# #

    @property
    def feature_config(self) -> Config:
        """Feature configuration for `${cls_name}`.

        ```python
        ${feature_config}
        ```

        This property returns `${cls_name}._feature_config`, which is (hybrid-) copied during instance
        creation. Modifying this configuration does not affect the class-level configuration.

        To change fields, modify the configuration in-place, override this property, or replace the
        instance variable `${cls_name}._feature_config`.

        Returns:
            Config: Feature configuration for the class.
        """
        return self._feature_config

    def use_feature_config_of(self, cls: tp.Type[DataT]) -> None:
        """Copy the feature configuration from another `Data` class.

        Args:
            cls (Type[Data]): Class from which to copy the feature configuration.

        Returns:
            None
        """
        self._feature_config = cls.feature_config.copy()

    @classmethod
    def modify_state(cls, rec_state: RecState) -> RecState:
        # Ensure backward compatibility
        if "_column_config" in rec_state.attr_dct and "_feature_config" not in rec_state.attr_dct:
            new_attr_dct = dict(rec_state.attr_dct)
            new_attr_dct["_feature_config"] = new_attr_dct.pop("_column_config")
            rec_state = RecState(
                init_args=rec_state.init_args,
                init_kwargs=rec_state.init_kwargs,
                attr_dct=new_attr_dct,
            )
        if "single_symbol" in rec_state.init_kwargs and "single_key" not in rec_state.init_kwargs:
            new_init_kwargs = dict(rec_state.init_kwargs)
            new_init_kwargs["single_key"] = new_init_kwargs.pop("single_symbol")
            rec_state = RecState(
                init_args=rec_state.init_args,
                init_kwargs=new_init_kwargs,
                attr_dct=rec_state.attr_dct,
            )
        if "symbol_classes" in rec_state.init_kwargs and "classes" not in rec_state.init_kwargs:
            new_init_kwargs = dict(rec_state.init_kwargs)
            new_init_kwargs["classes"] = new_init_kwargs.pop("symbol_classes")
            rec_state = RecState(
                init_args=rec_state.init_args,
                init_kwargs=new_init_kwargs,
                attr_dct=rec_state.attr_dct,
            )
        return rec_state

    @classmethod
    def fix_data_dict_type(cls, data: dict) -> tp.Union[feature_dict, symbol_dict]:
        """Ensure that the data dictionary conforms to the proper key dictionary type.

        If `data` is not an instance of `key_dict`, convert it using `symbol_dict`.

        Args:
            data (dict): Data dictionary.

        Returns:
            Union[feature_dict, symbol_dict]: Wrapped data dictionary.
        """
        checks.assert_instance_of(data, dict, arg_name="data")
        if not isinstance(data, key_dict):
            data = symbol_dict(data)
        return data

    @classmethod
    def fix_dict_types_in_kwargs(
        cls,
        data_type: tp.Type[tp.Union[feature_dict, symbol_dict]],
        **kwargs: tp.Kwargs,
    ) -> tp.Kwargs:
        """Adjust dictionary-type keyword arguments to conform to the specified `data_type`.

        For each attribute in `_key_dict_attrs` present in `kwargs`, replace None with an empty
        dictionary and convert the value to the proper type if necessary.

        Args:
            data_type (Type[Union[feature_dict, symbol_dict]]): Data dictionary type.
            **kwargs: Keyword arguments.

        Returns:
            Kwargs: Adjusted keyword arguments.
        """
        for attr in cls._key_dict_attrs:
            if attr in kwargs:
                attr_value = kwargs[attr]
                if attr_value is None:
                    attr_value = {}
                checks.assert_instance_of(attr_value, dict, arg_name=attr)
                if not isinstance(attr_value, key_dict):
                    attr_value = data_type(attr_value)
                if attr in cls._data_dict_type_attrs:
                    checks.assert_instance_of(attr_value, data_type, arg_name=attr)
                kwargs[attr] = attr_value
        return kwargs

    @hybrid_method
    def row_stack(
        cls_or_self: tp.MaybeType[DataT],
        *objs: tp.MaybeSequence[DataT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Stack multiple `Data` instances along rows.

        This method uses `vectorbtpro.base.wrapping.ArrayWrapper.row_stack` to combine
        the wrappers from each instance, validates that all instances share consistent data
        keys and dictionary types, and merges their data along with related attributes.

        Args:
            *objs (MaybeSequence[Data]): (Additional) `Data` instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `Data`.

        Returns:
            Data: New `Data` instance created by stacking the given instances.
        """
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Data):
                raise TypeError("Each object to be merged must be an instance of Data")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.row_stack(*[obj.wrapper for obj in objs], **wrapper_kwargs)

        keys = set()
        for obj in objs:
            keys = keys.union(set(obj.data.keys()))
        data_type = None
        for obj in objs:
            if len(keys.difference(set(obj.data.keys()))) > 0:
                if isinstance(obj.data, feature_dict):
                    raise ValueError("Objects to be merged must have the same features")
                else:
                    raise ValueError("Objects to be merged must have the same symbols")
            if data_type is None:
                data_type = type(obj.data)
            elif not isinstance(obj.data, data_type):
                raise TypeError("Objects to be merged must have the same dict type for data")
        if "data" not in kwargs:
            new_data = data_type()
            for k in objs[0].data.keys():
                new_data[k] = kwargs["wrapper"].row_stack_arrs(*[obj.data[k] for obj in objs], group_by=False)
            kwargs["data"] = new_data
        kwargs["data"] = cls.fix_data_dict_type(kwargs["data"])
        for attr in cls._key_dict_attrs:
            if attr not in kwargs:
                attr_data_type = None
                for obj in objs:
                    v = getattr(obj, attr)
                    if attr_data_type is None:
                        attr_data_type = type(v)
                    elif not isinstance(v, attr_data_type):
                        raise TypeError(f"Objects to be merged must have the same dict type for '{attr}'")
                kwargs[attr] = getattr(objs[-1], attr)

        kwargs = cls.resolve_row_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        kwargs = cls.fix_dict_types_in_kwargs(type(kwargs["data"]), **kwargs)
        return cls(**kwargs)

    @hybrid_method
    def column_stack(
        cls_or_self: tp.MaybeType[DataT],
        *objs: tp.MaybeSequence[DataT],
        wrapper_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Stack multiple `Data` instances along columns.

        This method concatenates multiple `Data` instances by stacking their underlying arrays
        along the column axis. It relies on `vectorbtpro.base.wrapping.ArrayWrapper.column_stack`
        to merge the wrappers.

        Args:
            *objs (MaybeSequence[Data]): (Additional) `Data` instances to stack.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            **kwargs: Keyword arguments for `Data`.

        Returns:
            Data: New `Data` instance created by stacking the given instances.
        """
        if not isinstance(cls_or_self, type):
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, Data):
                raise TypeError("Each object to be merged must be an instance of Data")
        if "wrapper" not in kwargs:
            if wrapper_kwargs is None:
                wrapper_kwargs = {}
            kwargs["wrapper"] = ArrayWrapper.column_stack(
                *[obj.wrapper for obj in objs],
                **wrapper_kwargs,
            )

        keys = set()
        for obj in objs:
            keys = keys.union(set(obj.data.keys()))
        data_type = None
        for obj in objs:
            if len(keys.difference(set(obj.data.keys()))) > 0:
                if isinstance(obj.data, feature_dict):
                    raise ValueError("Objects to be merged must have the same features")
                else:
                    raise ValueError("Objects to be merged must have the same symbols")
            if data_type is None:
                data_type = type(obj.data)
            elif not isinstance(obj.data, data_type):
                raise TypeError("Objects to be merged must have the same dict type for data")
        if "data" not in kwargs:
            new_data = data_type()
            for k in objs[0].data.keys():
                new_data[k] = kwargs["wrapper"].column_stack_arrs(*[obj.data[k] for obj in objs], group_by=False)
            kwargs["data"] = new_data
        kwargs["data"] = cls.fix_data_dict_type(kwargs["data"])
        for attr in cls._key_dict_attrs:
            if attr not in kwargs:
                attr_data_type = None
                for obj in objs:
                    v = getattr(obj, attr)
                    if attr_data_type is None:
                        attr_data_type = type(v)
                    elif not isinstance(v, attr_data_type):
                        raise TypeError(f"Objects to be merged must have the same dict type for '{attr}'")
                if (issubclass(data_type, feature_dict) and issubclass(attr_data_type, symbol_dict)) or (
                    issubclass(data_type, symbol_dict) and issubclass(attr_data_type, feature_dict)
                ):
                    kwargs[attr] = attr_data_type()
                    for obj in objs:
                        kwargs[attr].update(**getattr(obj, attr))

        kwargs = cls.resolve_column_stack_kwargs(*objs, **kwargs)
        kwargs = cls.resolve_stack_kwargs(*objs, **kwargs)
        kwargs = cls.fix_dict_types_in_kwargs(type(kwargs["data"]), **kwargs)
        return cls(**kwargs)

    def replace(self: DataT, **kwargs) -> DataT:
        if "wrapper" in kwargs and "data" not in kwargs:
            wrapper = kwargs["wrapper"]
            if isinstance(wrapper, dict):
                new_index = wrapper.get("index", self.wrapper.index)
                new_columns = wrapper.get("columns", self.wrapper.columns)
            else:
                new_index = wrapper.index
                new_columns = wrapper.columns
            data = self.config["data"]
            new_data = {}
            index_changed = False
            columns_changed = False
            for k, v in data.items():
                if isinstance(v, (pd.Series, pd.DataFrame)):
                    if not checks.is_index_equal(v.index, new_index):
                        v = v.copy(deep=False)
                        v.index = new_index
                        index_changed = True
                    if isinstance(v, pd.DataFrame):
                        if not checks.is_index_equal(v.columns, new_columns):
                            v = v.copy(deep=False)
                            v.columns = new_columns
                            columns_changed = True
                new_data[k] = v
            if index_changed or columns_changed:
                kwargs["data"] = self.fix_data_dict_type(new_data)
                if columns_changed:
                    rename = dict(zip(self.keys, new_columns))
                    for attr in self._key_dict_attrs:
                        if attr not in kwargs:
                            attr_value = getattr(self, attr)
                            if (self.feature_oriented and isinstance(attr_value, symbol_dict)) or (
                                self.symbol_oriented and isinstance(attr_value, feature_dict)
                            ):
                                kwargs[attr] = self.rename_in_dict(getattr(self, attr), rename)

        kwargs = self.fix_dict_types_in_kwargs(type(kwargs.get("data", self.data)), **kwargs)
        return Analyzable.replace(self, **kwargs)

    def indexing_func(self: DataT, *args, replace_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Perform indexing on the `Data` instance.

        This method applies positional and keyword arguments to index the underlying data.
        It retrieves indexing metadata via `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`,
        applies the specified indexing to each element in the data, and then constructs a new
        `Data` instance with an updated wrapper and modified data.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.
            replace_kwargs (KwargsLike): Keyword arguments for `Data.replace`.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.indexing_func_meta`.

        Returns:
            Data: New `Data` instance after indexing.
        """
        if replace_kwargs is None:
            replace_kwargs = {}
        wrapper_meta = self.wrapper.indexing_func_meta(*args, **kwargs)
        new_wrapper = wrapper_meta["new_wrapper"]
        new_data = self.dict_type()
        for k, v in self._data.items():
            if wrapper_meta["rows_changed"]:
                v = v.iloc[wrapper_meta["row_idxs"]]
            if wrapper_meta["columns_changed"]:
                v = v.iloc[:, wrapper_meta["col_idxs"]]
            new_data[k] = v
        attr_dicts = dict()
        attr_dicts["last_index"] = type(self.last_index)()
        for k in self.last_index:
            attr_dicts["last_index"][k] = min([self.last_index[k], new_wrapper.index[-1]])
        if wrapper_meta["columns_changed"]:
            new_symbols = new_wrapper.columns
            for attr in self._key_dict_attrs:
                attr_value = getattr(self, attr)
                if (self.feature_oriented and isinstance(attr_value, symbol_dict)) or (
                    self.symbol_oriented and isinstance(attr_value, feature_dict)
                ):
                    if attr in attr_dicts:
                        attr_dicts[attr] = self.select_from_dict(attr_dicts[attr], new_symbols)
                    else:
                        attr_dicts[attr] = self.select_from_dict(attr_value, new_symbols)
        return self.replace(wrapper=new_wrapper, data=new_data, **attr_dicts, **replace_kwargs)

    # ############# Iteration ############# #

    def items(
        self,
        over: str = "symbols",
        group_by: tp.GroupByLike = None,
        apply_group_by: bool = False,
        keep_2d: bool = False,
        key_as_index: bool = False,
    ) -> tp.Items:
        """Iterate over specific aspects of the `Data` instance.

        Depending on the value of `over`, this method iterates over different components of the data:

        * If `over` is "columns", "symbols", or "features" (depending on data orientation),
            it yields pairs of key and corresponding `Data` subsets.
        * If `over` is "keys", it iterates over keys with optional grouping.

        Args:
            over (str): Iteration mode.

                Allowed values include "columns", "symbols", "features", or "keys".
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            apply_group_by (bool): If True, applies the grouping to both iteration and the final output.

                If False, `group_by` is used solely as an iteration instruction.

                !!! note
                    Grouping is not supported when iterating over keys.
            keep_2d (bool): Whether to maintain the output data in a two-dimensional format.
            key_as_index (bool): Whether to return the yielded key as an index.

        Returns:
            Items: Iterator yielding pairs of key and corresponding `Data` subsets.
        """
        if (
            over.lower() == "columns"
            or (over.lower() == "symbols" and self.feature_oriented)
            or (over.lower() == "features" and self.symbol_oriented)
        ):
            for k, v in Analyzable.items(
                self,
                group_by=group_by,
                apply_group_by=apply_group_by,
                keep_2d=keep_2d,
                key_as_index=key_as_index,
            ):
                yield k, v
        elif (
            over.lower() == "keys"
            or (over.lower() == "features" and self.feature_oriented)
            or (over.lower() == "symbols" and self.symbol_oriented)
        ):
            if apply_group_by:
                raise ValueError("Cannot apply grouping to keys")
            if group_by is not None:
                key_wrapper = self.get_key_wrapper(group_by=group_by)
                if key_wrapper.get_ndim() == 1:
                    if key_as_index:
                        yield key_wrapper.get_columns(), self
                    else:
                        yield key_wrapper.get_columns()[0], self
                else:
                    for group, group_idxs in key_wrapper.grouper.iter_groups(key_as_index=key_as_index):
                        if keep_2d or len(group_idxs) > 1:
                            yield group, self.select_keys([self.keys[i] for i in group_idxs])
                        else:
                            yield group, self.select_keys(self.keys[group_idxs[0]])
            else:
                key_wrapper = self.get_key_wrapper(attach_classes=False)
                if key_wrapper.ndim == 1:
                    if key_as_index:
                        yield key_wrapper.columns, self
                    else:
                        yield key_wrapper.columns[0], self
                else:
                    for i in range(len(key_wrapper.columns)):
                        if key_as_index:
                            key = key_wrapper.columns[[i]]
                        else:
                            key = key_wrapper.columns[i]
                        if keep_2d:
                            yield key, self.select_keys([key])
                        else:
                            yield key, self.select_keys(key)
        else:
            raise ValueError(f"Invalid over: '{over}'")

    # ############# Getting ############# #

    def get_key_wrapper(
        self,
        keys: tp.Optional[tp.MaybeKeys] = None,
        attach_classes: bool = True,
        clean_index_kwargs: tp.KwargsLike = None,
        group_by: tp.GroupByLike = None,
        **kwargs,
    ) -> ArrayWrapper:
        """Return a new `vectorbtpro.base.wrapping.ArrayWrapper` instance with keys as columns.

        If `attach_classes` is True, this method stacks `Data.classes` over the keys using
        `vectorbtpro.base.indexes.stack_indexes`.

        Args:
            keys (Optional[MaybeKeys]): Feature or symbol identifier(s) to use as columns.

                If None, defaults to the object's keys.
            attach_classes (bool): Whether to attach classes from `Data.classes`.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.
            group_by (GroupByLike): Grouping specification.

                See `vectorbtpro.base.grouping.base.Grouper`.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.replace`.

        Returns:
            ArrayWrapper: New array wrapper updated with the specified keys and dimensions.
        """
        if clean_index_kwargs is None:
            clean_index_kwargs = {}
        if keys is None:
            keys = self.keys
            ndim = 1 if self.single_key else 2
        else:
            if self.has_multiple_keys(keys):
                ndim = 2
            else:
                keys = [keys]
                ndim = 1
            for key in keys:
                if self.feature_oriented:
                    self.assert_has_feature(key)
                else:
                    self.assert_has_symbol(key)
        new_columns = self.get_key_index(keys=keys)
        wrapper = self.wrapper.replace(
            columns=new_columns,
            ndim=ndim,
            grouper=None,
            **kwargs,
        )

        if attach_classes:
            classes = []
            all_have_classes = True
            for key in wrapper.columns:
                if key in self.classes:
                    key_classes = self.classes[key]
                    if len(key_classes) > 0:
                        classes.append(key_classes)
                    else:
                        all_have_classes = False
                else:
                    all_have_classes = False
            if len(classes) > 0 and not all_have_classes:
                if self.feature_oriented:
                    raise ValueError("Some features have classes while others not")
                else:
                    raise ValueError("Some symbols have classes while others not")
            if len(classes) > 0:
                classes_frame = pd.DataFrame(classes)
                if len(classes_frame.columns) == 1:
                    classes_columns = pd.Index(classes_frame.iloc[:, 0])
                else:
                    classes_columns = pd.MultiIndex.from_frame(classes_frame)
                new_columns = stack_indexes((classes_columns, wrapper.columns), **clean_index_kwargs)
                wrapper = wrapper.replace(columns=new_columns)
        if group_by is not None:
            wrapper = wrapper.replace(group_by=group_by)
        return wrapper

    @cached_property
    def key_wrapper(self) -> ArrayWrapper:
        """Key-based array wrapper.

        This property returns an `vectorbtpro.base.wrapping.ArrayWrapper` instance with columns
        determined by the object's keys via `get_key_wrapper`.

        Returns:
            ArrayWrapper: `vectorbtpro.base.wrapping.ArrayWrapper` instance with keys as columns.
        """
        return self.get_key_wrapper()

    def get_feature_wrapper(self, features: tp.Optional[tp.MaybeFeatures] = None, **kwargs) -> ArrayWrapper:
        """Return an array wrapper with features as columns.

        Args:
            features (Optional[MaybeFeatures]): Feature identifier(s) to use as columns.

                If None, defaults to a full selection.
            **kwargs: Keyword arguments for `Data.get_key_wrapper` when applicable.

        Returns:
            ArrayWrapper: `vectorbtpro.base.wrapping.ArrayWrapper` instance updated to have features as columns.
        """
        if self.feature_oriented:
            return self.get_key_wrapper(keys=features, **kwargs)
        wrapper = self.wrapper
        if features is not None:
            wrapper = wrapper[features]
        return wrapper

    @cached_property
    def feature_wrapper(self) -> ArrayWrapper:
        return self.get_feature_wrapper()

    def get_symbol_wrapper(self, symbols: tp.Optional[tp.MaybeSymbols] = None, **kwargs) -> ArrayWrapper:
        """Return an array wrapper with symbols as columns.

        Args:
            symbols (Optional[MaybeSymbols]): Symbol identifier(s) to use as columns.

                If None, defaults to a full selection.
            **kwargs: Keyword arguments for `Data.get_key_wrapper` when applicable.

        Returns:
            ArrayWrapper: `vectorbtpro.base.wrapping.ArrayWrapper` instance updated to have symbols as columns.
        """
        if self.symbol_oriented:
            return self.get_key_wrapper(keys=symbols, **kwargs)
        wrapper = self.wrapper
        if symbols is not None:
            wrapper = wrapper[symbols]
        return wrapper

    @cached_property
    def symbol_wrapper(self) -> ArrayWrapper:
        return self.get_symbol_wrapper()

    @property
    def ndim(self) -> int:
        """Number of dimensions based on the default symbol wrapper.

        Returns:
            int: Number of dimensions of the default symbol wrapper.
        """
        return self.symbol_wrapper.ndim

    @property
    def shape(self) -> tp.Shape:
        """Shape determined from the default symbol wrapper.

        Returns:
            Shape: Shape of the default symbol wrapper.
        """
        return self.symbol_wrapper.shape

    @property
    def shape_2d(self) -> tp.Shape:
        """Shape as if the object were two-dimensional, based on the default symbol wrapper.

        Returns:
            Shape: Two-dimensional shape of the default symbol wrapper.
        """
        return self.symbol_wrapper.shape_2d

    @property
    def columns(self) -> tp.Index:
        """Column index based on the default symbol wrapper.

        Returns:
            Index: Column index of the default symbol wrapper.
        """
        return self.symbol_wrapper.columns

    @property
    def index(self) -> tp.Index:
        """Index based on the default symbol wrapper.

        Returns:
            Index: Index of the default symbol wrapper.
        """
        return self.symbol_wrapper.index

    @property
    def freq(self) -> tp.Optional[tp.PandasFrequency]:
        """Frequency based on the default symbol wrapper.

        Returns:
            Optional[PandasFrequency]: Frequency of the default symbol wrapper.
        """
        return self.symbol_wrapper.freq

    @property
    def features(self) -> tp.List[tp.Feature]:
        if self.feature_oriented:
            return self.keys
        return self.wrapper.columns.tolist()

    @property
    def symbols(self) -> tp.List[tp.Symbol]:
        if self.feature_oriented:
            return self.wrapper.columns.tolist()
        return self.keys

    def resolve_feature(self, feature: tp.Feature, raise_error: bool = False) -> tp.Optional[tp.Feature]:
        """Return the feature from the instance that matches the provided feature.

        Args:
            feature (Feature): Feature identifier.
            raise_error (bool): Whether to raise an error if the feature is not found.

        Returns:
            Optional[Feature]: Matching feature if found; otherwise, None.
        """
        feature_idx = self.get_feature_idx(feature, raise_error=raise_error)
        if feature_idx == -1:
            return None
        return self.features[feature_idx]

    def resolve_symbol(self, symbol: tp.Symbol, raise_error: bool = False) -> tp.Optional[tp.Symbol]:
        """Return the symbol from the instance that matches the provided symbol.

        Args:
            symbol (Symbol): Symbol identifier.
            raise_error (bool): Whether to raise an error if the symbol is not found.

        Returns:
            Optional[Symbol]: Matching symbol if found; otherwise, None.
        """
        symbol_idx = self.get_symbol_idx(symbol, raise_error=raise_error)
        if symbol_idx == -1:
            return None
        return self.symbols[symbol_idx]

    def resolve_key(self, key: tp.Key, raise_error: bool = False) -> tp.Optional[tp.Key]:
        """Return the key from the instance that matches the provided key.

        If the instance is feature-oriented, the key is resolved as a feature;
        otherwise, it is resolved as a symbol.

        Args:
            key (Key): Feature or symbol identifier.
            raise_error (bool): Whether to raise an error if the key is not found.

        Returns:
            Optional[Key]: Matching key if found; otherwise, None.
        """
        if self.feature_oriented:
            return self.resolve_feature(key, raise_error=raise_error)
        return self.resolve_symbol(key, raise_error=raise_error)

    def resolve_column(self, column: tp.Column, raise_error: bool = False) -> tp.Optional[tp.Column]:
        """Return the column from the instance that matches the provided column.

        If the instance is feature-oriented, the column is resolved as a symbol;
        otherwise, it is resolved as a feature.

        Args:
            column (Column): Column identifier.
            raise_error (bool): Whether to raise an error if the column is not found.

        Returns:
            Optional[Column]: Matching column if found; otherwise, None.
        """
        if self.feature_oriented:
            return self.resolve_symbol(column, raise_error=raise_error)
        return self.resolve_feature(column, raise_error=raise_error)

    def resolve_features(self, features: tp.MaybeFeatures, raise_error: bool = True) -> tp.MaybeFeatures:
        """Return the feature(s) from the instance that match the provided input.

        If a single feature is provided, a single feature is returned;
        otherwise, a list of features is returned.

        Args:
            features (MaybeFeatures): Feature identifier(s).
            raise_error (bool): Whether to raise an error if a feature is not found.

        Returns:
            MaybeFeatures: Resolved feature(s), either as a single feature or a list of features.
        """
        if not self.has_multiple_keys(features):
            features = [features]
            single_feature = True
        else:
            single_feature = False
        new_features = []
        for feature in features:
            new_features.append(self.resolve_feature(feature, raise_error=raise_error))
        if single_feature:
            return new_features[0]
        return new_features

    def resolve_symbols(self, symbols: tp.MaybeSymbols, raise_error: bool = True) -> tp.MaybeSymbols:
        """Return the symbol(s) from the instance that match the provided input.

        Args:
            symbols (MaybeSymbols): Symbol identifier(s).
            raise_error (bool): Whether to raise an error if a symbol is not found.

        Returns:
            MaybeSymbols: Resolved symbol(s), either as a single symbol or a list of symbols.
        """
        if not self.has_multiple_keys(symbols):
            symbols = [symbols]
            single_symbol = True
        else:
            single_symbol = False
        new_symbols = []
        for symbol in symbols:
            new_symbols.append(self.resolve_symbol(symbol, raise_error=raise_error))
        if single_symbol:
            return new_symbols[0]
        return new_symbols

    def resolve_keys(self, keys: tp.MaybeKeys, raise_error: bool = True) -> tp.MaybeKeys:
        """Return the key(s) from the instance that match the provided input.

        For a feature-oriented instance, keys are resolved as features; otherwise,
        they are resolved as symbols.

        Args:
            keys (MaybeKeys): Feature or symbol identifier(s).
            raise_error (bool): Whether to raise an error if a key is not found.

        Returns:
            MaybeKeys: Resolved key(s), either as a single key or a list of keys.
        """
        if self.feature_oriented:
            return self.resolve_features(keys, raise_error=raise_error)
        return self.resolve_symbols(keys, raise_error=raise_error)

    def resolve_columns(self, columns: tp.MaybeColumns, raise_error: bool = True) -> tp.MaybeColumns:
        """Return the column(s) from the instance that match the provided input.

        If the instance is feature-oriented, columns are resolved as symbols;
        otherwise, they are resolved as features.

        Args:
            columns (MaybeColumns): Column identifier(s).
            raise_error (bool): Whether to raise an error if a column is not found.

        Returns:
            MaybeColumns: Resolved column(s), either as a single column or a list of columns.
        """
        if self.feature_oriented:
            return self.resolve_symbols(columns, raise_error=raise_error)
        return self.resolve_features(columns, raise_error=raise_error)

    def concat(
        self,
        keys: tp.Optional[tp.Symbols] = None,
        attach_classes: bool = True,
        clean_index_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[feature_dict, symbol_dict]:
        """Concatenate data associated with the specified keys along columns.

        Args:
            keys (Optional[Symbols]): Keys to select the data for concatenation.

                If not provided, uses `Data.keys`.
            attach_classes (bool): Whether to attach classes to the data using the key wrapper.
            clean_index_kwargs (KwargsLike): Keyword arguments for cleaning MultiIndex levels.

                See `vectorbtpro.base.indexes.clean_index`.
            **kwargs: Keyword arguments for `Data.get_key_wrapper`.

        Returns:
            Union[feature_dict, symbol_dict]: Dictionary-like structure containing the concatenated data.
        """
        key_wrapper = self.get_key_wrapper(
            keys=keys,
            attach_classes=attach_classes,
            clean_index_kwargs=clean_index_kwargs,
            **kwargs,
        )
        if keys is None:
            keys = self.keys

        new_data = self.column_type()
        first_data = self.data[keys[0]]
        if key_wrapper.ndim == 1:
            if isinstance(first_data, pd.Series):
                new_data[first_data.name] = key_wrapper.wrap(first_data.values, zero_to_none=False)
            else:
                for c in first_data.columns:
                    new_data[c] = key_wrapper.wrap(first_data[c].values, zero_to_none=False)
        else:
            if isinstance(first_data, pd.Series):
                columns = pd.Index([first_data.name])
            else:
                columns = first_data.columns
            for c in columns:
                col_data = []
                for k in keys:
                    if isinstance(self.data[k], pd.Series):
                        col_data.append(self.data[k].values)
                    else:
                        col_data.append(self.data[k][c].values)
                new_data[c] = key_wrapper.wrap(column_stack_arrays(col_data), zero_to_none=False)

        return new_data

    def get(
        self,
        features: tp.Optional[tp.MaybeFeatures] = None,
        symbols: tp.Optional[tp.MaybeSymbols] = None,
        feature: tp.Optional[tp.Feature] = None,
        symbol: tp.Optional[tp.Symbol] = None,
        squeeze_features: bool = False,
        squeeze_symbols: bool = False,
        per: str = "feature",
        as_dict: bool = False,
        **kwargs,
    ) -> tp.Union[tp.MaybeTuple[tp.SeriesFrame], dict]:
        """Retrieve one or more features and symbols from the data.

        Args:
            features (Optional[MaybeFeatures]): Feature identifier(s).

                If not provided, uses `Data.features`.
            symbols (Optional[MaybeSymbols]): Symbol identifier(s).

                If not provided, uses `Data.symbols`.
            feature (Optional[Feature]): Feature identifier.

                Cannot be used with `features`.
            symbol (Optional[Symbol]): Symbol identifier.

                Cannot be used with `symbols`.
            squeeze_features (bool): Whether to squeeze the features when only one element is present.
            squeeze_symbols (bool): Whether to squeeze the symbols when only one element is present.
            per (str): Grouping specification for data concatenation,
                such as "feature", "symbol", "column", or "key".
            as_dict (bool): If True, returns the data as a dictionary mapping keys to the data.
            **kwargs: Keyword arguments for `Data.concat`.

        Returns:
            Union[MaybeTuple[SeriesFrame], dict]: Retrieved data as a tuple or dictionary
                based on the provided parameters.
        """
        if features is not None and feature is not None:
            raise ValueError("Must provide either features or feature, not both")
        if symbols is not None and symbol is not None:
            raise ValueError("Must provide either symbols or symbol, not both")

        if feature is not None:
            features = feature
            single_feature = True
        else:
            if features is None:
                features = self.features
                single_feature = self.single_feature
                if single_feature:
                    features = features[0]
            else:
                single_feature = not self.has_multiple_keys(features)
            if not single_feature and squeeze_features and len(features) == 1:
                features = features[0]
                single_feature = True
        if symbol is not None:
            symbols = symbol
            single_symbol = True
        else:
            if symbols is None:
                symbols = self.symbols
                single_symbol = self.single_symbol
                if single_symbol:
                    symbols = symbols[0]
            else:
                single_symbol = not self.has_multiple_keys(symbols)
            if not single_symbol and squeeze_symbols and len(symbols) == 1:
                symbols = symbols[0]
                single_symbol = True

        if not single_feature:
            feature_idxs = [self.get_feature_idx(k, raise_error=True) for k in features]
            features = [self.features[i] for i in feature_idxs]
        else:
            feature_idxs = self.get_feature_idx(features, raise_error=True)
            features = self.features[feature_idxs]
        if not single_symbol:
            symbol_idxs = [self.get_symbol_idx(k, raise_error=True) for k in symbols]
            symbols = [self.symbols[i] for i in symbol_idxs]
        else:
            symbol_idxs = self.get_symbol_idx(symbols, raise_error=True)
            symbols = self.symbols[symbol_idxs]

        def _get_objs():
            if self.feature_oriented:
                if single_feature:
                    if self.single_symbol:
                        return list(self.data.values())[feature_idxs], features
                    return list(self.data.values())[feature_idxs].iloc[:, symbol_idxs], features
                if single_symbol:
                    concat_data = self.concat(keys=features, **kwargs)
                    return list(concat_data.values())[symbol_idxs], symbols
                if per.lower() in ("symbol", "column"):
                    concat_data = self.concat(keys=features, **kwargs)
                    return tuple([list(concat_data.values())[i] for i in symbol_idxs]), symbols
                if per.lower() in ("feature", "key"):
                    if self.single_feature:
                        if self.single_symbol:
                            return list(self.data.values())[feature_idxs], features
                        return list(self.data.values())[feature_idxs].iloc[:, symbol_idxs], features
                    if self.single_symbol:
                        return tuple([list(self.data.values())[i] for i in feature_idxs]), features
                    return tuple([list(self.data.values())[i].iloc[:, symbol_idxs] for i in feature_idxs]), features
                raise ValueError(f"Invalid per: '{per}'")
            else:
                if single_symbol:
                    if self.single_feature:
                        return self.data[self.symbols[symbol_idxs]], symbols
                    return self.data[self.symbols[symbol_idxs]].iloc[:, feature_idxs], symbols
                if single_feature:
                    concat_data = self.concat(keys=symbols, **kwargs)
                    return list(concat_data.values())[feature_idxs], features
                if per.lower() in ("feature", "column"):
                    concat_data = self.concat(keys=symbols, **kwargs)
                    return tuple([list(concat_data.values())[i] for i in feature_idxs]), features
                if per.lower() in ("symbol", "key"):
                    if self.single_symbol:
                        if self.single_feature:
                            return list(self.data.values())[symbol_idxs], symbols
                        return list(self.data.values())[symbol_idxs].iloc[:, feature_idxs], symbols
                    if self.single_feature:
                        return tuple([list(self.data.values())[i] for i in symbol_idxs]), symbols
                    return tuple([list(self.data.values())[i].iloc[:, feature_idxs] for i in symbol_idxs]), symbols
                raise ValueError(f"Invalid per: '{per}'")

        objs, keys = _get_objs()
        if as_dict:
            if isinstance(objs, tuple):
                return dict(zip(keys, objs))
            return {keys: objs}
        return objs

    # ############# Pre- and post-processing ############# #

    @classmethod
    def prepare_dt_index(
        cls,
        index: tp.Index,
        parse_dates: bool = False,
        tz_localize: tp.TimezoneLike = None,
        tz_convert: tp.TimezoneLike = None,
        force_tz_convert: bool = False,
        remove_tz: bool = False,
    ) -> tp.SeriesFrame:
        """Prepare a datetime index.

        Args:
            index (Index): Index to be processed.
            parse_dates (bool): If True, convert to a datetime index using `vectorbtpro.utils.datetime_.prepare_dt_index`.
            tz_localize (TimezoneLike): Timezone to localize a datetime-naive index.
            tz_convert (TimezoneLike): Timezone to convert a datetime-aware index.
            force_tz_convert (bool): If True, convert the timezone even if the index is not timezone-aware.
            remove_tz (bool): Whether to remove timezone information from the index.

        Returns:
            SeriesFrame: Processed datetime index.
        """
        if parse_dates:
            if not isinstance(index, (pd.DatetimeIndex, pd.MultiIndex)) and index.dtype == object:
                index = dt.prepare_dt_index(index)
        if isinstance(index, pd.DatetimeIndex):
            if index.tz is None and tz_localize is not None:
                index = index.tz_localize(dt.to_timezone(tz_localize))
            if tz_convert is not None:
                if index.tz is not None or force_tz_convert:
                    index = index.tz_convert(dt.to_timezone(tz_convert))
            if remove_tz and index.tz is not None:
                index = index.tz_localize(None)
        return index

    @classmethod
    def prepare_dt_column(
        cls,
        sr: tp.Series,
        parse_dates: bool = False,
        tz_localize: tp.TimezoneLike = None,
        tz_convert: tp.TimezoneLike = None,
        force_tz_convert: bool = False,
        remove_tz: bool = False,
    ) -> tp.Series:
        """Prepare a datetime column.

        Args:
            sr (Series): Series to be processed.
            parse_dates (bool): If True, convert to a datetime index using `vectorbtpro.utils.datetime_.prepare_dt_index`.
            tz_localize (TimezoneLike): Timezone to localize a datetime-naive index.
            tz_convert (TimezoneLike): Timezone to convert a datetime-aware index.
            force_tz_convert (bool): If True, convert the timezone even if the index is not timezone-aware.
            remove_tz (bool): Whether to remove timezone information from the index.

        Returns:
            Series: Series with its index converted to a datetime index if applicable,
                or the original series.
        """
        index = cls.prepare_dt_index(
            pd.Index(sr),
            parse_dates=parse_dates,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            force_tz_convert=force_tz_convert,
            remove_tz=remove_tz,
        )
        if isinstance(index, pd.DatetimeIndex):
            return pd.Series(index, index=sr.index, name=sr.name)
        return sr

    @classmethod
    def prepare_dt(
        cls,
        obj: tp.SeriesFrame,
        parse_dates: tp.Union[None, bool, tp.Sequence[str]] = True,
        to_utc: tp.Union[None, bool, str, tp.Sequence[str]] = True,
        remove_utc_tz: bool = False,
    ) -> tp.Frame:
        """Prepare datetime index and columns.

        Args:
            obj (SeriesFrame): Pandas Series or DataFrame.
            parse_dates (Union[None, bool, Sequence[str]]): Specifies whether to parse dates.

                If True, converts any index or column with an object data type to datetime.
                If a sequence of strings, converts only fields whose name is in the sequence.
            to_utc (Union[None, bool, str, Sequence[str]]): Specifies whether to localize or convert
                datetime fields to UTC.

                If True, converts both index and columns.
                If a string equal to "index" or "columns", converts only the corresponding field.
                If a sequence, converts fields whose names are in the sequence.
            remove_utc_tz (bool): Indicates whether to remove the timezone after converting to UTC.

        Returns:
            Frame: Pandas DataFrame or Series with prepared datetime indices and columns.
        """
        obj = obj.copy(deep=False)
        made_frame = False
        if isinstance(obj, pd.Series):
            obj = obj.to_frame()
            made_frame = True

        index_parse_dates = False
        if not isinstance(obj.index, pd.MultiIndex) and obj.index.dtype == object:
            if parse_dates is True:
                index_parse_dates = True
            elif checks.is_sequence(parse_dates) and obj.index.name in parse_dates:
                index_parse_dates = True
        if (
            to_utc is True
            or (isinstance(to_utc, str) and to_utc.lower() == "index")
            or (checks.is_sequence(to_utc) and obj.index.name in to_utc)
        ):
            index_tz_localize = "utc"
            index_tz_convert = "utc"
            index_remove_tz = remove_utc_tz
        else:
            index_tz_localize = None
            index_tz_convert = None
            index_remove_tz = False
        obj.index = cls.prepare_dt_index(
            obj.index,
            parse_dates=index_parse_dates,
            tz_localize=index_tz_localize,
            tz_convert=index_tz_convert,
            remove_tz=index_remove_tz,
        )

        for column_name in obj.columns:
            column_parse_dates = False
            if obj[column_name].dtype == object:
                if parse_dates is True:
                    column_parse_dates = True
                elif checks.is_sequence(parse_dates) and column_name in parse_dates:
                    column_parse_dates = True
            elif not hasattr(obj[column_name], "dt"):
                continue
            if (
                to_utc is True
                or (isinstance(to_utc, str) and to_utc.lower() == "columns")
                or (checks.is_sequence(to_utc) and column_name in to_utc)
            ):
                column_tz_localize = "utc"
                column_tz_convert = "utc"
                column_remove_tz = remove_utc_tz
            else:
                column_tz_localize = None
                column_tz_convert = None
                column_remove_tz = False
            obj[column_name] = cls.prepare_dt_column(
                obj[column_name],
                parse_dates=column_parse_dates,
                tz_localize=column_tz_localize,
                tz_convert=column_tz_convert,
                remove_tz=column_remove_tz,
            )

        if made_frame:
            obj = obj.iloc[:, 0]
        return obj

    @classmethod
    def prepare_tzaware_index(
        cls,
        obj: tp.SeriesFrame,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
    ) -> tp.SeriesFrame:
        """Prepare a timezone-aware index for a Pandas object.

        Args:
            obj (SeriesFrame): Pandas Series or DataFrame.
            tz_localize (Union[None, bool, TimezoneLike]): Flag or specification for timezone localization.

                Set to False to disable localization. See `Data.prepare_dt_index`.
            tz_convert (Union[None, bool, TimezoneLike]): Flag or specification for timezone conversion.

                Set to False to disable conversion. See `Data.prepare_dt_index`.

        Returns:
            SeriesFrame: Object with a timezone-aware index.

        !!! info
            For default settings, see `vectorbtpro._settings.data`.
        """
        obj = obj.copy(deep=False)
        tz_localize = cls.resolve_base_setting(tz_localize, "tz_localize")
        if isinstance(tz_localize, bool):
            if tz_localize:
                raise ValueError("tz_localize cannot be True")
            else:
                tz_localize = None
        tz_convert = cls.resolve_base_setting(tz_convert, "tz_convert")
        if isinstance(tz_convert, bool):
            if tz_convert:
                raise ValueError("tz_convert cannot be True")
            else:
                tz_convert = None
        obj.index = cls.prepare_dt_index(
            obj.index,
            parse_dates=True,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            force_tz_convert=True,
        )
        return obj

    @classmethod
    def align_index(
        cls,
        data: dict,
        missing: tp.Optional[str] = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> dict:
        """Align data to share a common index.

        Args:
            data (dict): Data dictionary.
            missing (Optional[str]): Specifies how to handle missing indices when aligning data.

                * 'nan': Set missing data points to NaN.
                * 'drop': Remove missing data points.
                * 'raise': Raise an error.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            dict: Dictionary with all data reindexed to a common index.

        !!! info
            For default settings, see `vectorbtpro._settings.data`.
        """
        missing = cls.resolve_base_setting(missing, "missing_index")
        silence_warnings = cls.resolve_base_setting(silence_warnings, "silence_warnings")

        index = None
        index_changed = False
        for k, obj in data.items():
            if index is None:
                index = obj.index
            else:
                if not checks.is_index_equal(index, obj.index, check_names=False):
                    if missing == "nan":
                        if not silence_warnings:
                            warn("Symbols have mismatching index. Setting missing data points to NaN.")
                        index = index.union(obj.index)
                        index_changed = True
                    elif missing == "drop":
                        if not silence_warnings:
                            warn("Symbols have mismatching index. Dropping missing data points.")
                        index = index.intersection(obj.index)
                        index_changed = True
                    elif missing == "raise":
                        raise ValueError("Symbols have mismatching index")
                    else:
                        raise ValueError(f"Invalid missing: '{missing}'")

        if not index_changed:
            return data
        new_data = {k: obj.reindex(index=index) for k, obj in data.items()}
        return type(data)(new_data)

    @classmethod
    def align_columns(
        cls,
        data: dict,
        missing: tp.Optional[str] = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> dict:
        """Align data to share a common set of columns.

        Args:
            data (dict): Data dictionary.
            missing (Optional[str]): Specifies how to handle missing columns when aligning data.

                * 'nan': Set missing data points to NaN.
                * 'drop': Remove missing data points.
                * 'raise': Raise an error.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            dict: Dictionary with all data reindexed to a common set of columns.
        """
        if len(data) == 1:
            return data

        missing = cls.resolve_base_setting(missing, "missing_columns")
        silence_warnings = cls.resolve_base_setting(silence_warnings, "silence_warnings")

        columns = None
        multiple_columns = False
        name_is_none = False
        columns_changed = False
        for k, obj in data.items():
            if isinstance(obj, pd.Series):
                if obj.name is None:
                    name_is_none = True
                obj = obj.to_frame()
            else:
                multiple_columns = True
            obj_columns = obj.columns
            if columns is None:
                columns = obj_columns
            else:
                if not checks.is_index_equal(columns, obj_columns, check_names=False):
                    if missing == "nan":
                        if not silence_warnings:
                            warn("Symbols have mismatching columns. Setting missing data points to NaN.")
                        columns = columns.union(obj_columns)
                        columns_changed = True
                    elif missing == "drop":
                        if not silence_warnings:
                            warn("Symbols have mismatching columns. Dropping missing data points.")
                        columns = columns.intersection(obj_columns)
                        columns_changed = True
                    elif missing == "raise":
                        raise ValueError("Symbols have mismatching columns")
                    else:
                        raise ValueError(f"Invalid missing: '{missing}'")

        if not columns_changed:
            return data
        new_data = {}
        for k, obj in data.items():
            if isinstance(obj, pd.Series):
                obj = obj.to_frame()
            obj = obj.reindex(columns=columns)
            if not multiple_columns:
                obj = obj[columns[0]]
                if name_is_none:
                    obj = obj.rename(None)
            new_data[k] = obj
        return type(data)(new_data)

    def switch_class(
        self,
        new_cls: tp.Type[DataT],
        clear_fetch_kwargs: bool = False,
        clear_returned_kwargs: bool = False,
        **kwargs,
    ) -> DataT:
        """Switch the class of the data instance.

        Args:
            new_cls (Type[Data]): New class to assign to the data instance.
            clear_fetch_kwargs (bool): If True, reset fetch keyword arguments for each symbol.
            clear_returned_kwargs (bool): If True, reset returned keyword arguments for each symbol.
            **kwargs: Keyword arguments for `Data.replace`.

        Returns:
            Data: New instance with the updated class.
        """
        if clear_fetch_kwargs:
            new_fetch_kwargs = type(self.fetch_kwargs)({k: {} for k in self.symbols})
        else:
            new_fetch_kwargs = copy_dict(self.fetch_kwargs)
        if clear_returned_kwargs:
            new_returned_kwargs = type(self.returned_kwargs)({k: {} for k in self.symbols})
        else:
            new_returned_kwargs = copy_dict(self.returned_kwargs)
        return self.replace(
            cls_=new_cls,
            fetch_kwargs=new_fetch_kwargs,
            returned_kwargs=new_returned_kwargs,
            **kwargs,
        )

    @classmethod
    def invert_data(cls, dct: tp.Dict[tp.Key, tp.SeriesFrame]) -> tp.Dict[tp.Key, tp.SeriesFrame]:
        """Invert data by swapping keys and columns.

        Args:
            dct (Dict[Key, SeriesFrame]): Dictionary mapping keys to Pandas Series or DataFrame.

        Returns:
            Dict[Key, SeriesFrame]: Inverted dictionary where keys and columns are swapped.
                If `dct` is an instance of `symbol_dict`, returns a `feature_dict`;
                if `dct` is an instance of `feature_dict`, returns a `symbol_dict`.
        """
        if len(dct) == 0:
            return dct
        new_dct = dict()
        for k, v in dct.items():
            if isinstance(v, pd.Series):
                if v.name not in new_dct:
                    new_dct[v.name] = []
                new_dct[v.name].append(v.rename(k))
            else:
                for c in v.columns:
                    if c not in new_dct:
                        new_dct[c] = []
                    new_dct[c].append(v[c].rename(k))
        new_dct2 = {}
        for k, v in new_dct.items():
            if len(v) == 1:
                new_dct2[k] = v[0]
            else:
                new_dct2[k] = pd.concat(v, axis=1)

        if isinstance(dct, symbol_dict):
            return feature_dict(new_dct2)
        if isinstance(dct, feature_dict):
            return symbol_dict(new_dct2)
        return new_dct2

    @hybrid_method
    def align_data(
        cls_or_self,
        data: dict,
        last_index: tp.Union[None, feature_dict, symbol_dict] = None,
        delisted: tp.Union[None, feature_dict, symbol_dict] = None,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> dict:
        """Align data by preparing indices and columns.

        Removes duplicate indices, prepares timezone-aware datetime indices,
        and aligns both indices and columns.

        Args:
            data (dict): Data dictionary.
            last_index (Union[None, feature_dict, symbol_dict]): Container to record
                the last datetime index for each key.
            delisted (Union[None, feature_dict, symbol_dict]): Container to track
                delisted status for each key.
            tz_localize (Union[None, bool, TimezoneLike]): Flag or specification for timezone localization.

                Set to False to disable localization. See `Data.prepare_dt_index`.
            tz_convert (Union[None, bool, TimezoneLike]): Flag or specification for timezone conversion.

                Set to False to disable conversion. See `Data.prepare_dt_index`.
            missing_index (Optional[str]): Specifies how to handle missing indices when aligning data.

                See `Data.align_index`.
            missing_columns (Optional[str]): Specifies how to handle missing columns when aligning data.

                See `Data.align_columns`.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

        Returns:
            dict: Aligned data dictionary with cleaned and ordered indices and columns.
        """
        if last_index is None:
            last_index = {}
        if delisted is None:
            delisted = {}
        if tz_localize is None and not isinstance(cls_or_self, type):
            tz_localize = cls_or_self.tz_localize
        if tz_convert is None and not isinstance(cls_or_self, type):
            tz_convert = cls_or_self.tz_convert
        if missing_index is None and not isinstance(cls_or_self, type):
            missing_index = cls_or_self.missing_index
        if missing_columns is None and not isinstance(cls_or_self, type):
            missing_columns = cls_or_self.missing_columns

        data = type(data)(data)
        for k, obj in data.items():
            obj = to_pd_array(obj)
            obj = cls_or_self.prepare_tzaware_index(obj, tz_localize=tz_localize, tz_convert=tz_convert)
            if obj.index.is_monotonic_decreasing:
                obj = obj.iloc[::-1]
            elif not obj.index.is_monotonic_increasing:
                obj = obj.sort_index()
            if obj.index.has_duplicates:
                obj = obj[~obj.index.duplicated(keep="last")]
            data[k] = obj
            if (isinstance(data, symbol_dict) and isinstance(last_index, symbol_dict)) or (
                isinstance(data, feature_dict) and isinstance(last_index, feature_dict)
            ):
                if k not in last_index:
                    last_index[k] = obj.index[-1]
            if (isinstance(data, symbol_dict) and isinstance(delisted, symbol_dict)) or (
                isinstance(data, feature_dict) and isinstance(delisted, feature_dict)
            ):
                if k not in delisted:
                    delisted[k] = False

        data = cls_or_self.align_index(data, missing=missing_index, silence_warnings=silence_warnings)
        data = cls_or_self.align_columns(data, missing=missing_columns, silence_warnings=silence_warnings)

        first_data = data[list(data.keys())[0]]
        if isinstance(first_data, pd.Series):
            columns = [first_data.name]
        else:
            columns = first_data.columns
        for k in columns:
            if (isinstance(data, symbol_dict) and isinstance(last_index, feature_dict)) or (
                isinstance(data, feature_dict) and isinstance(last_index, symbol_dict)
            ):
                if k not in last_index:
                    last_index[k] = first_data.index[-1]
            if (isinstance(data, symbol_dict) and isinstance(delisted, feature_dict)) or (
                isinstance(data, feature_dict) and isinstance(delisted, symbol_dict)
            ):
                if k not in delisted:
                    delisted[k] = False
        for obj in data.values():
            if isinstance(obj.index, pd.DatetimeIndex):
                obj.index.freq = obj.index.inferred_freq

        return data

    @classmethod
    def from_data(
        cls: tp.Type[DataT],
        data: tp.Union[dict, tp.SeriesFrame],
        columns_are_symbols: bool = False,
        invert_data: bool = False,
        single_key: bool = True,
        classes: tp.Optional[dict] = None,
        level_name: tp.Union[None, bool, tp.MaybeIterable[tp.Hashable]] = None,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        fetch_kwargs: tp.Optional[dict] = None,
        returned_kwargs: tp.Optional[dict] = None,
        last_index: tp.Optional[dict] = None,
        delisted: tp.Optional[dict] = None,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance from provided data.

        Args:
            data (Union[dict, SeriesFrame]): Dictionary or DataFrame/Series used to construct the instance.
            columns_are_symbols (bool): Flag indicating whether the columns represent symbols.
            invert_data (bool): Determines if the data dictionary should be inverted using `Data.invert_data`.
            single_key (bool): Specifies whether the instance should be treated as having a single key.

                Automatically set to False when multiple keys are present.
            classes (Optional[dict]): Mapping defining classes for data handling.

                See `Data.classes`.
            level_name (Union[None, bool, MaybeIterable[Hashable]]): Level name(s) for indexing.

                See `Data.level_name`.
            tz_localize (Union[None, bool, TimezoneLike]): Flag or specification for timezone localization.

                Set to False to disable localization. See `Data.prepare_dt_index`.
            tz_convert (Union[None, bool, TimezoneLike]): Flag or specification for timezone conversion.

                Set to False to disable conversion. See `Data.prepare_dt_index`.
            missing_index (Optional[str]): Specifies how to handle missing indices when aligning data.

                See `Data.align_index`.
            missing_columns (Optional[str]): Specifies how to handle missing columns when aligning data.

                See `Data.align_columns`.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            fetch_kwargs (Optional[dict]): Additional keyword arguments
                initially passed to `Data.fetch_symbol`.
            returned_kwargs (Optional[dict]): Additional keyword arguments
                returned by `Data.fetch_symbol`.
            last_index (Optional[dict]): Last fetched index per symbol.
            delisted (Optional[dict]): Indicator dictionary for delisted symbols.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.
            **kwargs: Keyword arguments for `Data.replace`.

        Returns:
            Data: New `Data` instance constructed from the provided data.

        !!! info
            For default settings, see `vectorbtpro._settings.data`.
        """
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if classes is None:
            classes = {}
        if fetch_kwargs is None:
            fetch_kwargs = {}
        if returned_kwargs is None:
            returned_kwargs = {}
        if last_index is None:
            last_index = {}
        if delisted is None:
            delisted = {}

        if columns_are_symbols and isinstance(data, symbol_dict):
            raise TypeError("Data cannot have the type symbol_dict when columns_are_symbols=True")
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if columns_are_symbols:
                data = feature_dict(feature=data)
            else:
                data = symbol_dict(symbol=data)
        checks.assert_instance_of(data, dict, arg_name="data")
        if not isinstance(data, key_dict):
            if columns_are_symbols:
                data = feature_dict(data)
            else:
                data = symbol_dict(data)
        if invert_data:
            data = cls.invert_data(data)
        if len(data) > 1:
            single_key = False
        checks.assert_instance_of(last_index, dict, arg_name="last_index")
        if not isinstance(last_index, key_dict):
            last_index = type(data)(last_index)
        checks.assert_instance_of(delisted, dict, arg_name="delisted")
        if not isinstance(delisted, key_dict):
            delisted = type(data)(delisted)

        data = cls.align_data(
            data,
            last_index=last_index,
            delisted=delisted,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            silence_warnings=silence_warnings,
        )
        first_data = data[list(data.keys())[0]]
        wrapper = ArrayWrapper.from_obj(first_data, **wrapper_kwargs)
        attr_dicts = cls.fix_dict_types_in_kwargs(
            type(data),
            classes=classes,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            last_index=last_index,
            delisted=delisted,
        )
        return cls(
            wrapper,
            data,
            single_key=single_key,
            level_name=level_name,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            **attr_dicts,
            **kwargs,
        )

    def invert(self: DataT, key_wrapper_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Invert the data and return a new instance.

        Args:
            key_wrapper_kwargs (KwargsLike): Keyword arguments for `Data.get_key_wrapper`.
            **kwargs: Keyword arguments for `Data.replace` for instance configuration.

        Returns:
            Data: New instance with the inverted data.
        """
        if key_wrapper_kwargs is None:
            key_wrapper_kwargs = {}
        new_data = self.concat(attach_classes=False)
        if "wrapper" not in kwargs:
            kwargs["wrapper"] = self.get_key_wrapper(**key_wrapper_kwargs)
        if "classes" not in kwargs:
            kwargs["classes"] = self.column_type()
        if "single_key" not in kwargs:
            kwargs["single_key"] = self.wrapper.ndim == 1
        if "level_name" not in kwargs:
            if isinstance(self.wrapper.columns, pd.MultiIndex):
                if self.wrapper.columns.names == [None] * self.wrapper.columns.nlevels:
                    kwargs["level_name"] = False
                else:
                    kwargs["level_name"] = self.wrapper.columns.names
            else:
                if self.wrapper.columns.name is None:
                    kwargs["level_name"] = False
                else:
                    kwargs["level_name"] = self.wrapper.columns.name
        return self.replace(data=new_data, **kwargs)

    def to_feature_oriented(self: DataT, **kwargs) -> DataT:
        """Convert the instance to the feature-oriented format.

        Args:
            **kwargs: Keyword arguments for `Data.invert`.

        Returns:
            Data: Instance in feature-oriented format.
        """
        if self.feature_oriented:
            if len(kwargs) > 0:
                return self.replace(**kwargs)
            return self
        return self.invert(**kwargs)

    def to_symbol_oriented(self: DataT, **kwargs) -> DataT:
        """Convert the instance to the symbol-oriented format.

        Args:
            **kwargs: Keyword arguments for `Data.invert`.

        Returns:
            Data: Instance in symbol-oriented format.
        """
        if self.symbol_oriented:
            if len(kwargs) > 0:
                return self.replace(**kwargs)
            return self
        return self.invert(**kwargs)

    @classmethod
    def has_key_dict(
        cls,
        arg: tp.Any,
        dict_type: tp.Optional[tp.Type[tp.Union[feature_dict, symbol_dict]]] = None,
    ) -> bool:
        """Check if the provided argument contains any data dictionary.

        Args:
            arg (Any): Argument to check for a data dictionary.
            dict_type (Optional[Type[Union[feature_dict, symbol_dict]]]): Dictionary type to validate against.

                Defaults to `key_dict` if not specified.

        Returns:
            bool: True if the argument contains a data dictionary, False otherwise.
        """
        if dict_type is None:
            dict_type = key_dict
        if isinstance(arg, dict_type):
            return True
        if isinstance(arg, dict):
            for k, v in arg.items():
                if isinstance(v, dict_type):
                    return True
        return False

    @hybrid_method
    def check_dict_type(
        cls_or_self,
        arg: tp.Any,
        arg_name: tp.Optional[str] = None,
        dict_type: tp.Optional[tp.Type[tp.Union[feature_dict, symbol_dict]]] = None,
    ) -> None:
        """Check if the provided argument conforms to a data dictionary type.

        Args:
            arg (Any): Argument to be validated.
            arg_name (Optional[str]): Name of the argument for error messaging.
            dict_type (Optional[Type[Union[feature_dict, symbol_dict]]]): Dictionary type to validate against.

                Defaults to `Data.dict_type` if not specified.

        Returns:
            None
        """
        if isinstance(cls_or_self, type):
            checks.assert_not_none(dict_type, arg_name="dict_type")
        if dict_type is None:
            dict_type = cls_or_self.dict_type
        if issubclass(dict_type, feature_dict):
            checks.assert_not_instance_of(arg, symbol_dict, arg_name=arg_name)
        if issubclass(dict_type, symbol_dict):
            checks.assert_not_instance_of(arg, feature_dict, arg_name=arg_name)

    @hybrid_method
    def select_key_kwargs(
        cls_or_self,
        key: tp.Key,
        kwargs: tp.KwargsLike,
        kwargs_name: str = "kwargs",
        dict_type: tp.Optional[tp.Type[tp.Union[feature_dict, symbol_dict]]] = None,
        check_dict_type: bool = True,
    ) -> tp.Kwargs:
        """Select the keyword arguments corresponding to a given key from a collection.

        Args:
            key (Key): Feature or symbol identifier.
            kwargs (KwargsLike): Dictionary of keyword arguments.
            kwargs_name (str): Name of the keyword arguments dictionary.
            dict_type (Optional[Type[Union[feature_dict, symbol_dict]]]): Dictionary type to validate against.

                Defaults to `Data.dict_type` if not specified.
            check_dict_type (bool): Flag to validate the type of dictionaries.

        Returns:
            Kwargs: Dictionary containing the filtered keyword arguments.
        """
        if isinstance(cls_or_self, type):
            checks.assert_not_none(dict_type, arg_name="dict_type")
        if dict_type is None:
            dict_type = cls_or_self.dict_type
        if kwargs is None:
            return {}
        if check_dict_type:
            cls_or_self.check_dict_type(kwargs, arg_name=kwargs_name, dict_type=dict_type)
        if type(kwargs) is key_dict or isinstance(kwargs, dict_type):
            if key not in kwargs:
                return {}
            kwargs = dict(kwargs[key])
        _kwargs = {}
        for k, v in kwargs.items():
            if check_dict_type:
                cls_or_self.check_dict_type(v, arg_name=f"{kwargs_name}[{k}]", dict_type=dict_type)
            if type(v) is key_dict or isinstance(v, dict_type):
                if key in v:
                    _kwargs[k] = v[key]
            else:
                _kwargs[k] = v
        return _kwargs

    @classmethod
    def select_feature_kwargs(cls, feature: tp.Feature, kwargs: tp.KwargsLike, **kwargs_) -> tp.Kwargs:
        """Select the keyword arguments associated with a feature.

        Args:
            feature (Feature): Feature identifier.
            kwargs (KwargsLike): Dictionary of keyword arguments.
            **kwargs_: Keyword arguments for `Data.select_key_kwargs`.

        Returns:
            Kwargs: Dictionary containing the filtered keyword arguments.
        """
        return cls.select_key_kwargs(feature, kwargs, dict_type=feature_dict, **kwargs_)

    @classmethod
    def select_symbol_kwargs(cls, symbol: tp.Symbol, kwargs: tp.KwargsLike, **kwargs_) -> tp.Kwargs:
        """Select the keyword arguments associated with a symbol.

        Args:
            symbol (Symbol): Symbol identifier.
            kwargs (KwargsLike): Dictionary of keyword arguments.
            **kwargs_: Keyword arguments for `Data.select_key_kwargs`.

        Returns:
            Kwargs: Dictionary containing the filtered keyword arguments.
        """
        return cls.select_key_kwargs(symbol, kwargs, dict_type=symbol_dict, **kwargs_)

    @hybrid_method
    def select_key_from_dict(
        cls_or_self,
        key: tp.Key,
        dct: key_dict,
        dct_name: str = "dct",
        dict_type: tp.Optional[tp.Type[tp.Union[feature_dict, symbol_dict]]] = None,
        check_dict_type: bool = True,
    ) -> tp.Any:
        """Select the value corresponding to a specified key in a dictionary.

        Args:
            key (Key): Feature or symbol identifier.
            dct (key_dict): Dictionary to search for the key.
            dct_name (str): Name of the dictionary.
            dict_type (Optional[Type[Union[feature_dict, symbol_dict]]]): Dictionary type to validate against.

                Defaults to `Data.dict_type` if not specified.
            check_dict_type (bool): Flag to validate the type of dictionaries.

        Returns:
            Any: Value associated with the specified key.
        """
        if isinstance(cls_or_self, type):
            checks.assert_not_none(dict_type, arg_name="dict_type")
        if dict_type is None:
            dict_type = cls_or_self.dict_type
        if check_dict_type:
            cls_or_self.check_dict_type(dct, arg_name=dct_name, dict_type=dict_type)
        return dct[key]

    @classmethod
    def select_feature_from_dict(cls, feature: tp.Feature, dct: feature_dict, **kwargs) -> tp.Any:
        """Select the value associated with a feature in the given dictionary.

        Args:
            feature (Feature): Feature identifier.
            dct (feature_dict): Dictionary containing feature values.
            **kwargs: Keyword arguments for `Data.select_key_kwargs`.

        Returns:
            Any: Value corresponding to the feature.
        """
        return cls.select_key_kwargs(feature, dct, dict_type=feature_dict, **kwargs)

    @classmethod
    def select_symbol_from_dict(cls, symbol: tp.Symbol, dct: symbol_dict, **kwargs) -> tp.Any:
        """Select the value associated with a symbol in the given dictionary.

        Args:
            symbol (Symbol): Symbol identifier.
            dct (symbol_dict): Dictionary containing symbol values.
            **kwargs: Keyword arguments for `Data.select_key_kwargs`.

        Returns:
            Any: Value corresponding to the symbol.
        """
        return cls.select_key_kwargs(symbol, dct, dict_type=symbol_dict, **kwargs)

    @classmethod
    def select_from_dict(cls, dct: dict, keys: tp.Keys, raise_error: bool = False) -> dict:
        """Create a new dictionary by selecting specified keys from the given dictionary.

        Args:
            dct (dict): Source dictionary.
            keys (Keys): Collection of keys to select.
            raise_error (bool): If True, raises an error when a key is missing.

        Returns:
            dict: Dictionary containing the selected keys and their corresponding values.
        """
        if raise_error:
            return type(dct)({k: dct[k] for k in keys})
        return type(dct)({k: dct[k] for k in keys if k in dct})

    @classmethod
    def get_intersection_dict(cls, dct: dict) -> dict:
        """Retrieve a dictionary of sub-keys and values that are common and identical across all entries.

        Args:
            dct (dict): Source dictionary containing nested dictionaries as values.

        Returns:
            dict: Dictionary of sub-keys with uniform corresponding values across all nested dictionaries.
        """
        dct_values = list(dct.values())
        overlapping_keys = set(dct_values[0].keys())
        for d in dct_values[1:]:
            overlapping_keys.intersection_update(d.keys())
        new_dct = dict()
        for i, k in enumerate(dct.keys()):
            for k2 in overlapping_keys:
                v2 = dct[k][k2]
                if i == 0 and k2 not in new_dct:
                    new_dct[k2] = v2
                elif k2 in new_dct and new_dct[k2] is not v2:
                    del new_dct[k2]
        return new_dct

    def select_keys(self: DataT, keys: tp.MaybeKeys, **kwargs) -> DataT:
        """Create a new `Data` instance containing only the selected keys.

        Args:
            keys (MaybeKeys): Feature or symbol identifier(s).
            **kwargs: Keyword arguments for `Data.replace`.

        Returns:
            Data: New `Data` instance with the selected keys.
        """
        keys = self.resolve_keys(keys)
        if self.has_multiple_keys(keys):
            single_key = False
        else:
            single_key = True
            keys = [keys]
        attr_dicts = dict()
        for attr in self._key_dict_attrs:
            attr_value = getattr(self, attr)
            if isinstance(attr_value, self.dict_type):
                attr_dicts[attr] = self.select_from_dict(attr_value, keys)

        return self.replace(
            data=self.select_from_dict(self.data, keys, raise_error=True),
            single_key=single_key,
            **attr_dicts,
            **kwargs,
        )

    def select_columns(self: DataT, columns: tp.MaybeColumns, **kwargs) -> DataT:
        """Create a new `Data` instance containing only the selected columns.

        Args:
            columns (MaybeColumns): Column identifier(s).
            **kwargs: Keyword arguments for `Data.indexing_func`.

        Returns:
            Data: New `Data` instance with the selected columns.
        """
        columns = self.resolve_columns(columns)

        def _pd_indexing_func(obj):
            return obj[columns]

        return self.indexing_func(_pd_indexing_func, replace_kwargs=kwargs)

    def select_feature_idxs(self: DataT, idxs: tp.MaybeSequence[int], **kwargs) -> DataT:
        if checks.is_int(idxs):
            features = self.features[idxs]
        else:
            features = [self.features[i] for i in idxs]
        if self.feature_oriented:
            return self.select_keys(features, **kwargs)
        return self.select_columns(features, **kwargs)

    def select_symbol_idxs(self: DataT, idxs: tp.MaybeSequence[int], **kwargs) -> DataT:
        if checks.is_int(idxs):
            symbols = self.symbols[idxs]
        else:
            symbols = [self.symbols[i] for i in idxs]
        if self.feature_oriented:
            return self.select_columns(symbols, **kwargs)
        return self.select_keys(symbols, **kwargs)

    def select(self: DataT, keys: tp.MaybeKeys, **kwargs) -> DataT:
        """Create a new `Data` instance with selected features or symbols based on the provided keys.

        Depending on the data orientation, delegates to either `Data.select_features` or `Data.select_symbols`.

        Args:
            keys (MaybeKeys): Feature or symbol identifier(s).
            **kwargs: Keyword arguments for the delegated function.

        Returns:
            Data: New `Data` instance with the selected features or symbols.
        """
        if not self.has_multiple_keys(keys):
            keys = [keys]
            single_key = True
        else:
            single_key = False
        feature_keys = set(self.resolve_features(keys, raise_error=False))
        symbol_keys = set(self.resolve_symbols(keys, raise_error=False))
        features_and_keys = set(self.features).intersection(feature_keys)
        symbols_and_keys = set(self.symbols).intersection(symbol_keys)
        if features_and_keys and not symbols_and_keys:
            if single_key:
                return self.select_features(keys[0], **kwargs)
            return self.select_features(keys, **kwargs)
        if symbols_and_keys and not features_and_keys:
            if single_key:
                return self.select_symbols(keys[0], **kwargs)
            return self.select_symbols(keys, **kwargs)
        raise ValueError("Cannot determine orientation. Use select_features or select_symbols.")

    def add_feature(
        self: DataT,
        feature: tp.Feature,
        data: tp.Union[None, tp.SeriesFrame, CustomTemplate] = None,
        pull_feature: bool = False,
        pull_kwargs: tp.KwargsLike = None,
        reuse_fetch_kwargs: bool = True,
        run_kwargs: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
        merge_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance with a new feature added to this instance.

        Args:
            feature (Feature): Feature identifier.
            data (Union[None, SeriesFrame, CustomTemplate]): Data corresponding to the feature.

                If None, data is retrieved using `Data.run` or, if `pull_feature` is True, via `Data.pull`.
            pull_feature (bool): Whether to use `Data.pull` to retrieve data when `data` is None.
            pull_kwargs (KwargsLike): Keyword arguments for `Data.pull`.
            reuse_fetch_kwargs (bool): Whether to reuse fetch kwargs from the current instance.
            run_kwargs (KwargsLike): Keyword arguments for `Data.run`.
            wrap_kwargs (KwargsLike): Keyword arguments for wrapping the result.

                See `vectorbtpro.base.wrapping.ArrayWrapper.wrap`.
            merge_kwargs (KwargsLike): Keyword arguments for `Data.merge`.
            **kwargs: Keyword arguments for `Data.from_data`.

        Returns:
            Data: New `Data` instance with the added feature.
        """
        if run_kwargs is None:
            run_kwargs = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}
        if data is None:
            if pull_feature:
                if isinstance(self.fetch_kwargs, feature_dict) and reuse_fetch_kwargs:
                    pull_kwargs = merge_dicts(self.get_intersection_dict(self.fetch_kwargs), pull_kwargs)
                data = type(self).pull(features=feature, **pull_kwargs).get(feature=feature)
            else:
                data = self.run(feature, **run_kwargs, unpack=True)
                data = self.symbol_wrapper.wrap(data, **wrap_kwargs)
        if isinstance(data, CustomTemplate):
            data = data.substitute(dict(data=self), eval_id="data")
        if isinstance(data, pd.Series) and self.symbol_wrapper.ndim == 1:
            data = data.copy(deep=False)
            data.name = self.symbols[0]
        for attr in self._key_dict_attrs:
            if attr in kwargs:
                checks.assert_not_instance_of(kwargs[attr], key_dict, arg_name=attr)
                kwargs[attr] = feature_dict({feature: kwargs[attr]})
        data = type(self).from_data(
            feature_dict({feature: data}),
            invert_data=not self.feature_oriented,
            **kwargs,
        )
        on_merge_conflict = {k: "error" for k in kwargs if k not in self._key_dict_attrs}
        on_merge_conflict["_def"] = "first"
        if merge_kwargs is None:
            merge_kwargs = {}
        return self.merge(data, on_merge_conflict=on_merge_conflict, **merge_kwargs)

    def add_symbol(
        self: DataT,
        symbol: tp.Symbol,
        data: tp.Union[None, tp.SeriesFrame, CustomTemplate] = None,
        pull_kwargs: tp.KwargsLike = None,
        reuse_fetch_kwargs: bool = True,
        merge_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance with a new symbol added to this instance.

        Args:
            symbol (Symbol): Symbol identifier.
            data (Union[None, SeriesFrame, CustomTemplate]): Data corresponding to the symbol.

                If None, data is retrieved using `Data.pull`.
            pull_kwargs (KwargsLike): Keyword arguments for `Data.pull`.
            reuse_fetch_kwargs (bool): Whether to reuse fetch kwargs from the current instance.
            merge_kwargs (KwargsLike): Keyword arguments for `Data.merge`.
            **kwargs: Keyword arguments for `Data.from_data`.

        Returns:
            Data: New `Data` instance with the added symbol.
        """
        if pull_kwargs is None:
            pull_kwargs = {}
        if data is None:
            if isinstance(self.fetch_kwargs, symbol_dict) and reuse_fetch_kwargs:
                pull_kwargs = merge_dicts(self.get_intersection_dict(self.fetch_kwargs), pull_kwargs)
            data = type(self).pull(symbols=symbol, **pull_kwargs).get(symbol=symbol)
        if isinstance(data, CustomTemplate):
            data = data.substitute(dict(data=self), eval_id="data")
        if isinstance(data, pd.Series) and self.feature_wrapper.ndim == 1:
            data = data.copy(deep=False)
            data.name = self.features[0]
        for attr in self._key_dict_attrs:
            if attr in kwargs:
                checks.assert_not_instance_of(kwargs[attr], key_dict, arg_name=attr)
                kwargs[attr] = symbol_dict({symbol: kwargs[attr]})
        data = type(self).from_data(
            symbol_dict({symbol: data}),
            invert_data=not self.symbol_oriented,
            **kwargs,
        )
        on_merge_conflict = {k: "error" for k in kwargs if k not in self._key_dict_attrs}
        on_merge_conflict["_def"] = "first"
        if merge_kwargs is None:
            merge_kwargs = {}
        return self.merge(data, on_merge_conflict=on_merge_conflict, **merge_kwargs)

    def add_key(
        self: DataT,
        key: tp.Key,
        data: tp.Union[None, tp.SeriesFrame, CustomTemplate] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance with a new key added to the current instance.

        Depending on the data orientation, delegates to either `Data.add_feature` or `Data.add_symbol`.

        Args:
            key (Key): Feature or symbol identifier.
            data (Union[None, SeriesFrame, CustomTemplate]): Data corresponding to the key.
            **kwargs: Keyword arguments for the delegated function.

        Returns:
            Data: New `Data` instance with the added key.
        """
        if self.feature_oriented:
            return self.add_feature(key, data=data, **kwargs)
        return self.add_symbol(key, data=data, **kwargs)

    def add_column(
        self: DataT,
        column: tp.Column,
        data: tp.Union[None, tp.SeriesFrame, CustomTemplate] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance with a new column added to the current instance.

        Depending on the data orientation, delegates to either `Data.add_symbol` or `Data.add_feature`.

        Args:
            column (Column): Column identifier.
            data (Union[None, SeriesFrame, CustomTemplate]): Data corresponding to the column.
            **kwargs: Keyword arguments for the delegated function.

        Returns:
            Data: New `Data` instance with the added key.
        """
        if self.feature_oriented:
            return self.add_symbol(column, data=data, **kwargs)
        return self.add_feature(column, data=data, **kwargs)

    def add(
        self: DataT,
        key: tp.Key,
        data: tp.Union[None, tp.SeriesFrame, CustomTemplate] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance with a new feature or symbol based on the data's orientation.

        Depending on the data orientation, delegates to either `Data.add_symbol` or `Data.add_feature`.
        The orientation is automatically determined by comparing data columns with existing features and symbols.

        Args:
            key (Key): Feature or symbol identifier.
            data (Union[None, SeriesFrame, CustomTemplate]): Data corresponding to the key.
            **kwargs: Keyword arguments for the delegated function.

        Returns:
            Data: New `Data` instance with the added feature or symbol.

        Raises:
            ValueError: If the orientation cannot be determined from the data.
        """
        if data is not None:
            if isinstance(data, CustomTemplate):
                data = data.substitute(dict(data=self), eval_id="data")
            if isinstance(data, pd.Series):
                columns = [data.name]
            else:
                columns = data.columns
            feature_columns = set(self.resolve_features(columns, raise_error=False))
            symbol_columns = set(self.resolve_symbols(columns, raise_error=False))
            features_and_columns = set(self.features).intersection(feature_columns)
            symbols_and_columns = set(self.symbols).intersection(symbol_columns)
            if features_and_columns and not symbols_and_columns:
                return self.add_symbol(key, data=data, **kwargs)
            if symbols_and_columns and not features_and_columns:
                return self.add_feature(key, data=data, **kwargs)
        raise ValueError("Cannot determine orientation. Use add_feature or add_symbol.")

    @classmethod
    def rename_in_dict(cls, dct: dict, rename: tp.Dict[tp.Key, tp.Key]) -> dict:
        """Rename keys in the given dictionary using a provided mapping.

        Args:
            dct (dict): Dictionary whose keys will be renamed.
            rename (Dict[Key, Key]): Mapping from old keys to new keys.

        Returns:
            dict: New dictionary with keys renamed according to the mapping.
        """
        return type(dct)({rename.get(k, k): v for k, v in dct.items()})

    def rename_keys(
        self: DataT,
        rename: tp.Union[tp.MaybeKeys, tp.Dict[tp.Key, tp.Key]],
        to: tp.Optional[tp.MaybeKeys] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance with keys renamed based on a provided mapping.

        If a value for `to` is provided, a one-to-one mapping is constructed.

        Args:
            rename (Union[MaybeKeys, Dict[Key, Key]]): Feature or symbol identifier(s) or a mapping.
            to (Optional[MaybeKeys]): New feature or symbol identifier(s) corresponding to the old one(s).
            **kwargs: Keyword arguments for `Data.replace`.

        Returns:
            Data: New `Data` instance with keys renamed.
        """
        if to is not None:
            if self.has_multiple_keys(to):
                rename = dict(zip(rename, to))
            else:
                rename = {rename: to}
        rename = dict(zip(self.resolve_keys(list(rename.keys())), rename.values()))
        attr_dicts = dict()
        for attr in self._key_dict_attrs:
            attr_value = getattr(self, attr)
            if isinstance(attr_value, self.dict_type):
                attr_dicts[attr] = self.rename_in_dict(attr_value, rename)
        return self.replace(data=self.rename_in_dict(self.data, rename), **attr_dicts, **kwargs)

    def rename_columns(
        self: DataT,
        rename: tp.Union[tp.MaybeColumns, tp.Dict[tp.Column, tp.Column]],
        to: tp.Optional[tp.MaybeColumns] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance with columns renamed based on a provided mapping.

        If a value for `to` is provided, a one-to-one mapping is constructed.

        Args:
            rename (Union[MaybeColumns, Dict[Column, Column]]): Column identifier(s) or a mapping.
            to (Optional[MaybeColumns]): New column identifier(s) corresponding to the old one(s).
            **kwargs: Keyword arguments for `Data.replace`.

        Returns:
            Data: New `Data` instance with columns renamed.
        """
        if to is not None:
            if self.has_multiple_keys(to):
                rename = dict(zip(rename, to))
            else:
                rename = {rename: to}
        rename = dict(zip(self.resolve_columns(list(rename.keys())), rename.values()))
        attr_dicts = dict()
        for attr in self._key_dict_attrs:
            attr_value = getattr(self, attr)
            if isinstance(attr_value, self.column_type):
                attr_dicts[attr] = self.rename_in_dict(attr_value, rename)
        new_wrapper = self.wrapper.replace(columns=self.wrapper.columns.map(lambda x: rename.get(x, x)))
        return self.replace(wrapper=new_wrapper, **attr_dicts, **kwargs)

    def rename_features(
        self: DataT,
        rename: tp.Union[tp.MaybeFeatures, tp.Dict[tp.Feature, tp.Feature]],
        to: tp.Optional[tp.MaybeFeatures] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance with features renamed based on a provided mapping.

        Depending on the data orientation, delegates to either `Data.rename_keys` or `Data.rename_columns`.

        Args:
            rename (Union[MaybeFeatures, Dict[Feature, Feature]]): Feature identifier(s) or a mapping.
            to (Optional[MaybeFeatures]): New feature identifier(s) corresponding to the old one(s).
            **kwargs: Keyword arguments for the delegated function.

        Returns:
            Data: New `Data` instance with features renamed.
        """
        if self.feature_oriented:
            return self.rename_keys(rename, to=to, **kwargs)
        return self.rename_columns(rename, to=to, **kwargs)

    def rename_symbols(
        self: DataT,
        rename: tp.Union[tp.MaybeSymbols, tp.Dict[tp.Symbol, tp.Symbol]],
        to: tp.Optional[tp.MaybeSymbols] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance with symbols renamed.

        Depending on the data orientation, delegates to either `Data.rename_columns` or `Data.rename_keys`.

        Args:
            rename (Union[MaybeSymbols, Dict[Symbol, Symbol]]): Symbol identifier(s) or a mapping.
            to (Optional[MaybeSymbols]): New symbol identifier(s) corresponding to the old one(s).
            **kwargs: Keyword arguments for the delegated function.

        Returns:
            Data: New `Data` instance with renamed symbols.
        """
        if self.feature_oriented:
            return self.rename_columns(rename, to=to, **kwargs)
        return self.rename_keys(rename, to=to, **kwargs)

    def rename(
        self: DataT,
        rename: tp.Union[tp.MaybeKeys, tp.Dict[tp.Key, tp.Key]],
        to: tp.Optional[tp.MaybeKeys] = None,
        **kwargs,
    ) -> DataT:
        """Create a new `Data` instance with features or symbols renamed by automatically determining orientation.

        Depending on the data orientation, delegates to either `Data.rename_features` or `Data.rename_symbols`.
        The orientation is automatically determined by comparing data columns with existing features and symbols.

        Args:
            rename (Union[MaybeKeys, Dict[Key, Key]]): Feature or symbol identifier(s) or a mapping.
            to (Optional[MaybeKeys]): New feature or symbol identifier(s) corresponding to the old one(s).
            **kwargs: Keyword arguments for the delegated function.

        Returns:
            Data: New `Data` instance with renamed features or symbols.
        """
        if to is not None:
            if self.has_multiple_keys(to):
                rename = dict(zip(rename, to))
            else:
                rename = {rename: to}
        feature_keys = set(self.resolve_features(list(rename.keys()), raise_error=False))
        symbol_keys = set(self.resolve_symbols(list(rename.keys()), raise_error=False))
        features_and_keys = set(self.features).intersection(feature_keys)
        symbols_and_keys = set(self.symbols).intersection(symbol_keys)
        if features_and_keys and not symbols_and_keys:
            return self.rename_features(rename, **kwargs)
        if symbols_and_keys and not features_and_keys:
            return self.rename_symbols(rename, **kwargs)
        raise ValueError("Cannot determine orientation. Use rename_features or rename_symbols.")

    def remove_features(self: DataT, features: tp.MaybeFeatures, **kwargs) -> DataT:
        """Create a new `Data` instance with the specified feature(s) removed.

        Args:
            features (MaybeFeatures): Feature identifier(s).
            **kwargs: Keyword arguments for `Data.select_feature_idxs`.

        Returns:
            Data: New `Data` instance with the remaining features.
        """
        if self.has_multiple_keys(features):
            remove_feature_idxs = [self.get_feature_idx(k, raise_error=True) for k in features]
        else:
            remove_feature_idxs = [self.get_feature_idx(features, raise_error=True)]
        keep_feature_idxs = [i for i in range(len(self.features)) if i not in remove_feature_idxs]
        if len(keep_feature_idxs) == 0:
            raise ValueError("No features will be left after this operation")
        return self.select_feature_idxs(keep_feature_idxs, **kwargs)

    def remove_symbols(self: DataT, symbols: tp.MaybeSymbols, **kwargs) -> DataT:
        """Create a new `Data` instance with the specified symbol(s) removed.

        Args:
            symbols (MaybeSymbols): Symbol identifier(s).
            **kwargs: Keyword arguments for `Data.select_symbol_idxs`.

        Returns:
            Data: New `Data` instance with the remaining symbols.
        """
        if self.has_multiple_keys(symbols):
            remove_symbol_idxs = [self.get_symbol_idx(k, raise_error=True) for k in symbols]
        else:
            remove_symbol_idxs = [self.get_symbol_idx(symbols, raise_error=True)]
        keep_symbol_idxs = [i for i in range(len(self.symbols)) if i not in remove_symbol_idxs]
        if len(keep_symbol_idxs) == 0:
            raise ValueError("No symbols will be left after this operation")
        return self.select_symbol_idxs(keep_symbol_idxs, **kwargs)

    def remove_keys(self: DataT, keys: tp.MaybeKeys, **kwargs) -> DataT:
        """Create a new `Data` instance with the specified key(s) removed.

        Depending on the data orientation, delegates to either `Data.remove_features` or `Data.remove_symbols`.

        Args:
            keys (MaybeKeys): Feature or symbol identifier(s).
            **kwargs: Keyword arguments for the delegated function.

        Returns:
            Data: New `Data` instance with the specified keys removed.
        """
        if self.feature_oriented:
            return self.remove_features(keys, **kwargs)
        return self.remove_symbols(keys, **kwargs)

    def remove_columns(self: DataT, columns: tp.MaybeColumns, **kwargs) -> DataT:
        """Create a new `Data` instance with the specified column(s) removed.

        Depending on the data orientation, delegates to either `Data.remove_symbols` or `Data.remove_features`.

        Args:
            columns (MaybeColumns): Column identifier(s).
            **kwargs: Keyword arguments for the delegated function.

        Returns:
            Data: New `Data` instance with the remaining columns.
        """
        if self.feature_oriented:
            return self.remove_symbols(columns, **kwargs)
        return self.remove_features(columns, **kwargs)

    def remove(self: DataT, keys: tp.MaybeKeys, **kwargs) -> DataT:
        """Create a new `Data` instance with the specified key(s) removed by automatically determining orientation.

        Depending on the data orientation, delegates to either `Data.remove_features` or `Data.remove_symbols`.
        The orientation is automatically determined by comparing data columns with existing features and symbols.

        Args:
            keys (MaybeKeys): Feature or symbol identifier(s).
            **kwargs: Keyword arguments for the delegated function.

        Returns:
            Data: New `Data` instance after removal of the specified key(s).
        """
        if not self.has_multiple_keys(keys):
            keys = [keys]
        feature_keys = set(self.resolve_features(keys, raise_error=False))
        symbol_keys = set(self.resolve_symbols(keys, raise_error=False))
        features_and_keys = set(self.features).intersection(feature_keys)
        symbols_and_keys = set(self.symbols).intersection(symbol_keys)
        if features_and_keys and not symbols_and_keys:
            return self.remove_features(keys, **kwargs)
        if symbols_and_keys and not features_and_keys:
            return self.remove_symbols(keys, **kwargs)
        raise ValueError("Cannot determine orientation. Use remove_features or remove_symbols.")

    @hybrid_method
    def merge(
        cls_or_self: tp.MaybeType[DataT],
        *datas: tp.MaybeSequence[DataT],
        rename: tp.Optional[tp.Dict[tp.Key, tp.Key]] = None,
        **kwargs,
    ) -> DataT:
        """Merge multiple `Data` instances into a single instance.

        Data from later instances overrides earlier instances. This method supports
        merging both symbols and features.

        Args:
            *datas (MaybeSequence[Data]): (Additional) data instances to merge.
            rename (Optional[Dict[Key, Key]]): Optional mapping for renaming keys during merging.
            **kwargs: Keyword arguments for `Data.resolve_merge_kwargs` and then `Data.from_data`.

        Returns:
            Data: Merged `Data` instance.
        """
        if len(datas) == 1 and not isinstance(datas[0], Data):
            datas = datas[0]
        datas = list(datas)
        if not isinstance(cls_or_self, type):
            datas = (cls_or_self, *datas)

        data_type = None
        data = {}
        single_key = True
        attr_dicts = dict()
        for instance in datas:
            if data_type is None:
                data_type = type(instance.data)
            elif not isinstance(instance.data, data_type):
                raise TypeError("Objects to be merged must have the same dict type for data")
            if not instance.single_key:
                single_key = False
            for k in instance.keys:
                if rename is None:
                    new_k = k
                else:
                    new_k = rename[k]
                if new_k in data:
                    obj1 = instance.data[k]
                    obj2 = data[new_k]
                    both_were_series = True
                    if isinstance(obj1, pd.Series):
                        obj1 = obj1.to_frame()
                    else:
                        both_were_series = False
                    if isinstance(obj2, pd.Series):
                        obj2 = obj2.to_frame()
                    else:
                        both_were_series = False
                    new_obj = obj1.combine_first(obj2)
                    new_columns = []
                    for c in obj2.columns:
                        new_columns.append(c)
                    for c in obj1.columns:
                        if c not in new_columns:
                            new_columns.append(c)
                    new_obj = new_obj[new_columns]
                    if new_obj.shape[1] == 1 and both_were_series:
                        new_obj = new_obj.iloc[:, 0]
                    data[new_k] = new_obj
                else:
                    data[new_k] = instance.data[k]
                for attr in cls_or_self._key_dict_attrs:
                    attr_value = getattr(instance, attr)
                    if (issubclass(data_type, symbol_dict) and isinstance(attr_value, symbol_dict)) or (
                        issubclass(data_type, feature_dict) and isinstance(attr_value, feature_dict)
                    ):
                        if k in attr_value:
                            if attr not in attr_dicts:
                                attr_dicts[attr] = type(attr_value)()
                            elif not isinstance(attr_value, type(attr_dicts[attr])):
                                raise TypeError(f"Objects to be merged must have the same dict type for '{attr}'")
                            attr_dicts[attr][new_k] = attr_value[k]
            for attr in cls_or_self._key_dict_attrs:
                attr_value = getattr(instance, attr)
                if (issubclass(data_type, symbol_dict) and isinstance(attr_value, feature_dict)) or (
                    issubclass(data_type, feature_dict) and isinstance(attr_value, symbol_dict)
                ):
                    if attr not in attr_dicts:
                        attr_dicts[attr] = type(attr_value)()
                    elif not isinstance(attr_value, type(attr_dicts[attr])):
                        raise TypeError(f"Objects to be merged must have the same dict type for '{attr}'")
                    attr_dicts[attr].update(**attr_value)

        if "missing_index" not in kwargs:
            kwargs["missing_index"] = "nan"
        if "missing_columns" not in kwargs:
            kwargs["missing_columns"] = "nan"
        kwargs = cls_or_self.resolve_merge_kwargs(
            *[instance.config for instance in datas],
            wrapper=None,
            data=data_type(data),
            single_key=single_key,
            **attr_dicts,
            **kwargs,
        )
        kwargs.pop("wrapper", None)
        return cls_or_self.from_data(**kwargs)

    # ############# Fetching ############# #

    @classmethod
    def fetch_feature(
        cls,
        feature: tp.Feature,
        **kwargs,
    ) -> tp.FeatureData:
        """Fetch a feature from the data.

        This method may also return a dictionary that is stored in `Data.returned_kwargs`. If the
        returned dictionary includes the keyword arguments `tz_localize`, `tz_convert`, or `freq`, they
        will be used to override global settings.

        Args:
            feature (Feature): Feature identifier.
            **kwargs: Keyword arguments for fetching the feature.

        Returns:
            FeatureData: Fetched data and (optionally) a metadata dictionary, or None.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    @classmethod
    def try_fetch_feature(
        cls,
        feature: tp.Feature,
        skip_on_error: bool = False,
        silence_warnings: bool = False,
        fetch_kwargs: tp.KwargsLike = None,
    ) -> tp.FeatureData:
        """Attempt to fetch a feature by calling `Data.fetch_feature`.

        Args:
            feature (Feature): Feature identifier.
            skip_on_error (bool): If True, do not raise an exception on error.
            silence_warnings (bool): Flag to suppress warning messages.
            fetch_kwargs (KwargsLike): Keyword arguments for `Data.fetch_feature`.

        Returns:
            FeatureData: Fetched data and (optionally) a metadata dictionary, or None.
        """
        if fetch_kwargs is None:
            fetch_kwargs = {}
        try:
            out = cls.fetch_feature(feature, **fetch_kwargs)
            if out is None:
                if not silence_warnings:
                    warn(f"Feature '{str(feature)}' returned None. Skipping.")
            return out
        except Exception as e:
            if not skip_on_error:
                raise e
            if not silence_warnings:
                warn(traceback.format_exc())
                warn(f"Feature '{str(feature)}' raised an exception. Skipping.")
        return None

    @classmethod
    def fetch_symbol(
        cls,
        symbol: tp.Symbol,
        **kwargs,
    ) -> tp.SymbolData:
        """Fetch a symbol.

        The method can return a dictionary accessible from `Data.returned_kwargs`. If the returned
        dictionary includes any of the following keyword arguments, they are used to override global settings:

        * `tz_localize`
        * `tz_convert`
        * `freq`

        Args:
            symbol (Symbol): Symbol identifier.
            **kwargs: Keyword arguments for fetching the symbol.

        Returns:
            SymbolData: Fetched data and (optionally) a metadata dictionary, or None.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    @classmethod
    def try_fetch_symbol(
        cls,
        symbol: tp.Symbol,
        skip_on_error: bool = False,
        silence_warnings: bool = False,
        fetch_kwargs: tp.KwargsLike = None,
    ) -> tp.SymbolData:
        """Attempt to fetch a symbol by calling `Data.fetch_symbol`.

        Args:
            symbol (Symbol): Symbol identifier.
            skip_on_error (bool): If True, do not raise an exception on error.
            silence_warnings (bool): Flag to suppress warning messages.
            fetch_kwargs (KwargsLike): Keyword arguments for `Data.fetch_symbol`.

        Returns:
            SymbolData: Fetched data and (optionally) a metadata dictionary, or None.
        """
        if fetch_kwargs is None:
            fetch_kwargs = {}
        try:
            out = cls.fetch_symbol(symbol, **fetch_kwargs)
            if out is None:
                if not silence_warnings:
                    warn(f"Symbol '{str(symbol)}' returned None. Skipping.")
            return out
        except Exception as e:
            if not skip_on_error:
                raise e
            if not silence_warnings:
                warn(traceback.format_exc())
                warn(f"Symbol '{str(symbol)}' raised an exception. Skipping.")
        return None

    @classmethod
    def resolve_keys_meta(
        cls,
        keys: tp.Union[None, dict, tp.MaybeKeys] = None,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[None, dict, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, dict, tp.MaybeSymbols] = None,
    ) -> tp.Kwargs:
        """Resolve metadata for keys.

        The method validates that only one of `keys`, `features`, or `symbols` is provided and
        determines whether the keys represent features. It also sets the corresponding dictionary type.
        The resolved metadata is returned as a dictionary with the keys, a flag for features, and the
        determined dictionary type.

        Args:
            keys (Union[None, dict, MaybeKeys]): Feature or symbol identifier(s).

                Depending on `keys_are_features`, these keys will be treated as features or symbols.
            keys_are_features (Optional[bool]): Flag indicating whether the keys represent features.
            features (Union[None, dict, MaybeFeatures]): Feature identifier(s).

                If provided as a dictionary, keys are treated as features and values as keyword arguments.

                !!! note
                    Tuple is interpreted as a single feature.
            symbols (Union[None, dict, MaybeSymbols]): Symbol identifier(s).

                If provided as a dictionary, keys are treated as symbols and values as keyword arguments.

                !!! note
                    Tuple is interpreted as a single symbol.

        Returns:
            Kwargs: Dictionary with metadata:

                * `keys`: Mapping of keys to use.
                * `keys_are_features`: Flag indicating whether the keys represent features.
                * `dict_type`: Dictionary type.
        """
        if keys is not None and features is not None:
            raise ValueError("Must provide either keys or features, not both")
        if keys is not None and symbols is not None:
            raise ValueError("Must provide either keys or symbols, not both")
        if features is not None and symbols is not None:
            raise ValueError("Must provide either features or symbols, not both")
        if keys is None:
            if features is not None:
                if isinstance(features, dict):
                    cls.check_dict_type(features, "features", dict_type=feature_dict)
                keys = features
                keys_are_features = True
                dict_type = feature_dict
            elif symbols is not None:
                if isinstance(symbols, dict):
                    cls.check_dict_type(symbols, "symbols", dict_type=symbol_dict)
                keys = symbols
                keys_are_features = False
                dict_type = symbol_dict
            else:
                keys = symbols
                keys_are_features = False
                dict_type = symbol_dict
        else:
            if isinstance(keys, feature_dict):
                if keys_are_features is not None and not keys_are_features:
                    raise TypeError("Keys are of type feature_dict but keys_are_features is False")
                keys_are_features = True
            elif isinstance(keys, symbol_dict):
                if keys_are_features is not None and keys_are_features:
                    raise TypeError("Keys are of type symbol_dict but keys_are_features is True")
                keys_are_features = False
            keys_are_features = cls.resolve_base_setting(keys_are_features, "keys_are_features")
            if keys_are_features:
                dict_type = feature_dict
            else:
                dict_type = symbol_dict
        return dict(
            keys=keys,
            keys_are_features=keys_are_features,
            dict_type=dict_type,
        )

    @classmethod
    def pull(
        cls: tp.Type[DataT],
        keys: tp.Union[None, dict, tp.MaybeKeys] = None,
        *,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[None, dict, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, dict, tp.MaybeSymbols] = None,
        classes: tp.Optional[tp.MaybeSequence[tp.Union[tp.Hashable, dict]]] = None,
        level_name: tp.Union[None, bool, tp.MaybeIterable[tp.Hashable]] = None,
        tz_localize: tp.Union[None, bool, tp.TimezoneLike] = None,
        tz_convert: tp.Union[None, bool, tp.TimezoneLike] = None,
        missing_index: tp.Optional[str] = None,
        missing_columns: tp.Optional[str] = None,
        wrapper_kwargs: tp.KwargsLike = None,
        skip_on_error: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        execute_kwargs: tp.KwargsLike = None,
        return_raw: bool = False,
        **kwargs,
    ) -> tp.PullOutput:
        """Pull data.

        Fetch each feature or symbol using `Data.fetch_feature`/`Data.fetch_symbol`
        and prepare it with `Data.from_data`.

        Iteration over features or symbols is performed using `vectorbtpro.utils.execution.execute`
        to allow parallelized execution.

        Args:
            keys (Union[None, dict, MaybeKeys]): Feature or symbol identifier(s).

                Depending on `keys_are_features`, these keys will be treated as features or symbols.
            keys_are_features (Optional[bool]): Flag indicating whether the keys represent features.
            features (Union[None, dict, MaybeFeatures]): Feature identifier(s).

                If provided as a dictionary, keys are treated as features and values as keyword arguments.

                !!! note
                    Tuple is interpreted as a single feature.
            symbols (Union[None, dict, MaybeSymbols]): Symbol identifier(s).

                If provided as a dictionary, keys are treated as symbols and values as keyword arguments.

                !!! note
                    Tuple is interpreted as a single symbol.
            classes (Optional[MaybeSequence[Union[Hashable, dict]]]): See `Data.classes`.

                Can be a single hashable, a dictionary with class names as keys, or a sequence of such.

                !!! note
                    Tuple is interpreted as a single class.
            level_name (Union[None, bool, MaybeIterable[Hashable]]): Used as in `Data.level_name`.
            tz_localize (Union[None, bool, TimezoneLike]): Flag or specification for timezone localization.

                Set to False to disable localization. See `Data.prepare_dt_index`.
            tz_convert (Union[None, bool, TimezoneLike]): Flag or specification for timezone conversion.

                Set to False to disable conversion. See `Data.prepare_dt_index`.
            missing_index (Optional[str]): Specifies how to handle missing indices when aligning data.

                See `Data.align_index`.
            missing_columns (Optional[str]): Specifies how to handle missing columns when aligning data.

                See `Data.align_columns`.
            wrapper_kwargs (KwargsLike): Keyword arguments for configuring the wrapper.

                See `vectorbtpro.base.wrapping.ArrayWrapper`.
            skip_on_error (Optional[bool]): Whether to skip pulling a feature or symbol if an exception occurs.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

                This flag is also forwarded to `Data.fetch_feature`/`Data.fetch_symbol` if applicable.
            execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                See `vectorbtpro.utils.execution.execute`.
            return_raw (bool): Return the raw outputs if True.
            **kwargs: Keyword arguments for `Data.fetch_feature`/`Data.fetch_symbol`.

                For features or symbols that require distinct keyword arguments, pass `key_dict` or
                `feature_dict`/`symbol_dict` for each.

        Returns:
            PullOutput: `Data` instance or a list of execution outputs if `return_raw` is True.

        !!! info
            For default settings, see `vectorbtpro._settings.data`.
        """
        keys_meta = cls.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
        )
        keys = keys_meta["keys"]
        keys_are_features = keys_meta["keys_are_features"]
        dict_type = keys_meta["dict_type"]

        fetch_kwargs = dict_type()
        if isinstance(keys, dict):
            new_keys = []
            for k, key_fetch_kwargs in keys.items():
                new_keys.append(k)
                fetch_kwargs[k] = key_fetch_kwargs
            keys = new_keys
            single_key = False
        elif cls.has_multiple_keys(keys):
            keys = list(keys)
            if len(set(keys)) < len(keys):
                raise ValueError("Duplicate keys provided")
            single_key = False
        else:
            single_key = True
            keys = [keys]
        if classes is not None:
            cls.check_dict_type(classes, arg_name="classes", dict_type=dict_type)
            if not isinstance(classes, key_dict):
                new_classes = {}
                single_class = checks.is_hashable(classes) or isinstance(classes, dict)
                if single_class:
                    for k in keys:
                        if isinstance(classes, dict):
                            new_classes[k] = classes
                        else:
                            if keys_are_features:
                                new_classes[k] = {"feature_class": classes}
                            else:
                                new_classes[k] = {"symbol_class": classes}
                else:
                    for i, k in enumerate(keys):
                        _classes = classes[i]
                        if not isinstance(_classes, dict):
                            if keys_are_features:
                                _classes = {"feature_class": _classes}
                            else:
                                _classes = {"symbol_class": _classes}
                        new_classes[k] = _classes
                classes = new_classes
        wrapper_kwargs = cls.resolve_base_setting(wrapper_kwargs, "wrapper_kwargs", merge=True)
        skip_on_error = cls.resolve_base_setting(skip_on_error, "skip_on_error")
        silence_warnings = cls.resolve_base_setting(silence_warnings, "silence_warnings")
        execute_kwargs = cls.resolve_base_setting(execute_kwargs, "execute_kwargs", merge=True)
        execute_kwargs = merge_dicts(dict(show_progress=not single_key), execute_kwargs)

        tasks = []
        if keys_are_features:
            func_arg_names = get_func_arg_names(cls.fetch_feature)
        else:
            func_arg_names = get_func_arg_names(cls.fetch_symbol)
        for k in keys:
            if keys_are_features:
                key_fetch_func = cls.try_fetch_feature
                key_fetch_kwargs = cls.select_feature_kwargs(k, kwargs)
            else:
                key_fetch_func = cls.try_fetch_symbol
                key_fetch_kwargs = cls.select_symbol_kwargs(k, kwargs)
            if "silence_warnings" in func_arg_names:
                key_fetch_kwargs["silence_warnings"] = silence_warnings
            if k in fetch_kwargs:
                key_fetch_kwargs = merge_dicts(key_fetch_kwargs, fetch_kwargs[k])

            tasks.append(
                Task(
                    key_fetch_func,
                    k,
                    skip_on_error=skip_on_error,
                    silence_warnings=silence_warnings,
                    fetch_kwargs=key_fetch_kwargs,
                )
            )
            fetch_kwargs[k] = key_fetch_kwargs

        key_index = cls.get_key_index(keys=keys, level_name=level_name, feature_oriented=keys_are_features)
        outputs = execute(tasks, size=len(keys), keys=key_index, **execute_kwargs)
        if return_raw:
            return outputs

        data = dict_type()
        returned_kwargs = dict_type()
        common_tz_localize = None
        common_tz_convert = None
        common_freq = None
        for i, out in enumerate(outputs):
            k = keys[i]
            if out is not None:
                if isinstance(out, tuple):
                    _data = out[0]
                    _returned_kwargs = out[1]
                else:
                    _data = out
                    _returned_kwargs = {}
                _data = to_any_array(_data)
                _tz = _returned_kwargs.pop("tz", None)
                _tz_localize = _returned_kwargs.pop("tz_localize", None)
                _tz_convert = _returned_kwargs.pop("tz_convert", None)
                _freq = _returned_kwargs.pop("freq", None)
                if _tz is not None:
                    if _tz_localize is None:
                        _tz_localize = _tz
                    if _tz_convert is None:
                        _tz_convert = _tz
                if _tz_localize is not None:
                    if common_tz_localize is None:
                        common_tz_localize = _tz_localize
                    elif common_tz_localize != _tz_localize:
                        raise ValueError("Returned objects have different timezones (tz_localize)")
                if _tz_convert is not None:
                    if common_tz_convert is None:
                        common_tz_convert = _tz_convert
                    elif common_tz_convert != _tz_convert:
                        if not silence_warnings:
                            warn(f"Returned objects have different timezones (tz_convert). Setting to UTC.")
                        common_tz_convert = "utc"
                if _freq is not None:
                    if common_freq is None:
                        common_freq = _freq
                    elif common_freq != _freq:
                        raise ValueError("Returned objects have different frequencies (freq)")
                if _data.size == 0:
                    if not silence_warnings:
                        if keys_are_features:
                            warn(f"Feature '{str(k)}' returned an empty array. Skipping.")
                        else:
                            warn(f"Symbol '{str(k)}' returned an empty array. Skipping.")
                else:
                    data[k] = _data
                    returned_kwargs[k] = _returned_kwargs
        if tz_localize is None and common_tz_localize is not None:
            tz_localize = common_tz_localize
        if tz_convert is None and common_tz_convert is not None:
            tz_convert = common_tz_convert
        if wrapper_kwargs.get("freq", None) is None and common_freq is not None:
            wrapper_kwargs["freq"] = common_freq

        if len(data) == 0:
            if keys_are_features:
                raise ValueError("No features could be fetched")
            else:
                raise ValueError("No symbols could be fetched")

        return cls.from_data(
            data,
            single_key=single_key,
            classes=classes,
            level_name=level_name,
            tz_localize=tz_localize,
            tz_convert=tz_convert,
            missing_index=missing_index,
            missing_columns=missing_columns,
            wrapper_kwargs=wrapper_kwargs,
            fetch_kwargs=fetch_kwargs,
            returned_kwargs=returned_kwargs,
            silence_warnings=silence_warnings,
        )

    @classmethod
    def download(cls: tp.Type[DataT], *args, **kwargs) -> tp.PullOutput:
        """Return a Data instance by calling `Data.pull` with the provided arguments.

        This method exists for backward compatibility; use `Data.pull` in new code.

        Args:
            *args: Positional arguments for `Data.pull`.
            **kwargs: Keyword arguments for `Data.pull`.

        Returns:
            PullOutput: `Data` instance or a list of execution outputs if `return_raw` is True.
        """
        return cls.pull(*args, **kwargs)

    @classmethod
    def fetch(cls: tp.Type[DataT], *args, **kwargs) -> tp.PullOutput:
        """Return a Data instance by calling `Data.pull` with the provided arguments.

        This method exists for backward compatibility; use `Data.pull` in new code.

        Args:
            *args: Positional arguments for `Data.pull`.
            **kwargs: Keyword arguments for `Data.pull`.

        Returns:
            PullOutput: `Data` instance or a list of execution outputs if `return_raw` is True.
        """
        return cls.pull(*args, **kwargs)

    @classmethod
    def from_data_str(cls: tp.Type[DataT], data_str: str) -> DataT:
        """Parse a `Data` instance from a string.

        Args:
            data_str (str): String representing the data instance, formatted as either
                "Class:Symbol" (e.g. "YFData:BTC-USD") or "Symbol". In the latter case, the default
                class `vectorbtpro.data.custom.yf.YFData` is used.

        Returns:
            Data: Parsed `Data` instance.
        """
        from vectorbtpro.data import custom

        if ":" in data_str:
            cls_name, symbol = data_str.split(":")
            cls_name = cls_name.strip()
            symbol = symbol.strip()
            return getattr(custom, cls_name).pull(symbol)
        return custom.YFData.pull(data_str.strip())

    # ############# Updating ############# #

    def update_feature(
        self,
        feature: tp.Feature,
        **kwargs,
    ) -> tp.FeatureData:
        """Update a feature.

        Args:
            feature (Feature): Feature identifier.
            **kwargs: Keyword arguments passed for feature update.

        Returns:
            FeatureData: Updated data and (optionally) a metadata dictionary, or None.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def try_update_feature(
        self,
        feature: tp.Feature,
        skip_on_error: bool = False,
        silence_warnings: bool = False,
        update_kwargs: tp.KwargsLike = None,
    ) -> tp.FeatureData:
        """Try to update a feature using `Data.update_feature`.

        Args:
            feature (Feature): Feature identifier.
            skip_on_error (bool): If True, do not raise an exception on error.
            silence_warnings (bool): Flag to suppress warning messages.
            update_kwargs (KwargsLike): Keyword arguments for `Data.update_feature`.

        Returns:
            FeatureData: Updated data and (optionally) a metadata dictionary, or None.
        """
        if update_kwargs is None:
            update_kwargs = {}
        try:
            out = self.update_feature(feature, **update_kwargs)
            if out is None:
                if not silence_warnings:
                    warn(f"Feature '{str(feature)}' returned None. Skipping.")
            return out
        except Exception as e:
            if not skip_on_error:
                raise e
            if not silence_warnings:
                warn(traceback.format_exc())
                warn(f"Feature '{str(feature)}' raised an exception. Skipping.")
        return None

    def update_symbol(
        self,
        symbol: tp.Symbol,
        **kwargs,
    ) -> tp.SymbolData:
        """Update a symbol.

        Args:
            symbol (Symbol): Symbol identifier.
            **kwargs: Keyword arguments passed for symbol update.

        Returns:
            SymbolData: Updated data and (optionally) a metadata dictionary, or None.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    def try_update_symbol(
        self,
        symbol: tp.Symbol,
        skip_on_error: bool = False,
        silence_warnings: bool = False,
        update_kwargs: tp.KwargsLike = None,
    ) -> tp.SymbolData:
        """Try to update a symbol using `Data.update_symbol`.

        Args:
            symbol (Symbol): Symbol identifier.
            skip_on_error (bool): If True, do not raise an exception on error.
            silence_warnings (bool): Flag to suppress warning messages.
            update_kwargs (KwargsLike): Keyword arguments for `Data.update_symbol`.

        Returns:
            SymbolData: Updated data and (optionally) a metadata dictionary, or None.
        """
        if update_kwargs is None:
            update_kwargs = {}
        try:
            out = self.update_symbol(symbol, **update_kwargs)
            if out is None:
                if not silence_warnings:
                    warn(f"Symbol '{str(symbol)}' returned None. Skipping.")
            return out
        except Exception as e:
            if not skip_on_error:
                raise e
            if not silence_warnings:
                warn(traceback.format_exc())
                warn(f"Symbol '{str(symbol)}' raised an exception. Skipping.")
        return None

    def update(
        self: DataT,
        *,
        concat: bool = True,
        skip_on_error: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        execute_kwargs: tp.KwargsLike = None,
        return_raw: bool = False,
        **kwargs,
    ) -> tp.PullOutput:
        """Update the data object.

        Fetch new data for each feature or symbol by calling the appropriate update function via
        `Data.update_feature` or `Data.update_symbol`.

        Args:
            concat (bool): Whether to concatenate existing data with updated/new data.
            skip_on_error (Optional[bool]): Whether to skip updating a feature or symbol if an exception occurs.
            silence_warnings (Optional[bool]): Flag to suppress warning messages.

                Also passed to `Data.update_feature` or `Data.update_symbol` if accepted by
                `Data.fetch_feature` or `Data.fetch_symbol`.
            execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                See `vectorbtpro.utils.execution.execute`.
            return_raw (bool): Whether to return the raw outputs from update operations.
            **kwargs: Keyword arguments for `Data.update_feature` or `Data.update_symbol`.

                If different features or symbols require distinct keyword arguments, provide them
                via `key_dict` or `feature_dict`/`symbol_dict` for each.

        Returns:
            PullOutput: `Data` instance or a list of execution outputs if `return_raw` is True.

        !!! note
            Returns a new `Data` instance instead of modifying the current data in place.
        """
        skip_on_error = self.resolve_base_setting(skip_on_error, "skip_on_error")
        silence_warnings = self.resolve_base_setting(silence_warnings, "silence_warnings")
        execute_kwargs = self.resolve_base_setting(execute_kwargs, "execute_kwargs", merge=True)
        execute_kwargs = merge_dicts(dict(show_progress=False), execute_kwargs)
        if self.feature_oriented:
            func_arg_names = get_func_arg_names(self.fetch_feature)
        else:
            func_arg_names = get_func_arg_names(self.fetch_symbol)
        if "show_progress" in func_arg_names and "show_progress" not in kwargs:
            kwargs["show_progress"] = False
        checks.assert_instance_of(self.last_index, self.dict_type, "last_index")
        checks.assert_instance_of(self.delisted, self.dict_type, "delisted")

        tasks = []
        key_indices = []
        for i, k in enumerate(self.keys):
            if not self.delisted.get(k, False):
                if self.feature_oriented:
                    key_update_func = self.try_update_feature
                    key_update_kwargs = self.select_feature_kwargs(k, kwargs)
                else:
                    key_update_func = self.try_update_symbol
                    key_update_kwargs = self.select_symbol_kwargs(k, kwargs)
                if "silence_warnings" in func_arg_names:
                    key_update_kwargs["silence_warnings"] = silence_warnings
                tasks.append(
                    Task(
                        key_update_func,
                        k,
                        skip_on_error=skip_on_error,
                        silence_warnings=silence_warnings,
                        update_kwargs=key_update_kwargs,
                    )
                )
                key_indices.append(i)

        outputs = execute(tasks, size=len(self.keys), keys=self.key_index, **execute_kwargs)
        if return_raw:
            return outputs

        new_data = {}
        new_last_index = {}
        new_returned_kwargs = {}
        i = 0
        for k, obj in self.data.items():
            if self.delisted.get(k, False):
                out = None
            else:
                out = outputs[i]
                i += 1
            skip_key = False
            if out is not None:
                if isinstance(out, tuple):
                    new_obj = out[0]
                    new_returned_kwargs[k] = out[1]
                else:
                    new_obj = out
                    new_returned_kwargs[k] = {}
                new_obj = to_any_array(new_obj)
                if new_obj.size == 0:
                    if not silence_warnings:
                        if self.feature_oriented:
                            warn(f"Feature '{str(k)}' returned an empty array. Skipping.")
                        else:
                            warn(f"Symbol '{str(k)}' returned an empty array. Skipping.")
                    skip_key = True
                else:
                    if not isinstance(new_obj, (pd.Series, pd.DataFrame)):
                        new_obj = to_pd_array(new_obj)
                        new_obj.index = pd.RangeIndex(
                            start=obj.index[-1],
                            stop=obj.index[-1] + new_obj.shape[0],
                            step=1,
                        )
                    new_obj = self.prepare_tzaware_index(
                        new_obj,
                        tz_localize=self.tz_localize,
                        tz_convert=self.tz_convert,
                    )
                    if new_obj.index.is_monotonic_decreasing:
                        new_obj = new_obj.iloc[::-1]
                    elif not new_obj.index.is_monotonic_increasing:
                        new_obj = new_obj.sort_index()
                    if new_obj.index.has_duplicates:
                        new_obj = new_obj[~new_obj.index.duplicated(keep="last")]
                    new_data[k] = new_obj
                    if len(new_obj.index) > 0:
                        new_last_index[k] = new_obj.index[-1]
                    else:
                        new_last_index[k] = self.last_index[k]
            else:
                skip_key = True
            if skip_key:
                new_data[k] = obj.iloc[0:0]
                new_last_index[k] = self.last_index[k]

        # Get the last index in the old data from where the new data should begin
        from_index = None
        for k, new_obj in new_data.items():
            if len(new_obj.index) > 0:
                index = new_obj.index[0]
            else:
                continue
            if from_index is None or index < from_index:
                from_index = index
        if from_index is None:
            if not silence_warnings:
                if self.feature_oriented:
                    warn(f"None of the features were updated")
                else:
                    warn(f"None of the symbols were updated")
            return self.copy()

        # Concatenate the updated old data and the new data
        for k, new_obj in new_data.items():
            if len(new_obj.index) > 0:
                to_index = new_obj.index[0]
            else:
                to_index = None
            obj = self.data[k]
            if isinstance(obj, pd.DataFrame) and isinstance(new_obj, pd.DataFrame):
                shared_columns = obj.columns.intersection(new_obj.columns)
                obj = obj[shared_columns]
                new_obj = new_obj[shared_columns]
            elif isinstance(new_obj, pd.DataFrame):
                if obj.name is not None:
                    new_obj = new_obj[obj.name]
                else:
                    new_obj = new_obj[0]
            elif isinstance(obj, pd.DataFrame):
                if new_obj.name is not None:
                    obj = obj[new_obj.name]
                else:
                    obj = obj[0]
            obj = obj.loc[from_index:to_index]
            new_obj = pd.concat((obj, new_obj), axis=0)
            if new_obj.index.has_duplicates:
                new_obj = new_obj[~new_obj.index.duplicated(keep="last")]
            new_data[k] = new_obj

        # Align the index and columns in the new data
        new_data = self.align_index(new_data, missing=self.missing_index, silence_warnings=silence_warnings)
        new_data = self.align_columns(new_data, missing=self.missing_columns, silence_warnings=silence_warnings)

        # Align the columns and data type in the old and new data
        for k, new_obj in new_data.items():
            obj = self.data[k]
            if isinstance(obj, pd.DataFrame) and isinstance(new_obj, pd.DataFrame):
                new_obj = new_obj[obj.columns]
            elif isinstance(new_obj, pd.DataFrame):
                if obj.name is not None:
                    new_obj = new_obj[obj.name]
                else:
                    new_obj = new_obj[0]
            if isinstance(obj, pd.DataFrame):
                new_obj = new_obj.astype(obj.dtypes, errors="ignore")
            else:
                new_obj = new_obj.astype(obj.dtype, errors="ignore")
            new_data[k] = new_obj

        if not concat:
            # Do not concatenate with the old data
            for k, new_obj in new_data.items():
                if isinstance(new_obj.index, pd.DatetimeIndex):
                    new_obj.index.freq = new_obj.index.inferred_freq
            new_index = new_data[self.keys[0]].index
            return self.replace(
                wrapper=self.wrapper.replace(index=new_index),
                data=self.dict_type(new_data),
                returned_kwargs=self.dict_type(new_returned_kwargs),
                last_index=self.dict_type(new_last_index),
            )

        # Append the new data to the old data
        for k, new_obj in new_data.items():
            obj = self.data[k]
            obj = obj.loc[:from_index]
            if obj.index[-1] == from_index:
                obj = obj.iloc[:-1]
            new_obj = pd.concat((obj, new_obj), axis=0)
            if isinstance(new_obj.index, pd.DatetimeIndex):
                new_obj.index.freq = new_obj.index.inferred_freq
            new_data[k] = new_obj

        new_index = new_data[self.keys[0]].index

        return self.replace(
            wrapper=self.wrapper.replace(index=new_index),
            data=self.dict_type(new_data),
            returned_kwargs=self.dict_type(new_returned_kwargs),
            last_index=self.dict_type(new_last_index),
        )

    # ############# Transforming ############# #

    def transform(
        self: DataT,
        transform_func: tp.Callable,
        *args,
        per_feature: bool = False,
        per_symbol: bool = False,
        pass_frame: bool = False,
        key_wrapper_kwargs: tp.KwargsLike = None,
        broadcast_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> DataT:
        """Transform data using a provided transformation function.

        Depending on the configuration, the transformation function is applied to:

        * the entire data if a single key is present or both `per_feature` and `per_symbol` are False,
        * each feature individually if `per_feature` is True and `per_symbol` is False,
        * each symbol individually if `per_symbol` is True and `per_feature` is False,
        * each feature-symbol combination if both `per_feature` and `per_symbol` are True.

        In cases with individual processing, if `pass_frame` is False, a Series is passed to
        `transform_func`; otherwise, a one-column DataFrame is passed.

        After transformation, the data is split by key and a new `Data` instance is created. The new data
        is aligned using `Data.align_data`. If the output is not a Pandas object, it is broadcast to match
        the original data using `broadcast_kwargs`.

        Args:
            transform_func (Callable): Function to apply to the data.
            *args: Positional arguments for `transform_func`.
            per_feature (bool): Process each feature separately.
            per_symbol (bool): Process each symbol separately.
            pass_frame (bool): Pass a one-column DataFrame instead of a Series when processing individual keys.
            key_wrapper_kwargs (KwargsLike): Keyword arguments for `Data.get_key_wrapper`.
            broadcast_kwargs (KwargsLike): Keyword arguments for broadcasting.

                See `vectorbtpro.base.reshaping.broadcast`.
            template_context (KwargsLike): Additional context for template substitution.
            **kwargs: Keyword arguments for `transform_func`.

        Returns:
            Data: New instance with the transformed data.

        !!! note
            The returned object retains the same type and dimensionality as the input.

            * Number and names of columns remain unchanged.
            * To remove columns, use indexing or `Data.select`.
            * To add columns, use column stacking or `Data.merge`.
            * Index may change freely.
        """
        if key_wrapper_kwargs is None:
            key_wrapper_kwargs = {}
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if template_context is None:
            template_context = {}

        def _transform(data, _template_context=None):
            _transform_func = substitute_templates(transform_func, _template_context, eval_id="transform_func")
            _args = substitute_templates(args, _template_context, eval_id="args")
            _kwargs = substitute_templates(kwargs, _template_context, eval_id="kwargs")
            out = _transform_func(data, *_args, **_kwargs)
            if not isinstance(out, (pd.Series, pd.DataFrame)):
                out = broadcast_to(out, data, **broadcast_kwargs)
            return out

        if (self.feature_oriented and (per_feature and not per_symbol)) or (
            self.symbol_oriented and (per_symbol and not per_feature)
        ):
            new_data = self.dict_type()
            for k in self.keys:
                if self.feature_oriented:
                    _template_context = merge_dicts(dict(key=k, feature=k), template_context)
                else:
                    _template_context = merge_dicts(dict(key=k, symbol=k), template_context)
                new_data[k] = _transform(self.data[k], _template_context)
                checks.assert_meta_equal(new_data[k], self.data[k], axis=1)
        elif (self.feature_oriented and (per_symbol and not per_feature)) or (
            self.symbol_oriented and (per_feature and not per_symbol)
        ):
            first_data = self.data[list(self.data.keys())[0]]
            if isinstance(first_data, pd.Series):
                concat_data = pd.concat(self.data.values(), axis=1)
                key_wrapper = self.get_key_wrapper(**key_wrapper_kwargs)
                concat_data.columns = key_wrapper.columns
                if self.feature_oriented:
                    _template_context = merge_dicts(
                        dict(column=self.wrapper.columns[0], symbol=self.wrapper.columns[0]),
                        template_context,
                    )
                else:
                    _template_context = merge_dicts(
                        dict(column=self.wrapper.columns[0], feature=self.wrapper.columns[0]),
                        template_context,
                    )
                new_concat_data = _transform(concat_data, _template_context)
                checks.assert_meta_equal(new_concat_data, concat_data, axis=1)
                new_data = self.dict_type()
                for i, k in enumerate(self.keys):
                    new_data[k] = new_concat_data.iloc[:, i]
                    new_data[k].name = first_data.name
            else:
                all_concat_data = []
                for i in range(len(first_data.columns)):
                    concat_data = pd.concat([self.data[k].iloc[:, [i]] for k in self.keys], axis=1)
                    key_wrapper = self.get_key_wrapper(**key_wrapper_kwargs)
                    concat_data.columns = key_wrapper.columns
                    if self.feature_oriented:
                        _template_context = merge_dicts(
                            dict(column=self.wrapper.columns[i], symbol=self.wrapper.columns[i]),
                            template_context,
                        )
                    else:
                        _template_context = merge_dicts(
                            dict(column=self.wrapper.columns[i], feature=self.wrapper.columns[i]),
                            template_context,
                        )
                    new_concat_data = _transform(concat_data, _template_context)
                    checks.assert_meta_equal(new_concat_data, concat_data, axis=1)
                    all_concat_data.append(new_concat_data)
                new_data = self.dict_type()
                for i, k in enumerate(self.keys):
                    new_objs = []
                    for c in range(len(first_data.columns)):
                        new_objs.append(all_concat_data[c].iloc[:, [i]])
                    new_data[k] = pd.concat(new_objs, axis=1)
                    new_data[k].columns = first_data.columns
        else:
            key_wrapper = self.get_key_wrapper(**key_wrapper_kwargs)
            concat_data = pd.concat(self.data.values(), axis=1, keys=key_wrapper.columns)
            if (self.feature_oriented and (per_feature and per_symbol)) or (
                self.symbol_oriented and (per_symbol and per_feature)
            ):
                new_concat_data = []
                for i in range(len(concat_data.columns)):
                    if self.feature_oriented:
                        _template_context = merge_dicts(
                            dict(
                                key=self.keys[i // len(self.wrapper.columns)],
                                column=self.wrapper.columns[i % len(self.wrapper.columns)],
                                feature=self.keys[i // len(self.wrapper.columns)],
                                symbol=self.wrapper.columns[i % len(self.wrapper.columns)],
                            ),
                            template_context,
                        )
                    else:
                        _template_context = merge_dicts(
                            dict(
                                key=self.wrapper.columns[i % len(self.wrapper.columns)],
                                column=self.keys[i // len(self.wrapper.columns)],
                                feature=self.wrapper.columns[i % len(self.wrapper.columns)],
                                symbol=self.keys[i // len(self.wrapper.columns)],
                            ),
                            template_context,
                        )
                    if pass_frame:
                        new_obj = _transform(concat_data.iloc[:, [i]], _template_context)
                        checks.assert_meta_equal(new_obj, concat_data.iloc[:, [i]], axis=1)
                    else:
                        new_obj = _transform(concat_data.iloc[:, i], _template_context)
                        checks.assert_meta_equal(new_obj, concat_data.iloc[:, i], axis=1)
                    new_concat_data.append(new_obj)
                new_concat_data = pd.concat(new_concat_data, axis=1)
            else:
                new_concat_data = _transform(concat_data)
                checks.assert_meta_equal(new_concat_data, concat_data, axis=1)
            native_concat_data = pd.concat(self.data.values(), axis=1, keys=None)
            new_concat_data.columns = native_concat_data.columns
            new_data = self.dict_type()
            first_data = self.data[list(self.data.keys())[0]]
            for i, k in enumerate(self.keys):
                if isinstance(first_data, pd.Series):
                    new_data[k] = new_concat_data.iloc[:, i]
                    new_data[k].name = first_data.name
                else:
                    start_i = first_data.shape[1] * i
                    stop_i = first_data.shape[1] * (1 + i)
                    new_data[k] = new_concat_data.iloc[:, start_i:stop_i]
                    new_data[k].columns = first_data.columns

        new_data = self.align_data(new_data)
        first_data = new_data[list(new_data.keys())[0]]
        new_wrapper = self.wrapper.replace(index=first_data.index)
        return self.replace(
            wrapper=new_wrapper,
            data=new_data,
        )

    def dropna(self: DataT, **kwargs) -> DataT:
        """Drop missing values from data.

        Calls `Data.transform` with a function that applies `dropna` on each Pandas Series or DataFrame.

        Args:
            **kwargs: Keyword arguments for `pd.Series.dropna` or `pd.DataFrame.dropna`.

        Returns:
            Data: New instance with missing values removed.
        """

        def _dropna(df, **_kwargs):
            return df.dropna(**_kwargs)

        return self.transform(_dropna, **kwargs)

    def mirror_ohlc(
        self: DataT,
        jitted: tp.JittedOption = None,
        chunked: tp.ChunkedOption = None,
        start_value: tp.ArrayLike = np.nan,
        ref_feature: tp.ArrayLike = -1,
    ) -> DataT:
        """Mirror OHLC features in the data.

        Args:
            jitted (JittedOption): Option to control JIT compilation.

                See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            chunked (ChunkedOption): Option to control chunked processing.

                See `vectorbtpro.utils.chunking.resolve_chunked_option`.
            start_value (ArrayLike): Initial value for the transformation.
            ref_feature (ArrayLike): Reference feature used for mirroring.

                Mapped using `vectorbtpro.ohlcv.enums.PriceFeature` if provided as a string.

        Returns:
            Data: New data instance with mirrored OHLC features.

        See:
            `vectorbtpro.ohlcv.nb.mirror_ohlc_nb`
        """
        if isinstance(ref_feature, str):
            ref_feature = map_enum_fields(ref_feature, PriceFeature)

        open_name = self.resolve_feature("Open")
        high_name = self.resolve_feature("High")
        low_name = self.resolve_feature("Low")
        close_name = self.resolve_feature("Close")

        func = jit_reg.resolve_option(mirror_ohlc_nb, jitted)
        func = ch_reg.resolve_option(func, chunked)
        new_open, new_high, new_low, new_close = func(
            self.symbol_wrapper.shape_2d,
            open=to_2d_array(self.get_feature(open_name)) if open_name is not None else None,
            high=to_2d_array(self.get_feature(high_name)) if high_name is not None else None,
            low=to_2d_array(self.get_feature(low_name)) if low_name is not None else None,
            close=to_2d_array(self.get_feature(close_name)) if close_name is not None else None,
            start_value=to_1d_array(start_value),
            ref_feature=to_1d_array(ref_feature),
        )

        def _mirror_ohlc(df, feature, **_kwargs):
            if open_name is not None and feature == open_name:
                return new_open
            if high_name is not None and feature == high_name:
                return new_high
            if low_name is not None and feature == low_name:
                return new_low
            if close_name is not None and feature == close_name:
                return new_close
            return df

        return self.transform(_mirror_ohlc, Rep("feature"), per_feature=True)

    def resample(self: DataT, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> DataT:
        """Perform resampling on `Data`.

        This method automatically resamples features such as "open", "high", "low", "close",
        "volume", "trade count", and "vwap" (case-insensitive). It searches for a `resample_func`
        in `Data.feature_config` for each feature, which should accept the `Data` instance, the feature
        object, and a resampler. If a feature-specific function is not provided, default resampling
        methods are used based on the feature name.

        Args:
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.
            wrapper_meta (DictLike): Metadata from the resampling operation on the wrapper.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.

        Returns:
            Data: New data instance with resampled features.
        """
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)

        def _resample_feature(obj, feature, symbol=None):
            resample_func = self.feature_config.get(feature, {}).get("resample_func", None)
            if resample_func is not None:
                if isinstance(resample_func, str):
                    return obj.vbt.resample_apply(wrapper_meta["resampler"], resample_func)
                return resample_func(self, obj, wrapper_meta["resampler"])
            if isinstance(feature, str) and feature.lower() == "open":
                return obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.first_reduce_nb)
            if isinstance(feature, str) and feature.lower() == "high":
                return obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.max_reduce_nb)
            if isinstance(feature, str) and feature.lower() == "low":
                return obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.min_reduce_nb)
            if isinstance(feature, str) and feature.lower() == "close":
                return obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.last_reduce_nb)
            if isinstance(feature, str) and feature.lower() == "volume":
                return obj.vbt.resample_apply(wrapper_meta["resampler"], generic_nb.sum_reduce_nb)
            if isinstance(feature, str) and feature.lower() == "trade count":
                return obj.vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.sum_reduce_nb,
                    wrap_kwargs=dict(dtype=int),
                )
            if isinstance(feature, str) and feature.lower() == "vwap":
                volume_obj = None
                for feature2 in self.features:
                    if isinstance(feature2, str) and feature2.lower() == "volume":
                        if self.feature_oriented:
                            volume_obj = self.data[feature2]
                        else:
                            volume_obj = self.data[symbol][feature2]
                if volume_obj is None:
                    raise ValueError("Volume is required to resample VWAP")
                return pd.DataFrame.vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.wmean_range_reduce_meta_nb,
                    to_2d_array(obj),
                    to_2d_array(volume_obj),
                    wrapper=self.wrapper[feature],
                )
            raise ValueError(f"Cannot resample feature '{feature}'. Specify resample_func in feature_config.")

        new_data = self.dict_type()
        if self.feature_oriented:
            for feature in self.features:
                new_data[feature] = _resample_feature(self.data[feature], feature)
        else:
            for symbol, obj in self.data.items():
                _new_obj = []
                for feature in self.features:
                    if self.single_feature:
                        _new_obj.append(_resample_feature(obj, feature, symbol=symbol))
                    else:
                        _new_obj.append(_resample_feature(obj[[feature]], feature, symbol=symbol))
                if self.single_feature:
                    new_obj = _new_obj[0]
                else:
                    new_obj = pd.concat(_new_obj, axis=1)
                new_data[symbol] = new_obj

        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            data=new_data,
        )

    def realign(
        self: DataT,
        rule: tp.Optional[tp.AnyRuleLike] = None,
        *args,
        wrapper_meta: tp.DictLike = None,
        ffill: bool = True,
        **kwargs,
    ) -> DataT:
        """Perform realignment on `Data`.

        This method realigns features based on a specified rule or the frequency defined in the wrapper.
        It checks for a `realign_func` in `Data.feature_config` for each feature, which should handle the
        realignment with a resampler and an option to forward-fill missing values. If no function is provided,
        it realigns the "open" feature using `vectorbtpro.generic.accessors.GenericAccessor.realign_opening` and
        applies `vectorbtpro.generic.accessors.GenericAccessor.realign_closing` to other features.

        Args:
            rule (Optional[AnyRuleLike]): Time rule for realignment.

                If not provided, uses `vectorbtpro.base.wrapping.ArrayWrapper.freq`.
            *args: Positional arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.
            wrapper_meta (DictLike): Metadata from the resampling operation on the wrapper.
            ffill (bool): Whether to forward-fill missing values during realignment.
            **kwargs: Keyword arguments for `vectorbtpro.base.wrapping.ArrayWrapper.resample_meta`.

        Returns:
            Data: New data instance with realigned features.
        """
        if rule is None:
            rule = self.wrapper.freq
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(rule, *args, **kwargs)

        def _realign_feature(obj, feature, symbol=None):
            realign_func = self.feature_config.get(feature, {}).get("realign_func", None)
            if realign_func is not None:
                if isinstance(realign_func, str):
                    return getattr(obj.vbt, realign_func)(wrapper_meta["resampler"], ffill=ffill)
                return realign_func(self, obj, wrapper_meta["resampler"], ffill=ffill)
            if isinstance(feature, str) and feature.lower() == "open":
                return obj.vbt.realign_opening(wrapper_meta["resampler"], ffill=ffill)
            return obj.vbt.realign_closing(wrapper_meta["resampler"], ffill=ffill)

        new_data = self.dict_type()
        if self.feature_oriented:
            for feature in self.features:
                new_data[feature] = _realign_feature(self.data[feature], feature)
        else:
            for symbol, obj in self.data.items():
                _new_obj = []
                for feature in self.features:
                    if self.single_feature:
                        _new_obj.append(_realign_feature(obj, feature, symbol=symbol))
                    else:
                        _new_obj.append(_realign_feature(obj[[feature]], feature, symbol=symbol))
                if self.single_feature:
                    new_obj = _new_obj[0]
                else:
                    new_obj = pd.concat(_new_obj, axis=1)
                new_data[symbol] = new_obj

        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            data=new_data,
        )

    # ############# Running ############# #

    @classmethod
    def try_run(
        cls,
        data: "Data",
        func_name: str,
        *args,
        raise_errors: bool = False,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Any:
        """Try to run a function on the given `Data` instance.

        Executes the `run` method on `data` with the provided arguments. If an exception is raised,
        the error is either re-raised or a warning is issued based on the `raise_errors` and
        `silence_warnings` flags.

        Args:
            data (Data): Data instance on which to execute the function.
            func_name (str): Name identifying the function to run.
            *args: Positional arguments for `Data.run`.
            raise_errors (bool): If True, raises any exceptions encountered.
            silence_warnings (bool): Flag to suppress warning messages.
            **kwargs: Keyword arguments for `Data.run`.

        Returns:
            Any: Result of executing `Data.run`, or `vectorbtpro.utils.execution.NoResult`
                if execution fails without raising an error.
        """
        try:
            return data.run(*args, **kwargs)
        except Exception as e:
            if raise_errors:
                raise e
            if not silence_warnings:
                warn(func_name + ": " + str(e))
        return NoResult

    @classmethod
    def select_run_func_args(cls, i: int, func_name: str, args: tp.Args) -> tuple:
        """Select positional arguments corresponding to a function index or name.

        Iterates over the provided arguments and, for items that are instances of `run_func_dict`,
        extracts the value associated with the given function name or index. If a default value is
        defined with the key "_def", that value is used when neither the function name nor the index
        is found. Items that are not instances of `run_func_dict` are returned unchanged.

        Args:
            i (int): Index used to look up an argument.
            func_name (str): Name used to look up an argument.
            args (Args): Tuple of positional arguments to search through.

        Returns:
            tuple: Tuple containing the selected arguments corresponding to the given function index or name.
        """
        _args = ()
        for v in args:
            if isinstance(v, run_func_dict):
                if func_name in v:
                    _args += (v[func_name],)
                elif i in v:
                    _args += (v[i],)
                elif "_def" in v:
                    _args += (v["_def"],)
            else:
                _args += (v,)
        return _args

    @classmethod
    def select_run_func_kwargs(cls, i: int, func_name: str, kwargs: tp.Kwargs) -> dict:
        """Select keyword arguments corresponding to a runnable function based on its index or name.

        Args:
            i (int): Index used to select the matching function arguments.
            func_name (str): Name of the function to match during argument selection.
            kwargs (Kwargs): Dictionary where each value can be:

                * `run_func_dict` mapping function names or indices to specific argument values.
                * `run_arg_dict` that updates arguments if its key matches the provided function name or index.
                * Direct value if no special type is applicable.

        Returns:
            dict: Dictionary containing the keyword arguments selected based
                on the provided index or function name.
        """
        _kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, run_func_dict):
                if func_name in v:
                    _kwargs[k] = v[func_name]
                elif i in v:
                    _kwargs[k] = v[i]
                elif "_def" in v:
                    _kwargs[k] = v["_def"]
            elif isinstance(v, run_arg_dict):
                if func_name == k or i == k:
                    _kwargs.update(v)
            else:
                _kwargs[k] = v
        return _kwargs

    def run(
        self,
        func: tp.MaybeIterable[tp.Union[tp.Hashable, tp.Callable]],
        *args,
        on_features: tp.Optional[tp.MaybeFeatures] = None,
        on_symbols: tp.Optional[tp.MaybeSymbols] = None,
        func_args: tp.ArgsLike = None,
        func_kwargs: tp.KwargsLike = None,
        magnet_kwargs: tp.KwargsLike = None,
        ignore_args: tp.Optional[tp.Sequence[str]] = None,
        rename_args: tp.DictLike = None,
        location: tp.Optional[str] = None,
        prepend_location: tp.Optional[bool] = None,
        unpack: tp.Union[bool, str] = False,
        concat: bool = True,
        data_kwargs: tp.KwargsLike = None,
        silence_warnings: bool = False,
        raise_errors: bool = False,
        execute_kwargs: tp.KwargsLike = None,
        filter_results: bool = True,
        raise_no_results: bool = True,
        merge_func: tp.MergeFuncLike = None,
        merge_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        return_keys: bool = False,
        _func_name: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.Any:
        """Run a function on data.

        This method applies a function or a collection of functions to the data contained in the
        current object. It automatically inspects the signature of the provided function and fills in
        parameters using object attributes (e.g., substituting `open` with `Data.open`).

        The `func` parameter can be one of the following:

        * Location to compute all indicators. See `vectorbtpro.indicators.factory.IndicatorFactory.list_locations`.
        * Indicator name. See `vectorbtpro.indicators.factory.IndicatorFactory.get_indicator`.
        * Simulation method from `vectorbtpro.portfolio.base.Portfolio`.
        * Any callable.
        * Iterable of the above, with results aggregated as columns in a DataFrame.

        Use `magnet_kwargs` to pass keyword arguments that are provided only if they are present in the
        function signature. Use `rename_args` to map argument names; for example, data can be passed under a
        different name as expected by a simulation method.

        Set `unpack` to True, "dict", or "frame" to automatically post-process the result using the corresponding
        methods: `vectorbtpro.indicators.factory.IndicatorBase.unpack`, `to_dict`, or `to_frame`.

        Args:
            func (MaybeIterable[Union[Hashable, Callable]]): Function, location, indicator name,
                or simulation method to run, or an iterable of such.
            *args: Positional arguments for the function.
            on_features (Optional[MaybeFeatures]): Features identifier(s) used to filter the data.
            on_symbols (Optional[MaybeSymbols]): Symbols identifier(s) used to filter the data.
            func_args (ArgsLike): Extra positional arguments for the function.
            func_kwargs (KwargsLike): Extra keyword arguments for the function.
            magnet_kwargs (KwargsLike): Keyword arguments injected only if they
                match the function signature.
            ignore_args (Optional[Sequence[str]]): Names of arguments to ignore when
                auto-assigning object attributes.
            rename_args (DictLike): Mapping of argument names to substitute names.
            location (Optional[str]): Identifier used to specify the location for indicator selection.
            prepend_location (Optional[bool]): When True, indicator names are prefixed with their location.
            unpack (Union[bool, str]): Determines the processing of the function output.

                Use True, "dict", or "frame" to apply post-processing.
            concat (bool): If True, concatenates results from multiple function calls.
            data_kwargs (KwargsLike): Keyword arguments for `Data.from_data`.
            silence_warnings (bool): Flag to suppress warning messages.
            raise_errors (bool): If True, raises any exceptions encountered.
            execute_kwargs (KwargsLike): Keyword arguments for the execution handler.

                See `vectorbtpro.utils.execution.execute`.
            filter_results (bool): Whether to filter out results that are `vectorbtpro.utils.execution.NoResult`.
            raise_no_results (bool): Flag indicating whether to raise a
                `vectorbtpro.utils.execution.NoResultsException` exception if no results remain.
            merge_func (MergeFuncLike): Function to merge the results.

                See `vectorbtpro.utils.merging.MergeFunc`.
            merge_kwargs (KwargsLike): Keyword arguments for `merge_func`.
            template_context (KwargsLike): Additional context for template substitution.
            return_keys (bool): If True, includes keys representing function names in the return.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: Result of applying the function(s) to the data. If `return_keys` is True,
                returns a tuple of the results and the corresponding function keys.
        """
        from vectorbtpro.indicators.factory import IndicatorBase, IndicatorFactory
        from vectorbtpro.indicators.talib_ import talib_func
        from vectorbtpro.portfolio.base import Portfolio

        if magnet_kwargs is None:
            magnet_kwargs = {}
        if data_kwargs is None:
            data_kwargs = {}
        if execute_kwargs is None:
            execute_kwargs = {}
        if merge_kwargs is None:
            merge_kwargs = {}

        _self = self
        if on_features is not None:
            _self = _self.select_features(on_features)
        if on_symbols is not None:
            _self = _self.select_symbols(on_symbols)

        if checks.is_complex_iterable(func):
            tasks = []
            keys = []
            for i, f in enumerate(func):
                _location = location
                if callable(f):
                    func_name = f.__name__
                elif isinstance(f, str):
                    if _location is not None:
                        func_name = f.lower().strip()
                        if func_name == "*":
                            func_name = "all"
                        if prepend_location is True:
                            func_name = _location + "_" + func_name
                    else:
                        _location, f = IndicatorFactory.split_indicator_name(f)
                        if f is None:
                            raise ValueError("Sequence of locations is not supported")
                        func_name = f.lower().strip()
                        if func_name == "*":
                            func_name = "all"
                        if _location is not None:
                            if prepend_location in (None, True):
                                func_name = _location + "_" + func_name
                else:
                    func_name = f
                new_args = _self.select_run_func_args(i, func_name, args)
                new_args = (_self, func_name, f, *new_args)
                new_kwargs = _self.select_run_func_kwargs(i, func_name, kwargs)
                if concat and _location == "talib_func":
                    new_kwargs["unpack_to"] = "frame"
                new_kwargs = {
                    **dict(
                        func_args=func_args,
                        func_kwargs=func_kwargs,
                        magnet_kwargs=magnet_kwargs,
                        ignore_args=ignore_args,
                        rename_args=rename_args,
                        location=_location,
                        prepend_location=prepend_location,
                        unpack="frame" if concat else unpack,
                        concat=concat,
                        data_kwargs=data_kwargs,
                        silence_warnings=silence_warnings,
                        raise_errors=raise_errors,
                        execute_kwargs=execute_kwargs,
                        merge_func=merge_func,
                        merge_kwargs=merge_kwargs,
                        template_context=template_context,
                        return_keys=return_keys,
                        _func_name=func_name,
                    ),
                    **new_kwargs,
                }

                tasks.append(Task(self.try_run, *new_args, **new_kwargs))
                keys.append(str(func_name))

            keys = pd.Index(keys, name="run_func")
            results = execute(tasks, size=len(keys), keys=keys, **execute_kwargs)
            if filter_results:
                try:
                    results, keys = filter_out_no_results(results, keys=keys)
                except NoResultsException as e:
                    if raise_no_results:
                        raise e
                    return NoResult
                no_results_filtered = True
            else:
                no_results_filtered = False

            if merge_func is None and concat:
                merge_func = "column_stack"
            if merge_func is not None:
                if is_merge_func_from_config(merge_func):
                    merge_kwargs = merge_dicts(
                        dict(
                            keys=keys,
                            filter_results=not no_results_filtered,
                            raise_no_results=raise_no_results,
                        ),
                        merge_kwargs,
                    )
                if isinstance(merge_func, MergeFunc):
                    merge_func = merge_func.replace(merge_kwargs=merge_kwargs, context=template_context)
                else:
                    merge_func = MergeFunc(merge_func, merge_kwargs=merge_kwargs, context=template_context)
                if return_keys:
                    return merge_func(results), keys
                return merge_func(results)
            if return_keys:
                return results, keys
            return results

        if isinstance(func, str):
            func_name = func.lower().strip()
            if func_name.startswith("from_") and getattr(Portfolio, func_name):
                func = getattr(Portfolio, func_name)
                if func_args is None:
                    func_args = ()
                if func_kwargs is None:
                    func_kwargs = {}
                pf = func(_self, *args, *func_args, **kwargs, **func_kwargs)
                if isinstance(pf, Portfolio) and unpack:
                    raise ValueError("Portfolio cannot be unpacked")
                return pf
            if location is None:
                location, func_name = IndicatorFactory.split_indicator_name(func_name)
            if location is not None and (func_name is None or func_name == "all" or func_name == "*"):
                matched_location = IndicatorFactory.match_location(location)
                if matched_location is not None:
                    location = matched_location
                if func_name == "all" or func_name == "*":
                    if prepend_location is None:
                        prepend_location = True
                else:
                    if prepend_location is None:
                        prepend_location = False
                if location == "talib_func":
                    indicators = IndicatorFactory.list_indicators("talib", prepend_location=False)
                else:
                    indicators = IndicatorFactory.list_indicators(location, prepend_location=False)
                return _self.run(
                    indicators,
                    *args,
                    func_args=func_args,
                    func_kwargs=func_kwargs,
                    magnet_kwargs=magnet_kwargs,
                    ignore_args=ignore_args,
                    rename_args=rename_args,
                    location=location,
                    prepend_location=prepend_location,
                    unpack=unpack,
                    concat=concat,
                    data_kwargs=data_kwargs,
                    silence_warnings=silence_warnings,
                    raise_errors=raise_errors,
                    execute_kwargs=execute_kwargs,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                    return_keys=return_keys,
                    **kwargs,
                )
            if location is not None:
                matched_location = IndicatorFactory.match_location(location)
                if matched_location is not None:
                    location = matched_location
                if location == "talib_func":
                    func = talib_func(func_name)
                else:
                    func = IndicatorFactory.get_indicator(func_name, location=location)
            else:
                func = IndicatorFactory.get_indicator(func_name)
        if isinstance(func, type) and issubclass(func, IndicatorBase):
            func = func.run

        with_kwargs = {}
        func_arg_names = get_func_arg_names(func)
        for arg_name in func_arg_names:
            real_arg_name = arg_name
            if ignore_args is not None:
                if arg_name in ignore_args:
                    continue
            if rename_args is not None:
                if arg_name in rename_args:
                    arg_name = rename_args[arg_name]
            if real_arg_name not in kwargs:
                if arg_name == "data":
                    with_kwargs[real_arg_name] = _self
                elif arg_name == "wrapper":
                    with_kwargs[real_arg_name] = _self.symbol_wrapper
                elif arg_name in ("input_shape", "shape"):
                    with_kwargs[real_arg_name] = _self.shape
                elif arg_name in ("target_shape", "shape_2d"):
                    with_kwargs[real_arg_name] = _self.shape_2d
                elif arg_name in ("input_index", "index"):
                    with_kwargs[real_arg_name] = _self.index
                elif arg_name in ("input_columns", "columns"):
                    with_kwargs[real_arg_name] = _self.columns
                elif arg_name == "freq":
                    with_kwargs[real_arg_name] = _self.freq
                elif arg_name == "hlc3":
                    with_kwargs[real_arg_name] = _self.hlc3
                elif arg_name == "ohlc4":
                    with_kwargs[real_arg_name] = _self.ohlc4
                elif arg_name == "returns":
                    with_kwargs[real_arg_name] = _self.returns
                else:
                    feature_idx = _self.get_feature_idx(arg_name)
                    if feature_idx != -1:
                        with_kwargs[real_arg_name] = _self.get_feature(feature_idx)
        kwargs = dict(kwargs)
        for k, v in magnet_kwargs.items():
            if k in func_arg_names:
                kwargs[k] = v
        new_args, new_kwargs = extend_args(func, args, kwargs, **with_kwargs)
        if func_args is None:
            func_args = ()
        if func_kwargs is None:
            func_kwargs = {}
        out = func(*new_args, *func_args, **new_kwargs, **func_kwargs)
        if isinstance(unpack, bool):
            if unpack:
                if isinstance(out, IndicatorBase):
                    out = out.unpack()
        elif isinstance(unpack, str) and unpack.lower() == "dict":
            if isinstance(out, IndicatorBase):
                out = out.to_dict()
            else:
                if _func_name is None:
                    feature_name = func.__name__
                else:
                    feature_name = _func_name
                out = {feature_name: out}
        elif isinstance(unpack, str) and unpack.lower() == "frame":
            if isinstance(out, IndicatorBase):
                out = out.to_frame()
            elif isinstance(out, pd.Series):
                out = out.to_frame()
        elif isinstance(unpack, str) and unpack.lower() == "data":
            if isinstance(out, IndicatorBase):
                out = feature_dict(out.to_dict())
            else:
                if _func_name is None:
                    feature_name = func.__name__
                else:
                    feature_name = _func_name
                out = feature_dict({feature_name: out})
            out = Data.from_data(out, **data_kwargs)
        else:
            raise ValueError(f"Invalid unpack: '{unpack}'")
        return out

    # ############# Persisting ############# #

    def resolve_key_arg(
        self,
        arg: tp.Any,
        k: tp.Key,
        arg_name: str,
        check_dict_type: bool = True,
        template_context: tp.KwargsLike = None,
        is_kwargs: bool = False,
    ) -> tp.Any:
        """Resolve an argument based on its type and key.

        Args:
            arg (Any): Input argument to process.
            k (Key): Key used for lookup when the argument is a key dictionary.
            arg_name (str): Name of the argument for validation and template evaluation.
            check_dict_type (bool): Flag to validate the type of dictionaries.
            template_context (KwargsLike): Additional context for template substitution.
            is_kwargs (bool): Specifies if the argument should be handled as keyword arguments.

        Returns:
            Any: Resolved argument with any applicable template substitutions.
        """
        if check_dict_type:
            self.check_dict_type(arg, arg_name=arg_name)
        if isinstance(arg, key_dict):
            _arg = arg[k]
        else:
            if is_kwargs:
                _arg = self.select_key_kwargs(k, arg, check_dict_type=check_dict_type)
            else:
                _arg = arg
        if isinstance(_arg, CustomTemplate):
            _arg = _arg.substitute(template_context, eval_id=arg_name)
        elif is_kwargs:
            _arg = substitute_templates(_arg, template_context, eval_id=arg_name)
        return _arg

    def to_csv(
        self,
        path_or_buf: tp.Union[tp.PathLike, feature_dict, symbol_dict, CustomTemplate] = ".",
        ext: tp.Union[str, feature_dict, symbol_dict, CustomTemplate] = "csv",
        mkdir_kwargs: tp.Union[tp.KwargsLike, feature_dict, symbol_dict, CustomTemplate] = None,
        check_dict_type: bool = True,
        template_context: tp.KwargsLike = None,
        return_meta: bool = False,
        **kwargs,
    ) -> tp.Union[None, feature_dict, symbol_dict]:
        """Save data to CSV file(s) using Pandas.DataFrame.to_csv.

        Uses `pd.DataFrame.to_csv` to write each DataFrame to a CSV file. If `path_or_buf` is a directory
        or does not include a file suffix, each feature or symbol is saved to an individual file.

        Args:
            path_or_buf (Union[PathLike, feature_dict, symbol_dict, CustomTemplate]): File path or buffer.

                If a directory is provided, each feature or symbol is saved in a separate file.
            ext (Union[str, feature_dict, symbol_dict, CustomTemplate]):
                File extension to use for CSV files when saving multiple files.
            mkdir_kwargs (Union[KwargsLike, feature_dict, symbol_dict, CustomTemplate]):
                Keyword arguments for directory creation.

                See `vectorbtpro.utils.path_.check_mkdir`.
            check_dict_type (bool): Flag to validate the type of dictionaries.
            template_context (KwargsLike): Additional context for template substitution.
            return_meta (bool): If True, returns metadata about the saved CSV file(s).
            **kwargs: Keyword arguments for `pd.DataFrame.to_csv`.

        Returns:
            Union[None, feature_dict, symbol_dict]: Dictionary with metadata if `return_meta` is True, or None.
        """
        meta = self.dict_type()
        for k, v in self.data.items():
            if self.feature_oriented:
                _template_context = merge_dicts(dict(data=v, key=k, feature=k), template_context)
            else:
                _template_context = merge_dicts(dict(data=v, key=k, symbol=k), template_context)
            _kwargs = self.select_key_kwargs(k, kwargs, check_dict_type=check_dict_type)
            sep = _kwargs.pop("sep", None)
            _path_or_buf = self.resolve_key_arg(
                path_or_buf,
                k,
                "path_or_buf",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            if isinstance(_path_or_buf, str):
                _path_or_buf = Path(_path_or_buf)
            if isinstance(_path_or_buf, Path):
                if (_path_or_buf.exists() and _path_or_buf.is_dir()) or _path_or_buf.suffix == "":
                    _ext = self.resolve_key_arg(
                        ext,
                        k,
                        "ext",
                        check_dict_type=check_dict_type,
                        template_context=_template_context,
                    )
                    _path_or_buf /= f"{k}.{_ext}"
                _mkdir_kwargs = self.resolve_key_arg(
                    mkdir_kwargs,
                    k,
                    "mkdir_kwargs",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                    is_kwargs=True,
                )
                check_mkdir(_path_or_buf.parent, **_mkdir_kwargs)
                if _path_or_buf.suffix.lower() == ".csv":
                    if sep is None:
                        sep = ","
                if _path_or_buf.suffix.lower() == ".tsv":
                    if sep is None:
                        sep = "\t"
                _path_or_buf = str(_path_or_buf)
            if sep is None:
                sep = ","
            meta[k] = {"path_or_buf": _path_or_buf, "sep": sep, **_kwargs}
            v.to_csv(**meta[k])

        if return_meta:
            return meta
        return None

    @classmethod
    def from_csv(cls: tp.Type[DataT], *args, fetch_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Load data from CSV files and convert the instance to the current class.

        Uses `vectorbtpro.data.custom.csv.CSVData` to load CSV data and then switches the resulting instance
        to the current class.

        Args:
            *args: Positional arguments for `vectorbtpro.data.custom.csv.CSVData.pull`.
            fetch_kwargs (KwargsLike): Keyword arguments originally used for fetching data.

                Refer to the fetching function for further details.
            **kwargs: Keyword arguments for `vectorbtpro.data.custom.csv.CSVData.pull`.

        Returns:
            Data: Instance of the class with the loaded data.
        """
        from vectorbtpro.data.custom.csv import CSVData

        if fetch_kwargs is None:
            fetch_kwargs = {}
        data = CSVData.pull(*args, **kwargs)
        data = data.switch_class(cls, clear_fetch_kwargs=True, clear_returned_kwargs=True)
        data = data.update_fetch_kwargs(**fetch_kwargs)
        return data

    def to_hdf(
        self,
        path_or_buf: tp.Union[tp.PathLike, feature_dict, symbol_dict, CustomTemplate] = ".",
        key: tp.Union[None, str, feature_dict, symbol_dict, CustomTemplate] = None,
        mkdir_kwargs: tp.Union[tp.KwargsLike, feature_dict, symbol_dict, CustomTemplate] = None,
        format: str = "table",
        check_dict_type: bool = True,
        template_context: tp.KwargsLike = None,
        return_meta: bool = False,
        **kwargs,
    ) -> tp.Union[None, feature_dict, symbol_dict]:
        """Save data to an HDF file using PyTables.

        Uses `pd.DataFrame.to_hdf` to write each DataFrame to an HDF file.
        If `path_or_buf` is a directory or does not have a file suffix, a file named
        after the class (with a .h5 extension) is created automatically.

        Args:
            path_or_buf (Union[PathLike, feature_dict, symbol_dict, CustomTemplate]): File path or buffer.

                If a directory is provided, each feature or symbol is saved in a separate file.
            key (Union[None, str, feature_dict, symbol_dict, CustomTemplate]):
                HDF key under which to store the data.

                If None, the key defaults to the string representation of the feature or symbol.
            mkdir_kwargs (Union[KwargsLike, feature_dict, symbol_dict, CustomTemplate]):
                Keyword arguments for directory creation.

                See `vectorbtpro.utils.path_.check_mkdir`.
            format (str): File format for HDF storage.
            check_dict_type (bool): Flag to validate the type of dictionaries.
            template_context (KwargsLike): Additional context for template substitution.
            return_meta (bool): If True, returns metadata about the saved HDF file(s).
            **kwargs: Keyword arguments for `pd.DataFrame.to_hdf`.

        Returns:
            Union[None, feature_dict, symbol_dict]: Dictionary with metadata if `return_meta` is True, or None.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tables")

        meta = self.dict_type()
        for k, v in self.data.items():
            if self.feature_oriented:
                _template_context = merge_dicts(dict(data=v, key=k, feature=k), template_context)
            else:
                _template_context = merge_dicts(dict(data=v, key=k, symbol=k), template_context)
            _path_or_buf = self.resolve_key_arg(
                path_or_buf,
                k,
                "path_or_buf",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            if isinstance(_path_or_buf, str):
                _path_or_buf = Path(_path_or_buf)
            if isinstance(_path_or_buf, Path):
                if (_path_or_buf.exists() and _path_or_buf.is_dir()) or _path_or_buf.suffix == "":
                    _path_or_buf /= type(self).__name__ + ".h5"
                _mkdir_kwargs = self.resolve_key_arg(
                    mkdir_kwargs,
                    k,
                    "mkdir_kwargs",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                    is_kwargs=True,
                )
                check_mkdir(_path_or_buf.parent, **_mkdir_kwargs)
                _path_or_buf = str(_path_or_buf)
            if key is None:
                _key = str(k)
            else:
                _key = self.resolve_key_arg(
                    key,
                    k,
                    "key",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                )
            _kwargs = self.select_key_kwargs(k, kwargs, check_dict_type=check_dict_type)
            meta[k] = {"path_or_buf": _path_or_buf, "key": _key, "format": format, **_kwargs}
            v.to_hdf(**meta[k])

        if return_meta:
            return meta
        return None

    @classmethod
    def from_hdf(cls: tp.Type[DataT], *args, fetch_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Load data from an HDF file and convert the instance to the current class.

        Uses `vectorbtpro.data.custom.hdf.HDFData` to load HDF data and then switches the resulting instance
        to the current class.

        Args:
            *args: Positional arguments for `vectorbtpro.data.custom.hdf.HDFData.pull`.
            fetch_kwargs (KwargsLike): Keyword arguments originally used for fetching data.

                Refer to the fetching function for further details.
            **kwargs: Keyword arguments for `vectorbtpro.data.custom.hdf.HDFData.pull`.

        Returns:
            Data: Instance of the class with the loaded data.
        """
        from vectorbtpro.data.custom.hdf import HDFData

        if fetch_kwargs is None:
            fetch_kwargs = {}
        data = HDFData.pull(*args, **kwargs)
        data = data.switch_class(cls, clear_fetch_kwargs=True, clear_returned_kwargs=True)
        data = data.update_fetch_kwargs(**fetch_kwargs)
        return data

    def to_feather(
        self,
        path_or_buf: tp.Union[tp.PathLike, feature_dict, symbol_dict, CustomTemplate] = ".",
        mkdir_kwargs: tp.Union[tp.KwargsLike, feature_dict, symbol_dict, CustomTemplate] = None,
        check_dict_type: bool = True,
        template_context: tp.KwargsLike = None,
        return_meta: bool = False,
        **kwargs,
    ) -> tp.Union[None, feature_dict, symbol_dict]:
        """Save data to Feather file(s) using PyArrow.

        Save the instance's data as one or multiple Feather file(s). If `path_or_buf` specifies
        a directory or lacks a file suffix, each feature (or symbol) is saved to a separate
        file with a name based on its key. Otherwise, the data is saved to a single file.

        Args:
            path_or_buf (Union[PathLike, feature_dict, symbol_dict, CustomTemplate]): File path or buffer.

                When a directory is provided or no file suffix is detected,
                the key is appended with a ".feather" extension to form the file name.
            mkdir_kwargs (Union[KwargsLike, feature_dict, symbol_dict, CustomTemplate]):
                Keyword arguments for directory creation.

                See `vectorbtpro.utils.path_.check_mkdir`.
            check_dict_type (bool): Flag to validate the type of dictionaries.
            template_context (KwargsLike): Additional context for template substitution.
            return_meta (bool): Flag specifying whether to return a metadata dictionary
                containing file paths and saving options.
            **kwargs:
                Keyword arguments passed to DataFrame.to_feather.

        Returns:
            Union[None, feature_dict, symbol_dict]: Dictionary with metadata if `return_meta` is True, or None.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("pyarrow")

        meta = self.dict_type()
        for k, v in self.data.items():
            if self.feature_oriented:
                _template_context = merge_dicts(dict(data=v, key=k, feature=k), template_context)
            else:
                _template_context = merge_dicts(dict(data=v, key=k, symbol=k), template_context)
            _path_or_buf = self.resolve_key_arg(
                path_or_buf,
                k,
                "path_or_buf",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            if isinstance(_path_or_buf, str):
                _path_or_buf = Path(_path_or_buf)
            if isinstance(_path_or_buf, Path):
                if (_path_or_buf.exists() and _path_or_buf.is_dir()) or _path_or_buf.suffix == "":
                    _path_or_buf /= f"{k}.feather"
                _mkdir_kwargs = self.resolve_key_arg(
                    mkdir_kwargs,
                    k,
                    "mkdir_kwargs",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                    is_kwargs=True,
                )
                check_mkdir(_path_or_buf.parent, **_mkdir_kwargs)
                _path_or_buf = str(_path_or_buf)
            _kwargs = self.select_key_kwargs(k, kwargs, check_dict_type=check_dict_type)
            meta[k] = {"path": _path_or_buf, **_kwargs}
            if isinstance(v, pd.Series):
                v = v.to_frame()
            try:
                v.to_feather(**meta[k])
            except Exception as e:
                if isinstance(e, ValueError) and "you can .reset_index()" in str(e):
                    v = v.reset_index()
                    v.to_feather(**meta[k])
                else:
                    raise e

        if return_meta:
            return meta
        return None

    @classmethod
    def from_feather(cls: tp.Type[DataT], *args, fetch_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Load data from Feather files and switch the class of the loaded data.

        Load data using `vectorbtpro.data.custom.feather.FeatherData.pull`, change its class
        to the current class, and update its fetch parameters.

        Args:
            *args: Positional arguments for `vectorbtpro.data.custom.feather.FeatherData.pull`.
            fetch_kwargs (KwargsLike): Keyword arguments originally used for fetching data.

                Refer to the fetching function for further details.
            **kwargs: Keyword arguments for `vectorbtpro.data.custom.feather.FeatherData.pull`.

        Returns:
            Data: Instance of the class containing the loaded data.
        """
        from vectorbtpro.data.custom.feather import FeatherData

        if fetch_kwargs is None:
            fetch_kwargs = {}
        data = FeatherData.pull(*args, **kwargs)
        data = data.switch_class(cls, clear_fetch_kwargs=True, clear_returned_kwargs=True)
        data = data.update_fetch_kwargs(**fetch_kwargs)
        return data

    def to_parquet(
        self,
        path_or_buf: tp.Union[tp.PathLike, feature_dict, symbol_dict, CustomTemplate] = ".",
        mkdir_kwargs: tp.Union[tp.KwargsLike, feature_dict, symbol_dict, CustomTemplate] = None,
        partition_cols: tp.Union[None, tp.List[str], feature_dict, symbol_dict, CustomTemplate] = None,
        partition_by: tp.Union[None, tp.AnyGroupByLike, feature_dict, symbol_dict, CustomTemplate] = None,
        period_index_to: tp.Union[str, tp.AnyGroupByLike, feature_dict, symbol_dict, CustomTemplate] = "str",
        groupby_kwargs: tp.Union[None, tp.AnyGroupByLike, feature_dict, symbol_dict, CustomTemplate] = None,
        keep_groupby_names: tp.Union[bool, feature_dict, symbol_dict, CustomTemplate] = False,
        engine: tp.Union[None, str, feature_dict, symbol_dict, CustomTemplate] = None,
        check_dict_type: bool = True,
        template_context: tp.KwargsLike = None,
        return_meta: bool = False,
        **kwargs,
    ) -> tp.Union[None, feature_dict, symbol_dict]:
        """Save data to Parquet file(s) using PyArrow or FastParquet.

        This method writes each feature or symbol's data to a Parquet file using Pandas'
        `DataFrame.to_parquet. If a directory path is provided via `path_or_buf`, a separate file
        is generated for each key. For a single file, specify the full file path; for multiple files,
        wrap the paths using `key_dict`.

        Args:
            path_or_buf (Union[PathLike, feature_dict, symbol_dict, CustomTemplate]): File path or buffer.

                If a directory is provided, each feature or symbol is saved in a separate file.
            mkdir_kwargs (Union[KwargsLike, feature_dict, symbol_dict, CustomTemplate]):
                Keyword arguments for directory creation.

                See `vectorbtpro.utils.path_.check_mkdir`.
            partition_cols (Union[None, List[str], feature_dict, symbol_dict, CustomTemplate]):
                Column names for partitioning the data.

                Must be None if `partition_by` is provided.
            partition_by (Union[None, AnyGroupByLike, feature_dict, symbol_dict, CustomTemplate]):
                Criteria for grouping the index before partitioning.

                When provided, the index is grouped using `vectorbtpro.base.wrapping.ArrayWrapper.get_index_grouper`
                with `groupby_kwargs`, and the resulting groups are added as partition columns.
            period_index_to (Union[str, AnyGroupByLike, feature_dict, symbol_dict, CustomTemplate]):
                Specifies how to convert a PeriodIndex.

                Use "str" to convert periods to strings or provide another method for timestamp conversion.
            groupby_kwargs (Union[None, AnyGroupByLike, feature_dict, symbol_dict, CustomTemplate]):
                Keyword arguments for `pandas.Series.groupby` and `pandas.Series.resample` methods.

                See `vectorbtpro.base.accessors.BaseIDXAccessor.get_grouper`.
            keep_groupby_names (Union[bool, feature_dict, symbol_dict, CustomTemplate]):
                Flag indicating whether to retain original group names when partitioning.
            engine (Union[None, str, feature_dict, symbol_dict, CustomTemplate]):
                Parquet engine to use; valid options are "pyarrow", "fastparquet", or "auto".
            check_dict_type (bool): Flag to validate the type of dictionaries.
            template_context (KwargsLike): Additional context for template substitution.
            return_meta (bool): If True, returns a metadata dictionary containing
                file paths and configuration settings.
            **kwargs: Keyword arguments for `DataFrame.to_parquet`.

        Returns:
            Union[None, feature_dict, symbol_dict]: Dictionary with metadata if `return_meta` is True, or None.
        """
        from vectorbtpro.utils.module_ import assert_can_import, assert_can_import_any
        from vectorbtpro.data.custom.parquet import ParquetData

        meta = self.dict_type()
        for k, v in self.data.items():
            if self.feature_oriented:
                _template_context = merge_dicts(dict(data=v, key=k, feature=k), template_context)
            else:
                _template_context = merge_dicts(dict(data=v, key=k, symbol=k), template_context)
            _partition_cols = self.resolve_key_arg(
                partition_cols,
                k,
                "partition_cols",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _partition_by = self.resolve_key_arg(
                partition_by,
                k,
                "partition_by",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            if _partition_cols is not None and _partition_by is not None:
                raise ValueError("Must use either partition_cols or partition_by, not both")
            _path_or_buf = self.resolve_key_arg(
                path_or_buf,
                k,
                "path_or_buf",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            if isinstance(_path_or_buf, str):
                _path_or_buf = Path(_path_or_buf)
            if isinstance(_path_or_buf, Path):
                if (_path_or_buf.exists() and _path_or_buf.is_dir()) or _path_or_buf.suffix == "":
                    if _partition_cols is not None or _partition_by is not None:
                        _path_or_buf /= f"{k}"
                    else:
                        _path_or_buf /= f"{k}.parquet"
                _mkdir_kwargs = self.resolve_key_arg(
                    mkdir_kwargs,
                    k,
                    "mkdir_kwargs",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                    is_kwargs=True,
                )
                check_mkdir(_path_or_buf.parent, **_mkdir_kwargs)
                _path_or_buf = str(_path_or_buf)
            _engine = self.resolve_key_arg(
                ParquetData.resolve_custom_setting(engine, "engine"),
                k,
                "engine",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            if _engine == "pyarrow":
                assert_can_import("pyarrow")
            elif _engine == "fastparquet":
                assert_can_import("fastparquet")
            elif _engine == "auto":
                assert_can_import_any("pyarrow", "fastparquet")
            else:
                raise ValueError(f"Invalid engine: '{_engine}'")
            if isinstance(v, pd.Series):
                v = v.to_frame()
            if _partition_by is not None:
                _period_index_to = self.resolve_key_arg(
                    period_index_to,
                    k,
                    "period_index_to",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                )
                _groupby_kwargs = self.resolve_key_arg(
                    groupby_kwargs,
                    k,
                    "groupby_kwargs",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                    is_kwargs=True,
                )
                _keep_groupby_names = self.resolve_key_arg(
                    keep_groupby_names,
                    k,
                    "keep_groupby_names",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                )
                v = v.copy(deep=False)
                grouper = self.wrapper.get_index_grouper(_partition_by, **_groupby_kwargs)
                partition_index = grouper.get_stretched_index()
                _partition_cols = []

                def _convert_period_index(index):
                    if _period_index_to == "str":
                        return index.map(str)
                    return index.to_timestamp(how=_period_index_to)

                if isinstance(partition_index, pd.MultiIndex):
                    for i in range(partition_index.nlevels):
                        partition_level = partition_index.get_level_values(i)
                        if _keep_groupby_names:
                            new_column_name = partition_level.name
                        else:
                            new_column_name = f"group_{i}"
                        if isinstance(partition_level, pd.PeriodIndex):
                            partition_level = _convert_period_index(partition_level)
                        v[new_column_name] = partition_level
                        _partition_cols.append(new_column_name)
                else:
                    if _keep_groupby_names:
                        new_column_name = partition_index.name
                    else:
                        new_column_name = "group"
                    if isinstance(partition_index, pd.PeriodIndex):
                        partition_index = _convert_period_index(partition_index)
                    v[new_column_name] = partition_index
                    _partition_cols.append(new_column_name)
            _kwargs = self.select_key_kwargs(k, kwargs, check_dict_type=check_dict_type)
            meta[k] = {"path": _path_or_buf, "partition_cols": _partition_cols, "engine": _engine, **_kwargs}
            v.to_parquet(**meta[k])

        if return_meta:
            return meta
        return None

    @classmethod
    def from_parquet(cls: tp.Type[DataT], *args, fetch_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Load data from Parquet and convert the object to the specified class using
        `vectorbtpro.data.custom.parquet.ParquetData`.

        Args:
            *args: Positional arguments for `vectorbtpro.data.custom.parquet.ParquetData.pull`.
            fetch_kwargs (KwargsLike): Keyword arguments originally used for fetching data.

                Refer to the fetching function for further details.
            **kwargs: Keyword arguments for `vectorbtpro.data.custom.parquet.ParquetData.pull`.

        Returns:
            Data: Instance of `cls` with data loaded from Parquet.
        """
        from vectorbtpro.data.custom.parquet import ParquetData

        if fetch_kwargs is None:
            fetch_kwargs = {}
        data = ParquetData.pull(*args, **kwargs)
        data = data.switch_class(cls, clear_fetch_kwargs=True, clear_returned_kwargs=True)
        data = data.update_fetch_kwargs(**fetch_kwargs)
        return data

    def to_sql(
        self,
        engine: tp.Union[None, str, EngineT, feature_dict, symbol_dict, CustomTemplate] = None,
        table: tp.Union[None, str, feature_dict, symbol_dict, CustomTemplate] = None,
        schema: tp.Union[None, str, feature_dict, symbol_dict, CustomTemplate] = None,
        to_utc: tp.Union[None, bool, str, tp.Sequence[str], feature_dict, symbol_dict, CustomTemplate] = None,
        remove_utc_tz: tp.Union[bool, feature_dict, symbol_dict, CustomTemplate] = True,
        attach_row_number: tp.Union[bool, feature_dict, symbol_dict, CustomTemplate] = False,
        from_row_number: tp.Union[None, int, feature_dict, symbol_dict, CustomTemplate] = None,
        row_number_column: tp.Union[None, str, feature_dict, symbol_dict, CustomTemplate] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        check_dict_type: bool = True,
        template_context: tp.KwargsLike = None,
        return_meta: bool = False,
        return_engine: bool = False,
        **kwargs,
    ) -> tp.Union[None, feature_dict, symbol_dict, EngineT]:
        """Save data to a SQL database using SQLAlchemy.

        This method saves each feature or symbol from the data into a separate SQL table using
        the Pandas DataFrame.to_sql method. When `engine` is provided as None or a string,
        it is resolved via `vectorbtpro.data.custom.sql.SQLData.resolve_engine` and may be
        disposed automatically unless overridden by `dispose_engine`.

        Args:
            engine (Union[None, str, EngineT, feature_dict, symbol_dict, CustomTemplate]):
                Database engine instance, a URL string, or a mapping for per-feature/symbol configuration.
            table (Union[None, str, feature_dict, symbol_dict, CustomTemplate]):
                Table name or mapping for assigning table names to each feature or symbol.
            schema (Union[None, str, feature_dict, symbol_dict, CustomTemplate]):
                SQL schema name or mapping; if the schema does not exist, a new one is created.
            to_utc (Union[None, bool, str, Sequence[str], feature_dict, symbol_dict, CustomTemplate]):
                Specifies whether to localize or convert datetime fields to UTC.

                See `Data.prepare_dt`.
            remove_utc_tz (Union[bool, feature_dict, symbol_dict, CustomTemplate]):
                Indicates whether to remove the timezone after converting to UTC.

                See `Data.prepare_dt`.
            attach_row_number (Union[bool, feature_dict, symbol_dict, CustomTemplate]):
                Specifies whether to attach a row number column to the data.
            from_row_number (Union[None, int, feature_dict, symbol_dict, CustomTemplate]):
                Starting row number for numbering if a row number column is attached.
            row_number_column (Union[None, str, feature_dict, symbol_dict, CustomTemplate]):
                Name of the column to use for row numbers.
            engine_config (KwargsLike): Additional configuration for the engine.
            dispose_engine (Optional[bool]): Flag indicating whether to dispose the engine after use.
            check_dict_type (bool): Flag to validate the type of dictionaries.
            template_context (KwargsLike): Additional context for template substitution.
            return_meta (bool): If True, returns metadata for each saved table.
            return_engine (bool): If True, returns the database engine used.
            **kwargs: Keyword arguments for `pd.DataFrame.to_sql`.

        Returns:
            Union[None, feature_dict, symbol_dict, Engine]: Dictionary with metadata if
                `return_meta` is True and/or engine if `return_engine` is True, or None.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from vectorbtpro.data.custom.sql import SQLData

        if engine_config is None:
            engine_config = {}
        if (engine is None or isinstance(engine, str)) and not self.has_key_dict(engine_config):
            engine_meta = SQLData.resolve_engine(
                engine=engine,
                return_meta=True,
                **engine_config,
            )
            engine = engine_meta["engine"]
            engine_name = engine_meta["engine_name"]
            should_dispose = engine_meta["should_dispose"]
            if dispose_engine is None:
                if return_meta or return_engine:
                    dispose_engine = False
                else:
                    dispose_engine = should_dispose
        else:
            engine_name = None
            if return_engine:
                raise ValueError("Engine can be returned only if URL was provided")

        meta = self.dict_type()
        for k, v in self.data.items():
            if self.feature_oriented:
                _template_context = merge_dicts(dict(data=v, key=k, feature=k), template_context)
            else:
                _template_context = merge_dicts(dict(data=v, key=k, symbol=k), template_context)
            _engine = self.resolve_key_arg(
                engine,
                k,
                "engine",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _engine_config = self.resolve_key_arg(
                engine_config,
                k,
                "engine_config",
                check_dict_type=check_dict_type,
                template_context=_template_context,
                is_kwargs=True,
            )
            if _engine is None or isinstance(_engine, str):
                _engine_meta = SQLData.resolve_engine(
                    engine=_engine,
                    return_meta=True,
                    **_engine_config,
                )
                _engine = _engine_meta["engine"]
                _engine_name = _engine_meta["engine_name"]
                _should_dispose = _engine_meta["should_dispose"]
                if dispose_engine is None:
                    if return_meta or return_engine:
                        _dispose_engine = False
                    else:
                        _dispose_engine = _should_dispose
                else:
                    _dispose_engine = dispose_engine
            else:
                _engine_name = engine_name
                if dispose_engine is None:
                    _dispose_engine = False
                else:
                    _dispose_engine = dispose_engine
            if table is None:
                _table = k
            else:
                _table = self.resolve_key_arg(
                    table,
                    k,
                    "table",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                )
            _schema = self.resolve_key_arg(
                SQLData.resolve_engine_setting(schema, "schema", engine_name=_engine_name),
                k,
                "schema",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _to_utc = self.resolve_key_arg(
                SQLData.resolve_engine_setting(to_utc, "to_utc", engine_name=_engine_name),
                k,
                "to_utc",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _remove_utc_tz = self.resolve_key_arg(
                remove_utc_tz,
                k,
                "remove_utc_tz",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _attach_row_number = self.resolve_key_arg(
                attach_row_number,
                k,
                "attach_row_number",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _from_row_number = self.resolve_key_arg(
                from_row_number,
                k,
                "from_row_number",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _row_number_column = self.resolve_key_arg(
                SQLData.resolve_engine_setting(row_number_column, "row_number_column", engine_name=_engine_name),
                k,
                "row_number_column",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            v = SQLData.prepare_dt(v, to_utc=_to_utc, remove_utc_tz=_remove_utc_tz, parse_dates=False)
            _kwargs = self.select_key_kwargs(k, kwargs, check_dict_type=check_dict_type)
            if _attach_row_number:
                v = v.copy(deep=False)
                if isinstance(v, pd.Series):
                    v = v.to_frame()
                if _from_row_number is None:
                    if not SQLData.has_table(_table, schema=_schema, engine=_engine):
                        _from_row_number = 0
                    elif _kwargs.get("if_exists", "fail") != "append":
                        _from_row_number = 0
                    else:
                        last_row_number = SQLData.get_last_row_number(
                            _table,
                            schema=_schema,
                            row_number_column=_row_number_column,
                            engine=_engine,
                        )
                        _from_row_number = last_row_number + 1
                v[_row_number_column] = np.arange(_from_row_number, _from_row_number + len(v.index))
            if _schema is not None:
                SQLData.create_schema(_schema, engine=_engine)
            meta[k] = {"name": _table, "con": _engine, "schema": _schema, **_kwargs}
            v.to_sql(**meta[k])
            if _dispose_engine:
                _engine.dispose()

        if return_meta:
            return meta
        if return_engine:
            return engine
        return None

    @classmethod
    def from_sql(cls: tp.Type[DataT], *args, fetch_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Load data from a SQL database and convert it to the current class.

        This method fetches data using `vectorbtpro.data.custom.sql.SQLData.pull` and
        then switches the loaded data to the current class. The `fetch_kwargs` are applied
        afterward to update fetch parameters.

        Args:
            *args: Positional arguments for `vectorbtpro.data.custom.sql.SQLData.pull`.
            fetch_kwargs (KwargsLike): Keyword arguments originally used for fetching data.

                Refer to the fetching function for further details.
            **kwargs: Keyword arguments for `vectorbtpro.data.custom.sql.SQLData.pull`.

        Returns:
            Data: Instance of the current class containing the loaded data.
        """
        from vectorbtpro.data.custom.sql import SQLData

        if fetch_kwargs is None:
            fetch_kwargs = {}
        data = SQLData.pull(*args, **kwargs)
        data = data.switch_class(cls, clear_fetch_kwargs=True, clear_returned_kwargs=True)
        data = data.update_fetch_kwargs(**fetch_kwargs)
        return data

    def to_duckdb(
        self,
        connection: tp.Union[None, str, DuckDBPyConnectionT, feature_dict, symbol_dict, CustomTemplate] = None,
        table: tp.Union[None, str, feature_dict, symbol_dict, CustomTemplate] = None,
        schema: tp.Union[None, str, feature_dict, symbol_dict, CustomTemplate] = None,
        catalog: tp.Union[None, str, feature_dict, symbol_dict, CustomTemplate] = None,
        write_format: tp.Union[None, str, feature_dict, symbol_dict, CustomTemplate] = None,
        write_path: tp.Union[tp.PathLike, feature_dict, symbol_dict, CustomTemplate] = ".",
        write_options: tp.Union[None, str, dict, feature_dict, symbol_dict, CustomTemplate] = None,
        mkdir_kwargs: tp.Union[tp.KwargsLike, feature_dict, symbol_dict, CustomTemplate] = None,
        to_utc: tp.Union[None, bool, str, tp.Sequence[str], feature_dict, symbol_dict, CustomTemplate] = None,
        remove_utc_tz: tp.Union[bool, feature_dict, symbol_dict, CustomTemplate] = True,
        if_exists: tp.Union[str, feature_dict, symbol_dict, CustomTemplate] = "fail",
        connection_config: tp.KwargsLike = None,
        check_dict_type: bool = True,
        template_context: tp.KwargsLike = None,
        return_meta: bool = False,
        return_connection: bool = False,
    ) -> tp.Union[None, feature_dict, symbol_dict, DuckDBPyConnectionT]:
        """Save data to a DuckDB database.

        This method saves the internal data to a DuckDB database. Each feature or symbol in the data
        dictionary is processed individually. The target for saving can be either a database table
        or a file, depending on the provided arguments.

        If `connection` is None or a string, the method resolves a connection using
        `vectorbtpro.data.custom.duckdb.DuckDBData.resolve_connection`. The connection is not
        disposed automatically if either `return_meta` or `return_connection` is True.

        When `write_format` is not provided and `write_path` specifies a directory, each feature
        or symbol is persisted to its own table (see https://duckdb.org/docs/guides/python/import_pandas).
        Specifying `catalog` or `schema` sets the default context for the connection, and a new schema
        is created if the given `schema` does not exist.

        The `if_exists` argument controls the behavior when a table with the same name already exists:

        * fail: Raise an error.
        * replace: Drop the existing table.
        * append: Append the new data to the existing table.

        If `write_format` is specified (as "csv", "parquet", or "json") and `write_path` is a
        directory or a path without a file suffix, each feature or symbol is saved to a separate file
        with the appropriate format extension. Data is written using the `COPY` command, and options
        for that command can be provided either as a dictionary or a preformatted string.

        For datetime handling, the parameters `to_utc` and `remove_utc_tz` are passed to `Data.prepare_dt`.

        Args:
            connection (Union[None, str, DuckDBPyConnection, feature_dict, symbol_dict, CustomTemplate]):
                Database connection string or instance.

                If None or a string, a connection is resolved via
                `vectorbtpro.data.custom.duckdb.DuckDBData.resolve_connection`.
            table (Union[None, str, feature_dict, symbol_dict, CustomTemplate]):
                Name of the target table.

                If not provided, the key from the data dictionary is used.
            schema (Union[None, str, feature_dict, symbol_dict, CustomTemplate]):
                Name of the schema to use.

                A new schema is created if it does not exist.
            catalog (Union[None, str, feature_dict, symbol_dict, CustomTemplate]):
                Name of the catalog to set as default for the connection.
            write_format (Union[None, str, feature_dict, symbol_dict, CustomTemplate]):
                Format for writing data to a file.

                Must be "csv", "parquet", or "json" if specified.
            write_path (Union[PathLike, feature_dict, symbol_dict, CustomTemplate]):
                File path or directory where data is saved.

                If a directory or a path without a file suffix, each feature or symbol is saved
                to a separate file with the format extension.
            write_options (Union[None, str, dict, feature_dict, symbol_dict, CustomTemplate]):
                Options to pass to the `COPY` command.

                Provide these as a dictionary or a preformatted string.
            mkdir_kwargs (Union[KwargsLike, feature_dict, symbol_dict, CustomTemplate]):
                Keyword arguments for directory creation.

                See `vectorbtpro.utils.path_.check_mkdir`.
            to_utc (Union[None, bool, str, Sequence[str], feature_dict, symbol_dict, CustomTemplate]):
                Specifies whether to localize or convert datetime fields to UTC.

                See `Data.prepare_dt`.
            remove_utc_tz (Union[bool, feature_dict, symbol_dict, CustomTemplate]):
                Indicates whether to remove the timezone after converting to UTC.

                See `Data.prepare_dt`.
            if_exists (Union[str, feature_dict, symbol_dict, CustomTemplate]):
                Action to take if the target table already exists.

                Options are "fail", "replace", or "append".
            connection_config (KwargsLike): Configuration parameters for creating a database connection.
            check_dict_type (bool): Flag to validate the type of dictionaries.
            template_context (KwargsLike): Additional context for template substitution.
            return_meta (bool): If True, returns metadata including all processed arguments
                per feature or symbol.
            return_connection (bool): If True and a connection string was provided, returns the
                resolved DuckDB connection instead of disposing it automatically.

        Returns:
            Union[feature_dict, symbol_dict, DuckDBPyConnection, None]: Metadata dictionary if
                `return_meta` is True. The resolved DuckDB connection is returned if
                `return_connection` is True, otherwise None.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("duckdb")
        from vectorbtpro.data.custom.duckdb import DuckDBData

        if connection_config is None:
            connection_config = {}
        if (connection is None or isinstance(connection, (str, Path))) and not self.has_key_dict(connection_config):
            connection_meta = DuckDBData.resolve_connection(
                connection=connection,
                read_only=False,
                return_meta=True,
                **connection_config,
            )
            connection = connection_meta["connection"]
            if return_meta or return_connection:
                should_close = False
            else:
                should_close = connection_meta["should_close"]
        elif return_connection:
            raise ValueError("Connection can be returned only if URL was provided")
        else:
            should_close = False

        meta = self.dict_type()
        for k, v in self.data.items():
            if self.feature_oriented:
                _template_context = merge_dicts(dict(data=v, key=k, feature=k), template_context)
            else:
                _template_context = merge_dicts(dict(data=v, key=k, symbol=k), template_context)
            _connection = self.resolve_key_arg(
                connection,
                k,
                "connection",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _connection_config = self.resolve_key_arg(
                connection_config,
                k,
                "connection_config",
                check_dict_type=check_dict_type,
                template_context=_template_context,
                is_kwargs=True,
            )
            if _connection is None or isinstance(_connection, (str, Path)):
                _connection_meta = DuckDBData.resolve_connection(
                    connection=_connection,
                    read_only=False,
                    return_meta=True,
                    **_connection_config,
                )
                _connection = _connection_meta["connection"]
                _should_close = _connection_meta["should_close"]
            else:
                _should_close = False
            if table is None:
                _table = k
            else:
                _table = self.resolve_key_arg(
                    table,
                    k,
                    "table",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                )
            _schema = self.resolve_key_arg(
                DuckDBData.resolve_custom_setting(schema, "schema"),
                k,
                "schema",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _catalog = self.resolve_key_arg(
                DuckDBData.resolve_custom_setting(catalog, "catalog"),
                k,
                "catalog",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _write_format = self.resolve_key_arg(
                write_format,
                k,
                "write_format",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _write_path = self.resolve_key_arg(
                write_path,
                k,
                "write_path",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _write_path = Path(_write_path)
            is_not_file = (_write_path.exists() and _write_path.is_dir()) or _write_path.suffix == ""
            if _write_format is not None and is_not_file:
                if _write_format.upper() == "CSV":
                    _write_path /= f"{k}.csv"
                elif _write_format.upper() == "PARQUET":
                    _write_path /= f"{k}.parquet"
                elif _write_format.upper() == "JSON":
                    _write_path /= f"{k}.json"
                else:
                    raise ValueError(f"Invalid write format: '{_write_format}'")
            if _write_path.suffix != "":
                _mkdir_kwargs = self.resolve_key_arg(
                    mkdir_kwargs,
                    k,
                    "mkdir_kwargs",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                    is_kwargs=True,
                )
                check_mkdir(_write_path.parent, **_mkdir_kwargs)
                _write_path = str(_write_path)
                use_write = True
            else:
                use_write = False
            _to_utc = self.resolve_key_arg(
                DuckDBData.resolve_custom_setting(to_utc, "to_utc"),
                k,
                "to_utc",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _remove_utc_tz = self.resolve_key_arg(
                remove_utc_tz,
                k,
                "remove_utc_tz",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            _if_exists = self.resolve_key_arg(
                if_exists,
                k,
                "if_exists",
                check_dict_type=check_dict_type,
                template_context=_template_context,
            )
            v = DuckDBData.prepare_dt(v, to_utc=_to_utc, remove_utc_tz=_remove_utc_tz, parse_dates=False)
            v = v.reset_index()
            if use_write:
                _write_options = self.resolve_key_arg(
                    write_options,
                    k,
                    "write_options",
                    check_dict_type=check_dict_type,
                    template_context=_template_context,
                    is_kwargs=isinstance(write_options, dict),
                )
                if _write_options is not None:
                    _write_options = DuckDBData.format_write_options(_write_options)
                if _write_format is not None and _write_options is not None and "FORMAT" not in _write_options:
                    _write_options = f"FORMAT {_write_format.upper()}, " + _write_options
                elif _write_format is not None and _write_options is None:
                    _write_options = f"FORMAT {_write_format.upper()}"
                _connection.register("_" + k, v)
                if _write_options is not None:
                    _connection.sql(f"COPY (SELECT * FROM \"_{k}\") TO '{_write_path}' ({_write_options})")
                else:
                    _connection.sql(f"COPY (SELECT * FROM \"_{k}\") TO '{_write_path}'")
                meta[k] = {"write_path": _write_path, "write_options": _write_options}
            else:
                if _catalog is not None:
                    _connection.sql(f"USE {_catalog}")
                elif _schema is not None:
                    catalogs = DuckDBData.list_catalogs(connection=_connection)
                    if len(catalogs) > 1:
                        raise ValueError("Please select a catalog")
                    _catalog = catalogs[0]
                    _connection.sql(f"USE {_catalog}")
                if _schema is not None:
                    _connection.sql(f'CREATE SCHEMA IF NOT EXISTS "{_schema}"')
                    _connection.sql(f"USE {_catalog}.{_schema}")
                append = False
                if _table in DuckDBData.list_tables(catalog=_catalog, schema=_schema, connection=_connection):
                    if _if_exists.lower() == "fail":
                        raise ValueError(f"Table '{_table}' already exists")
                    elif _if_exists.lower() == "replace":
                        _connection.sql(f'DROP TABLE "{_table}"')
                    elif _if_exists.lower() == "append":
                        append = True
                _connection.register("_" + k, v)
                if append:
                    _connection.sql(f'INSERT INTO "{_table}" SELECT * FROM "_{k}"')
                else:
                    _connection.sql(f'CREATE TABLE "{_table}" AS SELECT * FROM "_{k}"')
                meta[k] = {"table": _table, "schema": _schema, "catalog": _catalog}
                if _should_close:
                    _connection.close()

        if should_close:
            connection.close()
        if return_meta:
            return meta
        if return_connection:
            return connection
        return None

    @classmethod
    def from_duckdb(cls: tp.Type[DataT], *args, fetch_kwargs: tp.KwargsLike = None, **kwargs) -> DataT:
        """Load data from a DuckDB database using `vectorbtpro.data.custom.duckdb.DuckDBData` and
        switch the object's class.

        Args:
            *args: Positional arguments for `vectorbtpro.data.custom.duckdb.DuckDBData.pull`.
            fetch_kwargs (KwargsLike): Keyword arguments originally used for fetching data.

                Refer to the fetching function for further details.
            **kwargs: Keyword arguments for `vectorbtpro.data.custom.duckdb.DuckDBData.pull`.

        Returns:
            Data: Instance of the data with updated fetch parameters.
        """
        from vectorbtpro.data.custom.duckdb import DuckDBData

        if fetch_kwargs is None:
            fetch_kwargs = {}
        data = DuckDBData.pull(*args, **kwargs)
        data = data.switch_class(cls, clear_fetch_kwargs=True, clear_returned_kwargs=True)
        data = data.update_fetch_kwargs(**fetch_kwargs)
        return data

    # ############# Querying ############# #

    def sql(
        self,
        query: str,
        dbcon: tp.Optional[DuckDBPyConnectionT] = None,
        database: str = ":memory:",
        db_config: tp.KwargsLike = None,
        alias: str = "",
        params: tp.KwargsLike = None,
        other_objs: tp.Optional[dict] = None,
        date_as_object: bool = False,
        align_dtypes: bool = True,
        squeeze: bool = True,
        **kwargs,
    ) -> tp.SeriesFrame:
        """Execute a SQL query on the current instance using DuckDB.

        Establish a database connection, register the instance's data (retrieved via `Data.get`)
        along with any additional objects, and execute the specified SQL query. Optionally,
        align the data types with the original columns and adjust the index. If `squeeze` is True,
        convert a one-column DataFrame into a Series.

        Args:
            query (str): SQL query to execute.
            dbcon (Optional[DuckDBPyConnection]): DuckDB connection to use; if not provided,
                a new connection is established.
            database (str): Identifier or path of the target database.
            db_config (KwargsLike): Configuration parameters for establishing the DuckDB connection.
            alias (str): Alias assigned to the query result.
            params (KwargsLike): Parameters to substitute in the SQL query.
            other_objs (Optional[dict]): Additional objects to register within the database.
            date_as_object (bool): Whether to return date columns as objects.
            align_dtypes (bool): Whether to align result column data types with the original data.
            squeeze (bool): Flag indicating whether to convert a single-column DataFrame to a Series.
            **kwargs: Keyword arguments for `Data.get`.

        Returns:
            SeriesFrame: Query result as a DataFrame or a Series if squeezed.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("duckdb")
        from duckdb import connect

        if db_config is None:
            db_config = {}
        if dbcon is None:
            dbcon = connect(database=database, read_only=False, config=db_config)
        if params is None:
            params = {}

        dtypes = {}
        objs = self.get(**kwargs, as_dict=True)
        for k, v in objs.items():
            if not checks.is_default_index(v.index):
                v = v.reset_index()
            if isinstance(v, pd.Series):
                v = v.to_frame()
            for c in v.columns:
                dtypes[c] = v[c].dtype
            dbcon.register(k, v)
        if other_objs is not None:
            checks.assert_instance_of(other_objs, dict, arg_name="other_objs")
            for k, v in other_objs.items():
                if not checks.is_default_index(v.index):
                    v = v.reset_index()
                if isinstance(v, pd.Series):
                    v = v.to_frame()
                for c in v.columns:
                    dtypes[c] = v[c].dtype
                dbcon.register(k, v)
        df = dbcon.sql(query, alias=alias, params=params).df(date_as_object=date_as_object)
        if align_dtypes:
            for c in df.columns:
                if c in dtypes:
                    df[c] = df[c].astype(dtypes[c])
        if isinstance(self.index, pd.MultiIndex):
            if set(self.index.names) <= set(df.columns):
                df = df.set_index(self.index.names)
        else:
            if self.index.name is not None and self.index.name in df.columns:
                df = df.set_index(self.index.name)
            elif "index" in df.columns:
                df = df.set_index("index")
                df.index.name = None
        if squeeze and len(df.columns) == 1:
            df = df.iloc[:, 0]
        return df

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Default configuration for `Data.stats`.

        Merges the defaults from `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats_defaults`
        with the `stats` configuration from `vectorbtpro._settings.data`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the stats builder.
        """
        return merge_dicts(Analyzable.stats_defaults.__get__(self), self.get_base_settings()["stats"])

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start_index=dict(
                title="Start Index",
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags="wrapper",
            ),
            end_index=dict(
                title="End Index",
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags="wrapper",
            ),
            total_duration=dict(
                title="Total Duration",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            total_features=dict(
                title="Total Features",
                check_is_feature_oriented=True,
                calc_func=lambda self: len(self.features),
                agg_func=None,
                tags="data",
            ),
            total_symbols=dict(
                title="Total Symbols",
                check_is_symbol_oriented=True,
                calc_func=lambda self: len(self.symbols),
                tags="data",
            ),
            null_counts=dict(
                title="Null Counts",
                calc_func=lambda self, group_by: {
                    k: v.isnull().vbt(wrapper=self.wrapper).sum(group_by=group_by) for k, v in self.data.items()
                },
                agg_func=lambda x: x.sum(),
                tags="data",
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot(
        self,
        column: tp.Optional[tp.Column] = None,
        feature: tp.Optional[tp.Feature] = None,
        symbol: tp.Optional[tp.Symbol] = None,
        feature_map: tp.KwargsLike = None,
        plot_volume: tp.Optional[bool] = None,
        base: tp.Optional[float] = None,
        **kwargs,
    ) -> tp.Union[tp.BaseFigure, tp.TraceUpdater]:
        """Plot one feature across multiple symbols or generate an OHLC(V) chart for a single symbol.

        Args:
            column (Column): Column identifier.

                Used depending on the data orientation.
            feature (Feature): Feature identifier.
            symbol (Symbol): Symbol identifier.
            feature_map (KwargsLike): Dictionary mapping feature names to OHLCV components.

                Applied only when generating an OHLC(V) chart.
            plot_volume (bool): Whether to plot volume below.

                Applied only when generating an OHLC(V) chart.
            base (float): Initial base value for rebasing the feature series.

                !!! note
                    The feature must contain price data.

                    Applied only when plotting lines.
            **kwargs: Keyword arguments for
                `vectorbtpro.generic.accessors.GenericAccessor.plot` for lines and to
                `vectorbtpro.ohlcv.accessors.OHLCVDFAccessor.plot` for OHLC(V).

        Returns:
            Union[BaseFigure, TraceUpdater]: Plot figure or trace updater instance.

        Examples:
            Plot the lines of one feature across all symbols:

            ```pycon
            >>> from vectorbtpro import *

            >>> start = '2021-01-01 UTC'  # crypto is in UTC
            >>> end = '2021-06-01 UTC'
            >>> data = vbt.YFData.pull(['BTC-USD', 'ETH-USD', 'ADA-USD'], start=start, end=end)
            ```

            [=100% "100%"]{: .candystripe .candystripe-animate }

            ```pycon
            >>> data.plot(feature='Close', base=1).show()
            ```

            Plot OHLC(V) of one symbol (only if data contains the respective features):

            ![](/assets/images/api/data_plot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/data_plot.dark.svg#only-dark){: .iimg loading=lazy }

            ```pycon
            >>> data.plot(symbol='BTC-USD').show()
            ```

            ![](/assets/images/api/data_plot_ohlcv.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/data_plot_ohlcv.dark.svg#only-dark){: .iimg loading=lazy }
        """
        if column is not None:
            if self.feature_oriented:
                if symbol is not None:
                    raise ValueError("Either column or symbol can be provided, not both")
                symbol = column
            else:
                if feature is not None:
                    raise ValueError("Either column or feature can be provided, not both")
                feature = column
        if feature is None and self.has_ohlc:
            data = self.get(symbols=symbol, squeeze_symbols=True)
            if isinstance(data, tuple):
                raise ValueError("Cannot plot OHLC of multiple symbols. Select one symbol.")
            return data.vbt.ohlcv(feature_map=feature_map).plot(plot_volume=plot_volume, **kwargs)
        data = self.get(features=feature, symbols=symbol, squeeze_features=True, squeeze_symbols=True)
        if isinstance(data, tuple):
            raise ValueError("Cannot plot multiple features and symbols. Select one feature or symbol.")
        if base is not None:
            data = data.vbt.rebase(base)
        return data.vbt.lineplot(**kwargs)

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Default configuration for `Data.plots`.

        Merges the defaults from `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots_defaults`
        with the `plots` configuration from `vectorbtpro._settings.data`.

        Returns:
            Kwargs: Dictionary containing the default configuration for the plots builder.
        """
        return merge_dicts(Analyzable.plots_defaults.__get__(self), self.get_base_settings()["plots"])

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=RepEval(
                """
                if symbols is None:
                    symbols = self.symbols
                if not self.has_multiple_keys(symbols):
                    symbols = [symbols]
                [
                    dict(
                        check_is_not_grouped=True,
                        plot_func="plot",
                        plot_volume=False,
                        symbol=s,
                        title=s,
                        pass_add_trace_kwargs=True,
                        xaxis_kwargs=dict(rangeslider_visible=False, showgrid=True),
                        yaxis_kwargs=dict(showgrid=True),
                        tags="data",
                    )
                    for s in symbols
                ]""",
                context=dict(symbols=None),
            )
        ),
    )

    @property
    def subplots(self) -> Config:
        return self._subplots

    @classmethod
    def build_feature_config_doc(cls, source_cls: tp.Optional[type] = None) -> str:
        """Build and return the documentation for the feature configuration.

        Args:
            source_cls (Optional[type]): Source class providing the original configuration.

        Returns:
            str: Formatted feature configuration documentation.
        """
        if source_cls is None:
            source_cls = Data
        return string.Template(inspect.cleandoc(get_dict_attr(source_cls, "feature_config").__doc__)).substitute(
            {"feature_config": cls.feature_config.prettify_doc(), "cls_name": cls.__name__},
        )

    @classmethod
    def override_feature_config_doc(cls, __pdoc__: dict, source_cls: tp.Optional[type] = None) -> None:
        """Override the feature configuration documentation for the subclass.

        Args:
            __pdoc__ (dict): Dictionary mapping objects to their documentation strings.
            source_cls (Optional[type]): Source class providing the original configuration.

        Returns:
            None
        """
        __pdoc__[cls.__name__ + ".feature_config"] = cls.build_feature_config_doc(source_cls=source_cls)


Data.override_feature_config_doc(__pdoc__)
Data.override_metrics_doc(__pdoc__)
Data.override_subplots_doc(__pdoc__)
