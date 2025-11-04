# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `SyntheticData` class for generating synthetic data."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.custom import CustomData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "SyntheticData",
]

__pdoc__ = {}


class SyntheticData(CustomData):
    """Data class for generating synthetic data.

    Provides a framework for generating and updating synthetic data for features and symbols.
    Subclasses should implement the `generate_key` method, which is used by other class methods.

    See:
        * `SyntheticData.fetch_key` for argument details.

    !!! info
        For default settings, see `custom.synthetic` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.synthetic")

    @classmethod
    def generate_key(cls, key: tp.Key, index: tp.Index, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        """Abstract method to generate synthetic data for a given key.

        Args:
            key (Key): Feature or symbol identifier.
            index (Index): Datetime index over which data will be generated.
            key_is_feature (bool): Flag indicating whether the key represents a feature.
            **kwargs: Additional keyword arguments.

        Returns:
            KeyData: Generated data and a metadata dictionary.

        !!! abstract
            This method should be overridden in a subclass.
        """
        raise NotImplementedError

    @classmethod
    def generate_feature(cls, feature: tp.Feature, index: tp.Index, **kwargs) -> tp.FeatureData:
        """Abstract method to generate synthetic data for a feature.

        Calls `SyntheticData.generate_key` with `key_is_feature=True`.

        Args:
            feature (Feature): Feature identifier.
            index (Index): Datetime index over which synthetic data is generated.
            **kwargs: Keyword arguments for `SyntheticData.generate_key`.

        Returns:
            FeatureData: Generated data and a metadata dictionary.
        """
        return cls.generate_key(feature, index, key_is_feature=True, **kwargs)

    @classmethod
    def generate_symbol(cls, symbol: tp.Symbol, index: tp.Index, **kwargs) -> tp.SymbolData:
        """Abstract method to generate synthetic data for a symbol.

        Calls `SyntheticData.generate_key` with `key_is_feature=False`.

        Args:
            symbol (Symbol): Symbol identifier.
            index (Index): Datetime index over which synthetic data is generated.
            **kwargs: Keyword arguments for `SyntheticData.generate_key`.

        Returns:
            SymbolData: Generated data and a metadata dictionary.
        """
        return cls.generate_key(symbol, index, key_is_feature=False, **kwargs)

    @classmethod
    def fetch_key(
        cls,
        key: tp.Symbol,
        key_is_feature: bool = False,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        periods: tp.Optional[int] = None,
        timeframe: tp.Optional[tp.FrequencyLike] = None,
        tz: tp.TimezoneLike = None,
        normalize: tp.Optional[bool] = None,
        inclusive: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.KeyData:
        """Generate synthetic data for a given key (feature or symbol).

        Generates a datetime index using `vectorbtpro.utils.datetime_.date_range`.

        Args:
            key (Symbol): Identifier of the feature or symbol.
            key_is_feature (bool): Flag indicating whether the key represents a feature.
            start (Optional[DatetimeLike]): Start datetime (e.g., "2024-01-01", "1 year ago").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            end (Optional[DatetimeLike]): End datetime (e.g., "2025-01-01", "now").

                See `vectorbtpro.utils.datetime_.to_timestamp`.
            periods (Optional[int]): Number of periods for the datetime index.
            timeframe (Optional[FrequencyLike]): Timeframe specification (e.g., "daily", "15 minutes").

                See `vectorbtpro.utils.datetime_.to_freq`.
            tz (TimezoneLike): Timezone specification (e.g., "UTC", "America/New_York").

                See `vectorbtpro.utils.datetime_.to_timezone`.
            normalize (Optional[bool]): If True, normalizes the datetime index.
            inclusive (Optional[str]): Inclusivity setting for the datetime range.
            **kwargs: Keyword arguments for `SyntheticData.generate_feature` or
                `SyntheticData.generate_symbol`.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        normalize = cls.resolve_custom_setting(normalize, "normalize")
        inclusive = cls.resolve_custom_setting(inclusive, "inclusive")

        index = dt.date_range(
            start=start,
            end=end,
            periods=periods,
            freq=timeframe,
            normalize=normalize,
            inclusive=inclusive,
        )
        if tz is None:
            tz = index.tz
        if len(index) == 0:
            raise ValueError("Date range is empty")
        if key_is_feature:
            return cls.generate_feature(key, index, **kwargs), dict(tz=tz, freq=timeframe)
        return cls.generate_symbol(key, index, **kwargs), dict(tz=tz, freq=timeframe)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Generate synthetic data for a feature.

        Calls `SyntheticData.fetch_key` with `key_is_feature=True`.

        Args:
            feature (Feature): Feature identifier.
            **kwargs: Keyword arguments for `SyntheticData.fetch_key`.

        Returns:
            FeatureData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(feature, key_is_feature=True, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Generate synthetic data for a symbol.

        Calls `SyntheticData.fetch_key` with `key_is_feature=False`.

        Args:
            symbol (Symbol): Symbol identifier.
            **kwargs: Keyword arguments for `SyntheticData.fetch_key`.

        Returns:
            SymbolData: Fetched data and a metadata dictionary.
        """
        return cls.fetch_key(symbol, key_is_feature=False, **kwargs)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        """Update synthetic data for a given key (feature or symbol).

        Fetches the latest start datetime from the object's settings, merges it with additional arguments,
        and updates the synthetic data.

        Args:
            key (Key): Feature or symbol identifier.
            key_is_feature (bool): Flag indicating whether the key represents a feature.
            **kwargs: Keyword arguments for `SyntheticData.fetch_feature` or `SyntheticData.fetch_symbol`.

        Returns:
            KeyData: Updated data and a metadata dictionary.
        """
        fetch_kwargs = self.select_fetch_kwargs(key)
        fetch_kwargs["start"] = self.select_last_index(key)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if key_is_feature:
            return self.fetch_feature(key, **kwargs)
        return self.fetch_symbol(key, **kwargs)

    def update_feature(self, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        return self.update_key(feature, key_is_feature=True, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        return self.update_key(symbol, key_is_feature=False, **kwargs)
