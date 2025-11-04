# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `CustomData` class for accessing custom data using a dedicated custom settings path."""

import fnmatch
import re

from vectorbtpro import _typing as tp
from vectorbtpro.data.base import Data

__all__ = [
    "CustomData",
]

__pdoc__ = {}


class CustomData(Data):
    """Data class for accessing custom data using a dedicated custom settings path.

    !!! info
        For default settings, see `custom` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom=None)

    @classmethod
    def get_custom_settings(cls, *args, **kwargs) -> dict:
        """Return custom settings by calling `CustomData.get_settings` with `path_id="custom"`.

        Args:
            *args: Positional arguments for `CustomData.get_settings`.
            **kwargs: Keyword arguments for `CustomData.get_settings`.

        Returns:
            dict: Custom settings.
        """
        return cls.get_settings(*args, path_id="custom", **kwargs)

    @classmethod
    def has_custom_settings(cls, *args, **kwargs) -> bool:
        """Return whether custom settings exist by calling `CustomData.has_settings` with `path_id="custom"`.

        Args:
            *args: Positional arguments for `CustomData.has_settings`.
            **kwargs: Keyword arguments for `CustomData.has_settings`.

        Returns:
            bool: True if custom settings exist, False otherwise.
        """
        return cls.has_settings(*args, path_id="custom", **kwargs)

    @classmethod
    def get_custom_setting(cls, *args, **kwargs) -> tp.Any:
        """Return a custom setting by calling `CustomData.get_setting` with `path_id="custom"`.

        Args:
            *args: Positional arguments for `CustomData.get_setting`.
            **kwargs: Keyword arguments for `CustomData.get_setting`.

        Returns:
            Any: Requested custom setting.
        """
        return cls.get_setting(*args, path_id="custom", **kwargs)

    @classmethod
    def has_custom_setting(cls, *args, **kwargs) -> bool:
        """Return whether a custom setting exists by calling `CustomData.has_setting` with `path_id="custom"`.

        Args:
            *args: Positional arguments for `CustomData.has_setting`.
            **kwargs: Keyword arguments for `CustomData.has_setting`.

        Returns:
            bool: True if the custom setting exists, False otherwise.
        """
        return cls.has_setting(*args, path_id="custom", **kwargs)

    @classmethod
    def resolve_custom_setting(cls, *args, **kwargs) -> tp.Any:
        """Return the resolved custom setting by calling `CustomData.resolve_setting` with `path_id="custom"`.

        Args:
            *args: Positional arguments for `CustomData.resolve_setting`.
            **kwargs: Keyword arguments for `CustomData.resolve_setting`.

        Returns:
            Any: Resolved custom setting.
        """
        return cls.resolve_setting(*args, path_id="custom", **kwargs)

    @classmethod
    def set_custom_settings(cls, *args, **kwargs) -> None:
        """Set custom settings by calling `CustomData.set_settings` with `path_id="custom"`.

        Args:
            *args: Positional arguments for `CustomData.set_settings`.
            **kwargs: Keyword arguments for `CustomData.set_settings`.

        Returns:
            None
        """
        cls.set_settings(*args, path_id="custom", **kwargs)

    @staticmethod
    def key_match(key: str, pattern: str, use_regex: bool = False) -> tp.Optional[re.Match]:
        """Return a match result indicating whether the given key matches the specified pattern.

        If `use_regex` is True, the pattern is interpreted as a regular expression.

        Otherwise, a glob-style pattern is used.

        Args:
            key (str): Key to evaluate.
            pattern (str): Pattern to compare against.
            use_regex (bool): Flag indicating whether the pattern is a regular expression.

        Returns:
            Optional[re.Match]: Match object if the key matches the pattern; otherwise, None.
        """
        if use_regex:
            return re.match(pattern, key)
        return re.match(fnmatch.translate(pattern), key)
