# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing the `RemoteData` class for fetching remote data."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.custom import CustomData

__all__ = [
    "RemoteData",
]

__pdoc__ = {}


class RemoteData(CustomData):
    """Data class for fetching remote data.

    !!! info
        For default settings, see `custom.remote` in `vectorbtpro._settings.data`.
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.remote")
