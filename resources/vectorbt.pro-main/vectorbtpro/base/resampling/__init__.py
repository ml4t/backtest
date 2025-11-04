# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package providing classes and utilities for resampling.

!!! info
    For default settings, see `vectorbtpro._settings.resampling`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.base.resampling.base import *
    from vectorbtpro.base.resampling.nb import *
