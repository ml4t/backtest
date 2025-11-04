# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package providing classes and utilities for portfolio optimization.

!!! info
    For default settings, see `vectorbtpro._settings.pfopt`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.portfolio.pfopt.base import *
    from vectorbtpro.portfolio.pfopt.nb import *
    from vectorbtpro.portfolio.pfopt.records import *
