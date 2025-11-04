# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package providing utilities for working with generic time series.

In contrast to the `vectorbtpro.base` sub-package, this module focuses on data manipulation and analysis.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.generic.accessors import *
    from vectorbtpro.generic.analyzable import *
    from vectorbtpro.generic.decorators import *
    from vectorbtpro.generic.drawdowns import *
    from vectorbtpro.generic.nb import *
    from vectorbtpro.generic.plots_builder import *
    from vectorbtpro.generic.plotting import *
    from vectorbtpro.generic.price_records import *
    from vectorbtpro.generic.ranges import *
    from vectorbtpro.generic.sim_range import *
    from vectorbtpro.generic.splitting import *
    from vectorbtpro.generic.stats_builder import *

__exclude_from__all__ = [
    "enums",
]

__import_if_installed__ = dict()
__import_if_installed__["plotting"] = "plotly"
