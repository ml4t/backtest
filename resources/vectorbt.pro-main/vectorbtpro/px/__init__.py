# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package for plotting with Plotly Express."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.px.accessors import *
    from vectorbtpro.px.decorators import *

__import_if_installed__ = dict()
__import_if_installed__["accessors"] = "plotly"
__import_if_installed__["decorators"] = "plotly"
