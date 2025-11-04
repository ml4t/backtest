# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package for working with records.

Records provide a secondary data representation in vectorbtpro for efficiently storing sparse
event data—such as drawdowns, orders, trades, and positions—without converting them back to
a matrix form, thereby reducing memory usage.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.records.base import *
    from vectorbtpro.records.chunking import *
    from vectorbtpro.records.col_mapper import *
    from vectorbtpro.records.mapped_array import *
    from vectorbtpro.records.nb import *
