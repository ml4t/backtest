# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package for working with returns.

Provides common financial risk and performance metrics modeled after
[empyrical](https://github.com/quantopian/empyrical), an adapter for quantstats,
and additional return-based features.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.returns.accessors import *
    from vectorbtpro.returns.nb import *
    from vectorbtpro.returns.qs_adapter import *

__exclude_from__all__ = [
    "enums",
]

__import_if_installed__ = dict()
__import_if_installed__["qs_adapter"] = "quantstats"
