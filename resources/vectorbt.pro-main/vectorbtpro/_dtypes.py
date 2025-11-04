# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing default data types.

!!! info
    For default settings, see `vectorbtpro._settings.numpy`.
"""

import numpy as np

from vectorbtpro._settings import settings

__all__ = [
    "int_",
    "float_",
]

__pdoc__ = {}

int_ = settings["numpy"]["int_"]
"""_"""

__pdoc__[int_.__name__] = "Default integer data type retrieved from `vectorbtpro._settings.numpy`."
if np.issubdtype(int_, np.integer):
    __pdoc__[f"{int_.__name__}.bit_count"] = False

float_ = settings["numpy"]["float_"]
"""_"""

__pdoc__[float_.__name__] = (
    "Default floating point data type retrieved from `vectorbtpro._settings.numpy`."
)
if np.issubdtype(float_, np.floating):
    __pdoc__[f"{float_.__name__}.as_integer_ratio"] = False
    __pdoc__[f"{float_.__name__}.is_integer"] = False
