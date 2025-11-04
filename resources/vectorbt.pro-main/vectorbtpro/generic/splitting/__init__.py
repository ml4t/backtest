# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Package providing utilities for splitting."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.generic.splitting.base import *
    from vectorbtpro.generic.splitting.decorators import *
    from vectorbtpro.generic.splitting.nb import *
    from vectorbtpro.generic.splitting.purged import *
    from vectorbtpro.generic.splitting.sklearn_ import *

__import_if_installed__ = dict()
__import_if_installed__["sklearn_"] = "sklearn"
